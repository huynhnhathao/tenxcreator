"""
API to use the bark model
"""

import contextlib
import os
import re
import logging
import funcy


from typing_extensions import Annotated, Literal, Optional, Union, Dict, List, Tuple

from pydantic.types import *
from pydantic import validate_call

import numpy as np

import torch
from dataclasses import dataclass


from core.gvar import torch_models, ModelEnum, env
from transformers import BertTokenizer
from core.model import GPT


"""
BARK is a text-to-audio model. It is a combination of 3 smaller models. This module
provides convenient methods to use BARK to generate audio from texts.
"""

SEMANTIC_VOCAB_SIZE = 10_000
CUR_PATH = os.path.dirname(os.path.abspath(__file__))


# for the BERT model to get semantic tokens from raw texts
TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599


class InferenceContext:
    def __init__(self, benchmark=False):
        # we can't expect inputs to be the same length, so disable benchmarking by default
        self._chosen_cudnn_benchmark = benchmark
        self._cudnn_benchmark = None

    def __enter__(self):
        self._cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._chosen_cudnn_benchmark

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.backends.cudnn.benchmark = self._cudnn_benchmark


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


if (
    torch.cuda.is_available()
    and hasattr(torch.cuda, "amp")
    and hasattr(torch.cuda.amp, "autocast")
    and hasattr(torch.cuda, "is_bf16_supported")
    and torch.cuda.is_bf16_supported()
):
    autocast = funcy.partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)
else:

    @contextlib.contextmanager
    def autocast():
        yield


@contextlib.contextmanager
def _inference_mode():
    with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
        yield


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BarkConfig:
    """Configuration for BARK model parameters"""

    context_window_size: int = 1024
    semantic_rate_hz: float = 49.9
    semantic_vocab_size: int = 10_000
    codebook_size: int = 1024
    n_coarse_codebooks: int = 2
    n_fine_codebooks: int = 8
    coarse_rate_hz: float = 75
    sample_rate: int = 24_000


@dataclass
class BarkInferenceParams:
    temp: float = (0.7,)
    top_k: Optional[int] = (None,)
    top_p: Optional[int] = (None,)
    silent: bool = (False,)
    min_eos_p: float = (0.2,)
    max_gen_duration_s: Optional[float] = (None,)
    allow_early_stop: bool = (True,)
    use_kv_caching: bool = (False,)
    device: torch.device = torch.device(env.DEVICE)


@dataclass
class BarkAudioPrompt:
    """
    To create a custom audio prompt to feed in the bark model, you need to have
    all fields in this class
    """

    semantic_prompt: np.ndarray
    coarse_prompt: np.ndarray
    fine_prompt: np.ndarray


# Create a default configuration instance
bark_config = BarkConfig()

# Supported languages for BARK
SUPPORTED_LANGS: List[Tuple[str, str]] = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]


def text_to_semantic(
    text: str,
    prompt_voice_name: Optional[str],
    history_prompt: BarkAudioPrompt = None,
    temp: float = 0.7,
    silent: bool = False,
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        prompt_voice_name: name of the voice from the prompt lib to be used as the audio prompt
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """

    if prompt_voice_name:
        history_prompt = _load_history_prompt(prompt_voice_name)
        semantic_history = history_prompt.semantic_prompt
        assert (
            len(semantic_history.shape) == 1
            and len(semantic_history) > 0
            and semantic_history.min() >= 0
            and semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
        ), "invalid semantic history prompt"

    x_semantic = _text_to_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True,
    )
    return x_semantic


@validate_call
def _text_to_semantic(
    text: str,
    semantic_history: np.ndarray,  # pass in an empty array if you don't have a semantic prompt
    inference_config: BarkInferenceConfig,
):
    """Generate semantic tokens from text.

    Args:

        text: text to generate semantic tokens
        semantic_history: the audio prompt to feed the model, the generated audio will based on this speaker characteristics.
            Pass in an empty array if you don't have one
    """
    text = _preprocess_texts(text)
    assert len(text) > 0, f"invalid input text {text}"
    text_model_info = (
        ModelEnum.BARK_TEXT_SMALL if env.SUNO_USE_SMALL_MODELS else ModelEnum.BARK_TEXT
    )

    text_model = torch_models.get_model(text_model_info)
    assert text_model.model is not None, "text model is None"
    assert text_model.preprocessor is not None, "tokenizer for the text model is None"

    model: GPT = text_model.model
    tokenizer: BertTokenizer = text_model.preprocessor

    tokenized_text = tokenizer.encode(text, add_special_tokens=False)

    encoded_text = np.array(tokenized_text) + TEXT_ENCODING_OFFSET

    # encoded_text's length must be less than 256 as from the original implementation
    encoded_text = trim_or_pad_array(encoded_text, TEXT_PAD_TOKEN, 256)

    if semantic_history is None:
        semantic_history = np.array([])

    semantic_history = trim_or_pad_array(semantic_history, SEMANTIC_PAD_TOKEN, 256)

    input_tensor = torch.from_numpy(
        np.hstack(
            [encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])]
        ).astype(np.int64)
    )[None]

    assert input_tensor.shape[1] == 256 + 256 + 1, "unexpected tensor shape"

    with _inference_mode():
        output = _inference_bark(
            model,
            input_tensor,
        )

    assert all(0 <= output) and all(
        output < SEMANTIC_VOCAB_SIZE
    ), "unexpected output tokens"
    _clear_cuda_cache()
    return output


def _inference_bark(
    model: torch.Module, input: torch.Tensor, params: BarkInferenceParams
) -> torch.Tensor:
    input = input.to(params.device)
    n_tot_steps = 768
    # custom tqdm updates since we don't know when eos will occur
    pbar = tqdm.tqdm(disable=silent, total=n_tot_steps)
    pbar_state = 0
    tot_generated_duration_s = 0
    kv_cache = None
    for n in range(n_tot_steps):
        if use_kv_caching and kv_cache is not None:
            x_input = x[:, [-1]]
        else:
            x_input = x
        logits, kv_cache = model(
            x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
        )
        relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
        if allow_early_stop:
            relevant_logits = torch.hstack(
                (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
            )
        if top_p is not None:
            # faster to convert to numpy
            original_device = relevant_logits.device
            relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
            sorted_indices = np.argsort(relevant_logits)[::-1]
            sorted_logits = relevant_logits[sorted_indices]
            cumulative_probs = np.cumsum(softmax(sorted_logits))
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False
            relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
            relevant_logits = torch.from_numpy(relevant_logits)
            relevant_logits = relevant_logits.to(original_device)
        if top_k is not None:
            v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
            relevant_logits[relevant_logits < v[-1]] = -float("Inf")
        probs = F.softmax(relevant_logits / temp, dim=-1)
        item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)
        if allow_early_stop and (
            item_next == SEMANTIC_VOCAB_SIZE
            or (
                min_eos_p is not None and probs[-1] >= min_eos_p
            )  # probs[-1] is probability of the end of sentence token
        ):
            # eos found, so break
            pbar.update(n - pbar_state)
            break
        x = torch.cat((x, item_next[None]), dim=1)
        tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
        if (
            max_gen_duration_s is not None
            and tot_generated_duration_s > max_gen_duration_s
        ):
            pbar.update(n - pbar_state)
            break
        if n == n_tot_steps - 1:
            pbar.update(n - pbar_state)
            break
        del logits, relevant_logits, probs, item_next

        if n > pbar_state:
            if n > pbar.total:
                pbar.total = n
            pbar.update(n - pbar_state)
        pbar_state = n
    pbar.total = n
    pbar.refresh()
    pbar.close()
    out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1 :]


# (from the original implementation) the tokenized text array length must be 256
# trim or pad the array to get that length
# array is expected to be 1D
def trim_or_pad_array(
    array: np.ndarray, pad_token: int, max_length: int = 256
) -> np.ndarray:
    if len(array) > max_length:
        return array[:max_length]

    array = np.pad(
        array,
        (0, max_length - len(array)),
        constant_values=pad_token,
        mode="constant",
    )
    return array


# preprocess the texts for the generate_text_semantic model
def _preprocess_texts(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@validate_call
def _load_history_prompt(file_path: str) -> BarkAudioPrompt:
    history_prompt: Dict[str, np.ndarray] = {}
    if file_path.endswith(".npz"):
        history_prompt = np.load(file_path)
    else:
        # make sure this works on non-ubuntu
        file_path = os.path.join(*file_path.split("/"))
        history_prompt = np.load(
            os.path.join(CUR_PATH, "assets", "prompts", f"{file_path}.npz")
        )
    # expecting a dictionary with 3 keys: semantic_prompt, coarse_prompt, fine_prompt
    return BarkAudioPrompt(
        history_prompt.get("semantic_prompt", np.array([])),
        history_prompt.get("coarse_prompt", np.array([])),
        history_prompt.get("fine_prompt", np.array([])),
    )


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
