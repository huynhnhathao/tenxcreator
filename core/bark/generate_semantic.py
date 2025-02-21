import os
import re

import numpy as np
import torch
from transformers import BertTokenizer

from typing_extensions import List, Dict, Optional, Union

from pydantic import validate_call

from core.gvar import torch_models, ModelEnum, env
from core.bark.custom_context import inference_mode
from core.bark.utils import _clear_cuda_cache
from core.model import GPT

SEMANTIC_VOCAB_SIZE = 10_000
CUR_PATH = os.path.dirname(os.path.abspath(__file__))


# for the BERT model to get semantic tokens from raw texts
TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599


@validate_call
def generate_semantic_tokens_from_text(
    text: str,
    semantic_prompt: Union[np.ndarray, None] = None,
    temperature: float = 0.7,
    top_k: Union[int, None] = None,
    top_p: Union[int, None] = None,
    silent: Union[bool, None] = False,
    min_eos_p: float = 0.2,
    max_gen_duration_second: Union[float, None] = None,
    allow_early_stop: bool = True,
    use_kv_caching: bool = False,
) -> torch.Tensor:
    """
    Generate semantic tokens from given text and semantic prompt.
    The semantic prompt if provided will be concatenated to the generated semantic tokens of the given text.
    """

    # trim white spaces and replace redundant white space characters
    text = _preprocess_texts(text)
    assert len(text) > 0, f"invalid input text {text}"

    # load the GPT style model that generate semantic token from text
    # and the BERT tokenizer to memory
    text_model_info = (
        ModelEnum.BARK_TEXT_SMALL if env.SUNO_USE_SMALL_MODELS else ModelEnum.BARK_TEXT
    )

    text_model = torch_models.get_model(text_model_info)
    assert text_model.model is not None, "text model is None"
    assert text_model.preprocessor is not None, "tokenizer for the text model is None"

    model: GPT = text_model.model
    tokenizer: BertTokenizer = text_model.preprocessor

    # tokenize the given text using the BERT tokenizer
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)

    encoded_text = np.array(tokenized_text) + TEXT_ENCODING_OFFSET

    # encoded_text's length must has length 256 as from the original implementation
    # pad to the right if the encoded_text is shorter, trim on the right if it is longer than 256 tokens
    encoded_text = trim_or_pad_array(encoded_text, TEXT_PAD_TOKEN, 256)

    # semantic prompt also need to be an array of 256 discrete tokens
    if semantic_prompt is None:
        semantic_prompt = np.array([])

    semantic_history = trim_or_pad_array(semantic_history, SEMANTIC_PAD_TOKEN, 256)

    # final input is the concatenation of the input encoded text and the semantic tokens array
    # create a new axis to the tensor by indexing [None]
    input_tensor = torch.from_numpy(
        np.hstack(
            [encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])]
        ).astype(np.int64)
    )[None]

    assert (
        input_tensor.shape[1] == 256 + 256 + 1
    ), f"expecting tensor shape [1, 513], received {input_tensor.shape}"

    with inference_mode():
        output: torch.Tensor = _inference_bark(
            model,
            input_tensor,
        )

    validate_semantic_token_output(output)

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


def validate_semantic_token_output(output: torch.Tensor) -> None:
    assert all(0 <= output) and all(
        output < SEMANTIC_VOCAB_SIZE
    ), "unexpected output tokens"


# preprocess the texts for the generate_text_semantic model
def _preprocess_texts(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
