import os
import re
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from transformers import BertTokenizer
from typing_extensions import List, Tuple, Optional, Union, Sequence

from core.gvar import torch_models, ModelEnum, env
from core.bark.custom_context import inference_mode
from core.bark.utils import _clear_cuda_cache
from core.model import GPT
from core.bark.constants import *
from core.types.bark import BarkPrompt


def generate_semantic_tokens_from_text(
    text: str,
    semantic_prompt: Union[torch.Tensor, None] = None,
    temperature: float = 0.7,
    top_k: Union[int, None] = None,
    top_p: Union[int, None] = None,
    silent: Union[bool, None] = False,
    min_eos_p: float = 0.2,
    max_gen_duration_second: Union[float, None] = None,
    allow_early_stop: bool = True,
    use_kv_caching: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Generate semantic tokens from given text and optional semantic prompt.

    This function processes input text and generates corresponding semantic tokens using a GPT-style model.
    The semantic tokens can be used for further audio generation. An optional semantic prompt can be provided
    to influence the generation process.

    Args:
        text (str): Input text to generate semantic tokens for. Must be non-empty.
        semantic_prompt (Union[torch.Tensor, None]): Optional array of semantic tokens to use as prompt.
            If provided, these tokens will be concatenated with the generated tokens.
        temperature (float): Sampling temperature for token generation. Higher values produce more random outputs.
            Defaults to 0.7.
        top_k (Union[int, None]): If set, limits sampling to top-k tokens. Defaults to None.
        top_p (Union[int, None]): If set, uses nucleus sampling with this probability threshold. Defaults to None.
        silent (Union[bool, None]): If True, suppresses progress output. Defaults to False.
        min_eos_p (float): Stop generating new token if the probability of the EOS token is greater than or equal to this p. Defaults to 0.2.
        max_gen_duration_second (Union[float, None]): Maximum duration in seconds for the audio to be generated, set as an early stopping condition. Defaults to None.
        allow_early_stop (bool): Whether to allow early stopping based on EOS probability. Defaults to True.
        use_kv_caching (bool): Whether to use key-value caching for faster generation. Defaults to False.

    Returns:
        torch.Tensor: Generated semantic tokens as a 1D tensor.

    Raises:
        AssertionError: If input text is empty or model/tokenizer is not properly loaded.
        ValueError: If generated tokens are outside valid range.

    Notes:
        - The function handles both text preprocessing and token generation.
        - Input text is trimmed and normalized before processing.
        - Both text and semantic prompts are padded/trimmed to 256 tokens.
        - CUDA cache is cleared after generation to manage GPU memory.
    """

    # trim white spaces and replace redundant white space characters
    text = _preprocess_texts(text)
    assert len(text) > 0, f"invalid input text {text}"

    if semantic_prompt is None:
        semantic_prompt = torch.tensor([])
    else:
        assert isinstance(
            semantic_prompt, torch.Tensor
        ), f"expecting semantic_prompt of type torch.Tensor, received {type(semantic_prompt)}"

    # load the GPT style model that generate semantic token from text
    # and the BERT tokenizer to memory
    text_model_info = (
        ModelEnum.BARK_TEXT_SMALL.value
        if env.SUNO_USE_SMALL_MODELS
        else ModelEnum.BARK_TEXT.value
    )

    text_model = torch_models.get_model(text_model_info)
    assert text_model.model is not None, "text model is None"
    assert text_model.preprocessor is not None, "tokenizer for the text model is None"

    assert isinstance(
        text_model.model, GPT
    ), f"expecting model of type GPT, got {type(text_model.model)}"

    assert isinstance(
        text_model.preprocessor, BertTokenizer
    ), f"expecting preprocessor of type BertTokenizer, got {type(text_model.preprocessor)}"

    model: GPT = text_model.model
    tokenizer: BertTokenizer = text_model.preprocessor

    # tokenize the given text using the BERT tokenizer
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)

    encoded_text = np.array(tokenized_text) + TEXT_ENCODING_OFFSET

    # encoded_text's length must has length 256 as from the original implementation
    # pad to the right if the encoded_text is shorter, trim on the right if it is longer than 256 tokens
    encoded_text = trim_or_pad_array(encoded_text, TEXT_PAD_TOKEN, 256)

    # semantic prompt also need to be an array of 256 discrete tokens
    semantic_prompt = trim_or_pad_array(semantic_prompt, SEMANTIC_PAD_TOKEN, 256)

    # final input is the concatenation of the input encoded text and the semantic tokens array
    # create a new axis to the tensor by indexing [None]
    input_tensor = torch.hstack(
        [encoded_text, semantic_prompt, torch.tensor([SEMANTIC_INFER_TOKEN])]
    ).to(torch.int64)[None]

    assert (
        input_tensor.shape[1] == 256 + 256 + 1
    ), f"expecting tensor shape [1, 513], received {input_tensor.shape}"

    with inference_mode():
        output: torch.Tensor = _generate_semantic(
            model,
            input_tensor,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            silent=silent,
            min_eos_p=min_eos_p,
            max_gen_duration_s=max_gen_duration_second,
            allow_early_stop=allow_early_stop,
            use_kv_caching=use_kv_caching,
        )

        validate_semantic_token_output(output)

    _clear_cuda_cache()

    return output


def _generate_semantic(
    model: GPT,  # Assuming GPT is a custom class from your model.py
    input_tensor: torch.Tensor,
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    silent: bool = False,
    min_eos_p: float = 0.2,
    max_gen_duration_s: Optional[float] = None,
    allow_early_stop: bool = True,
    use_kv_caching: bool = False,
) -> torch.Tensor:
    """
    Perform autoregressive inference to generate semantic tokens using a GPT-style model.

    Args:
        model: The GPT model for generating semantic tokens.
        input_tensor: Initial input tensor of shape [1, 513] (text + history + infer token).
        temperature: Sampling temperature for randomness control.
        top_k: Number of top logits to consider during sampling.
        top_p: Cumulative probability threshold for nucleus sampling.
        silent: If True, suppresses progress bar output.
        min_eos_p: Minimum EOS probability to stop generation. Increasing this probability translate to generating longer sequence
        max_gen_duration_s: Maximum duration in seconds for generation.
        allow_early_stop: Whether to stop on EOS token or probability threshold.
        use_kv_caching: Whether to use key-value caching for efficiency.

    Returns:
        torch.Tensor: Generated semantic tokens (shape [seq_len]).
    """
    # Move input tensor to the model's device (e.g., GPU)
    device = next(model.parameters()).device
    x = input_tensor.to(device)

    # Maximum number of tokens to generate (hardcoded as in original)
    max_steps = 768

    # Initialize progress bar for user feedback (custom due to unpredictable stopping)
    progress_bar = tqdm(
        total=max_steps, disable=silent, desc="Generating semantic tokens"
    )
    last_progress = 0

    # Track generation duration in seconds (based on SEMANTIC_RATE_HZ = 49.9)
    total_duration_s = 0.0
    duration_per_step = 1 / SEMANTIC_RATE_HZ  # ~0.02004 seconds per token

    # Key-value cache for attention optimization
    kv_cache = None

    # Autoregressive generation loop
    for step in range(max_steps):
        # Determine input based on KV caching
        if use_kv_caching and kv_cache is not None:
            # Use only the last token with cached attention states
            x_input = x[:, [-1]]  # Shape [1, 1]
        else:
            # Use full sequence (recomputes attention each time)
            x_input = x  # Shape [1, seq_len]

        # Forward pass through the model
        logits, kv_cache = model(
            x_input,
            merge_context=True,  # Merges text and semantic history context
            past_kv=kv_cache,  # Previous attention states
            use_cache=use_kv_caching,  # Enables caching if requested
        )

        # Sample the next token and check for early stopping
        next_token, should_stop = _sample_next_token(
            logits,
            temperature,
            top_k,
            top_p,
            SEMANTIC_VOCAB_SIZE,
            allow_early_stop,
            min_eos_p,
            SEMANTIC_PAD_TOKEN,
        )

        # Check stopping conditions
        if should_stop:
            progress_bar.update(step - last_progress + 1)
            break
        if max_gen_duration_s is not None and total_duration_s > max_gen_duration_s:
            progress_bar.update(step - last_progress + 1)
            break
        if step == max_steps - 1:
            progress_bar.update(step - last_progress + 1)
            break

        # Append the new token to the sequence
        x = torch.cat((x, next_token[None]), dim=1)

        # Update duration and progress
        total_duration_s += duration_per_step
        if step > last_progress:
            progress_bar.update(step - last_progress)
            last_progress = step

        # Clean up tensors to manage memory
        del logits, next_token

    # Finalize progress bar
    progress_bar.total = step + 1
    progress_bar.close()

    # Extract generated tokens (skip initial 513 context tokens)
    output = x[:, 256 + 256 + 1 :].detach().cpu()

    return output


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    vocab_size: int,
    allow_early_stop: bool,
    min_eos_p: Optional[float],
    eos_token: int,
) -> Tuple[torch.Tensor, bool]:
    """
    Sample the next token from logits with optional top-k, top-p filtering and early stopping.

    Args:
        logits: Tensor of shape [batch, seq, vocab_size] containing model predictions.
        temperature: Controls randomness of sampling (lower = more deterministic).
        top_k: If set, keeps only the top-k logits.
        top_p: If set, applies nucleus (top-p) filtering.
        vocab_size: Size of the semantic vocabulary (e.g., SEMANTIC_VOCAB_SIZE).
        allow_early_stop: Whether to check for EOS token or probability threshold.
        min_eos_p: Minimum probability for EOS to trigger early stop.
        eos_token: Token ID representing end-of-sequence.

    Returns:
        Tuple[next_token, should_stop]:
            - next_token: Sampled token (shape [1]).
            - should_stop: Whether to stop generation (EOS detected).
    """
    # Extract logits for the last position in the sequence
    relevant_logits = logits[0, 0, :vocab_size]

    # Append EOS logit if early stopping is allowed
    if allow_early_stop:
        eos_logit = logits[0, 0, eos_token]
        relevant_logits = torch.hstack((relevant_logits, eos_logit))

    # Apply top-p (nucleus) filtering for diversity
    if top_p is not None:
        # Convert to NumPy for faster sorting (optimization from original)
        original_device = relevant_logits.device
        logits_np = relevant_logits.detach().cpu().type(torch.float32).numpy()
        sorted_indices = np.argsort(logits_np)[::-1]  # Descending order
        sorted_logits = logits_np[sorted_indices]
        cumulative_probs = np.cumsum(
            F.softmax(torch.from_numpy(sorted_logits), dim=-1).numpy()
        )
        indices_to_remove = cumulative_probs > top_p
        indices_to_remove[1:] = indices_to_remove[
            :-1
        ].copy()  # Shift to keep at least one
        indices_to_remove[0] = False  # Ensure top token stays
        logits_np[sorted_indices[indices_to_remove]] = -np.inf
        relevant_logits = torch.from_numpy(logits_np).to(original_device)

    # Apply top-k filtering for diversity
    if top_k is not None:
        top_values, _ = torch.topk(
            relevant_logits, min(top_k, relevant_logits.size(-1))
        )
        relevant_logits[relevant_logits < top_values[-1]] = -float("Inf")

    # Compute probabilities with temperature scaling
    probs = F.softmax(relevant_logits / temperature, dim=-1)

    # Sample the next token
    next_token = torch.multinomial(probs, num_samples=1).to(torch.int32)

    # Check for early stopping conditions
    should_stop = False
    if allow_early_stop:
        is_eos_token = (
            next_token.item() == vocab_size
        )  # EOS token is vocab_size when appended
        eos_prob_high = min_eos_p is not None and probs[-1] >= min_eos_p
        should_stop = is_eos_token or eos_prob_high

    return next_token, should_stop


def validate_semantic_token_output(output: torch.Tensor) -> None:
    assert torch.all(
        (0 <= output) & (output <= SEMANTIC_VOCAB_SIZE)
    ), "unexpected output tokens"


# preprocess the texts for the generate_text_semantic model
def _preprocess_texts(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# (from the original implementation) the tokenized text array length must be 256
# trim or pad the array to get that length
# array is expected to be 1D
# def trim_or_pad_array(
#     array: np.ndarray, pad_token: int, max_length: int = 256
# ) -> np.ndarray:
#     if len(array) > max_length:
#         return array[-max_length:]

#     array = np.pad(
#         array,
#         (0, max_length - len(array)),
#         constant_values=pad_token,
#         mode="constant",
#     )
#     return array


def trim_or_pad_array(
    array: Union[np.ndarray, torch.Tensor], pad_token: int, max_length: int = 256
) -> torch.Tensor:
    # Convert np.ndarray to torch.Tensor if necessary
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    else:  # Already a torch.Tensor
        tensor = array

    # Get the current length
    current_length = tensor.shape[0]

    if current_length > max_length:
        # Trim from the end (last max_length elements)
        return tensor[-max_length:]

    elif current_length < max_length:
        # Pad with pad_token
        padding = (
            0,
            max_length - current_length,
        )  # Left pad 0, right pad to max_length
        return torch.nn.functional.pad(
            tensor, padding, mode="constant", value=pad_token
        )

    # If length equals max_length, just return as is
    return tensor
