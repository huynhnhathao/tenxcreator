from typing_extensions import Union

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from core.bark.data_types import BarkPrompt
from core.bark.constants import *
from core.bark.custom_context import inference_mode
from core.bark import FineGPT
from core.gvar import ModelEnum, env, torch_models


def generate_fine_tokens_from_coarse(
    coarse_tokens: torch.Tensor,
    history_prompt: Union[BarkPrompt, None] = None,
    temperature: float = 0.5,
    silent: bool = False,
) -> torch.Tensor:
    """
    Generate fine-grained audio codes from coarse audio codes using the BARK fine model.

    This function takes coarse tokens (representing a partial set of audio codebooks) and
    autoregressively predicts the remaining fine tokens, optionally conditioning on a history
    prompt. The process involves sliding a context window over the sequence, predicting 512
    timesteps at a time based on a 1024-timestep input.

    Args:
        coarse_tokens (torch.Tensor): Coarse audio codes with shape (n_coarse, sequence_length),
            where n_coarse <= N_FINE_CODEBOOKS - 1 and values are in [0, CODEBOOK_SIZE - 1].
        history_prompt (BarkPrompt, optional): Historical fine tokens for conditioning, or None.
        temperature (float): Sampling temperature for fine token prediction; if None, uses argmax.
        silent (bool): If True, suppresses progress bar output.

    Returns:
        torch.Tensor: Fine audio codes with shape (N_FINE_CODEBOOKS, sequence_length),
            matching the input sequence_length.

    Raises:
        AssertionError: If input validation fails for coarse_tokens or history_prompt.
    """
    # Validate inputs
    _validate_coarse_tokens(coarse_tokens=coarse_tokens)
    history_fine_tokens = _validate_and_load_history(history_prompt=history_prompt)
    n_coarse = coarse_tokens.shape[0]
    sequence_length = coarse_tokens.shape[1]

    # Load the fine model
    model_info = (
        ModelEnum.BARK_FINE_SMALL.value
        if env.SUNO_USE_SMALL_MODELS
        else ModelEnum.BARK_FINE.value
    )
    model_wrapper = torch_models.get_model(model_info)
    model: FineGPT = model_wrapper.model
    assert isinstance(model, FineGPT), "Expected FineGPT model type"
    device = next(model.parameters()).device

    # stack coarse tokens with padding for remaining codebooks across the codebook dimension
    # e.g original coarse_token shape (2, T), after vstack shape: (8, T) where codebook size = 8
    input_tensor = torch.vstack(
        (
            coarse_tokens,
            torch.full(
                (N_FINE_CODEBOOKS - n_coarse, sequence_length),
                CODEBOOK_SIZE,
                dtype=torch.int32,
                device=device,
            ),
        )
    )

    # Prepend history if provided. Maximum history time step is 512
    # this is a horizontal prepend on the left of the previous padded input tensor
    # output tensor: (8, history_timestep + coarse_timestep), history_timestep <= 512
    n_history = 0
    if history_fine_tokens is not None:
        history_limit = min(history_fine_tokens.shape[1], 512)
        history_slice = history_fine_tokens[:, -history_limit:].to(
            device, dtype=torch.int32
        )
        input_tensor = torch.hstack((history_slice, input_tensor))
        n_history = history_limit

    # right Pad if total_length (history_timestep + coarse_timestep) is less than model context (1024)
    total_length = input_tensor.shape[1]
    padding_needed = max(0, 1024 - total_length)
    if padding_needed > 0:
        padding = torch.full(
            (N_FINE_CODEBOOKS, padding_needed),
            CODEBOOK_SIZE,
            dtype=torch.int32,
            device=device,
        )
        input_tensor = torch.hstack((input_tensor, padding))
        total_length = input_tensor.shape[1]

    # Calculate number of prediction loops
    context_window = 1024  # Model's input context size
    prediction_step = 512  # Number of new timesteps predicted per loop
    remaining_length = max(0, sequence_length - (context_window - n_history))
    extra_loops = (remaining_length + prediction_step - 1) // prediction_step
    n_loops = 1 + extra_loops  # Total loops: initial + extra

    # Process sequence in sliding windows
    input_tensor = input_tensor.T  # Shape: (total_length, N_FINE_CODEBOOKS)
    with inference_mode():
        for loop_idx in tqdm(
            range(n_loops), disable=silent, desc="Generating fine tokens"
        ):
            # Define window boundaries
            # the last loop, by using window_start = (total_length - context_window),
            # the input will be: input_tensor[:, -1024:, :], the last context_window timestep of the input
            window_start = min(
                loop_idx * prediction_step, total_length - context_window
            )

            fill_start = min(
                n_history + loop_idx * prediction_step, total_length - prediction_step
            )
            fill_offset = fill_start - window_start
            window_end = window_start + context_window

            # Extract input window
            # Shape: (1, 1024, N_FINE_CODEBOOKS)
            input_window = input_tensor[window_start:window_end, :].unsqueeze(0)

            # Predict fine codebooks autoregressively
            for codebook_idx in range(n_coarse, N_FINE_CODEBOOKS):
                logits = model(
                    codebook_idx, input_window
                )  # Shape: (1, 1024, vocab_size)
                if temperature is None:
                    preds = torch.argmax(
                        logits[0, fill_offset:, :CODEBOOK_SIZE], dim=-1
                    )
                else:
                    scaled_logits = logits[0, :, :CODEBOOK_SIZE] / temperature
                    probs = F.softmax(scaled_logits, dim=-1)
                    preds = torch.multinomial(
                        probs[fill_offset:], num_samples=1
                    ).squeeze(-1)
                input_window[0, fill_offset:, codebook_idx] = preds.to(torch.int32)

                # Update main tensor with predictions
                fill_length = min(prediction_step, total_length - fill_start)
                input_tensor[fill_start : fill_start + fill_length, codebook_idx] = (
                    input_window[
                        0, fill_offset : fill_offset + fill_length, codebook_idx
                    ]
                )

    # Extract final result, removing history and padding
    # Shape: (N_FINE_CODEBOOKS, sequence_length)
    fine_tokens = input_tensor.T[:, n_history : n_history + sequence_length]

    # Verify output shape matches input sequence length
    assert fine_tokens.shape[1] == sequence_length, "Output length mismatch"

    return fine_tokens


def _validate_coarse_tokens(coarse_tokens: torch.Tensor) -> None:
    """Validate coarse token tensor properties."""
    assert isinstance(
        coarse_tokens, torch.Tensor
    ), "coarse_tokens must be a torch.Tensor"
    assert len(coarse_tokens.shape) == 2, "coarse_tokens must be 2D"
    assert (
        1 <= coarse_tokens.shape[0] <= N_FINE_CODEBOOKS - 1
    ), "Invalid number of coarse codebooks"
    assert coarse_tokens.shape[1] > 0, "Sequence length must be positive"
    assert (
        coarse_tokens.min() >= 0 and coarse_tokens.max() <= CODEBOOK_SIZE - 1
    ), "Token values out of range"


def _validate_and_load_history(
    history_prompt: Union[BarkPrompt, None]
) -> Union[torch.Tensor, None]:
    """Validate and load history prompt if provided."""
    if history_prompt is None:
        return None

    history_fine_tokens = history_prompt.fine_prompt
    assert isinstance(
        history_fine_tokens, torch.Tensor
    ), "history_prompt.fine_prompt must be a torch.Tensor"
    assert len(history_fine_tokens.shape) == 2, "History must be 2D"
    assert (
        history_fine_tokens.shape[0] == N_FINE_CODEBOOKS
    ), "History must have all fine codebooks"
    assert history_fine_tokens.shape[1] > 0, "History must not empty"
    assert (
        history_fine_tokens.min() >= 0
        and history_fine_tokens.max() <= CODEBOOK_SIZE - 1
    ), "History values out of range"
    return history_fine_tokens
