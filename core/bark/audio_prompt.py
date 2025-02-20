import numpy as np
import torch
import torchaudio


from pydantic import validate_call

"""
To create a prompt for BARK from an arbitrary audio file we need to use the 3 models
of BARK to forward the audio and collect 3 components of the prompt: a semantic prompt
from the text model, a coarse prompt from the coarse model and the fine prompt from the fine model.
"""


class BarkPrompt:
    semantic_prompt: torch.Tensor
    coarse_prompt: torch.Tensor
    fine_prompt: torch.Tensor


@validate_call
def create_prompt_from_audio(
    audio_file_path: str, max_duration: int, sample_rate: int
) -> BarkPrompt:
    wav, sr = torchaudio.load(audio_file_path)
