"""
API to use the bark model
"""

from typing_extensions import Annotated, Literal, Optional

from pydantic.types import *
from pydantic import validate_call

import numpy as np

import torch
from dataclasses import dataclass
from typing import List, Tuple


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


# reference the bark model from gvar.models.torch_models, then use it to do forward pass
@validate_call
def preprocess_audio_prompt(audio: np.ndarray) -> np.ndarray:
    """
    Given the audio numpy array, preprocess it to match the expected audio prompt for BARK

    """
    pass


@validate_call
def generate_audio(
    texts: list[str], prompt: Optional[np.ndarray], device: torch.device
) -> np.ndarray:
    """
    Given the preprocessed audio, use the bark model to generate the final audio
    """
    pass
