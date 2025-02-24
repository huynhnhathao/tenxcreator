import numpy as np

from typing_extensions import Optional

import torch

from core.api.types import *

"""
Convenient functions to generate text to audio
"""


# reference the bark model from gvar.models.torch_models, then use it to do forward pass
def preprocess_audio_prompt(audio: np.ndarray) -> np.ndarray:
    """
    Given the audio numpy array, preprocess it to match the expected audio prompt for BARK

    """
    pass


# for now we will loop over each text to generate audio, later should support batch inference
def text_to_audio(
    texts: list[str],
    prompt: Optional[np.ndarray],
    device: torch.device,
    model: T2AModel,
) -> np.ndarray:
    """
    Given the preprocessed audio, use the bark model to generate the final audio
    """

    # get 3 models from gvar
    # preprocess the input texts to the format that the first model expected
    # forward the text through 3 models and return the final audio wave as a np.ndarray

    pass
