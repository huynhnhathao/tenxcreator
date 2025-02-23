import torch
import numpy as np

from torch import nn

from core.gvar import torch_models, ModelEnum, env


def decode_fine_tokens_to_audio(fine_tokens: torch.Tensor) -> np.ndarray:
    """
    Decode the given fine_tokens using the Encodec's decoder
    Returns the audio sample array as an np.ndarray
    """
    model_info = ModelEnum.ENCODEC24k.value

    model_wrapper = torch_models.get_model(model_info)
    model: nn.Module = model_wrapper.model

    device = next(model.parameters()).device
