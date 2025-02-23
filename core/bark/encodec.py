import torch
import numpy as np

from torch import nn
from encodec import EncodecModel
from core.gvar import torch_models, ModelEnum, env


def decode_fine_tokens_to_audio(fine_tokens: torch.Tensor) -> np.ndarray:
    """
    Decode the given fine_tokens using the Encodec's decoder
    Returns the audio sample array as an np.ndarray
    """
    model_info = ModelEnum.ENCODEC24k.value

    model_wrapper = torch_models.get_model(model_info)
    model: EncodecModel = model_wrapper.model

    device = next(model.parameters()).device

    input_tensor = fine_tokens[None].transpose(0, 1).to(device)

    emb = model.quantizer.decode(input_tensor)

    output: torch.Tensor = model.decoder(emb)
    audio_arr = output.detach().cpu().numpy().squeeze()

    del input_tensor, emb, output

    return audio_arr
