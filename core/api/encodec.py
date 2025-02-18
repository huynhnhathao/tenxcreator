import numpy as np
import torch
from transformers import EncodecModel, AutoProcessor


# reference the encodec model from gvar.model.torch_models
# then use it to encode/decode the given audio
def encode_audio(audio: np.ndarray, device: torch.device) -> np.ndarray:
    pass


def decode_audio(data: np.ndarray) -> np.ndarray:
    pass
