import torch
import numpy as np

from torch import nn
from encodec import EncodecModel
from encodec.utils import convert_audio
from core.gvar import torch_models, ModelEnum, env
from core.bark.custom_context import inference_mode


def encodec_decode_fine_tokens_to_audio(fine_tokens: torch.Tensor) -> np.ndarray:
    """
    expecting fine_tokens shape [codebook_size, timestep], concretely [8, 75*duration_in_sec]
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


def encodec_encode_audio(
    audio_sample: torch.Tensor,
    audio_sample_rate: int,
) -> torch.Tensor:
    """
    Encode the given audio sample using the encodec model
    Returns codes as a tensor shape [n_q, T] where n_q typically is 8 and T is the compressed time step dimension (75 per second for 24khz model)
    """
    model_wrapper = torch_models.get_model(ModelEnum.ENCODEC24k.value)
    model: EncodecModel = model_wrapper.model

    wav = convert_audio(
        audio_sample, audio_sample_rate, model.sample_rate, model.channels
    )
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with inference_mode():
        encoded_frames = model.encode(wav)

    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

    return codes[0, :, :]
