from core.bark.generate_semantic import generate_semantic_tokens_from_text

import numpy as np
import torch
import torchaudio

from typing_extensions import List, Tuple, Optional, Union, Sequence

from core.utils import read_audio_file

from core.gvar import env


class BarkPrompt:
    semantic_prompt: torch.Tensor
    coarse_prompt: torch.Tensor
    fine_prompt: torch.Tensor


def generate_audio(
    text: str,
    prompt: Union[str, None] = None,
    temperature: float = 0.7,
    top_k: Union[int, None] = None,
    top_p: Union[int, None] = None,
    silent: Union[bool, None] = False,
    min_eos_p: float = 0.2,
    max_gen_duration_second: Union[float, None] = None,
    allow_early_stop: bool = True,
    use_kv_caching: bool = False,
) -> np.ndarray:
    """
    Generate audio from text with an optional audio prompt
    Args:
        text (str): Input text to generate audio. Must be non-empty.
        prompt (Union[str, None]): optional path to a prompt file of type .npz that will be used as the audio prompt
        temperature (float): Sampling temperature for token generation. Higher values produce more random outputs.
            Defaults to 0.7.
        top_k (Union[int, None]): If set, limits sampling to top-k tokens. Defaults to None.
        top_p (Union[int, None]): If set, uses nucleus sampling with this probability threshold. Defaults to None.
        silent (Union[bool, None]): If True, suppresses progress output. Defaults to False.
        min_eos_p (float): Stop generating new token if the probability of the EOS token is greater than or equal to this p. Defaults to 0.2.
        max_gen_duration_second (Union[float, None]): Maximum duration in seconds for the audio to be generated, set as an early stopping condition. Defaults to None.
        allow_early_stop (bool): Whether to allow early stopping based on EOS probability. Defaults to True.
        use_kv_caching (bool): Whether to use key-value caching for faster generation. Defaults to False.

    """

    prompt = load_bark_audio_prompt()


def load_bark_audio_prompt(path: str) -> BarkPrompt:
    pass


def create_prompt_from_audio(
    audio_file_path: str, max_duration: int, sample_rate: int
) -> BarkPrompt:
    wav, sr = torchaudio.load(audio_file_path)
    pass
