import os
from dataclasses import dataclass, asdict

from core.bark.generate_semantic import generate_semantic_tokens_from_text
from core.bark.generate_coarse import generate_coarse_tokens_from_semantic
from core.bark.generate_fine import generate_fine_tokens_from_coarse

import numpy as np
import torch
import torchaudio

from typing_extensions import List, Tuple, Optional, Union, Sequence

from core.bark.data_types import BarkPrompt
from core.utils import read_audio_file
from core.gvar import env


CUR_PATH = os.path.dirname(os.path.abspath(__file__))


@dataclass
class GenerateAudioConfig:
    temperature: float = 0.7
    top_k: Union[int, None] = None
    top_k: Union[int, None] = None
    top_p: Union[int, None] = None
    silent: Union[bool, None] = False
    min_eos_p: float = 0.2
    max_gen_duration_second: Union[float, None] = None
    allow_early_stop: bool = True
    use_kv_caching: bool = True

    max_coarse_history: int = 630
    sliding_window_length: int = 60

    generate_fine_temperature: float = 0.5


generation_config = GenerateAudioConfig()


def generate_audio(
    text: str,
    prompt_file_path: Union[str, None] = None,
    generation_config: GenerateAudioConfig = generation_config,
) -> np.ndarray:
    """
    Generate audio from text with an optional audio prompt
    Args:
        text (str): Input text to generate audio. Must be non-empty.
        prompt (Union[str, None]): optional path to a prompt file of type .npz that will be used as the audio prompt
        generation_config: configurations to generate audio

    """

    prompt = load_bark_audio_prompt(prompt_file_path)

    semantic_tokens = generate_semantic_tokens_from_text(
        text, prompt.semantic_prompt, **asdict(generation_config)
    )

    # coarse token generation
    coarse_tokens = generate_coarse_tokens_from_semantic(
        semantic_tokens, prompt, **asdict(generation_config)
    )

    # fine token generation
    fine_tokens = generate_fine_tokens_from_coarse(
        coarse_tokens,
        prompt,
        generation_config.generate_fine_temperature,
        generation_config.silent,
    )

    # decoding the codes

    print(fine_tokens)

    # return the final audio

    return np.ndarray([])


def load_bark_audio_prompt(file_path: str) -> BarkPrompt:
    """
    Load a saved audio prompt from a .npz file. The file is expected to have 3 keys,
    semantic_prompt, coarse_prompt, fine_prompt, each of them is a np.ndarray
    """
    assert isinstance(
        file_path, str
    ), f"expecting a string type argument, received {type(file_path)} of value {file_path}"

    if file_path.endswith(".npz"):
        prompt = np.load(file_path)
    else:
        file_path = os.path.join(*file_path.split("/"))
        prompt = np.load(
            os.path.join(CUR_PATH, "assets", "prompts", f"{file_path}.npz")
        )

    assert (
        prompt["semantic_prompt"] is not None
        and prompt["coarse_prompt"] is not None
        and prompt["fine_prompt"] is not None
    ), f"invalid prompt data {prompt}"

    return BarkPrompt(
        torch.from_numpy(prompt["semantic_prompt"]),
        torch.from_numpy(prompt["coarse_prompt"]),
        torch.from_numpy(prompt["fine_prompt"]),
    )


def create_prompt_from_audio(
    audio_file_path: str, max_duration: int, sample_rate: int, save_prompt: bool = True
) -> BarkPrompt:
    """
    To create a prompt from an audio file, we need to forward the text transcript and the
    audio samples of the prompt through the 3 models of Bark to get 3 arrays: semantic, coarse and fine
    """
    wav, sr = torchaudio.load(audio_file_path)
    pass
