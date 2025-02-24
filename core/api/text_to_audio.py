import os
import numpy as np

from typing_extensions import Optional, Union, List

import torch

from core.api.models import *
from core.gvar.model import TorchModels
from core.bark.generate_audio import (
    generate_audio as bark_generate_audio,
    load_bark_audio_prompt,
)
from core.types.bark import BarkPrompt
from core.utils import normalize_whitespace, save_audio_file, read_audio_file
from core.bark import generate_semantic_tokens_from_text, encodec_encode_audio

"""
Convenient functions to generate text to audio
"""

_t2a_dispatcher = {TextToAudioModel.BARK: bark_generate_audio}


def create_bark_prompt(audio_prompt: RawAudioPrompt) -> BarkPrompt:
    """
    Turn raw audio into valid BARK prompt. When given a raw audio file, use this function
    to generate a valid BARK prompt
    """

    # validate data
    transcript = normalize_whitespace(audio_prompt.transcript)
    assert len(transcript) > 0, "invalid transcript"
    # read the audio
    raw_audio = read_audio_file(
        audio_prompt.audio_file_path,
        audio_prompt.sample_rate,
        audio_prompt.channels,
        max_duration=audio_prompt.max_duration,
    )

    # a 1D tensor contains generated semantic tokens from the transcript
    semantic_tokens = generate_semantic_tokens_from_text(transcript)

    # generate codebook tokens using encodec
    # assuming 24khz sample rate, will get back later if needed for 48khz
    codes = encodec_encode_audio(torch.from_numpy(raw_audio), audio_prompt.sample_rate)

    return BarkPrompt(semantic_tokens, codes[:2, :], codes)


# for now we will loop over each text to generate audio, later should support batch inference
def text_to_audio(
    texts: list[str],
    audio_prompt: Union[RawAudioPrompt, str, None] = None,
    sample_rate: int = 24000,
    device: Optional[torch.device] = None,
    save_path: str = "./artifact",
) -> List[np.ndarray]:
    """
    Generate audio using a raw audio file as prompt

    Args:
        - texts: texts to generate audio
        - audio_prompt: can be path to the raw audio file or path to the processed prompt for bark
        - device: device to run inference
        - save_path: path to save the final audio results
    """

    # if the prompt given is a raw audio file, need to turn it into valid prompt
    # and save it for later reference
    if isinstance(audio_prompt, RawAudioPrompt):
        prompt = create_bark_prompt(audio_prompt)
    elif isinstance(audio_prompt, str):
        prompt = load_bark_audio_prompt(audio_prompt)

    results = []
    for text in texts:
        audio = bark_generate_audio(text, prompt)
        results.append(audio)

    for text, audio in zip(texts, results):
        file_name = "_".join(text.split(" ")[:-5])
        file_path = os.path.join(save_path, file_name)
        save_audio_file(audio, sample_rate, file_path)


def text_to_audio(input_data: TextToAudioInput) -> List[np.ndarray]:
    """
    Generate audio from text using an optional audio prompt.

    Args:
        input_data (TextToAudioInput): Validated input data containing texts, optional prompt, and settings.

    Returns:
        List[np.ndarray]: List of generated audio arrays.

    Notes:
        This is a placeholder implementation. Replace with actual audio generation logic (e.g., Bark or another model).
    """
    # Simulate audio generation based on input texts
    audio_arrays = []
    for text in input_data.texts:
        # Placeholder: Generate random audio array (replace with real implementation)
        audio_array = np.random.rand(input_data.sample_rate * 5)  # 5 seconds of audio
        audio_arrays.append(audio_array)

    return audio_arrays
