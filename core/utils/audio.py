"""
Helpful functions to process audio
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from typing_extensions import Annotated, Literal, Optional

from pydantic.types import *
from pydantic import validate_call

AudioChannel = Literal[1, 2]


@validate_call
def read_audio_file(
    path: Annotated[str, StringConstraints(min_length=1)],
    target_sample_rate: PositiveInt = 24000,
    channels: AudioChannel = 1,
    normalize: bool = True,
    max_duration: Optional[PositiveFloat] = None,
) -> np.ndarray:
    """Read and process an audio file from the filesystem.

    Args:
        path: Path to the audio file (supports common formats like WAV, FLAC, OGG)
        target_sample_rate: Target sample rate to resample to (default: 44100)
        channels: Number of output channels (1 for mono, 2 for stereo)
        normalize: Whether to normalize audio to [-1, 1] range
        max_duration: Maximum duration in seconds (truncates longer files on the right)

    Returns:
        np.ndarray: Processed audio samples as a numpy array

    Raises:
        RuntimeError: If the file cannot be read or is not a valid audio file
        ValueError: If invalid parameters are provided
    """
    try:
        # Read audio file with original sample rate
        data, original_sample_rate = sf.read(path, always_2d=True)

        # Resample if needed
        if original_sample_rate != target_sample_rate:
            ratio = target_sample_rate / original_sample_rate
            data = resample_poly(data, int(ratio * 1000000), 1000000, axis=0)

        # Convert channels
        if channels == 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1, keepdims=True)
        elif channels == 2 and data.shape[1] == 1:
            data = np.tile(data, (1, 2))

        # Truncate if max_duration is set
        if max_duration is not None:
            max_samples = int(target_sample_rate * max_duration)
            data = data[:max_samples]

        # Normalize audio
        if normalize:
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val

        return data.squeeze()

    except Exception as e:
        raise RuntimeError(f"Failed to process audio file {path}: {str(e)}")


def save_audio_file(audio_array, sample_rate, file_path, format="WAV"):
    """
    Save an audio array to a file.

    Parameters:
    - audio_array: numpy array or list containing the audio samples
    - sample_rate: int, the sample rate of the audio (e.g., 44100 Hz)
    - file_path: str, path where the file will be saved (e.g., 'output.wav')
    - format: str, audio file format (e.g., 'WAV', 'FLAC', 'OGG'), default is 'WAV'
    """
    try:
        sf.write(file_path, audio_array, sample_rate, format=format)
        print(f"Audio saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving audio file: {e}")
