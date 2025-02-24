from enum import Enum
from dataclasses import dataclass


@dataclass
class RawAudioPrompt:
    """
    A raw audio prompt for BARK requires a transcript of what was spoken in the audio
    Args:
        transcript: text transcript of what was spoken in the audio
        audio_file_path: path to the audio file
        sample_rate: target sample rate to read the audio
        channels: 1 for mono, 2 for stereo
        max_duration: trim on the right of the audio if it is longer than this
    """

    transcript: str
    audio_file_path: str
    sample_rate: int
    channels: int
    max_duration: int


class TextToAudioModel(Enum):
    BARK = "BARK"
