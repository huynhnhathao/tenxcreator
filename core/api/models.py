from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field
from typing import Optional, Union, List


class RawAudioPrompt(BaseModel):
    """Model for validating raw audio prompt inputs."""

    transcript: str = Field(
        ..., min_length=1, description="Transcript of the prompt audio"
    )
    audio_file_path: str = Field(..., description="Path to the audio file")
    sample_rate: int = Field(..., ge=1, description="Sample rate of the audio in Hz")
    channels: int = Field(
        ..., ge=1, le=2, description="Number of audio channels (1=mono, 2=stereo)"
    )
    max_duration: int = Field(
        ..., ge=1, description="Maximum duration of the audio in seconds"
    )

    def get_default_prompt_name(self) -> str:
        num_word_in_name = 5
        words = self.transcript.split(" ")
        if len(words) > num_word_in_name:
            num_word_in_name = len(words)
        name = "_".join(words[:num_word_in_name])

        return name


class TextToAudioInput(BaseModel):
    """Model for validating inputs to the text-to-audio generation function."""

    texts: List[str] = Field(
        ..., min_items=1, description="List of text strings to convert to audio"
    )
    audio_prompt: Optional[Union[RawAudioPrompt, str]] = Field(
        None, description="Optional audio prompt (raw or file path)"
    )
    sample_rate: int = Field(
        default=24000, ge=1, description="Sample rate for generated audio"
    )
    device: Optional[str] = Field(
        None, description="Device to use for generation (e.g., 'cuda', 'cpu')"
    )
    save_path: str = Field(
        default="./artifact", description="Directory to save generated audio files"
    )


class TextToAudioModel(Enum):
    BARK = "BARK"
