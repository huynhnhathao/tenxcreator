import pytest
import numpy as np
import soundfile as sf
import tempfile
import os
from core.utils.audio import read_audio_file

from pydantic import ValidationError


@pytest.fixture
def temp_audio_file():
    """Fixture to create temporary audio files for testing"""
    files_to_cleanup = []

    def _create_audio(sample_rate=44100, channels=1, duration=1.0):
        # Create a simple sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

        if channels == 2:
            data = np.column_stack((data, data))  # Make stereo

        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(f.name, data, sample_rate)
        files_to_cleanup.append(f.name)
        return f.name

    yield _create_audio

    # Clean up all created files
    for file_path in files_to_cleanup:
        try:
            os.unlink(file_path)
        except:
            pass


def test_read_mono_audio(temp_audio_file):
    """Test reading a mono audio file"""
    path = temp_audio_file(channels=1)  # Call the function to create the audio file
    data = read_audio_file(path)
    assert isinstance(data, np.ndarray)
    assert data.ndim == 1  # Mono should return 1D array
    assert len(data) == 44100  # 1 second at 44100 Hz


def test_read_stereo_audio(temp_audio_file):
    """Test reading a stereo audio file"""
    path = temp_audio_file(channels=2)  # Call the function to create the audio file
    data = read_audio_file(path, channels=2)
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert data.shape == (44100, 2)  # 1 second at 44100 Hz, 2 channels


def test_resampling(temp_audio_file):
    """Test resampling to different sample rates"""
    path = temp_audio_file()  # Call the function to create the audio file
    data = read_audio_file(path, target_sample_rate=16000)
    assert len(data) == 16000  # 1 second at 16000 Hz


def test_channel_conversion(temp_audio_file):
    """Test mono to stereo and stereo to mono conversion"""
    # Test mono to stereo
    path = temp_audio_file(channels=2)  # Call the function to create the audio file
    data = read_audio_file(path, channels=2)
    assert data.shape == (44100, 2)

    # Test stereo to mono
    path = temp_audio_file(channels=2)  # Call the function to create the audio file
    data = read_audio_file(path, channels=1)
    assert data.ndim == 1


def test_normalization(temp_audio_file):
    """Test audio normalization"""
    path = temp_audio_file(channels=1)  # Call the function to create the audio file
    data = read_audio_file(path, normalize=True)
    assert np.max(np.abs(data)) <= 1.0
    assert np.isclose(np.max(np.abs(data)), 1.0, atol=1e-5)


def test_max_duration(temp_audio_file):
    """Test duration truncation"""
    path = temp_audio_file(duration=2)  # Call the function to create the audio file
    data = read_audio_file(path, max_duration=1.0)
    assert len(data) == 44100  # Should be truncated to 1 second


def test_invalid_file():
    """Test handling of invalid file paths"""
    with pytest.raises(RuntimeError):
        read_audio_file("nonexistent_file.wav")


def test_invalid_parameters(temp_audio_file):
    """Test validation of invalid parameters"""
    path = temp_audio_file(channels=1)  # Call the function to create the audio file
    with pytest.raises(ValidationError):
        read_audio_file(path, channels=3)  # Invalid channel count

    with pytest.raises(ValidationError):
        read_audio_file(path, target_sample_rate=-1)  # Invalid sample rate

    with pytest.raises(ValidationError):
        read_audio_file(path, max_duration=-1.0)  # Invalid duration
