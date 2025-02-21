import os
import logging
from dotenv import load_dotenv
from enum import Enum, StrEnum
from typing import ClassVar

import torch
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def grab_best_device(use_gpu: bool) -> str:
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    elif (
        torch.backends.mps.is_available() and use_gpu and globals()["GLOBAL_ENABLE_MPS"]
    ):
        device = "mps"
    else:
        device = "cpu"
    return device


class EnvVars:
    """Class to store and manage environment variables with type hints and documentation"""

    # List of environment variables to load with their default values
    _VARS: ClassVar[dict] = {
        "GLOBAL_ENABLE_MPS": True,  # enable pytorch's metal performance shaders on Apple silicon chips
        "AUDIO_SAMPLE_RATE": 24000,  # default sample rate for all audio files
        "SUNO_USE_SMALL_MODELS": False,  # if use small models when working with BARK
        "CACHE_DIR": os.path.join(
            os.path.expanduser("~"), ".cache/tenxcreator"
        ),  # everything that worth caching will be stored in this directory
        "LOG_LEVEL": "INFO",
        "USE_GPU": True,
    }

    # Load environment variables from .env file
    load_dotenv()

    # Load and validate environment variables
    for var_name, default_value in _VARS.items():
        value = os.getenv(var_name)
        if value is None:
            logger.info(
                f"{var_name} not found in environment, using default: {default_value}"
            )
            globals()[var_name] = default_value
        else:
            # Convert to appropriate type based on default value
            if isinstance(default_value, bool):
                globals()[var_name] = value.lower() in ("true", "1", "t")
            elif isinstance(default_value, int):
                globals()[var_name] = int(value)
            elif isinstance(default_value, float):
                globals()[var_name] = float(value)
            else:
                globals()[var_name] = value
            logger.info(f"{var_name} loaded from environment: {globals()[var_name]}")

    # Set logging level from environment
    logging.getLogger().setLevel(globals()["LOG_LEVEL"])

    # Expose variables as class attributes with type hints
    GLOBAL_ENABLE_MPS: bool = globals()["GLOBAL_ENABLE_MPS"]
    AUDIO_SAMPLE_RATE: int = globals()["AUDIO_SAMPLE_RATE"]
    SUNO_USE_SMALL_MODELS: bool = globals()["SUNO_USE_SMALL_MODELS"]
    CACHE_DIR: str = globals()["CACHE_DIR"]
    LOG_LEVEL: str = globals()["LOG_LEVEL"]
    DEVICE: str = grab_best_device(globals()["USE_GPU"])


# Create instance to load and expose variables
env = EnvVars()


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_cached_or_download_model_from_hf(
    repo_id: str, file_name: str, cache_dir: str = env.CACHE_DIR
) -> str:
    """Download a model from Hugging Face Hub if not already cached.

    Args:
        repo_id: The repository ID on Hugging Face Hub (e.g., 'suno/bark')
        file_name: The name of the model file to download
        cache_dir: Directory to store cached models (defaults to env.CACHE_DIR)

    Returns:
        str: The full path to the downloaded/cached model file

    Raises:
        OSError: If there are issues creating the cache directory
        RuntimeError: If the download fails
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Check if file already exists in cache
    cached_path = os.path.join(cache_dir, file_name)
    if os.path.exists(cached_path):
        return cached_path

    # Download if not cached
    try:
        hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=cache_dir)
        return cached_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model {repo_id}/{file_name}: {str(e)}")
