import os
import logging
from dotenv import load_dotenv
from enum import Enum, StrEnum
from typing import ClassVar

import torch

from utils import grab_best_device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
