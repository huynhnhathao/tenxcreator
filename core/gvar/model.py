from dataclasses import dataclass
from enum import Enum
import torch

import logging
from core.gvar.utils import get_cached_or_download_model_from_hf

logger = logging.getLogger(__name__)


"""
When starting the application, all loaded models are stored in memory in this module
"""


@dataclass
class ModelPath:
    repo_id: str
    file_name: str


class ModelName(Enum):
    """Enumeration of all supported model names with their paths"""

    BARK_TEXT_SMALL = ModelPath(repo_id="suno/bark", file_name="text.pt")
    BARK_COARSE_SMALL = ModelPath(repo_id="suno/bark", file_name="coarse.pt")
    BARK_FINE_SMALL = ModelPath(repo_id="suno/bark", file_name="fine.pt")
    BARK_TEXT = ModelPath(repo_id="suno/bark", file_name="text_2.pt")
    BARK_COARSE = ModelPath(repo_id="suno/bark", file_name="coarse_2.pt")
    BARK_FINE = ModelPath(repo_id="suno/bark", file_name="fine_2.pt")

    @classmethod
    def get_path(cls, model_name: str) -> ModelPath:
        """Get the ModelPath for a given model name"""
        try:
            return cls[model_name].value
        except KeyError:
            raise ValueError(f"Unknown model name: {model_name}")

    @classmethod
    def get_repo_id(cls, model_name: str) -> str:
        """Get the repo ID for a given model name"""
        return cls.get_path(model_name).repo_id

    @classmethod
    def get_file_name(cls, model_name: str) -> str:
        """Get the file name for a given model name"""
        return cls.get_path(model_name).file_name


# TODO: a model loaded and cached could be either a file saved by torch.load or a statedict file
# need to inspect the file to load them correctly
class TorchModels:
    def __init__(self):
        self._models = {}

    def get_model(self, model_name: str) -> torch.Module:
        # Validate model name
        if model_name not in ModelName.__members__:
            raise ValueError(
                f"Invalid model name: {model_name}. Must be one of {list(ModelName.__members__.keys())}"
            )

        # Check if model already loaded
        if model_name in self._models.keys():
            return self._models[model_name]

        # Get model path info
        model_path = ModelName.get_path(model_name)

        # Download or get cached model path
        logger.info(f"Loading model {model_name} from cache or downloading...")
        model_file = get_cached_or_download_model_from_hf(
            repo_id=model_path.repo_id, file_name=model_path.file_name
        )

        # Load the model
        logger.info(f"Loading model {model_name} from {model_file}")
        model = torch.load(model_file)

        # Store in memory for future use
        self._models[model_name] = model
        return model


torch_models = TorchModels()
