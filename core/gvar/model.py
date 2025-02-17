from dataclasses import dataclass
from enum import Enum
import torch

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


class TorchModels:
    models: dict[str, torch.Module]

    def model_is_loaded(self, model_name: str) -> bool:
        return model_name in self.models.keys()

    # store a model to the global scope
    def store_model(self, model_name: str, model: torch.Module) -> None:
        self.models[model_name] = model

    def get_model(self, model_name: str) -> torch.Module:
        pass


models = TorchModels()
