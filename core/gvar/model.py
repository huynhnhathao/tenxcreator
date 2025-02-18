from dataclasses import dataclass
from enum import Enum
import torch

import logging
from core.gvar.utils import get_cached_or_download_model_from_hf
from transformers import EncodecModel

from typing_extensions import Callable

from model_config import GPTConfig

logger = logging.getLogger(__name__)


"""
When starting the application, all loaded models are stored in memory defined in this module
All models' weights will be downloaded from huggingface hub once and cached to local filesystem
"""


@dataclass
class ModelInfo:
    repo_id: str  # e.g suno/bark
    file_name: str  # e.g text.pt
    checkpoint_name: str  # e.g bert-based-uncased
    config_class: type
    model_class: type
    preprocessor_class: type
    model_type: str
    # if this is true, we instantiate a model of class model_class first
    # then call model.load_state_dict(downloaded_file)
    # otherwise we use torch.load(downloaded_file), the latter returns a dict with more
    # data than just the model's state_dict
    use_load_state_dict: bool


class Model:
    model: torch.Module
    config: Callable
    preprocessor: Callable  # a tokenizer if model is a text processor,


class ModelEnum(Enum):
    """Enumeration of all supported model names with their paths"""

    BARK_TEXT_SMALL = ModelInfo(
        repo_id="suno/bark", file_name="text.pt", model_type="text"
    )
    BARK_COARSE_SMALL = ModelInfo(
        repo_id="suno/bark", file_name="coarse.pt", model_type="coarse"
    )
    BARK_FINE_SMALL = ModelInfo(
        repo_id="suno/bark", file_name="fine.pt", model_type="fine"
    )
    BARK_TEXT = ModelInfo(repo_id="suno/bark", file_name="text_2.pt", model_type="text")
    BARK_COARSE = ModelInfo(
        repo_id="suno/bark", file_name="coarse_2.pt", model_type="coarse"
    )
    BARK_FINE = ModelInfo(repo_id="suno/bark", file_name="fine_2.pt", model_type="fine")
    ENCODEC = ModelInfo(checkpoint_name="facebook/encodec_24khz", model_type="encodec")

    @classmethod
    def get_model_info(cls, model_name: str) -> ModelInfo:
        """Get the ModelPath for a given model name"""
        try:
            return cls[model_name].value
        except KeyError:
            raise ValueError(f"Unknown model name: {model_name}")


# TODO: a model loaded and cached could be either a file saved by torch.load or a state_dict.pt file
# need to inspect the file to load them correctly
class TorchModels:
    def __init__(self):
        # map from the model_name to the instance of the model
        self._models: dict[ModelInfo, Model] = {}

    def get_model(self, model_info: ModelInfo) -> Model:
        """
        Get a model and its preprocessor from the memory if already loaded, else load it from cache or download it from the hf hub
        Args
            - model_name: name of the model, must be one of the option in the ModelEnum enum
        """
        # Check if model already loaded, return it
        if model_info in self._models.keys():
            return self._models[model_info]

        if model_info.checkpoint_name == "" and (
            model_info.repo_id == "" and model_info.file_name == ""
        ):
            raise ValueError(
                "either checkpoint_name or repo_id and file_name must be provided"
            )

        # if checkpoint_name is provided, we load model via transformers.from_pretrained(checkpoint_name)
        # transformers model cache their models separately
        if model_info.checkpoint_name:
            return load_transformers_model(model_info)

        # if the repo_id and the file_name are specified, use them to download the model from hugging face
        # or load from cached folder if already downloaded
        if model_info.repo_id and model_info.file_name:
            model_file_path = get_cached_or_download_model_from_hf(
                repo_id=model_info.repo_id, file_name=model_info.file_name
            )
            return load_model_from_file(model_info, model_file_path)


# load the model, its configuration and preprocessor
def load_model_from_file(model_info: ModelInfo, model_file_path: str) -> Model:
    if model_info.repo_id == "suno/bark":
        return load_bark_model(model_info, model_file_path)

    raise ValueError(f"unknown how to load model {model_info}")


def load_transformers_model(model_info: ModelInfo) -> Model:
    pass


def load_bark_model(model_info: ModelInfo, model_file_path: str) -> Model:
    pass


torch_models = TorchModels()
