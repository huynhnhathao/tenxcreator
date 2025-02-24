from typing_extensions import Optional, Callable
from dataclasses import dataclass
from enum import Enum
from transformers import BertTokenizer
from encodec import EncodecModel

import torch

from core.model.bark import GPT
from core.gvar.common import clear_cuda_cache

# Memory threshold (in percentage) to trigger unloading of models when memory usage gets too high
MEMORY_THRESHOLD = (
    0.9  # 90% of available memory; applies to GPU unless offloaded to CPU
)


class EncodecModelType(Enum):
    """Enumeration for Encodec model types based on sampling rate."""

    ENCODEC24 = "24khz"  # 24 kHz model
    ENCODEC48 = "48khz"  # 48 kHz model


class EncodecTargetBandwidth(float, Enum):
    """Enumeration for supported Encodec bandwidths in kbps."""

    BANDWIDTH_1_5 = 1.5  # 1.5 kbps (n_q = 2)
    BANDWIDTH_3 = 3  # 3 kbps (n_q = 4)
    BANDWIDTH_6 = 6  # 6 kbps (n_q = 8)
    BANDWIDTH_12 = 12  # 12 kbps (n_q = 16)
    BANDWIDTH_24 = 24  # 24 kbps (n_q = 32)


@dataclass(frozen=True)
class ModelInfo:
    """Data structure to hold metadata about a model."""

    # Hugging Face repository ID (e.g., "suno/bark")
    repo_id: Optional[str] = None
    # Filename of the model weights (e.g., "text.pt")
    file_name: Optional[str] = None
    # Pretrained checkpoint name (e.g., "facebook/encodec_24khz")
    checkpoint_name: Optional[str] = None
    # Configuration class for the model
    config_class: Optional[type] = None
    # Model class to instantiate
    model_class: Optional[type] = None
    # Preprocessor class (e.g., tokenizer)
    preprocessor_class: Optional[type] = None
    # Type of model (e.g., "text", "coarse", "encodec")
    model_type: Optional[str] = None
    # define the function that load the model
    load_model: Optional[Callable] = None


@dataclass
class Model:
    """Container for a loaded model, its configuration, and preprocessor."""

    model: Callable  # The PyTorch model instance
    config: Optional[Callable] = None  # Model configuration object
    # Preprocessor (e.g., tokenizer for text models)
    preprocessor: Optional[Callable] = None


def _load_encodec_model(device: torch.device) -> Model:
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval()
    model.to(device)
    clear_cuda_cache()
    return Model(model)


class ModelEnum(Enum):
    """
    Enumeration of supported models with their metadata.
    Each entry maps to a ModelInfo object defining how to load the model.
    """

    BARK_TEXT_SMALL = ModelInfo(
        repo_id="suno/bark",
        file_name="text.pt",
        model_type="text",
        model_class=GPT,
        preprocessor_class=BertTokenizer,
    )
    BARK_COARSE_SMALL = ModelInfo(
        repo_id="suno/bark", file_name="coarse.pt", model_type="coarse"
    )
    BARK_FINE_SMALL = ModelInfo(
        repo_id="suno/bark", file_name="fine.pt", model_type="fine"
    )
    ENCODEC24k = ModelInfo(
        checkpoint_name="facebook/encodec_24khz",
        model_type="encodec",
        load_model=_load_encodec_model,
    )

    BARK_TEXT = ModelInfo(repo_id="suno/bark", file_name="text_2.pt", model_type="text")
    BARK_COARSE = ModelInfo(
        repo_id="suno/bark", file_name="coarse_2.pt", model_type="coarse"
    )
    BARK_FINE = ModelInfo(repo_id="suno/bark", file_name="fine_2.pt", model_type="fine")

    @classmethod
    def get_model_info(cls, model_name: str) -> ModelInfo:
        """
        Retrieve ModelInfo for a given model name.

        Args:
            model_name (str): Name of the model (e.g., "BARK_TEXT_SMALL")

        Returns:
            ModelInfo: Metadata for the requested model

        Raises:
            ValueError: If the model name is not recognized
        """
        try:
            return cls[model_name].value
        except KeyError:
            raise ValueError(f"Unknown model name: {model_name}")
