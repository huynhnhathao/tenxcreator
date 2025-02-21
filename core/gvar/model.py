import os
from dataclasses import dataclass
from enum import Enum

import torch

import logging
from core.gvar.common import get_cached_or_download_model_from_hf, clear_cuda_cache
from transformers import BertTokenizer
from encodec import EncodecModel

from typing_extensions import Callable, Dict, Any, Optional, Literal

from pydantic import validate_call

from core.bark.model import GPTConfig, FineGPTConfig, GPT, FineGPT

from core.gvar.common import env


logger = logging.getLogger(__name__)


"""
When starting the application, all loaded models are stored in memory defined in this module
All models' weights will be downloaded from huggingface hub once and cached to local filesystem
"""


class EncodecModelType(Enum):
    ENCODEC24 = "24khz"
    ENCODEC48 = "48khz"


# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
class EncodecTargetBandwidth(float, Enum):
    BANDWIDTH_1_5 = 1.5
    BANDWIDTH_3 = 3
    BANDWIDTH_6 = 6
    BANDWIDTH_12 = 12
    BANDWIDTH_24 = 24


@dataclass
class ModelInfo:
    repo_id: Optional[str] = None  # e.g suno/bark
    file_name: Optional[str] = None  # e.g text.pt
    checkpoint_name: Optional[str] = None  # e.g bert-based-uncased
    config_class: Optional[type] = None
    model_class: Optional[type] = None
    preprocessor_class: Optional[type] = None
    model_type: Optional[str] = None
    # if this is true, we instantiate a model of class model_class first
    # then call model.load_state_dict(downloaded_file)
    # otherwise we use torch.load(downloaded_file), the latter returns a dict with more
    # data than just the model's state_dict
    use_load_state_dict: Optional[bool] = False


@dataclass
class Model:
    model: Callable
    config: Optional[Callable]
    preprocessor: Optional[Callable]  # a tokenizer if model is a text processor,


class ModelEnum(Enum):
    """Enumeration of all supported model names with their paths"""

    BARK_TEXT_SMALL = ModelInfo(
        repo_id="suno/bark",
        file_name="text.pt",
        model_type="text",
    )
    BARK_COARSE_SMALL = ModelInfo(
        repo_id="suno/bark",
        file_name="coarse.pt",
        model_type="coarse",
    )
    BARK_FINE_SMALL = ModelInfo(
        repo_id="suno/bark", file_name="fine.pt", model_type="fine"
    )
    BARK_TEXT = ModelInfo(repo_id="suno/bark", file_name="text_2.pt", model_type="text")
    BARK_COARSE = ModelInfo(
        repo_id="suno/bark", file_name="coarse_2.pt", model_type="coarse"
    )
    BARK_FINE = ModelInfo(repo_id="suno/bark", file_name="fine_2.pt", model_type="fine")

    ENCODEC24k = ModelInfo(
        checkpoint_name="facebook/encodec_24khz", model_type="encodec"
    )

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
    if not os.path.exists(model_file_path):
        raise RuntimeError(
            f"not found a file at {model_file_path} but it should be there at this point"
        )

    # we know that bark's models are saved by torch.save()
    checkpoint = torch.load(
        model_file_path, map_location=torch.device(env.DEVICE), weights_only=False
    )
    if model_info.model_type not in ["text", "coarse", "fine"]:
        raise ValueError(f"unknown how to load model_type {model_info.model_type}")

    ConfigClass, ModelClass = (
        (GPTConfig, GPT)
        if model_info.model_type in ["text", "coarse"]
        else (FineGPTConfig, FineGPT)
    )

    model_args = checkpoint["model_args"]
    if "input_vocab_size" not in model_args:
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        del model_args["vocab_size"]

    conf = ConfigClass(**checkpoint["model_args"])
    model = ModelClass(conf)

    state_dict: Dict[str, Any] = checkpoint["model"]  # type of state_dict?
    # this fix is specific to this model
    state_dict = _update_bark_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=False)

    n_params = model.get_num_params()
    val_loss = checkpoint["best_val_loss"].item()
    logger.info(
        f"model loaded: {round(n_params/1e6,1)}M params, {round(val_loss,3)} loss"
    )
    model.eval()
    del checkpoint, state_dict
    clear_cuda_cache()

    if model_info.model_type == "text":
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        return Model(model, conf, tokenizer)

    return Model(model)


def _update_bark_state_dict(model: GPT, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    unwanted_prefix = "_orig_mod."
    # make a copy of the state_dict's keys,
    # it is dangerous to loop over a list while mutating that list
    keys = list(state_dict.keys())

    for key in keys:
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)

    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = set([k for k in extra_keys if not k.endswith(".attn.bias")])
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set([k for k in missing_keys if not k.endswith(".attn.bias")])
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    return state_dict


# load the Facebook's encodec model
@validate_call
def _load_codec_model(
    model_type: EncodecModelType = EncodecModelType.ENCODEC24,
    target_bandwidth: EncodecTargetBandwidth = EncodecTargetBandwidth.BANDWIDTH_6,
) -> None:
    if model_type == EncodecModelType.ENCODEC24:
        assert target_bandwidth in [
            1.5,
            3,
            6,
            12,
        ], "target_bandwidth of a 24khz model must be one of [1.5, 3, 6, 12], received {target_bandwidth}"
    else:
        assert target_bandwidth in [
            3,
            6,
            12,
            24,
        ], f"target_bandwidth of a 48khz model must be one of [3, 6, 12, 24], received {target_bandwidth}"

    model = (
        EncodecModel.encodec_model_24khz()
        if model_type == EncodecModelType.ENCODEC24
        else EncodecModel.encodec_model_48khz()
    )

    model.encode()
    return Model(
        model,
        None,
    )


torch_models = TorchModels()
