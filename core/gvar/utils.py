import os

import torch
from huggingface_hub import hf_hub_download
from core.gvar import *


# load a model to memory and store to the gvar.model module namespace
# if a model doesn't exist in the cache, download it from huggingface hub
# do nothing if the model already loaded in the global scope
def load_model(model_name: str) -> None:
    pass


def grab_best_device(use_gpu: bool) -> torch.device:
    if torch.cuda.device_count() > 0 and use_gpu:
        device = torch.device("cuda")
    elif (
        torch.backends.mps.is_available() and use_gpu and globals()["GLOBAL_ENABLE_MPS"]
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def download_model_from_hugging_face(
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
