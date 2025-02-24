import torch
from dataclasses import dataclass
import numpy as np

from core.utils import save_dict_to_msgpack, load_dict_from_msgpack


@dataclass
class BarkPrompt:
    """
    semantic_prompt shape: (T)
    coarse_prompt shape: (2, T)
    fine_prompt shape: (8, T)
    those T are different depends on the rate of token type per second
    """

    semantic_prompt: torch.Tensor
    coarse_prompt: torch.Tensor
    fine_prompt: torch.Tensor

    def save_prompt(self, file_path: str) -> bool:
        """
        Save all 3 prompts to disk. Return True if success, False if error
        """
        data = {
            "semantic_prompt": self.semantic_prompt.detach().cpu().tolist(),
            "coarse_prompt": self.coarse_prompt.detach().cpu().tolist(),
            "fine_prompt": self.fine_prompt.detach().cpu().tolist(),
        }

        if not file_path.endswith(".msgpack"):
            file_path += ".msgpack"

        return save_dict_to_msgpack(dictionary=data, file_path=file_path)

    @classmethod
    def load_prompt(self, file_path: str, device: torch.device) -> "BarkPrompt":
        """
        Load a prompt from disk. File to load could be a .npz or a .msgpack
        """

        if file_path.endswith(".msgpack"):
            prompt = load_dict_from_msgpack(file_path=file_path)
        elif file_path.endswith(".npz"):
            prompt = np.load(file_path)
        else:
            raise ValueError("don't know how to load this file")

        assert (
            prompt["semantic_prompt"] is not None
            and prompt["coarse_prompt"] is not None
            and prompt["fine_prompt"] is not None
        ), f"invalid prompt data {prompt}"

        semantic_prompt = torch.tensor(prompt["semantic_prompt"]).to(
            device=device, dtype=torch.int32
        )
        coarse_prompt = torch.tensor(prompt["coarse_prompt"]).to(
            device=device, dtype=torch.int32
        )
        fine_prompt = torch.tensor(prompt["fine_prompt"]).to(
            device=device, dtype=torch.int32
        )
        return BarkPrompt(semantic_prompt, coarse_prompt, fine_prompt)
