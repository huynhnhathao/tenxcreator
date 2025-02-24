import torch
from dataclasses import dataclass


@dataclass
class BarkPrompt:
    semantic_prompt: torch.Tensor
    coarse_prompt: torch.Tensor
    fine_prompt: torch.Tensor

    @classmethod
    def empty(cls) -> "BarkPrompt":
        """
        Create and return an empty BarkPrompt instance with zero tensors

        Returns:
            BarkPrompt: Empty prompt with zero-initialized tensors
        """
        return cls(
            semantic_prompt=torch.zeros(0, dtype=torch.int32),
            coarse_prompt=torch.zeros((2, 0), dtype=torch.int32),
            fine_prompt=torch.zeros((8, 0), dtype=torch.int32),
        )
