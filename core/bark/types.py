import torch
from dataclasses import dataclass


@dataclass
class BarkPrompt:
    semantic_prompt: torch.Tensor
    coarse_prompt: torch.Tensor
    fine_prompt: torch.Tensor
