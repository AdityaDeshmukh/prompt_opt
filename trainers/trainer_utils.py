import torch
from torch import nn, optim
import numpy as np
import random
from typing import Optional

def create_optimizer(model: nn.Module, learning_rate: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=learning_rate)

def step_optimizer(
    optimizer: optim.Optimizer,
    model: nn.Module,
    gradient_clip: bool,
    gradient_clip_norm: float
) -> None:
    if gradient_clip:
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

def set_random_seed(seed: Optional[int]) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
