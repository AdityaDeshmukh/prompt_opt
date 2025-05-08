import torch
from torch import nn
from typing import Dict, List, Any, Tuple

class BaseScoreModule(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_scores(
        self,
        batch: Dict[str, Any],
        output_tokens: List[List[str]],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns:

        scores: torch.Tensor
        scores_log: Dict[str, Any]
        """
        raise NotImplementedError

    def _pre_steps(self, step: int) -> None:
        """Does what a module needs to do at the beginning of a training step

        Examples include syncing with target model for a Q-learning module"""
        pass