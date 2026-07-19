import torch
from torch import nn
from typing import Dict, List, Tuple, Optional

from transformers.pytorch_utils import Conv1D

from .lm_adaptor_model import LMAdaptorModel
from omegaconf import DictConfig


def _target_in_out(module: nn.Module) -> Tuple[int, int]:
    if isinstance(module, Conv1D):
        # GPT-2 Conv1D: weight [d_in, d_out]
        return module.weight.shape[0], module.weight.shape[1]
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    raise TypeError(f"Unsupported LoRA target module: {type(module)}")


class DeltaRegistry:
    """Mutable holder for the currently active per-sample LoRA deltas.

    Shared by reference between the wrapped target modules and the model;
    copy.deepcopy of the whole model clones it consistently, keeping
    policy/reference deltas independent.
    """
    def __init__(self):
        self.deltas: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def __deepcopy__(self, memo):
        # deltas are transient per-rollout tensors (possibly non-leaf, which
        # deepcopy cannot handle); the copy starts empty and is recomputed by
        # _prepare_conditioning. Memoized so model/wrapper copies still share
        # one registry.
        new = DeltaRegistry()
        memo[id(self)] = new
        return new


class HyperLoRAWrapper(nn.Module):
    """Wraps a Conv1D/Linear; adds a per-sample low-rank delta to its output.

    out = base(x) + scale * ((x @ A_b^T) @ B_b^T)  with A_b [B, r, d_in] and
    B_b [B, d_out, r] taken from the registry (generated per batch from
    lambda). Holds no parameters of its own.
    """
    def __init__(self, base: nn.Module, name: str, registry: DeltaRegistry,
                 scale: float):
        super().__init__()
        self.base = base
        self.name = name
        self.scale = scale
        object.__setattr__(self, '_registry', registry)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        entry = self._registry.deltas.get(self.name)
        if entry is None:
            return out
        A, B = entry  # [B, r, d_in], [B, d_out, r]
        squeeze = (x.dim() == 2)
        if squeeze:
            x = x.unsqueeze(1)
            out = out.unsqueeze(1)
        low = torch.einsum('btd,brd->btr', x, A)
        delta = torch.einsum('btr,bor->bto', low, B)
        out = out + self.scale * delta
        if squeeze:
            out = out.squeeze(1)
        return out


class LoRAHyperNet(nn.Module):
    """lambda [B, 1] -> per-target LoRA factors A [B, r, d_in], B [B, d_out, r].

    B-heads are zero-initialized so the policy exactly equals the frozen base
    model at initialization.
    """
    def __init__(self, target_shapes: Dict[str, Tuple[int, int]], rank: int,
                 embed_dim: int):
        super().__init__()
        self.rank = rank
        self.target_shapes = dict(target_shapes)
        self.trunk = nn.Sequential(
            nn.Linear(1, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.heads_A = nn.ModuleDict()
        self.heads_B = nn.ModuleDict()
        for name, (d_in, d_out) in self.target_shapes.items():
            key = name.replace('.', '/')
            head_A = nn.Linear(embed_dim, rank * d_in)
            nn.init.normal_(head_A.weight, std=0.02)
            nn.init.zeros_(head_A.bias)
            head_B = nn.Linear(embed_dim, d_out * rank)
            nn.init.zeros_(head_B.weight)
            nn.init.zeros_(head_B.bias)
            self.heads_A[key] = head_A
            self.heads_B[key] = head_B

    def forward(self, lmbda: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        if lmbda.dim() == 1:
            lmbda = lmbda.unsqueeze(-1)
        e = self.trunk(lmbda)  # [B, embed_dim]
        out = {}
        for name, (d_in, d_out) in self.target_shapes.items():
            key = name.replace('.', '/')
            A = self.heads_A[key](e).view(-1, self.rank, d_in)
            B = self.heads_B[key](e).view(-1, d_out, self.rank)
            out[name] = (A, B)
        return out


class LMHyperLoRAModel(LMAdaptorModel):
    """lambda-conditioned LoRA hypernetwork on a frozen causal LM.

    A small hypernetwork maps the constraint value lambda to per-sample
    rank-r LoRA deltas on the attention projections; logits come from the
    frozen lm_head. The trainable parameters are the hypernetwork only.
    """
    def __init__(self, config: "DictConfig"):
        # skip LMAdaptorModel.__init__ (which builds the MLP head); build the
        # shared backbone plumbing directly
        nn.Module.__init__(self)
        self._init_backbone(config)
        if self.fluent:
            raise NotImplementedError(
                "fluent=true is not supported with adaptor_type=hyperlora")

        rank = int(config.get('lora_rank', 4))
        alpha = float(config.get('lora_alpha', 8.0))
        embed_dim = int(config.get('lora_embed_dim', 32))
        targets: List[str] = list(config.get('lora_targets', ['attn.c_attn']))
        scale = alpha / rank

        registry = DeltaRegistry()
        object.__setattr__(self, 'delta_registry', registry)

        target_shapes: Dict[str, Tuple[int, int]] = {}
        wrapped = 0
        for name, module in list(self.backbone.named_modules()):
            if not any(name.endswith(t) for t in targets):
                continue
            parent_name, _, child_name = name.rpartition('.')
            parent = self.backbone.get_submodule(parent_name) if parent_name \
                else self.backbone
            target_shapes[name] = _target_in_out(module)
            setattr(parent, child_name,
                    HyperLoRAWrapper(module, name, registry, scale))
            wrapped += 1
        if wrapped == 0:
            raise ValueError(f"No modules matched lora_targets={targets}")
        print(f"HyperLoRA: wrapped {wrapped} modules: {sorted(target_shapes)}")

        self.hyper = LoRAHyperNet(target_shapes, rank, embed_dim).to(self.device)
        print("Number of parameters in hyperlora:",
              sum(p.numel() for p in self.hyper.parameters()))

    def _prepare_conditioning(self, lmbda: torch.tensor) -> None:
        self.delta_registry.deltas = self.hyper(lmbda)

    def _finish_conditioning(self) -> None:
        # drop the per-call deltas so a stale (wrong-lambda / wrong-batch)
        # delta can never leak into a later backbone forward
        self.delta_registry.deltas = {}

    def _adapted_logits(self, lmbda: torch.tensor, state: torch.Tensor) -> torch.Tensor:
        # conditioning happens inside the backbone via the LoRA deltas; the
        # frozen head maps the modulated hidden state to logits
        return self.lm_head(state)
