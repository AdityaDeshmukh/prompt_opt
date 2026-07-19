"""Unified loss family for the GRPO-vs-R-REBEL comparison.

R-REBEL fits  r_i - r_j  ~  beta * (logratio_i - logratio_j)  over all
unordered pairs within a group of G outputs sharing a source x, with a
selectable discrepancy d. Every variant applies the same reward
preprocessing (score_scale * score_scaler_fnc(lambda), then optional
per-group std scaling). Note: for *_std variants the std division exactly
cancels any positive per-group scale, so score_scale and the lambda scaler
only affect the non-std variants (and GRPO's advantage normalization
cancels them likewise) — the regression target is scale-normalized, which
is the paper's r/std preprocessing.

Legacy pre-v2 losses live in legacy_losses.py.
"""
from typing import Tuple, Dict, Any

import torch
import torch.nn.functional as F

from losses import loss_utils

STD_EPS = 1e-8


def score_scaler_fnc(lmbda_):
    return 2.2111*10**lmbda_ - 2.1111


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V] -> mean per-token entropy of the policy
    logp = F.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1).mean()


def _pairwise_diff_triu(x: torch.Tensor) -> torch.Tensor:
    # x: [S, G] -> all unordered-pair differences [S, G*(G-1)/2]
    G = x.shape[-1]
    i, j = torch.triu_indices(G, G, offset=1, device=x.device)
    return x[:, i] - x[:, j]


def _rrebel_core(
        lmbda: torch.Tensor,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        scores_tensor: torch.Tensor,
        num_src: int,
        d_kind: str,
        beta: float,
        score_scale: float,
        reward_std_scale: bool,
        ent_coef: float,
        huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    lmbda_ = torch.repeat_interleave(
        lmbda, scores_tensor.shape[0]//num_src).view(num_src, -1)
    h = get_log_prob_diff(logits=logits, logits_=logits_, actions=actions)
    h = beta*h.contiguous().view(num_src, -1)

    scores = scores_tensor.contiguous().view(num_src, -1)
    scores = score_scale*score_scaler_fnc(lmbda_)*scores
    if reward_std_scale:
        scores = scores / (scores.std(dim=-1, keepdim=True) + STD_EPS)

    input = _pairwise_diff_triu(h)
    target = _pairwise_diff_triu(scores)

    if d_kind == 'l2':
        loss = F.mse_loss(input=input, target=target)
    elif d_kind == 'l1':
        loss = F.l1_loss(input=input, target=target)
    elif d_kind == 'huber':
        loss = F.huber_loss(input=input, target=target, delta=huber_delta)
    elif d_kind == 'cauchy':
        c2 = 0.5  # c = sqrt(0.5)
        loss = torch.log1p(0.5*torch.pow(input-target, 2.)/c2).mean()
    else:
        raise ValueError(f"Unknown d_kind: {d_kind}")

    loss_log = {"loss": loss, "pair_residual": (input-target).abs().mean()}

    if ent_coef > 0:
        entropy = _entropy_from_logits(logits)
        loss = loss - ent_coef*entropy
        loss_log["entropy"] = entropy
        loss_log["loss"] = loss

    return loss, loss_log


def make_rrebel_loss(d_kind: str, reward_std_scale: bool, use_entropy: bool):
    def _loss(
            lmbda: torch.Tensor,
            logits: torch.Tensor,
            logits_: torch.Tensor,
            actions: torch.LongTensor,
            scores_tensor: torch.Tensor,
            content_tensor: torch.Tensor,
            style_tensor: torch.Tensor,
            num_src: int,
            beta: float = 0.5,
            score_scale: float = 0.1,
            ent_coef: float = 0.0,
            huber_delta: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return _rrebel_core(
            lmbda=lmbda, logits=logits, logits_=logits_, actions=actions,
            scores_tensor=scores_tensor, num_src=num_src, d_kind=d_kind,
            beta=beta, score_scale=score_scale,
            reward_std_scale=reward_std_scale,
            ent_coef=(ent_coef if use_entropy else 0.0),
            huber_delta=huber_delta)
    _loss.__name__ = f"rrebel_{d_kind}{'_std' if reward_std_scale else ''}" \
                     f"{'_ent' if use_entropy else ''}_loss"
    return _loss


def grpo_v2_loss(
        lmbda: torch.Tensor,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        scores_tensor: torch.Tensor,
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        num_src: int,
        beta: float = 0.0,
        score_scale: float = 0.1,
        ent_coef: float = 0.0,
        huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """GRPO with (i) epsilon-guarded per-group advantage normalization and
    (ii) beta taken from config (0 keeps the standard KL-free GRPO). The
    single-update ratio==1 construction is canonical; its gradient equals
    REINFORCE with group-normalized advantage."""
    num_repeats = scores_tensor.shape[0]//num_src
    scores = scores_tensor.contiguous().view(num_src, num_repeats)
    advantage = (scores - scores.mean(dim=-1, keepdim=True)) \
        / (scores.std(dim=-1, keepdim=True) + STD_EPS)
    advantage = advantage.view(-1, 1)

    per_token_log_p = get_per_token_log_prob(logits=logits, actions=actions)
    ratio = torch.exp(per_token_log_p - per_token_log_p.detach())
    term_1 = ratio * advantage
    term_2 = ratio.clamp(1 - 0.2, 1 + 0.2) * advantage
    loss = -torch.min(term_1, term_2)

    if beta > 0:
        per_token_log_p_ref = get_per_token_log_prob(logits=logits_,
                                                     actions=actions)
        log_ratio = per_token_log_p_ref - per_token_log_p
        kl_penalty = log_ratio.exp() - log_ratio - 1
        loss = loss + beta*kl_penalty

    # fixed-length prompts: token-mean then member/group mean
    loss = loss.mean()
    loss_log = {"loss": loss, "advantage_abs": advantage.abs().mean()}
    return loss, loss_log


# --------------------------------------------------------------------------
# Shared log-probability helpers (used by both families).
# --------------------------------------------------------------------------

def get_log_prob(
        logits: torch.Tensor,
        actions: torch.LongTensor
) -> torch.Tensor:
    q = loss_utils.gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    v = logits.logsumexp(dim=-1)
    a = q - v
    log_prob = a.sum(dim=-1)
    return log_prob

def get_per_token_log_prob(
        logits: torch.Tensor,
        actions: torch.LongTensor
) -> torch.Tensor:
    q = loss_utils.gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    v = logits.logsumexp(dim=-1)
    per_token_log_prob = q - v
    return per_token_log_prob


def get_log_prob_diff(
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor
) -> torch.Tensor:
    log_prob = get_log_prob(logits=logits, actions=actions)
    log_prob_ref = get_log_prob(logits=logits_, actions=actions)

    return log_prob - log_prob_ref

def get_per_token_log_prob_diff(
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor
) -> torch.Tensor:
    per_token_log_prob = get_per_token_log_prob(logits=logits, actions=actions)
    per_token_log_prob_ref = get_per_token_log_prob(logits=logits_, actions=actions)

    return per_token_log_prob - per_token_log_prob_ref

def get_pairwise_diff(input: torch.Tensor) -> torch.Tensor:
    output = []
    for i in range(input.shape[0]):
        res = input[i].unsqueeze(1) - input[i]
        output.append(res.flatten())
    return torch.stack(output)
