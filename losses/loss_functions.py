import torch
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any

from losses import loss_utils


def score_scaler_fnc(lmbda_):
    return 2.2111*10**lmbda_ - 2.1111
def plgo_loss(
        lmbda: torch.Tensor,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        scores_tensor: torch.Tensor,
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        num_src: int,
        beta: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

    lmbda_ = torch.repeat_interleave(lmbda,scores_tensor.shape[0]//num_src).view(num_src, -1)
    H = get_log_prob_diff(logits=logits, logits_=logits_, actions=actions)

    # H = beta*H.contiguous().view(num_src, -1) 
    H = 0.5*beta*H.contiguous().view(num_src, -1) + 0.5*score_scaler_fnc(lmbda_)*scores_tensor.contiguous().view(num_src, -1)

    scores_tensor = scores_tensor.contiguous().view(num_src, -1)
    _, indices = torch.sort(scores_tensor)

    T = torch.gather(H, dim=-1, index=indices)
    U = torch.logcumsumexp(T, dim=-1)
    T = U-T

    loss = T.sum(dim=-1).mean()
    loss_log = {
        "loss": loss
    }
    return loss, loss_log

def plgo_b_loss(
        lmbda: torch.Tensor,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        scores_tensor: torch.Tensor,
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        num_src: int,
        score_scaler: float,
        beta: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

    lmbda_ = torch.repeat_interleave(lmbda,scores_tensor.shape[0]//num_src).view(num_src, -1)
    H = get_log_prob_diff(logits=logits, logits_=logits_, actions=actions)
    H = beta*H.contiguous().view(num_src, -1)

    scores_tensor = score_scaler_fnc(lmbda_)*scores_tensor.contiguous().view(num_src, -1)
    _, indices = torch.sort(scores_tensor, dim=-1)

    T = torch.gather(H, dim=-1, index=indices)
    U = torch.logcumsumexp(T, dim=-1)
    input = T-U
    input = input.sum(dim=-1)

    target_A = torch.gather(scores_tensor, dim=-1, index=indices)
    target_B = torch.logcumsumexp(target_A, dim=-1)
    target = target_A-target_B
    target = target.sum(dim=-1)

    loss = F.mse_loss(input=input, target=target)
    loss_log = {
        "loss": loss
    }

    return loss, loss_log

def drgo_loss(
        lmbda: torch.Tensor,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        scores_tensor: torch.Tensor,
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        num_src: int,
        beta: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

    lmbda_ = torch.repeat_interleave(lmbda,scores_tensor.shape[0]//num_src).view(num_src, -1)
    H = get_log_prob_diff(logits=logits, logits_=logits_, actions=actions)
    H = beta*H.contiguous().view(num_src, -1)

    scores_tensor = score_scaler_fnc(lmbda_)*scores_tensor.contiguous().view(num_src, -1)
    _, indices = torch.sort(scores_tensor, dim=-1)

    T = torch.gather(H, dim=-1, index=indices)
    U = torch.gather(scores_tensor, dim=-1, index=indices)

    loss = F.mse_loss(input=get_pairwise_diff(T), target=get_pairwise_diff(U))
    loss_log = {
        "loss": loss
    }

    return loss, loss_log

def drgo_new_loss(
        lmbda: torch.Tensor,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        scores_tensor: torch.Tensor,
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        num_src: int,
        beta: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

    # lmbda_ = torch.repeat_interleave(lmbda,scores_tensor.shape[0]//num_src).view(num_src, -1)
    h = get_log_prob_diff(logits=logits, logits_=logits_, actions=actions)
    h = beta*h.contiguous().view(num_src, -1)

    scores_tensor = scores_tensor.contiguous().view(num_src, -1)
    scores_tensor = score_scaler_fnc(lmbda)*scores_tensor
    scores_tensor /= scores_tensor.std(dim=-1, keepdim=True)
    _, indices = torch.sort(scores_tensor, dim=-1)

    t = torch.gather(h, dim=-1, index=indices)
    u = torch.logcumsumexp(t, dim=-1)
    t = u-t

    loss = F.huber_loss(input=get_pairwise_diff(h), target=get_pairwise_diff(scores_tensor), delta=10) + 0.1*t.sum(dim=-1).mean()
    # loss = F.mse_loss(input=get_pairwise_diff(T), target=get_pairwise_diff(U))
    loss_log = {
        "loss": loss
    }

    return loss, loss_log

def drgo_cauchy_loss(
        lmbda: torch.Tensor,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        scores_tensor: torch.Tensor,
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        num_src: int,
        beta: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

    lmbda_ = torch.repeat_interleave(lmbda,scores_tensor.shape[0]//num_src).view(num_src, -1)
    h = get_log_prob_diff(logits=logits, logits_=logits_, actions=actions)
    h = beta*h.contiguous().view(num_src, -1)

    scores_tensor = scores_tensor.contiguous().view(num_src, -1)
    scores_tensor = score_scaler_fnc(lmbda_)*scores_tensor
    # scores_tensor /= scores_tensor.std(dim=-1, keepdim=True)

    c = math.sqrt(0.5)
    input = get_pairwise_diff(h)
    target = get_pairwise_diff(scores_tensor)
    loss = torch.log(1+0.5*torch.pow((input-target)/c,2.)).mean(dim=-1).mean()
    # loss = F.mse_loss(input=get_pairwise_diff(T), target=get_pairwise_diff(U))
    loss_log = {
        "loss": loss
    }

    return loss, loss_log

def drgo_l1_loss(
        lmbda: torch.Tensor,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        scores_tensor: torch.Tensor,
        content_tensor: torch.Tensor,    
        style_tensor: torch.Tensor,
        num_src: int,
        beta: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:    

    lmbda_ = torch.repeat_interleave(lmbda,scores_tensor.shape[0]//num_src).view(num_src, -1)    
    h = get_log_prob_diff(logits=logits, logits_=logits_, actions=actions)
    h = beta*h.contiguous().view(num_src, -1)

    scores_tensor = scores_tensor.contiguous().view(num_src, -1)
    scores_tensor = score_scaler_fnc(lmbda_)*scores_tensor
    # scores_tensor /= scores_tensor.std(dim=-1, keepdim=True)

    loss = F.l1_loss(input=get_pairwise_diff(h), target=get_pairwise_diff(scores_tensor))
    # loss = F.mse_loss(input=get_pairwise_diff(T), target=get_pairwise_diff(U))
    loss_log = {
        "loss": loss
    }
    return loss, loss_log

def drgo_regularized_loss(
        lmbda: torch.Tensor,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        scores_tensor: torch.Tensor,
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        num_src: int,
        beta: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

    lmbda_ = torch.repeat_interleave(lmbda,scores_tensor.shape[0]//num_src).view(num_src, -1)
    h = get_log_prob_diff(logits=logits, logits_=logits_, actions=actions)
    scores_tensor = score_scaler_fnc(lmbda_)*scores_tensor.contiguous().view(num_src, -1)

    h = beta*h.contiguous().view(num_src, -1)
    _, indices = torch.sort(scores_tensor)

    t = torch.gather(0.5*h+0.5*scores_tensor, dim=-1, index=indices)
    u = torch.logcumsumexp(t, dim=-1)
    t = u-t

    loss = F.mse_loss(input=get_pairwise_diff(h), target=get_pairwise_diff(scores_tensor)) + 0.1*t.sum(dim=-1).mean()
    loss_log = {
        "loss": loss
    }
    return loss, loss_log

def grpo_loss(
        lmbda: torch.Tensor,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        scores_tensor: torch.Tensor,
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        num_src: int,
        beta: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
    num_repeats = scores_tensor.shape[0]//num_src
    lmbda_ = torch.repeat_interleave(lmbda,num_repeats).view(num_src, -1)
    scores_tensor = score_scaler_fnc(lmbda_)*scores_tensor.contiguous().view(num_src, num_repeats)
    advantage = (scores_tensor-scores_tensor.mean(dim=-1, keepdim=True))/scores_tensor.std(dim=-1, keepdim=True)
    advantage = advantage.view(-1)

    per_token_log_p = get_per_token_log_prob(logits=logits, actions=actions).transpose(0,1)
    num_tokens = per_token_log_p.shape[0]
    old_per_token_log_p = per_token_log_p.detach()
    per_token_log_p_ref = get_per_token_log_prob(logits=logits_, actions=actions).transpose(0,1)

    log_ratio = per_token_log_p_ref - per_token_log_p
    kl_penalty = log_ratio.exp() - log_ratio - 1

    ratio = torch.exp(per_token_log_p - old_per_token_log_p)
    
    term_1 = ratio * advantage
    term_2 = ratio.clamp(1 - 0.2, 1 + 0.2) * advantage
    loss = -torch.min(term_1, term_2) + beta*kl_penalty
    loss = loss.transpose(0,1).contiguous().view(num_src, num_repeats, num_tokens)
    loss = (loss.sum(dim=-1).sum(dim=-1)/(num_repeats*num_tokens)).mean()
    loss_log = {
        "loss": loss
    }

    return loss, loss_log

def kl_ub_loss(
        lmbda: torch.Tensor,
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor,
        scores_tensor: torch.Tensor,
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        num_src: int,
        beta: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

    num_repeat = scores_tensor.shape[0]//num_src
    lmbda_ = torch.repeat_interleave(lmbda,num_repeat).view(num_src, -1)
    scores_tensor = score_scaler_fnc(lmbda_)*scores_tensor.contiguous().view(num_src, -1)

    log_p = get_log_prob(logits=logits, actions=actions).view(num_src, -1)
    log_p_star = get_log_prob(logits=logits_, actions=actions).view(num_src, -1) + 1/beta*(scores_tensor - beta*(torch.logsumexp(1/beta*scores_tensor, dim=-1, keepdim=True)-math.log(num_repeat)))

    log_ratio = log_p_star - log_p
    kl_loss = log_ratio.exp() - log_ratio - 1

    loss = kl_loss.mean(dim=-1).mean()

    loss_log = {
        "loss": loss
    }

    return loss, loss_log

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
