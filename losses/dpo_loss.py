import torch
import numpy as np
import torch.nn.functional as F
from functools import partial

from typing import List, Tuple, Dict, Any, Optional

from losses import loss_utils
from utils import utils


def dro_score_loss(
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

    log_Q = get_log_prob(logits=logits_, actions=actions)
    log_Q = F.log_softmax(log_Q+1/beta*scores_tensor)

    log_P = F.log_softmax(get_log_prob(logits=logits, actions=actions))
    log_M = torch.logsumexp(torch.stack((log_P, log_Q), dim=-1), dim=-1)-np.log(2)

    log_M = log_M.contiguous().view(num_src, -1)
    log_P = log_P.contiguous().view(num_src, -1)
    # log_Q = log_Q.contiguous().view(num_src, -1)
    loss = F.kl_div(input=log_M, target=log_P, log_target=True, reduction='batchmean') + \
            F.kl_div(input=log_M, target=log_Q, log_target=True, reduction='batchmean')
    # loss = (log_P-log_M).mean(dim=-1).mean(dim=-1)
    # H = get_log_prob_diff(logits=logits, logits_=logits_, actions=actions)
    # H = beta*H.contiguous().view(num_src, -1)
    # loss = H - scores_tensor.contiguous().view(num_src, -1)
    # loss = loss.mean(dim=-1).mean(dim=-1)
    loss_log = {
        "loss": loss.item()
    }
    return loss, loss_log

def dpo_score_loss(
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

    lmbda_ = torch.repeat_interleave(lmbda,content_tensor.shape[0]//num_src).view(num_src, -1)
    H = get_log_prob_diff(logits=logits, logits_=logits_, actions=actions)
    # H = beta*H.contiguous().view(num_src, -1) 
    
    H = 0.5*beta*H.contiguous().view(num_src, -1) + 0.5*score_scaler*(9*lmbda_+1)*scores_tensor.contiguous().view(num_src, -1)

    # scores = torch.where(content_tensor > lmbda_, style_tensor, content_tensor-lmbda_)
    # scores = scores.contiguous().view(num_src, -1)
    # _, indices = torch.sort(scores)

    scores_tensor = scores_tensor.contiguous().view(num_src, -1)
    _, indices = torch.sort(scores_tensor)

    T = torch.gather(H, dim=-1, index=indices)
    U = torch.logcumsumexp(T, dim=-1)
    T = U-T

    loss = T.sum(dim=-1).mean()
    loss_log = {
        "loss": loss
    }
    # loss = beta*H.sum()
    # loss_log = {
    #     "loss": loss
    # }
    # return loss, loss_log

    # _, indices = torch.sort(torch.stack(scores_list))
    # print(indices)
    # print(torch.stack(scores_list))
    # T = []
    # for i in indices:
    #     H = get_log_prob_diff(logits=logits_list[i], logits_=logits_list_[i], actions=actions_list[i])
    #     T.append(beta*H)

    # T = torch.stack(T)
    # U = torch.logcumsumexp(T, dim=0)
    # T = U-T
    # loss = T.sum()

    # loss_log = {
    #     "loss": loss
    # }
    # return loss, loss_log
    return loss, loss_log

def get_log_prob(
        logits: torch.Tensor,
        actions: torch.LongTensor
) -> torch.Tensor:
    Q = loss_utils.gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    V = logits.logsumexp(dim=-1)
    A = Q - V
    L = A.sum(dim=-1)
    return L
    
    
def get_log_prob_diff(
        logits: torch.Tensor,
        logits_: torch.Tensor,
        actions: torch.LongTensor
) -> torch.Tensor:
    L = get_log_prob(logits=logits, actions=actions)
    L_ = get_log_prob(logits=logits_, actions=actions)

    H = L - L_
    return H

def kl_div_loss(
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

    lmbda_ = torch.repeat_interleave(lmbda,content_tensor.shape[0]//num_src).view(num_src, -1)
    H = get_log_prob_diff(logits=logits, logits_=logits_, actions=actions)
    H = beta*H.contiguous().view(num_src, -1)

    scores_tensor = score_scaler*(9*lmbda_+1)*scores_tensor.contiguous().view(num_src, -1)
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