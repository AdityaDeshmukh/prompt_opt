import torch
def _top_k_logits(
    logits: torch.Tensor,
    k: int
) -> torch.Tensor:
    r"""Adapted from
    https://github.com/openai/gpt-2/blob/master/src/sample.py#L63-L77
    """
    if k == 0:
        # no truncation
        return logits

    values, _ = torch.topk(logits, k=k)
    min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_values,
        torch.full_like(logits, float('-inf')), logits)

def _top_km_logits(
    logits: torch.Tensor,
    k: int,
    m: int | torch.Tensor
) -> torch.Tensor:
    r"""Adapted from
    https://github.com/openai/gpt-2/blob/master/src/sample.py#L63-L77
    """
    if k == 0 or k >= logits.shape[-1]:
        # no truncation
        return logits
    if isinstance(m, int):
        values, _ = torch.topk(logits, k=k+m)
        min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
        if m == 0:
            return torch.where(
                logits < min_values,
                torch.full_like(logits, float('-inf')), logits)
        max_values: torch.Tensor = values[:, m].unsqueeze(-1)
        return torch.where(
            torch.logical_or(logits < min_values, logits > max_values),
            torch.full_like(logits, float('-inf')), logits)
    else:
        min_values = []
        max_values = []
        for i in range(m.size(0)):
            values, _ = torch.topk(logits[i], k=k+m[i])
            min_values.append(values[-1])
            max_values.append(values[m[i]])
        min_values = torch.stack(min_values).unsqueeze(-1)
        max_values = torch.stack(max_values).unsqueeze(-1)
        return torch.where(
            torch.logical_or(logits < min_values, logits > max_values),
            torch.full_like(logits, float('-inf')), logits)

def _top_p_logits(
    logits: torch.Tensor,
    p: float
) -> torch.Tensor:
    r"""Adapted from
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317#file-top-k-top-p-py-L16-L27"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the
    # threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    for idx in range(logits.size(0)):
        batch_indices = sorted_indices[idx, sorted_indices_to_remove[idx]]
        logits[idx, batch_indices] = float("-inf")
    return logits
