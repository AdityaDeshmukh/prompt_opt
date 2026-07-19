"""Task-LM generation backends.

The task LM (gpt2-xl by default) is frozen and inference-only. Generating
`num_samples` continuations for every prompt row is ~98% of a training step
under the plain-transformers backend (measured 2026-07-15), so this is the
one component worth a specialized engine.

Backends implement a single method:

    generate(templates, max_new_tokens, num_samples, top_k, top_p)
        -> List[List[str]]   # [n_rows][num_samples] raw continuations

- HFTaskLM: transformers `model.generate` with left padding, rows sorted and
  chunked by `max_gen_batch_size` so each chunk only decodes as far as its
  longest member (the v2 path, kept as a dependency-free fallback).
- VLLMTaskLM: vLLM engine — continuous batching, paged KV cache, and
  copy-on-write prompt sharing for n>1. No manual chunking or sorting; every
  row is one request with its own max_tokens.
"""
import math
from typing import List, Optional

import torch


class HFTaskLM:
    def __init__(
        self,
        model: str,
        dtype: str,
        device: torch.device,
        pad_token: str,
        max_gen_batch_size: int = 400,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model,
                                                       pad_token=pad_token)
        # left padding is mandatory for batched decoder-only generation
        self.tokenizer.padding_side = 'left'
        torch_dtype = (getattr(torch, dtype)
                       if torch.cuda.is_available() else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch_dtype).to(self.device)
        self.model.eval()
        self.max_gen_batch_size = max_gen_batch_size

    @torch.no_grad()
    def generate(
        self,
        templates: List[str],
        max_new_tokens: List[int],
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
    ) -> List[List[str]]:
        n_rows = len(templates)
        # sort rows by their own budget so each chunk decodes only as far as
        # its longest member; truncate back per row afterwards
        order = sorted(range(n_rows), key=lambda i: max_new_tokens[i])
        results: List[Optional[List[str]]] = [None] * n_rows
        rows_per_chunk = max(1, self.max_gen_batch_size // num_samples)
        pad_token_id = self.tokenizer.pad_token_id

        for start in range(0, n_rows, rows_per_chunk):
            idxs = order[start:start + rows_per_chunk]
            chunk_templates = [templates[i] for i in idxs]
            chunk_max_new = max(max_new_tokens[i] for i in idxs)

            enc = self.tokenizer(chunk_templates, return_tensors='pt',
                                 padding=True).to(self.device)
            out = self.model.generate(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                do_sample=True,
                max_new_tokens=chunk_max_new,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_samples,
                pad_token_id=pad_token_id,
            )
            input_len = enc['input_ids'].shape[1]
            gen_ids = out[:, input_len:]

            for row_pos, i in enumerate(idxs):
                row_ids = gen_ids[row_pos*num_samples:(row_pos+1)*num_samples,
                                  :max_new_tokens[i]]
                results[i] = self.tokenizer.batch_decode(
                    row_ids, skip_special_tokens=True)

        return results


class VLLMTaskLM:
    def __init__(
        self,
        model: str,
        dtype: str,
        gpu_memory_utilization: float = 0.35,
        seed: Optional[int] = None,
        enforce_eager: bool = False,
    ):
        # import here so the hf backend has no vllm dependency
        from vllm import LLM, SamplingParams
        self._SamplingParams = SamplingParams
        kwargs = dict(model=model,
                      dtype=dtype,
                      gpu_memory_utilization=gpu_memory_utilization,
                      enable_prefix_caching=True,
                      enforce_eager=enforce_eager,
                      disable_log_stats=True)
        if seed is not None:  # vllm requires an int; omit to use its default
            kwargs['seed'] = seed
        self.llm = LLM(**kwargs)

    def generate(
        self,
        templates: List[str],
        max_new_tokens: List[int],
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
    ) -> List[List[str]]:
        # one request per row with its own token budget; vLLM handles
        # batching/scheduling internally and returns in request order
        params = [self._SamplingParams(
                      n=num_samples,
                      temperature=1.0,
                      top_k=(top_k if top_k else -1),  # -1 disables top-k
                      top_p=top_p,
                      max_tokens=m)
                  for m in max_new_tokens]
        outs = self.llm.generate(templates, params, use_tqdm=False)
        return [[o.text for o in out.outputs] for out in outs]


def build_task_lm(
    backend: str,
    model: str,
    dtype: str,
    device: torch.device,
    pad_token: str,
    max_gen_batch_size: int = 400,
    vllm_gpu_memory_utilization: float = 0.35,
    vllm_seed: Optional[int] = None,
):
    """Build the requested backend; fall back to transformers if vllm is
    requested but not installed (keeps the old env usable)."""
    if backend == 'vllm':
        try:
            return VLLMTaskLM(model, dtype=dtype,
                              gpu_memory_utilization=vllm_gpu_memory_utilization,
                              seed=vllm_seed)
        except ImportError:
            print("[task_lm] vllm not installed - falling back to the "
                  "transformers backend (task_lm_backend=hf)")
    elif backend != 'hf':
        raise ValueError(f"Unknown task_lm_backend: {backend}")
    return HFTaskLM(model, dtype=dtype, device=device, pad_token=pad_token,
                    max_gen_batch_size=max_gen_batch_size)
