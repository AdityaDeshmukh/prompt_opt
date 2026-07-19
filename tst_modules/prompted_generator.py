"""Prompted task-LM generation: templating, length control, postprocessing.

The actual token generation is delegated to a backend (vLLM or transformers,
see task_lm.py); this class owns everything benchmark-specific — the prompt
template, per-source output-length budgets, and output truncation rules.
"""
import math
from typing import Optional, List

import torch
from transformers import AutoConfig, AutoTokenizer

from .task_lm import build_task_lm


class PromptedGenerator:
    def __init__(
        self,
        model: str,
        template: str,
        end_punct: str,
        pad_token: str,
        device_id,
        lower_outputs: bool,
        control_output_length: bool,
        dtype: str = 'bfloat16',
        max_gen_batch_size: int = 400,
        backend: str = 'hf',
        vllm_gpu_memory_utilization: float = 0.35,
        vllm_seed: Optional[int] = None,
    ):
        self.device = torch.device(device_id) if not isinstance(device_id, torch.device) \
            else device_id
        # tokenizer only for source-length budgets; the backend owns decoding
        self.tokenizer = AutoTokenizer.from_pretrained(model,
                                                       pad_token=pad_token)
        self.n_positions = AutoConfig.from_pretrained(model).max_position_embeddings

        self.template = template
        self.end_punct = end_punct
        self.lower_outputs = lower_outputs
        self.control_output_length = control_output_length

        self.task_lm = build_task_lm(
            backend, model, dtype=dtype, device=self.device,
            pad_token=pad_token, max_gen_batch_size=max_gen_batch_size,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_seed=vllm_seed)

    def _get_max_new_tokens(
        self,
        seq_len: int,
        control_output_length: Optional[bool] = None
    ) -> Optional[int]:
        if control_output_length is None:
            control_output_length = self.control_output_length
        if control_output_length:
            # This hack tends to speed up generation compared to default
            return max(1.5 * seq_len, seq_len + 10)
        else:
            return None

    def sample_generate_grouped(
        self,
        prompts: List[str],
        source_texts: List[str],
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
    ) -> List[List[str]]:
        """Generate num_samples continuations for every (prompt, source) row.

        Each row gets its own max_new_tokens budget from its source length
        (ceil for exact parity with the historical float cap: a cap of e.g.
        13.5 effectively allowed 14 tokens).
        """
        assert len(prompts) == len(source_texts)

        templates = [self.template.format(prompt=p, sentence_1=s)
                     for p, s in zip(prompts, source_texts)]
        src_lens = [len(self.tokenizer(s)['input_ids']) for s in source_texts]
        max_new = [self._get_max_new_tokens(l, control_output_length)
                   for l in src_lens]
        if any(m is None for m in max_new):
            # no length control: cap by what actually fits the context after
            # the full formatted template (not just the source sentence)
            tmpl_lens = [len(self.tokenizer(t)['input_ids']) for t in templates]
            max_new = [max(1, min(self.n_positions // 2,
                                  self.n_positions - l - 1))
                       for l in tmpl_lens]
        max_new = [int(math.ceil(m)) for m in max_new]

        raw = self.task_lm.generate(templates, max_new, num_samples,
                                    top_k, top_p)
        return [[self.postprocess_output(t, lower_outputs=lower_outputs)
                 for t in row] for row in raw]

    def sample_generate(
        self,
        prompt: str,
        source_text: str,
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
        **kwargs
    ) -> List[str]:
        return self.sample_generate_grouped(
            [prompt], [source_text], num_samples, top_k, top_p,
            lower_outputs=lower_outputs,
            control_output_length=control_output_length)[0]

    def sample_generate_batch(
        self,
        prompt: str,
        source_texts: List[str],
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
        **kwargs
    ) -> List[List[str]]:
        return self.sample_generate_grouped(
            [prompt] * len(source_texts), source_texts, num_samples,
            top_k, top_p, lower_outputs=lower_outputs,
            control_output_length=control_output_length)

    def postprocess_output(
        self,
        text: str,
        end_punct: Optional[str] = None,
        lower_outputs: Optional[bool] = None
    ) -> str:
        if end_punct is None:
            end_punct = self.end_punct
        if lower_outputs is None:
            lower_outputs = self.lower_outputs

        try:
            end = text.index(end_punct)
        except ValueError:
            end = len(text)
        text = text[:end].strip()

        try:
            end = text.index('.')
        except ValueError:
            end = len(text)
        try:
            end = min(end, text.index('!'))
        except ValueError:
            end = end
        try:
            end = min(end, text.index('?'))
        except ValueError:
            end = end

        text = text[:end+1].strip()
        if lower_outputs:
            text = text.lower()

        return text
