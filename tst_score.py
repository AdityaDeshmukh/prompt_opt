import torch
import itertools
from typing import List, Tuple, Union, Dict, Any, Optional
from transformers import AutoTokenizer
from collections import defaultdict
from tst_modules import PromptedGenerator, TextStyleTransferObjectives
from omegaconf import DictConfig
from scores import BaseScore


class PromptedTextStyleTransferScore(BaseScore):
    def __init__(
        self,
        config: "DictConfig"
    ):

        generator_device = torch.device(config.device_scorer if torch.cuda.is_available() else 'cpu')  # TODO
        score_device = torch.device(config.device_scorer if torch.cuda.is_available() else 'cpu')  # TODO
        task_lm = config.task_lm
        style_classifier = config.style_classifier
        # Loading generator model
        print('Task LM:', task_lm)
        self.tokenizer = AutoTokenizer.from_pretrained(config.policy_lm)
        self.generator = PromptedGenerator(
            task_lm, config.template, config.end_punct,
            config.pad_token, generator_device,
            config.lower_outputs, config.control_output_length,
            dtype=config.get('task_lm_dtype', 'bfloat16'),
            max_gen_batch_size=config.get('max_gen_batch_size', 400),
            backend=config.get('task_lm_backend', 'hf'),
            vllm_gpu_memory_utilization=config.get(
                'vllm_gpu_memory_utilization', 0.35),
            vllm_seed=config.get('vllm_seed', None))
        self.top_k = config.task_top_k
        self.top_p = 1.0
        self.num_samples = config.num_samples
        self.num_bootstraps = config.num_bootstraps

        style_tokenizer = config.style_tokenizer
        # Loading reward models
        if style_tokenizer is None:
            style_tokenizer = style_classifier
        self.objectives = TextStyleTransferObjectives(
            style_classifier,
            style_tokenizer,
            config.style_batch_size,
            score_device,
            scorer_dtype=config.get('scorer_dtype', 'float16'))

        # Misc. training details
        self.num_repeats = config.num_repeats
        self._counter = 0
        self.tokens_explored = set()
        # limit per-step console output (the old per-prompt print produced
        # ~240 lines/step -> 300+ MB logs per run)
        self.print_every = config.get('print_every', 50)
        self.print_examples = config.get('print_examples', 3)

    def forward(
        self,
        source_texts: List[str],
        target_labels: List[str],
        lmbda: torch.tensor,
        output_tokens: List[List[str]],
        to_tensor: bool,
        mode: str
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        if mode == 'train':
            self._counter += 1
            source_strs = self._repeat_texts(source_texts)
            target_labels = self._repeat_texts(target_labels)
            lmbdas = lmbda.repeat_interleave(self.num_repeats)
        elif mode == "infer":
            source_strs = source_texts
            lmbdas = lmbda
        else:
            raise ValueError

        prompt_tokens = output_tokens
        prompt_strs = self._convert_tokens_to_string(prompt_tokens)
        assert len(prompt_strs) == len(source_strs)

        n_rows = len(prompt_strs)
        n_score = self.num_samples
        k_score = self.num_bootstraps
        N = n_score * k_score

        # 1) generate all hypotheses in batched calls: [n_rows][N]
        hypos_nested = self.generator.sample_generate_grouped(
            prompt_strs, source_strs, N, self.top_k, self.top_p)

        # 2) score all n_rows*N outputs in one batched pass
        flat_srcs = [s for s in source_strs for _ in range(N)]
        flat_labels = [l for l in target_labels for _ in range(N)]
        flat_hypos = [h for row in hypos_nested for h in row]
        content_flat, style_flat = self.objectives.compute_scores_flat(
            flat_srcs, flat_hypos, flat_labels)
        content_mat = content_flat.view(n_rows, N)   # [n_rows, N]
        style_mat = style_flat.view(n_rows, N)

        # 3) constrained reward, vectorized:
        #    score = style           if content >= 100*lambda
        #            0.01*(content-100*lambda)  otherwise
        lmbda_col = (lmbdas.detach().cpu().float() * 100).view(n_rows, 1)
        scores_mat = torch.where(content_mat >= lmbda_col, style_mat,
                                 0.01 * (content_mat - lmbda_col))

        if k_score > 1:
            # bootstrap: split each row's N=n*k samples into k segments,
            # take max per segment, average the maxes
            seg = scores_mat.view(n_rows, k_score, n_score)
            mean_scores = seg.max(dim=-1).values.mean(dim=-1)
        else:
            mean_scores = scores_mat.mean(dim=-1)

        mean_contents = content_mat.mean(dim=-1)
        mean_styles = style_mat.mean(dim=-1)

        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)
        quantities_to_log['mean_content'].append(mean_contents.mean())
        quantities_to_log['mean_style'].append(mean_styles.mean())
        quantities_to_log['mean_score'].append(mean_scores.mean())

        if mode == 'train' and self._counter % self.print_every == 0:
            right_frac = (content_mat >= lmbda_col).float().mean(dim=-1)
            # rows are src-major blocks of num_repeats: stride so each
            # printed example is a different source (and lambda)
            n_examples = min(self.print_examples,
                             max(1, n_rows // self.num_repeats))
            for i in range(0, n_examples * self.num_repeats,
                           self.num_repeats):
                masked_style = torch.where(content_mat[i] >= lmbda_col[i],
                                           style_mat[i],
                                           torch.zeros_like(style_mat[i]))
                top_index = masked_style.argmax()
                print(self._counter, '|', prompt_strs[i], '|',
                      source_strs[i], '|', hypos_nested[i][top_index], '|',
                      'Lambda:', round(lmbdas[i].item(), 2), '|',
                      'Top Content:', round(content_mat[i][top_index].item(), 2), '|',
                      'Top Style:', round(style_mat[i][top_index].item(), 2), '|',
                      'Mean Content:', round(mean_contents[i].item(), 2), '|',
                      'Mean Style:', round(mean_styles[i].item(), 2), '|',
                      'Mean Score:', round(mean_scores[i].item(), 3), '|',
                      'Right side:', round(right_frac[i].item() * 100, 1), '% |')

        scores_tensor = mean_scores
        content_tensor = mean_contents
        style_tensor = mean_styles
        self.tokens_explored = \
            self.tokens_explored.union(*[set(p) for p in prompt_tokens])
        quantities_to_log['num_tokens_explored'].append(
            torch.tensor(len(self.tokens_explored)).float())

        scores_log = dict(
            (score_key, torch.stack(score_vals, dim=0).mean())
            for score_key, score_vals in quantities_to_log.items())

        if to_tensor is True:
            return scores_tensor, content_tensor, style_tensor, scores_log
        else:
            return scores_tensor.tolist(), content_tensor.tolist(), style_tensor.tolist(), scores_log

    def _repeat_texts(
        self,
        texts: List[str],
        num_repeats: Optional[int] = None
    ) -> List[str]:
        if num_repeats is None:
            num_repeats = self.num_repeats
        return list(itertools.chain(*[[s for _ in range(num_repeats)]
                                      for s in texts]))

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self.tokenizer.convert_tokens_to_string(s)
                for s in tokens]
