import torch
import itertools
from typing import List, Tuple, Union, Dict, Any, Optional
from transformers import AutoTokenizer
from collections import defaultdict
from tst_modules import PromptedGenerator, TextStyleTransferObjectives
from omegaconf import DictConfig
from scores import BaseScore

# Magic variable
SUPPORTED_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                 'gpt2-large', 'gpt2-xl']


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
        assert task_lm in SUPPORTED_LMS
        print('Task LM:', task_lm)
        self.tokenizer = AutoTokenizer.from_pretrained(task_lm)
        self.generator = PromptedGenerator(task_lm, config.template, config.end_punct,
                                           config.pad_token, generator_device,
                                           config.lower_outputs, config.control_output_length)
        self.top_k = config.task_top_k
        self.top_p = 1.0
        self.num_samples = config.num_samples
        self.num_bootstraps = config.num_bootstraps

        style_tokenizer = config.style_tokenizer
        # Loading reward models
        if style_tokenizer is None: 
            style_tokenizer = style_classifier
        self.objectives = TextStyleTransferObjectives(style_classifier,
                                                        style_tokenizer,
                                                        config.style_batch_size,
                                                        score_device)

        # Misc. training details
        self.num_repeats = config.num_repeats
        self._counter = 0
        self.tokens_explored = set()

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
        # print(prompt_strs)
        # print(source_strs)
        
        assert len(prompt_strs) == len(source_strs)

        n_score = self.num_samples
        k_score = self.num_bootstraps
        N = n_score * k_score

        content_scores_list: List[torch.Tensor] = []
        style_scores_list: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        # input_scores: Dict[str, List[float]] = defaultdict(list)
        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for i, (lmbda, prompt, src, label) in enumerate(zip(lmbdas, prompt_strs,
                                                     source_strs,
                                                     target_labels)):
            hypos = self.generator.sample_generate(prompt, src, N,
                                                   self.top_k, self.top_p)
            content_scores, style_scores = \
                self.objectives.compute_sample_scores(lmbda, src, hypos, label)

            content_scores = torch.tensor(content_scores).float()
            style_scores = torch.tensor(style_scores).float()

            lmbda_ = lmbda.clone().detach().cpu()*100
            
            scores = torch.where(content_scores >= lmbda_, style_scores, 0.01*(content_scores-lmbda_))
            
            # w_scores, content_scores, style_scores = \
            #     self.objectives.compute_sample_scores(lmbda, src, hypos, label)
            # print('lmbda:', lmbda)
            # print('content_scores:', content_scores)
            # print('style_scores:', style_scores)
            # print('------------------------------------')
            # Bootstrap the max w_score for k times and average
            # print('lmbda:', lmbda)
            # print('content_scores:', content_scores)
            # print('style_scores:', style_scores)
            if k_score > 1:
                bootstrap_max_scores: List[float] = \
                    self._boostrap_max_scores_k_times(scores, k_score)
            
                # Average boostrap max score as the final score
                mean_score = torch.Tensor(bootstrap_max_scores).float().mean()
            else:
                mean_score = scores.mean()

            # Keep track of each input's max scores to compute z-score
            # input_scores[src] += bootstrap_max_scores

            # Take the max of the sub-list scores to print as example
            # max_score = max(bootstrap_max_scores)
            # top_index = w_scores.index(max_score)
            

            right_content_scores = torch.where(content_scores > lmbda_, content_scores, 0)
            right_style_scores = torch.where(content_scores > lmbda_, style_scores, 0)
            
            top_index = right_style_scores.argmax()
            # Log relevant quantities
            # content = torch.tensor(content_scores).float().mean()
            # style = torch.tensor(style_scores).float().mean()
            # mean_score = torch.tensor(w_scores).float().mean()
            top_content = content_scores[top_index]
            top_style = style_scores[top_index]
            # quantities_to_log['mean_content'].append(content)
            # quantities_to_log['mean_style'].append(style)
            # quantities_to_log["w_score"].append(score)
            # quantities_to_log["mean_w_score"].append(mean_score)
            # quantities_to_log["top_content"].append(top_content)
            # quantities_to_log["top_style"].append(top_style)
            # mean_score = scores.mean()
            mean_content = content_scores.mean()
            mean_style = style_scores.mean()
            quantities_to_log['mean_content'].append(mean_content)
            quantities_to_log['mean_style'].append(mean_style)
            quantities_to_log['mean_score'].append(mean_score)

            print(self._counter, '|', prompt_tokens[i], '|',
                  prompt, '|', src, '|', hypos[top_index], '|',
                  'Lambda:', round(lmbda.item(), 2), '|',
                  'Top Content:', round(top_content.item(), 2), '|',
                  'Top Style:', round(top_style.item(), 2), '|',
                  'Mean Content:', round(mean_content.item(), 2), '|',
                  'Mean Style:', round(mean_style.item(), 2), '|',
                #   'Top Weighted Score:', round(max_score.item(), 2), '|',
                  'Mean Score:', round(mean_score.item(), 3), '|',
                  'Percentage on right side:', round(right_content_scores.nonzero().numel()/right_content_scores.shape[0]*100, 3), '% |')

            # scores.append(score)
            scores_list.append(mean_score)
            content_scores_list.append(mean_content)
            style_scores_list.append(mean_style)


        scores_tensor = torch.stack(scores_list)
        content_tensor = torch.stack(content_scores_list)
        style_tensor = torch.stack(style_scores_list)
        self.tokens_explored = \
            self.tokens_explored.union(*[set(p) for p in prompt_tokens])
        quantities_to_log['num_tokens_explored'].append(
            torch.tensor(len(self.tokens_explored)).float())

        scores_log = dict(
            (score_key, torch.stack(score_vals, dim=0).mean())
            for score_key, score_vals in quantities_to_log.items())
        # score_log = quantities_to_log

        if to_tensor is True:
            return scores_tensor, content_tensor, style_tensor, scores_log
        else:
            return scores_tensor.tolist(), content_tensor.tolist(), style_tensor.tolist(), scores_log

    def _boostrap_max_scores_k_times(
        self,
        scores: List[float],
        k: int
    ) -> List[float]:
        # Segment list rewards into k equal sub-lists
        l = len(scores)
        assert l % k == 0, f'l={l}, k={k}'
        segmented_scores = [scores.tolist()[i*l//k:(i+1)*l//k]
                             for i in range(k)]  # [k, l/k]
        # We use different rewards for each bootstrap for now
        bootstrap_scores = segmented_scores

        # For each sub-list, take the max as the sub-reward
        values, indices = (torch.tensor(bootstrap_scores)
                           .float().max(axis=1))
        # Take numbers from the original list to avoid numerical issues
        bootstrap_max_scores = [bootstrap_scores[i][index]
                                 for i, index in enumerate(indices)]

        return bootstrap_max_scores

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