import torch
from torch import nn
import torch.distributions as D

import numpy as np
from typing import Optional, List, Dict, Union

from transformers import pipeline, AutoTokenizer

from .base_model import BaseModel
from .model_utils import _top_k_logits, _top_p_logits, _top_km_logits
from omegaconf import DictConfig

SUPPORTED_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                 'gpt2-large', 'gpt2-xl']

LM_HIDDEN_SIZES = {'distilgpt2': 768,
                   'gpt2': 768,
                   'gpt2-medium': 1024,
                   'gpt2-large': 1280,
                   'gpt2-xl': 1600}
            
class LMAdaptorModel(BaseModel):
    """Uses an MLP to modify the hidden states of an pre-trained LM

    The modified hidden state can then be passed into the original LM head
    to obtain output token logits. 
    
    Inspired by Houlsby et al. (2019): https://arxiv.org/abs/1902.00751
    """
    def __init__(
        self,
        config: "DictConfig"
    ):
        super().__init__()

        policy_lm = config.policy_lm
        assert policy_lm in SUPPORTED_LMS  # TODO: Support more LMs
        model = policy_lm
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            pad_token='<|endoftext|>')
        self.generator = pipeline("text-generation",
                                  tokenizer=self.tokenizer,
                                  model=model,
                                  device=self.device)
        for param in self.generator.model.parameters():
            param.requires_grad = False

        print("Number of parameters in generator:", sum(p.numel() for p in self.generator.model.parameters()))
        self.logit_bias: float = config.logit_bias
        self.fluent: bool = config.fluent
        self.fluent_top_k: int = config.fluent_top_k
        self.max_decoding_length: int = config.prompt_length
        self.eos_token_id: Optional[int] = config.eos_token_id
        self.num_repeats: int = config.num_repeats
        self.train_batch_size: int = config.train_batch_size
        self.explore: bool = config.explore
        self.mix_eps: float = config.mix_eps
        self.top_p: Optional[float] = config.top_p
        self.top_k: Optional[int] = config.top_k
        
        model_dim = LM_HIDDEN_SIZES[model]
        self.mlp = _build_hyper_mlp(in_dim=model_dim,
                                    out_dim=model_dim,
                                    hidden_size=config.hidden_size).to(self.device)
        print("Number of parameters in mlp:", sum(p.numel() for p in self.mlp.parameters()))

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.0001)
                m.bias.data.fill_(-0.0001)
        self.mlp.apply(_init_weights)

    def _mlp_forward(self, lmbda: torch.tensor, state: torch.Tensor) -> torch.Tensor:
        mlp_output = self.mlp(lmbda, state)
        logits = self.generator.model.lm_head(mlp_output)

        if self.fluent:
            lm_logits = self.generator.model.lm_head(state)
            values, _ = torch.topk(lm_logits, k=self.fluent_top_k)
            min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
            logits = torch.where(lm_logits < min_values,
                                 torch.full_like(logits, float('-inf')),
                                 logits)

        return logits

    def teacher_forcing(
        self,
        lmbda: torch.tensor,
        source_texts: List[str],
        sample_ids: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        state, past_key_values = self._get_generation_cache(source_texts)

        sample_logits = []
        for i in range(sample_ids.shape[-1]):
            logits = self._mlp_forward(lmbda, state)
            # logits = logits + self.logit_bias

            actions = sample_ids[:, i]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            sample_logits.append(logits.unsqueeze(dim=1))
            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)

        sample_logits = torch.cat(sample_logits, dim=1)
        output = dict(sample_logits=sample_logits,
                      sample_ids=sample_ids)
        return output

    def sample(
        self,
        lmbda: torch.tensor,
        source_texts: List[str],
        top_k: Optional[int],
        top_p: float,
        max_new_tokens: Optional[int],
        eos_token_id: Optional[int],
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        if eos_token_id is not None:
            raise NotImplementedError(
                "Only support fixed length prompt for now")

        state, past_key_values = self._get_generation_cache(source_texts)
        sample_tokens = [[] for _ in source_texts]
        sample_ids, sample_logits = [], []
        if self.explore:
            skip = torch.zeros(max_new_tokens, dtype=torch.int)
            mix_probs = torch.ones(self.train_batch_size, dtype=torch.float)*self.mix_eps
            mix_probs = torch.stack([1-mix_probs, mix_probs], dim=-1).to(self.device)
            mix = D.Categorical(probs = mix_probs) # exploration: (1-eps)*p + eps*uniform
        for i in range(max_new_tokens):
            logits = self._mlp_forward(lmbda, state)  # [batch_size, vocab_size]
            actions = torch.zeros(logits.size(0), dtype=torch.int)
            for j in range(self.num_repeats):
                if self.explore:
                    # skip[0], skip[1] = j//4, j%4
                    skip = torch.randint(0, j//4+1, (self.train_batch_size,))
                    sampling_logits = _top_km_logits(logits[j::self.num_repeats], k=self.top_k, m=skip)
                    # sampling_logits = _top_k_logits(logits[j::self.num_repeats], k=self.top_k)
                    comp_logits = torch.stack([sampling_logits, torch.where(sampling_logits==float('-inf'), float('-inf'), 0.).to(self.device)], dim=1)
                    comp = D.Categorical(logits = comp_logits)
                    mix_dist = D.MixtureSameFamily(mix, comp)
                    actions[j::self.num_repeats] = mix_dist.sample()
                else:
                    sampling_logits = _top_k_logits(logits[j::self.num_repeats], k=self.top_k)
                    actions[j::self.num_repeats] = D.Categorical(logits=sampling_logits).sample()
            
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            for s, t in zip(sample_tokens, tokens): 
                s.append(t)
            sample_ids.append(actions.unsqueeze(dim=1))  # [batch_size, 1]
            sample_logits.append(logits.unsqueeze(dim=1))
            # [batch_size, 1, vocab_size]

            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)

        # [batch_size, prompt_length]
        sample_ids = torch.cat(sample_ids, dim=1)
        # [batch_size, prompt_length, vocab_size]
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens
                                        for _ in range(sample_ids.shape[0])])
                          .to(self.device))

        output = dict(sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_ids=sample_ids,
                      sample_lengths=sample_lengths)
        return output

    def greedy_search(self,
                      lmbda: torch.tensor,
                      source_texts: List[str],
                      max_new_tokens: Optional[int],
                      eos_token_id: Optional[int],
                      **kwargs):
        if eos_token_id is not None:
            raise NotImplementedError(
                "Only support fixed length prompt for now")

        state, past_key_values = self._get_generation_cache(source_texts)
        sample_tokens = [[] for _ in source_texts]
        sample_ids, sample_logits = [], []
        for i in range(max_new_tokens):
            logits = self._mlp_forward(lmbda,state)
            # logits = logits + self.logit_bias
            # print(logits[:, 4:].min().item(), logits.max().item())
            sampling_logits = _top_k_logits(logits, k=3)

            # actions = logits.argmax(dim=-1)  # [batch_size]
            actions = D.Categorical(logits=sampling_logits).sample()
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0] for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t]) for t in tokens]

            for s, t in zip(sample_tokens, tokens): 
                s.append(t)
            sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))

            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)
        # [batch_size, prompt_length]
        sample_ids = torch.cat(sample_ids, dim=1)
        # [batch_size, prompt_length, vocab_size]
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens
                                        for _ in range(sample_ids.shape[0])])
                          .to(self.device))

        output = dict(sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_ids=sample_ids,
                      sample_lengths=sample_lengths)
        return output

    def _get_generation_cache(self,
                              source_texts: List[str],
                              past_key_values=None):
        token_encoding = (self.generator
                          .tokenizer(source_texts,
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt')
                          .to(self.device))
        input_ids = token_encoding['input_ids']
        input_lengths = token_encoding['attention_mask'].sum(dim=1)
        outputs = self.generator.model.transformer(input_ids,
                                                   past_key_values=past_key_values,
                                                   use_cache=True)
        last_token_hidden_state = \
            outputs.last_hidden_state[np.arange(input_ids.shape[0]),
                                      (input_lengths - 1)]
        past_key_values = outputs.past_key_values
        return last_token_hidden_state, past_key_values

    def generate(
        self,
        lmbda: torch.tensor,
        source_texts: List[str],
        do_sample: bool,
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
        max_new_tokens: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        assert num_beams == 1, "Beam search not supported yet"
        if max_new_tokens is None:
            max_new_tokens = self.max_decoding_length
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        is_greedy_gen_mode = (do_sample == False) and (num_beams == 1)
        is_sample_gen_mode = (do_sample == True) and (num_beams == 1)
        assert is_greedy_gen_mode or is_sample_gen_mode

        if is_greedy_gen_mode:
            return self.greedy_search(lmbda = lmbda,
                                      source_texts=source_texts,
                                      max_new_tokens=max_new_tokens,
                                      eos_token_id=eos_token_id)
        elif is_sample_gen_mode:
            return self.sample(lmbda = lmbda, 
                               source_texts=source_texts,
                               top_k=top_k,
                               top_p=top_p,
                               max_new_tokens=max_new_tokens,
                               eos_token_id=eos_token_id)

class HyperNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size):
        super().__init__()
        self.r = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(in_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_dim)
        self.gelu = nn.GELU()
        
        self.lmbda_w1 = nn.Linear(1, in_dim*hidden_size)
        self.lmbda_b1 = nn.Linear(1, hidden_size)
        
        self.lmbda_t1 = nn.Linear(in_dim, out_dim)
        self.lmbda_t2 = nn.Linear(hidden_size, out_dim)
        self.lambda_t3 = nn.Linear(out_dim, out_dim)

        self.lmbda_u3 = nn.Linear(self.r, out_dim)
        self.lmbda_v3 = nn.Linear(hidden_size, self.r)
        self.lmbda_b3 = nn.Linear(1, out_dim)

    def forward(self, lmbda: torch.tensor, x: torch.Tensor) -> torch.Tensor:
        w1 = self.lmbda_w1(lmbda)
        w1 = w1.contiguous().view(-1, self.hidden_size, self.in_dim)
        b1 = self.lmbda_b1(lmbda)
        x = torch.matmul(w1, x.unsqueeze(-1)).squeeze(-1) + b1 + self.layer1(x)
        x = self.gelu(x)
        
        w1 = self.gelu(w1)
        b1 = self.gelu(b1)
        w2 = self.lmbda_t1(w1)
        w2 = w2.contiguous().view(-1, self.out_dim, self.hidden_size)
        b2 = self.lmbda_t2(b1)
        x = torch.matmul(w2, x.unsqueeze(-1)).squeeze(-1) + b2 + self.layer2(x)
        x = self.gelu(x)

        w2 = self.gelu(w2)
        v3 = self.lmbda_v3(w2)
        v3 = self.gelu(v3)
        u3 = self.lmbda_u3(v3).contiguous().view(-1, self.out_dim, self.r)
        v3 = v3.contiguous().view(-1, self.r, self.out_dim)
        y = torch.matmul(v3,x.unsqueeze(-1))
        y = self.gelu(y)
        b2 = self.gelu(b2)
        x = x + torch.matmul(u3,y).squeeze(-1) + self.lambda_t3(b2)
        # x = x + torch.matmul(u3,y).squeeze(-1) + self.lmbda_b3(lmbda)
        return x
    

def _build_hyper_mlp(in_dim, out_dim, hidden_size):
    return HyperNet(in_dim, out_dim, hidden_size)