import torch
from torch import nn
import torch.distributions as D

import numpy as np
from typing import Optional, List, Dict, Union

from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_model import BaseModel
from .model_utils import _top_k_logits, _top_p_logits, _top_km_logits_vec
from omegaconf import DictConfig


class LMAdaptorModel(BaseModel):
    """Uses an MLP to modify the hidden states of an pre-trained LM

    The modified hidden state can then be passed into the original LM head
    to obtain output token logits.

    Inspired by Houlsby et al. (2019): https://arxiv.org/abs/1902.00751

    Decoding is id-based with an explicit attention mask: sampled token ids
    are fed straight back into the backbone (no token->string->retokenize
    round trip), and padded source positions are masked out of the KV cache.
    """
    def __init__(
        self,
        config: "DictConfig"
    ):
        super().__init__()
        self._init_backbone(config)
        self._mlp_weights = None

        model_dim = self.model.config.hidden_size
        self.mlp = _build_hyper_mlp(in_dim=model_dim,
                                    out_dim=model_dim,
                                    hidden_size=config.hidden_size).to(self.device)
        print("Number of parameters in mlp:", sum(p.numel() for p in self.mlp.parameters()))

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.0001)
                m.bias.data.fill_(-0.0001)
        self.mlp.apply(_init_weights)

    def _init_backbone(self, config: "DictConfig") -> None:
        policy_lm = config.policy_lm
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = config.pad_token
        self.tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(policy_lm).to(self.device)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        # Bypass nn.Module registration for the frozen backbone (the old code
        # held it inside a pipeline object, which was also unregistered): it
        # keeps checkpoints adaptor-only (old checkpoints stay loadable),
        # keeps .train() from enabling backbone dropout, and keeps the
        # optimizer's param list trainable-only.
        object.__setattr__(self, 'model', model)
        # generic accessors (GPT-2 exposes .transformer, Llama/Qwen .model)
        backbone = getattr(model, 'transformer', None)
        if backbone is None:
            backbone = model.get_decoder()
        object.__setattr__(self, 'backbone', backbone)
        object.__setattr__(self, 'lm_head', model.get_output_embeddings())

        print("Number of parameters in generator:",
              sum(p.numel() for p in self.model.parameters()))
        self.logit_bias: float = config.logit_bias
        self.fluent: bool = config.fluent
        self.fluent_top_k: int = config.fluent_top_k
        self.max_decoding_length: int = config.prompt_length
        self.eos_token_id: Optional[int] = config.eos_token_id
        self.num_repeats: int = config.num_repeats
        self.explore: bool = config.explore
        self.mix_eps: float = config.mix_eps
        self.top_p: Optional[float] = config.top_p
        self.top_k: Optional[int] = config.top_k

    def _prepare_conditioning(self, lmbda: torch.tensor) -> None:
        """Hook called once at the start of every rollout/teacher-forcing
        pass: precompute the lambda-only HyperNet weights so the per-token
        forward reuses them instead of regenerating the large weight matrices
        at every decoding step."""
        self._mlp_weights = self.mlp.precompute(lmbda)

    def _finish_conditioning(self) -> None:
        """Counterpart hook called before each rollout entry point returns;
        drops the per-call precomputed weights."""
        self._mlp_weights = None

    def _adapted_logits(self, lmbda: torch.tensor, state: torch.Tensor) -> torch.Tensor:
        mlp_output = self.mlp.apply_weights(self._mlp_weights, state)
        logits = self.lm_head(mlp_output)

        if self.fluent:
            lm_logits = self.lm_head(state)
            values, _ = torch.topk(lm_logits, k=self.fluent_top_k)
            min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
            logits = torch.where(lm_logits < min_values,
                                 torch.full_like(logits, float('-inf')),
                                 logits)

        return logits

    def _mlp_forward(self, lmbda: torch.tensor, state: torch.Tensor) -> torch.Tensor:
        # old name; delegates so subclasses dispatch correctly
        return self._adapted_logits(lmbda, state)

    def _init_cache(self, source_texts: List[str]) -> Dict:
        """Encode the (left-padded) sources once; return decoding state."""
        enc = self.tokenizer(source_texts,
                             padding=True,
                             truncation=True,
                             return_tensors='pt').to(self.device)
        attention_mask = enc['attention_mask']
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        outputs = self.backbone(enc['input_ids'],
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                use_cache=True)
        # left padding: the last real token sits at index -1 for every row
        state = outputs.last_hidden_state[:, -1]
        return {
            'state': state,
            'past_key_values': outputs.past_key_values,
            'attention_mask': attention_mask,
            'next_position': attention_mask.sum(dim=-1),
        }

    def _step_cache(self, cache: Dict, actions: torch.LongTensor) -> Dict:
        """Advance the decoding state by one sampled token per row."""
        attention_mask = torch.cat(
            [cache['attention_mask'],
             torch.ones_like(cache['attention_mask'][:, :1])], dim=-1)
        outputs = self.backbone(actions.view(-1, 1),
                                attention_mask=attention_mask,
                                position_ids=cache['next_position'].view(-1, 1),
                                past_key_values=cache['past_key_values'],
                                use_cache=True)
        return {
            'state': outputs.last_hidden_state[:, -1],
            'past_key_values': outputs.past_key_values,
            'attention_mask': attention_mask,
            'next_position': cache['next_position'] + 1,
        }

    def _ids_to_tokens(self, sample_ids: torch.LongTensor) -> List[List[str]]:
        return [self.tokenizer.convert_ids_to_tokens(row)
                for row in sample_ids.tolist()]

    def teacher_forcing(
        self,
        lmbda: torch.tensor,
        source_texts: List[str],
        sample_ids: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        self._prepare_conditioning(lmbda)
        cache = self._init_cache(source_texts)

        sample_logits = []
        for i in range(sample_ids.shape[-1]):
            logits = self._adapted_logits(lmbda, cache['state'])
            sample_logits.append(logits.unsqueeze(dim=1))
            cache = self._step_cache(cache, sample_ids[:, i])

        sample_logits = torch.cat(sample_logits, dim=1)
        self._finish_conditioning()
        output = dict(sample_logits=sample_logits,
                      sample_ids=sample_ids)
        return output

    def _sample_actions(self, logits: torch.Tensor) -> torch.LongTensor:
        """Sample one action per row; vectorized over the whole batch."""
        B = logits.size(0)
        if self.explore:
            # per-repeat exploration schedule: row with repeat index j
            # excludes the top-m tokens, m ~ Unif{0..j//4}
            rep_idx = torch.arange(B, device=logits.device) % self.num_repeats
            highs = rep_idx // 4 + 1
            m = (torch.rand(B, device=logits.device) * highs).long()
            sampling_logits = _top_km_logits_vec(logits, k=self.top_k, m=m)
            uniform_logits = torch.where(
                sampling_logits == float('-inf'),
                torch.full_like(sampling_logits, float('-inf')),
                torch.zeros_like(sampling_logits))
            comp_logits = torch.stack([sampling_logits, uniform_logits], dim=1)
            eps = torch.full((B,), self.mix_eps, device=logits.device)
            mix = D.Categorical(probs=torch.stack([1 - eps, eps], dim=-1))
            comp = D.Categorical(logits=comp_logits)
            return D.MixtureSameFamily(mix, comp).sample()
        else:
            sampling_logits = _top_k_logits(logits, k=self.top_k)
            return D.Categorical(logits=sampling_logits).sample()

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

        self._prepare_conditioning(lmbda)
        cache = self._init_cache(source_texts)
        sample_ids, sample_logits = [], []
        for i in range(max_new_tokens):
            logits = self._adapted_logits(lmbda, cache['state'])  # [B, vocab]
            actions = self._sample_actions(logits)
            sample_ids.append(actions.unsqueeze(dim=1))   # [B, 1]
            sample_logits.append(logits.unsqueeze(dim=1))
            cache = self._step_cache(cache, actions)

        # [batch_size, prompt_length]
        sample_ids = torch.cat(sample_ids, dim=1)
        # [batch_size, prompt_length, vocab_size]
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = torch.full((sample_ids.shape[0],), max_new_tokens,
                                    device=self.device)

        self._finish_conditioning()
        output = dict(sample_tokens=self._ids_to_tokens(sample_ids),
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

        self._prepare_conditioning(lmbda)
        cache = self._init_cache(source_texts)
        sample_ids, sample_logits = [], []
        for i in range(max_new_tokens):
            logits = self._adapted_logits(lmbda, cache['state'])
            sampling_logits = _top_k_logits(logits, k=3)
            actions = D.Categorical(logits=sampling_logits).sample()
            sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))
            cache = self._step_cache(cache, actions)

        sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = torch.full((sample_ids.shape[0],), max_new_tokens,
                                    device=self.device)

        self._finish_conditioning()
        output = dict(sample_tokens=self._ids_to_tokens(sample_ids),
                      sample_logits=sample_logits,
                      sample_ids=sample_ids,
                      sample_lengths=sample_lengths)
        return output

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

    def precompute(self, lmbda: torch.tensor) -> Dict[str, torch.Tensor]:
        """Everything that depends only on lambda (not on the hidden state).

        These are the large per-sample weight matrices; computing them once
        per rollout instead of once per decoded token cuts the retained
        autograd graph (and the compute) by the prompt length (~5x), which is
        what makes the policy fit alongside the scorer on a single GPU.
        """
        w1 = self.lmbda_w1(lmbda)
        w1 = w1.contiguous().view(-1, self.hidden_size, self.in_dim)
        b1 = self.lmbda_b1(lmbda)

        w1g = self.gelu(w1)
        b1g = self.gelu(b1)
        w2 = self.lmbda_t1(w1g)
        w2 = w2.contiguous().view(-1, self.out_dim, self.hidden_size)
        b2 = self.lmbda_t2(b1g)

        w2g = self.gelu(w2)
        v3 = self.lmbda_v3(w2g)
        v3 = self.gelu(v3)
        u3 = self.lmbda_u3(v3).contiguous().view(-1, self.out_dim, self.r)
        v3 = v3.contiguous().view(-1, self.r, self.out_dim)
        t3 = self.lambda_t3(self.gelu(b2))
        return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2,
                'v3': v3, 'u3': u3, 't3': t3}

    def apply_weights(self, W: Dict[str, torch.Tensor],
                      x: torch.Tensor) -> torch.Tensor:
        """The hidden-state-dependent path, reusing the precomputed weights."""
        x = torch.matmul(W['w1'], x.unsqueeze(-1)).squeeze(-1) + W['b1'] + self.layer1(x)
        x = self.gelu(x)
        x = torch.matmul(W['w2'], x.unsqueeze(-1)).squeeze(-1) + W['b2'] + self.layer2(x)
        x = self.gelu(x)
        y = torch.matmul(W['v3'], x.unsqueeze(-1))
        y = self.gelu(y)
        x = x + torch.matmul(W['u3'], y).squeeze(-1) + W['t3']
        return x

    def forward(self, lmbda: torch.tensor, x: torch.Tensor) -> torch.Tensor:
        # equivalent to the original single-shot forward; kept for callers
        # that do not use the precompute/apply split
        return self.apply_weights(self.precompute(lmbda), x)


def _build_hyper_mlp(in_dim, out_dim, hidden_size):
    return HyperNet(in_dim, out_dim, hidden_size)
