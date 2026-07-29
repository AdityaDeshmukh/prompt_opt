import os
import torch
import copy
from typing import Optional, List, Dict, Any, Union, Tuple
from omegaconf import DictConfig
from models import BaseModel
from modules import BaseScoreModule
from scores import BaseScore
from utils import utils
from losses import *

_MEM_DEBUG = os.environ.get('MEM_DEBUG', '0') == '1'


def _mem_report(tag: str) -> None:
    if not _MEM_DEBUG or not torch.cuda.is_available():
        return
    parts = []
    for d in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(d) / 1e9
        peak = torch.cuda.max_memory_allocated(d) / 1e9
        parts.append(f"cuda:{d} alloc={alloc:.1f}G peak={peak:.1f}G")
    print(f"[MEM] {tag}: " + " | ".join(parts), flush=True)


ALGOS = {
            # legacy losses (kept for checkpoint/back compatibility)
            "plgo": plgo_loss,
            "plgo_b": plgo_b_loss,
            "grpo_legacy": grpo_loss,
            "drgo": drgo_loss,
            "drgo_new": drgo_new_loss,
            "drgo_regularized": drgo_regularized_loss,
            "kl": kl_ub_loss,
            "drgo_cauchy": drgo_cauchy_loss,
            "l1": drgo_l1_loss,
            # unified family: identical reward scaling, selectable d, epsilon-
            # guarded std scaling, optional entropy bonus (see loss_functions)
            "grpo": grpo_v2_loss,
            "rrebel_l2": make_rrebel_loss('l2', reward_std_scale=False,
                                          use_entropy=False),
            "rrebel_l1_std": make_rrebel_loss('l1', reward_std_scale=True,
                                              use_entropy=False),
            "rrebel_l1_ent": make_rrebel_loss('l1', reward_std_scale=True,
                                              use_entropy=True),
            "rrebel_huber_std": make_rrebel_loss('huber', reward_std_scale=True,
                                                 use_entropy=False),
        }
class ScoreLossModule(BaseScoreModule):
    def __init__(
        self,
        model: BaseModel,
        score: Optional[BaseScore],
        config: "DictConfig"
    ):
        super().__init__()
        # Initialize self._model and self._score
        assert not (config.top_k is not None and config.top_p < 1.0), \
               "Only one of top_k or top_p should be selected"
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self._model = model
        print("Number of parameters:", sum(p.numel() for p in model.parameters()))
        self._ref_model = copy.deepcopy(self._model)
        for param in self._ref_model.parameters():
            param.requires_grad = False

        self._score = score

        self._top_k: Optional[int] = config.top_k
        self._top_p: float = config.top_p
        self._num_beams: int = config.num_beams
        self.num_repeats: int = config.num_repeats
        self.update_steps: int = config.update_steps
        self.algo: str = config.algo

        # loss hyperparameters (shared across the unified family)
        self.beta: float = float(config.get('beta', 0.5))
        self.score_scale: float = float(config.get('score_scale', 0.1))
        self.ent_coef: float = float(config.get('ent_coef', 0.01))
        self.huber_delta: float = float(config.get('huber_delta', 1.0))
        grpo_beta = float(config.get('grpo_beta', 0.0))
        # How often the KL reference is re-anchored to the current policy.
        # R-REBEL *requires* a short-lag ref -- that is REBEL's construction
        # (it regresses the logratio against reward differences w.r.t. the
        # recent policy), so it keeps update_steps.
        # GRPO does NOT. Sharing update_steps=150 made GRPO's trust region
        # inert: a hard deepcopy zeroes the KL *and its gradient* at every
        # refresh, so with ~80 refreshes over a 12k run the policy/ref drift
        # stayed ~1% while total drift from init was unbounded -> entropy
        # collapse inside 500 steps (2-19 distinct prompts vs R-REBEL's ~97).
        # Reference practice: TRL GRPOTrainer defaults sync_ref_model=False
        # (fixed anchor) and, when enabled, syncs every 512 steps with a SOFT
        # TR-DPO mixup rather than a hard copy; verl and OpenRLHF keep the ref
        # frozen; DeepSeekMath Alg.1 re-anchors once per OUTER iteration
        # (~2x per run), not per inner step. 0 = never re-anchor (default).
        self.ref_sync_steps: int = self.update_steps
        if self.algo == 'grpo':
            self.beta = grpo_beta
            self.ref_sync_steps = int(config.get('grpo_ref_sync_steps', 0))
            # entropy bonus is opt-in for grpo and defaults OFF, so the
            # baseline stays plain GRPO; ent_coef (0.01) belongs to the
            # rrebel *_ent variant and must not leak in implicitly.
            self.ent_coef = float(config.get('grpo_ent_coef', 0.0))
            print(f"NOTE: algo=grpo -> grpo_v2_loss (eps-guarded std, "
                  f"beta={self.beta}, ent_coef={self.ent_coef}, "
                  f"ref_sync_steps={self.ref_sync_steps} "
                  f"[0=fixed anchor]). Use algo=grpo_legacy for pre-2026-07.")
        # ref teacher-forcing is skipped when the loss never reads logits_
        self._needs_ref = not (self.algo == 'grpo' and grpo_beta == 0.0)

    def _pre_steps(self, step: int) -> None:
        # ref_sync_steps <= 0 => fixed anchor: keep the reference captured at
        # __init__ (or restored from the checkpoint, since _ref_model is a
        # registered submodule and round-trips through state_dict, so the
        # anchor survives the 4h resubmit chain).
        if not self._needs_ref or self.ref_sync_steps <= 0:
            return None
        if step % self.ref_sync_steps == 0:
            self._ref_model = copy.deepcopy(self._model)
            for param in self._ref_model.parameters():
                param.requires_grad = False
        return None

    def forward(self, lmbda: torch.tensor, batch: Dict[str, Any]) -> Tuple[Union[torch.Tensor, Dict],
                                                      Dict[str, Any]]:
        loss_list = []
        score_log_list = []
        _loss, scores_log = self._forward(lmbda=lmbda, batch=batch)
        loss_list.append(_loss)
        score_log_list.append(scores_log)

        # https://discuss.pytorch.org/t/get-the-mean-from-a-list-of-tensors/31989/2
        loss = torch.mean(torch.stack(loss_list))
        score_log = utils.unionize_dicts(score_log_list)
        scores_log['lmbda'] = lmbda

        return loss, score_log

    def _forward(
        self,
        lmbda: torch.tensor,
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict]:

        (logits, logits_, output_tokens, output_ids, sequence_lengths) = \
                self._decode_sampling(lmbda=lmbda, batch=batch)
        _mem_report("after policy rollout (graph retained)")

        score_tensor, content_tensor, style_tensor, scores_log = \
        self.compute_scores(lmbda = lmbda, batch=batch,
                                output_tokens=output_tokens,
                                mode="train")
        _mem_report("after task-LM generation + scoring")
        loss_func = ALGOS[self.algo]
        loss_kwargs = dict(
            lmbda=lmbda,
            logits=logits,
            logits_=logits_,
            actions=output_ids,
            scores_tensor=score_tensor,
            content_tensor=content_tensor,
            style_tensor=style_tensor,
            num_src=len(batch['source_texts']))
        if self.algo in ('grpo', 'rrebel_l2', 'rrebel_l1_std',
                         'rrebel_l1_ent', 'rrebel_huber_std'):
            loss_kwargs.update(beta=self.beta,
                               score_scale=self.score_scale,
                               ent_coef=self.ent_coef,
                               huber_delta=self.huber_delta)
        loss, loss_log = loss_func(**loss_kwargs)

        utils.add_prefix_to_dict_keys_inplace(
            scores_log, prefix="scores/")
        utils.add_prefix_to_dict_keys_inplace(
            loss_log, prefix="loss/")
        
        loss_log = utils.unionize_dicts([
            scores_log,
            loss_log
        ])

        return loss, loss_log 

    def compute_scores(
        self,
        lmbda: torch.tensor,
        batch: Dict[str, Any],
        output_tokens: List[List[str]],
        to_tensor: bool = True,
        mode: str = "infer"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        scores_tensor, content_tensor, style_tensor, scores_log = self._score(
            **batch,
            lmbda = lmbda,
            output_tokens=output_tokens,
            to_tensor=to_tensor,
            mode=mode)
        
        scores_tensor = scores_tensor.to(self.device)
        content_tensor = content_tensor.to(self.device) 
        style_tensor = style_tensor.to(self.device)
        #         
        # return content_tensor, style_tensor, scores_log
        return scores_tensor, content_tensor, style_tensor, scores_log

    def infer(
        self,
        lmbda: torch.tensor,
        batch: Dict[str, Any]
    ) -> Dict[str, Union[torch.Tensor, torch.LongTensor, List[List[str]]]]:
        
        return self._model.generate(**batch,
                                    lmbda=lmbda,
                                    do_sample=False,
                                    top_k=3,
                                    top_p=self._top_p,
                                    num_beams=self._num_beams,
                                    num_repeats = self.num_repeats,
                                    infer=True)

    def _decode_sampling(
        self,
        lmbda: torch.tensor,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]],
               torch.LongTensor, torch.LongTensor]:
        
        outputs = self._model.generate(**batch,
                                       lmbda=lmbda,
                                       do_sample=True,
                                       top_k=self._top_k,
                                       top_p=self._top_p,
                                       num_beams=self._num_beams,
                                       num_repeats = self.num_repeats)

        if self._needs_ref:
            batch_ = {k: v for k, v in batch.items()}
            batch_.update(outputs)
            outputs_ = self._ref_model.teacher_forcing(
                lmbda=lmbda, **batch_, num_repeats=self.num_repeats)
            ref_logits = outputs_['sample_logits'].contiguous()
        else:
            ref_logits = outputs['sample_logits'].detach()

        return (outputs['sample_logits'].contiguous(),
                ref_logits,
                outputs['sample_tokens'],
                outputs['sample_ids'].contiguous(),
                outputs['sample_lengths'].contiguous())