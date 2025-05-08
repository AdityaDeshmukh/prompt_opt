import torch
import copy
from typing import Optional, List, Dict, Any, Union, Tuple
from omegaconf import DictConfig
from models import BaseModel
from modules import BaseScoreModule
from scores import BaseScore
from utils import utils
from losses import dpo_score_loss, kl_div_loss



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
        self.score_scaler: float = config.score_scaler
        
    def _pre_steps(self, step: int) -> None:
        if step % self.update_steps == 0:
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
        
        score_tensor, content_tensor, style_tensor, scores_log = \
        self.compute_scores(lmbda = lmbda, batch=batch, 
                                output_tokens=output_tokens,
                                mode="train")
        
        loss, loss_log = kl_div_loss(
            lmbda = lmbda,
            logits=logits,
            logits_=logits_,
            actions=output_ids,
            scores_tensor = score_tensor,
            content_tensor = content_tensor,
            style_tensor = style_tensor,
            num_src=len(batch['source_texts']),
            score_scaler = self.score_scaler)

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

        batch_ = {k: v for k, v in batch.items()}
        batch_.update(outputs)

        outputs_ = self._ref_model.teacher_forcing(lmbda=lmbda, **batch_, num_repeats=self.num_repeats)

        return (outputs['sample_logits'].contiguous(),
                outputs_['sample_logits'].contiguous(),
                outputs['sample_tokens'],
                outputs['sample_ids'].contiguous(),
                outputs['sample_lengths'].contiguous())