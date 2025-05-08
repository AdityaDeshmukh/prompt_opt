import torch
from typing import Optional, List, Any, Dict
from .base_model import BaseModel
from omegaconf import DictConfig
class SinglePromptModel(BaseModel):
    def __init__(
        self,
        model: BaseModel,
        config: "DictConfig"
    ):
        super().__init__()
        self._model = model
        self.prompt_length: int = config.prompt_length
        self.prompt_infer_batch_size: int = config.prompt_infer_batch_size
        self.source_str: str = config.source_str
        self.input_conditioning = config.input_conditioning

    def _get_prompt_source(self, batch_size: int) -> List[str]:
        return [self.source_str for _ in range(batch_size)]
    
    def _do_source_reps(
        self, 
        source_texts: List[str], 
        num_reps: int
    ) -> List[str]:
        source_reps = []
        for text in source_texts: 
            for _ in range(num_reps): 
                source_reps.append(text)
        return source_reps

    def generate(
        self,
        source_texts: List[str],
        lmbda: torch.tensor,
        do_sample: bool,
        top_k: Optional[int],
        top_p: Optional[float],
        num_beams: Optional[int],
        num_repeats: int,
        max_new_tokens: Optional[int] = None,
        infer: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        if infer: 
            batch_size = min(self.prompt_infer_batch_size, len(source_texts))
            num_repeats = 1
        else: 
            batch_size = len(source_texts)*num_repeats
        # print("Batch size:", batch_size)
        if self.input_conditioning: 
            prompt_source = self._do_source_reps(source_texts, num_repeats)
        else:
            prompt_source = self._get_prompt_source(batch_size=batch_size)
        lmbda=lmbda.repeat_interleave(num_repeats)
        if lmbda.dim() == 0:
            lmbda = lmbda.unsqueeze(-1).unsqueeze(-1)
        if lmbda.dim() == 1:
            lmbda = lmbda.unsqueeze(-1)
        if max_new_tokens is None: 
            max_new_tokens = self.prompt_length
        return self._model.generate(lmbda=lmbda,
                                    source_texts=prompt_source,
                                    do_sample=do_sample,
                                    top_k=top_k,
                                    top_p=top_p,
                                    num_beams=num_beams,
                                    max_new_tokens=max_new_tokens,
                                    **kwargs)

    def teacher_forcing(
        self,
        lmbda: torch.tensor,
        source_texts: List[str],
        sample_ids: torch.LongTensor,
        num_repeats: int,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = len(source_texts)*num_repeats
        lmbda=lmbda.repeat_interleave(num_repeats)
        if lmbda.dim() == 0:
            lmbda = lmbda.unsqueeze(-1).unsqueeze(-1)
        if lmbda.dim() == 1:
            lmbda = lmbda.unsqueeze(-1)
        if self.input_conditioning: 
            prompt_source = self._do_source_reps(source_texts, num_repeats)
        else:
            prompt_source = self._get_prompt_source(batch_size=batch_size)
        return self._model.teacher_forcing(lmbda=lmbda,
                                           source_texts=prompt_source,
                                           sample_ids=sample_ids,
                                           **kwargs)