import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import pipeline
from bert_score import BERTScorer
from typing import Tuple, List, Union

class TextStyleTransferObjectives:
    def __init__(
        self,
        style_classifier: str,
        style_tokenizer: str,
        style_batch_size: int,
        device_id: int,
        scorer_dtype: str = 'float16',
    ):
        self.device = device_id
        use_half = (scorer_dtype in ('float16', 'bfloat16')
                    and torch.cuda.is_available())
        torch_dtype = getattr(torch, scorer_dtype) if use_half else torch.float32
        self.style_classifier = pipeline("sentiment-analysis",
                                         model=style_classifier,
                                         tokenizer=style_tokenizer,
                                         device=self.device,
                                         torch_dtype=torch_dtype)
        self.style_batch_size = style_batch_size
        self.bert_scorer = BERTScorer('roberta-large',
                                      device=self.device,
                                      rescale_with_baseline=True,
                                      lang='en')
        if use_half:
            self.bert_scorer._model.to(torch_dtype)

    def compute_scores_flat(
        self,
        source_texts: List[str],
        generated_texts: List[str],
        target_labels: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score all (source, hypo, label) rows in one batched pass.

        Returns (content_scores, style_scores), float tensors on CPU with one
        entry per row, on the same 0-100 scale as compute_sample_scores.
        """
        assert len(source_texts) == len(generated_texts) == len(target_labels)

        # Content preservation reward (bert_score batches + length-sorts
        # internally; one call over all rows). Empty candidates are scored 0
        # directly: bert_score gave them raw 0 (negative after baseline
        # rescale, clamped to 0 below), and its empty-string path uses a
        # tokenizer API removed in transformers 5.x.
        nonempty = [i for i, t in enumerate(generated_texts) if t.strip()]
        ctc_scores = torch.zeros(len(generated_texts))
        if nonempty:
            sub = self.bert_scorer.score(
                [generated_texts[i] for i in nonempty],
                [source_texts[i] for i in nonempty],
                batch_size=max(self.style_batch_size, 64))[2]
            ctc_scores[nonempty] = sub.float()
        content_scores = ctc_scores.clamp(min=0) * 100

        # Style probability reward
        hypo_dataset = ListDataset(generated_texts)
        style_scores = []
        for i, c in enumerate(self.style_classifier(hypo_dataset,
                                                    batch_size=self.style_batch_size,
                                                    truncation=True)):
            prob = ((c['label'] == target_labels[i]) * c['score']
                    + (c['label'] != target_labels[i]) * (1 - c['score']))
            style_scores.append(prob * 100)
        style_scores = torch.tensor(style_scores).float()

        return content_scores, style_scores

    def compute_sample_scores(
        self,
        lmbda: torch.tensor,
        source_text: str,
        generated_texts: List[str],
        target_label: str
    ) -> Tuple[List[float]]:
        srcs = [source_text for _ in generated_texts]
        labels = [target_label for _ in generated_texts]
        content_scores, style_scores = self.compute_scores_flat(
            srcs, generated_texts, labels)
        return content_scores.tolist(), style_scores.tolist()

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return self.data_list.__len__()
