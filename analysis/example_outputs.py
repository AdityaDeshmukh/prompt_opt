"""Worked example for the paper: ONE input sentence, the prompt the policy
emits at several floors, and what the frozen task model writes with it.

For each requested test sentence and floor: the policy's most likely prompt
(true argmax decode), N=50 task-LM rewrites with their content and sentiment
scores, and the rewrite closest to the prompt's mean scores (a representative
sample rather than the best one). Evaluation settings match eval_full.slurm
(vLLM, top-k=10 task-LM sampling, fp32 scorers).

    python analysis/example_outputs.py algo=rrebel_huber_std checkpoint_path=... \
        "+ex_sents=[403,438,352]" "+ex_floors=[0,30,50,70,90]" +ex_out=results/examples/x.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from trainers import ScoreTrainer
from modules import ScoreLossModule
from models import build_policy_model, SinglePromptModel
from tst_helpers import make_text_style_transfer_datasets, get_style_classifier
from tst_score import PromptedTextStyleTransferScore


@hydra.main(version_base=None, config_path="../", config_name="tst_config")
def main(config: "DictConfig"):
    _, _, test_dataset = make_text_style_transfer_datasets(config)
    policy_model = build_policy_model(config)
    prompt_model = SinglePromptModel(policy_model, config)
    config.style_classifier = get_style_classifier('train', config)
    score_module = PromptedTextStyleTransferScore(config)
    module = ScoreLossModule(prompt_model, score_module, config)
    module = ScoreTrainer(module, None, test_dataset, config).module.eval()   # loads ckpt
    batch = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset))))
    sources, labels = list(batch['source_texts']), list(batch['target_labels'])
    N = int(config.num_samples)
    out = {"checkpoint": str(config.checkpoint_path), "examples": []}
    for b in config.ex_sents:
        ex = {"src_index": int(b), "source": sources[b], "floors": []}
        for t in config.ex_floors:
            lam = torch.tensor([t / 100.0], device=module.device)
            dec = config.get("ex_decode", "argmax")   # argmax | sample (v5 eval) | top3 (v4 eval)
            with torch.no_grad():
                if dec == "sample":
                    o = prompt_model.generate(source_texts=[sources[b]], lmbda=lam, do_sample=True,
                                              top_k=int(config.top_k), top_p=1.0, num_beams=1,
                                              num_repeats=1, infer=True)
                else:
                    o = prompt_model.generate(source_texts=[sources[b]], lmbda=lam, do_sample=False, top_k=3,
                                              top_p=1.0, num_beams=1, num_repeats=1, infer=True,
                                              argmax=(dec == "argmax"))
            toks = o["sample_tokens"][0]
            pstr = score_module._convert_tokens_to_string([toks])[0]
            hyp = score_module.generator.sample_generate_grouped(
                [pstr], [sources[b]], N, score_module.top_k, score_module.top_p)[0]
            c, s = score_module.objectives.compute_scores_flat([sources[b]] * N, hyp, [labels[b]] * N)
            c, s = c.numpy().astype(float), s.numpy().astype(float)
            # representative: nearest to the mean in (content, sentiment)
            k = int(np.argmin((c - c.mean()) ** 2 + (s - s.mean()) ** 2))
            ex["floors"].append({"tau": int(t), "prompt": pstr, "prompt_tokens": list(toks),
                                 "mean_content": float(c.mean()), "mean_sentiment": float(s.mean()),
                                 "frac_meet": float((c >= t).mean()),
                                 "representative": {"text": hyp[k], "content": float(c[k]), "sentiment": float(s[k])},
                                 "samples": [{"text": h, "content": float(ci), "sentiment": float(si)}
                                             for h, ci, si in zip(hyp, c, s)]})
            if len(config.ex_sents) <= 10: print(f"[{b}] tau={t:3d} prompt={pstr!r}  E[c]={c.mean():5.1f} E[s]={s.mean():5.1f}  "
                  f"e.g. {hyp[k]!r} (c={c[k]:.0f}, s={s[k]:.0f})", flush=True)
        out["examples"].append(ex)
    os.makedirs(os.path.dirname(config.ex_out), exist_ok=True)
    json.dump(out, open(config.ex_out, "w"), indent=1)
    print("wrote", config.ex_out)


if __name__ == "__main__":
    main()
