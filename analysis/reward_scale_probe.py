"""Measure the within-group reward spread sigma(x, lambda) under the TRAINING
sampler, to ground the theory of R-REBEL-std (paper Sec. "Theory").

For each test source x and each lambda on the grid, draw G=num_repeats prompts
exactly as a training step does (ScoreLossModule._decode_sampling: top-k=200
sampling from the lambda-conditioned policy) and score each with the same
N=50-sample Monte-Carlo constrained reward. Nothing is updated; this is one
training forward pass per (x, lambda) group with gradients disabled.

Output: one JSON per checkpoint with, per group, the G rewards and the
per-prompt content / style / feasible means, so any statistic of the group
(std, plug-in KL of the R-REBEL target, ...) can be computed offline.

    python analysis/reward_scale_probe.py algo=rrebel_l1_std \
        checkpoint_path=.../ckpt.step.12000.pth max_size=40 \
        "eval_lmbdas=[0.0,0.1,...,0.9]" +probe_out=results/v4/reward_scale_probe/x.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydra
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
    # ScoreTrainer loads the checkpoint exactly as run_eval.py does
    trainer = ScoreTrainer(module, None, test_dataset, config)
    module = trainer.module.eval()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    grid = [float(v) for v in config.eval_lmbdas]
    G = int(config.num_repeats)

    groups, n_seen = [], 0
    for batch in DataLoader(test_dataset, batch_size=config.eval_batch_size):
        n = len(batch['source_texts'])
        for lam in grid:
            lmbda = torch.full((n,), lam, device=device)
            with torch.no_grad():
                _, _, tokens, _, _ = module._decode_sampling(lmbda=lmbda, batch=batch)
                s, c, st, _ = module.compute_scores(
                    lmbda=lmbda, batch=batch, output_tokens=tokens, mode="train")
            s, c, st = (t.view(n, G).cpu().tolist() for t in (s, c, st))
            for i in range(n):
                groups.append({'src': n_seen + i, 'lmbda': lam,
                               'reward': s[i], 'content': c[i], 'style': st[i]})
        n_seen += n
        print(f"probed {n_seen} sources", flush=True)

    out = config.probe_out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'checkpoint': str(config.checkpoint_path), 'G': G,
               'N': int(config.num_samples), 'grid': grid, 'groups': groups},
              open(out, 'w'))
    print(f"wrote {len(groups)} groups -> {out}")


if __name__ == "__main__":
    main()
