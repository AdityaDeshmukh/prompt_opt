"""Diagnose the geometry of the tradeoff curve: why several floors tau land on
the same point, and why a point's expected content E[c] is not its tau.

One GPU job per arm (slurm/frontier_probe.slurm), two measurements:

 A. Policy map (policy only, no task LM).
    * Fine lambda grid 0, 0.005, ..., 0.995 x all 500 test sentences: the
      ARGMAX prompt and, at each of its 5 positions, the top1-top2 logit
      margin and the top-3-truncated probability of the argmax token.
    * 20-point eval grid: the probability of every menu prompt under the
      evaluator's decoder. LMAdaptorModel.greedy_search does NOT take the
      argmax: it samples from the top-3-truncated policy (the argmax line has
      been commented out since the repo's first commit). So the "greedy"
      prompt of every eval is one draw from a <=3^5-prompt distribution, and
      the per-(tau) point is a mixture over that distribution.
 B. Menu cross-evaluation. The menu is every prompt the step-12000 evals
    emitted at least 25 times (5 seeds x 10000 rows) for ANY arm -- it covers
    >=98% of every arm's eval rows -- plus this arm's argmax prompts that occur
    in >=5 of the 10000 (sentence, eval-lambda) cells. Job k scores menu[k::4]
    plus its own argmax extras on all 500 test sentences with N=50 task-LM
    samples, and keeps PER-SAMPLE content and sentiment, so any reward (any
    tau, any penalty constant, per-sample vs mean floor) can be recomputed
    offline by analysis/frontier_diagnosis.py.

    python analysis/frontier_probe.py algo=rrebel_l1_std checkpoint_path=... \
        +probe_arm=rrebel_l1_std +probe_out=results/v4/frontier_probe
"""
import os, sys, json, collections, time
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ['rrebel_l1_std', 'rrebel_huber_std', 'grpo_ent', 'grpo_baseref']
EVAL_GRID = [round(0.05 * i, 2) for i in range(20)]
FINE_GRID = [round(0.005 * i, 3) for i in range(200)]
MENU_MIN_COUNT = 25      # over 5 seeds x 10000 rows
EXTRA_MIN_COUNT = 5      # argmax prompts, over 10000 (sentence, eval-lambda) cells
TOPK_EVAL = 3            # hard-coded in LMAdaptorModel.greedy_search


def shared_menu():
    """Deterministic union menu from the archived step-12000 test evals."""
    menu, seen = [], set()
    for arm in ARMS:
        cnt = collections.Counter()
        for seed in range(5):
            f = f'{ROOT}/results/v4/v4_{arm}_seed0/test_full/output.step.12000.seed{seed}.json'
            cnt.update(tuple(p) for p in json.load(open(f))['output_tokens'])
        for p, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            if n >= MENU_MIN_COUNT and p not in seen:
                menu.append(p); seen.add(p)
    return menu


def policy_pass(lm, sources, lam, ids=None, chunk=125):
    """Chunked over sources: the HyperNet materializes two 2048x768 weight
    matrices PER ROW, ~6 GB for 500 rows, next to vLLM's reserved memory."""
    outs = [_policy_pass(lm, sources[i:i + chunk], lam,
                         None if ids is None else ids[i:i + chunk])
            for i in range(0, len(sources), chunk)]
    return tuple(torch.cat(xs, 0) for xs in zip(*outs))


@torch.no_grad()
def _policy_pass(lm, sources, lam, ids=None):
    """One decoding pass over `sources` at a single lambda. ids=None: argmax
    decode. Otherwise teacher-force `ids` [n,5]. Returns ids, per-position
    top-3-truncated log-prob of the chosen token (-inf if outside the top 3),
    and the top1-top2 logit margin."""
    n = len(sources)
    lmbda = torch.full((n, 1), lam, device=lm.device)
    lm._prepare_conditioning(lmbda)
    cache = lm._init_cache(sources)
    out_ids, lp3, margin = [], [], []
    for t in range(5):
        logits = lm._adapted_logits(lmbda, cache['state']).float()
        top_v, top_i = torch.topk(logits, k=TOPK_EVAL, dim=-1)
        a = top_i[:, 0] if ids is None else ids[:, t]
        in_top = (top_i == a[:, None])
        logp_top = torch.log_softmax(top_v, dim=-1)
        lp = torch.where(in_top.any(-1), (logp_top * in_top).sum(-1),
                         torch.full((n,), float('-inf'), device=lm.device))
        out_ids.append(a); lp3.append(lp); margin.append(top_v[:, 0] - top_v[:, 1])
        cache = lm._step_cache(cache, a)
    lm._finish_conditioning()
    st = lambda xs: torch.stack(xs, 1).cpu()
    return st(out_ids), st(lp3), st(margin)


@hydra.main(version_base=None, config_path="../", config_name="tst_config")
def main(config: "DictConfig"):
    arm, out_dir = config.probe_arm, config.probe_out
    k = ARMS.index(arm)
    os.makedirs(out_dir, exist_ok=True)
    _, _, test_dataset = make_text_style_transfer_datasets(config)
    policy_model = build_policy_model(config)
    prompt_model = SinglePromptModel(policy_model, config)
    config.style_classifier = get_style_classifier('train', config)
    score_module = PromptedTextStyleTransferScore(config)
    module = ScoreLossModule(prompt_model, score_module, config)
    trainer = ScoreTrainer(module, None, test_dataset, config)   # loads ckpt
    module = trainer.module.eval()
    lm = module._model._model
    tok = lm.tokenizer
    batch = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset))))
    sources, labels = list(batch['source_texts']), list(batch['target_labels'])
    n_src = len(sources)
    assert n_src == 500, n_src

    # ---- A1. argmax map on the fine grid --------------------------------
    t0 = time.time()
    fine_ids = np.zeros((len(FINE_GRID), n_src, 5), np.int32)
    fine_lp3 = np.zeros((len(FINE_GRID), n_src, 5), np.float32)
    fine_margin = np.zeros((len(FINE_GRID), n_src, 5), np.float32)
    for a, lam in enumerate(FINE_GRID):
        i, lp, m = policy_pass(lm, sources, lam)
        fine_ids[a], fine_lp3[a], fine_margin[a] = i.numpy(), lp.numpy(), m.numpy()
    print(f"A1 fine argmax map: {time.time() - t0:.0f}s", flush=True)

    # ---- menu = shared union + this arm's frequent argmax prompts ---------
    menu = shared_menu()
    idx_eval = [FINE_GRID.index(l) for l in EVAL_GRID]
    amax = collections.Counter(
        tuple(tok.convert_ids_to_tokens(fine_ids[a, b].tolist()))
        for a in idx_eval for b in range(n_src))
    extras = [p for p, n in sorted(amax.items(), key=lambda kv: (-kv[1], kv[0]))
              if n >= EXTRA_MIN_COUNT and p not in set(menu)]
    full_menu = menu + extras
    menu_ids = torch.tensor([tok.convert_tokens_to_ids(list(p)) for p in full_menu])
    for p, row in zip(full_menu, menu_ids.tolist()):   # token strings round-trip
        assert tuple(tok.convert_ids_to_tokens(row)) == p, p
    print(f"menu: {len(menu)} shared + {len(extras)} argmax extras", flush=True)

    # ---- A2. eval-decoder probability of every menu prompt ------------------
    t0 = time.time()
    menu_lp3 = np.zeros((len(full_menu), len(EVAL_GRID), n_src), np.float32)
    for j in range(len(full_menu)):
        ids = menu_ids[j].to(lm.device).expand(n_src, 5)
        for a, lam in enumerate(EVAL_GRID):
            _, lp, _ = policy_pass(lm, sources, lam, ids=ids)
            menu_lp3[j, a] = lp.sum(-1).numpy()
    print(f"A2 menu log-probs: {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(
        f'{out_dir}/{arm}.policy.npz', fine_grid=np.array(FINE_GRID),
        eval_grid=np.array(EVAL_GRID), fine_ids=fine_ids, fine_lp3=fine_lp3,
        fine_margin=fine_margin, menu_lp3=menu_lp3,
        menu=np.array(['\x1f'.join(p) for p in full_menu], dtype=object),
        n_shared=len(menu), checkpoint=str(config.checkpoint_path))

    # ---- B. per-sample cross-evaluation of this job's menu chunk ------------
    mine = list(range(k, len(menu), 4)) + list(range(len(menu), len(full_menu)))
    N = int(config.num_samples)
    C = np.zeros((len(mine), n_src, N), np.float16)
    S = np.zeros((len(mine), n_src, N), np.float16)
    gen, obj = score_module.generator, score_module.objectives
    t0 = time.time()
    for r, j in enumerate(mine):
        pstr = score_module._convert_tokens_to_string([list(full_menu[j])])[0]
        hyp = gen.sample_generate_grouped([pstr] * n_src, sources, N,
                                          score_module.top_k, score_module.top_p)
        c, s = obj.compute_scores_flat(
            [x for x in sources for _ in range(N)],
            [h for row in hyp for h in row],
            [l for l in labels for _ in range(N)])
        C[r], S[r] = c.view(n_src, N).numpy(), s.view(n_src, N).numpy()
        print(f"B {r + 1}/{len(mine)} {pstr!r}: c={C[r].astype(float).mean():.1f} "
              f"s={S[r].astype(float).mean():.1f} ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(
        f'{out_dir}/{arm}.menu.npz', menu_index=np.array(mine),
        menu=np.array(['\x1f'.join(full_menu[j]) for j in mine], dtype=object),
        content=C, style=S, N=N, vllm_seed=int(config.vllm_seed))
    print(f"wrote {out_dir}/{arm}.policy.npz and {arm}.menu.npz")


if __name__ == "__main__":
    main()
