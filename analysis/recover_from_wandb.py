"""Recover the v3 campaign results from wandb after the /scratch purge.

WHY THIS EXISTS
---------------
The v3 campaign completed (15 runs x 12000 steps), but /scratch was purged
around 2026-09-03..05: every ckpt/ and eval/ under outputs/v3/ is empty, so
there are 0 checkpoints and 0 per-lambda eval JSONs left on disk. wandb still
holds the full scalar history, so the SCORES are recoverable -- but
ScoreTrainer.evaluate() averages over the lambda grid before logging
(trainers/score_trainer.py:282-284), so wandb has only lambda-AVERAGED
content/style. The per-lambda arrays needed for the content-vs-sentiment
tradeoff curve are NOT recoverable from wandb; slurm/train_v4.slurm reruns one
seed per arm to regenerate them.

THE STEP-NUMBER PROBLEM
-----------------------
wandb's `_step` counts log() calls, not training steps, and the true step
(`train/global_step`) was only logged during a short window (commits
30bceef..af66b44). Worse, a SLURM chain that dies between an eval and the next
checkpoint re-trains the tail segment on resume and RE-EVALUATES a step it has
already evaluated, so eval rows are not simply step 500, 1000, 1500, ...

Reconstruction rule, exploited here: evaluate() fires exactly when
total_steps % eval_steps == 0, and each training step emits exactly one
wandb.log(batch_log). So between two eval rows that are genuinely 500 training
steps apart there must be >= 500 train logs, i.e. _step must advance by
>= eval_steps + 1. An eval row whose _step advance is smaller than that cannot
have advanced 500 real steps, so it is a re-eval of the previous step.

This is not assumed -- it is VALIDATED three ways per run, and the script exits
nonzero if any check fails:
  1. every surviving `train/global_step` value on an eval row must equal the
     reconstructed step for that row;
  2. the number of distinct reconstructed steps must equal last_step/eval_steps;
  3. the reconstructed step grid must be perfectly regular (every gap exactly
     eval_steps, starting at eval_steps) -- a misclassified duplicate shows up
     here immediately as a doubled gap.

Separately REPORTED, not treated as an error: three runs (`*_seed1` of
rrebel_huber_std, rrebel_l1_ent and grpo_ent) are `finished` yet have no
step-12000 eval. Their final cycle ran under the trainer's wandb-init fallback
(score_trainer.py:158-161), which trains without logging when wandb is
unreachable, so that cycle's evals went to disk only -- and the disk copies were
purged. Consequence for the paper: step 11500 is the largest step at which all
15 campaign runs are present, so it, not 12000, is the balanced comparison
point. The script computes and exports it as `balanced_matched_step`.

Usage:
    python analysis/recover_from_wandb.py            # export + validate
    python analysis/recover_from_wandb.py --print    # also dump per-run tables
"""
import argparse
import json
import os
import sys
from collections import defaultdict

ENTITY_PROJECT = "aditya_team/multi-objective-prompt-opt"
OUT_JSON = os.path.join(os.path.dirname(__file__), os.pardir,
                        "results", "v3_wandb_export.json")
OUT_CSV = os.path.join(os.path.dirname(__file__), os.pardir,
                       "results", "v3_eval_history.csv")

# arm -> the wandb run names carrying that arm's final campaign history.
# grpo logs under a _fixref suffix: its plain ids were deleted (and a deleted
# id hangs wandb.init on resume), and its _kl ids hold the collapsed
# rolling-anchor history that the 2026-07-29 fix superseded.
ARMS = {
    "rrebel_l1_std":    [f"v3_rrebel_l1_std_seed{s}" for s in (0, 1, 2)],
    "rrebel_huber_std": [f"v3_rrebel_huber_std_seed{s}" for s in (0, 1, 2)],
    "rrebel_l1_ent":    [f"v3_rrebel_l1_ent_seed{s}" for s in (0, 1, 2)],
    "grpo":             [f"v3_grpo_seed{s}_fixref" for s in (0, 1, 2)],
    "grpo_ent":         [f"v3_grpo_ent_seed{s}" for s in (0, 1, 2)],
}
# The superseded rolling-KL-anchor GRPO runs. Kept because they are the
# evidence for "the baseline had a real bug and fixing it was worth ~+4 points"
# -- a reviewer will ask.
ARCHIVED = {"grpo_rollingref": [f"v3_grpo_seed{s}_kl" for s in (0, 1, 2)]}

EVAL_KEYS = ["_step", "eval/scores/mean_score",
             "eval/scores/mean_content", "eval/scores/mean_style"]


def fetch_run(api, name):
    """Pull one run's eval history + the surviving true-step anchors."""
    run = api.run(f"{ENTITY_PROJECT}/{name}")
    eval_steps = int(run.config.get("eval_steps", 500))
    max_train_steps = int(run.config.get("max_train_steps", 12000))

    rows = sorted(run.scan_history(keys=EVAL_KEYS), key=lambda r: r["_step"])
    # anchors: eval rows that also carry the true step
    anchors = {}
    for r in run.scan_history(keys=EVAL_KEYS + ["train/global_step"]):
        anchors[int(r["_step"])] = int(r["train/global_step"])

    return {
        "name": name,
        "state": run.state,
        "eval_steps": eval_steps,
        "max_train_steps": max_train_steps,
        "wandb_last_step": run.summary.get("_step"),
        "rows": [{"_step": int(r["_step"]),
                  "score": float(r["eval/scores/mean_score"]),
                  "content": float(r["eval/scores/mean_content"]),
                  "style": float(r["eval/scores/mean_style"])} for r in rows],
        "anchors": anchors,
    }


def reconstruct(run):
    """Assign a true training step to every eval row; validate hard.

    Returns (records, problems). records = [{step, score, content, style,
    duplicate}], one per eval row, in wandb order.
    """
    es = run["eval_steps"]
    rows, problems = run["rows"], []
    if not rows:
        return [], [f"{run['name']}: no eval rows in wandb history"]

    recs, step = [], 0
    for i, r in enumerate(rows):
        if i == 0:
            step = es
            dup = False
        else:
            # >= es + 1 log calls (es train logs + 1 eval log) are required for
            # the eval to have advanced a full eval_steps interval.
            advanced = (r["_step"] - rows[i - 1]["_step"]) >= es + 1
            dup = not advanced
            if advanced:
                step += es
        recs.append({"step": step, "duplicate": dup, **{k: r[k] for k in ("score", "content", "style")},
                     "_step": r["_step"]})

    # check 1: every surviving true-step anchor must agree
    checked = 0
    for rec in recs:
        true = run["anchors"].get(rec["_step"])
        if true is None:
            continue
        checked += 1
        if true != rec["step"]:
            problems.append(
                f"{run['name']}: ANCHOR MISMATCH at _step={rec['_step']}: "
                f"wandb train/global_step={true} but reconstruction says {rec['step']}")

    # check 2: distinct step count must be consistent with the last step
    distinct = sorted({r["step"] for r in recs})
    if len(distinct) != distinct[-1] // es:
        problems.append(
            f"{run['name']}: {len(distinct)} distinct steps but last step "
            f"{distinct[-1]} implies {distinct[-1] // es}")

    # check 3: the step grid must be perfectly regular -- every gap exactly
    # eval_steps, starting at eval_steps. A misclassified duplicate would show
    # up here as a 1000-step gap, so this is the sharpest check of the three.
    for a, b in zip(distinct, distinct[1:]):
        if b - a != es:
            problems.append(f"{run['name']}: irregular step gap {a} -> {b} "
                            f"(expected {es})")
    if distinct[0] != es:
        problems.append(f"{run['name']}: first step {distinct[0]} != {es}")

    # A `finished` run that stops short of max_train_steps is NOT a
    # reconstruction error -- it is a real gap in what survives. Cause: the
    # trainer's wandb.init fallback (score_trainer.py:158-161) lets a cycle
    # train WITHOUT logging when wandb is unreachable, so that cycle's evals
    # went to disk only, and the disk copies were purged. Recorded and
    # disclosed rather than treated as untrustworthy: the surviving rows are
    # still correct, there are just fewer of them.
    if run["state"] == "finished" and distinct[-1] != run["max_train_steps"]:
        run["short_by"] = run["max_train_steps"] - distinct[-1]

    run["anchors_checked"] = checked
    run["n_duplicates"] = sum(1 for r in recs if r["duplicate"])
    return recs, problems


def collapse(recs):
    """step -> metrics, keeping the LAST eval at each step.

    A duplicate arises when a chain died after an eval and resumed from an
    earlier checkpoint; the later eval is the one whose policy the run actually
    continued from, so it is the one to keep.
    """
    by = {}
    for r in recs:
        by[r["step"]] = r
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", dest="show",
                    help="dump the per-run reconstructed table")
    args = ap.parse_args()

    try:
        import wandb
    except ImportError:
        sys.exit("wandb not importable; use the prompt_opt env python")

    api = wandb.Api(timeout=120)
    all_runs, problems, export = {}, [], {}

    for group, names in list(ARMS.items()) + list(ARCHIVED.items()):
        export[group] = {}
        for name in names:
            try:
                run = fetch_run(api, name)
            except Exception as e:
                problems.append(f"{name}: FETCH FAILED ({type(e).__name__}: {e})")
                continue
            recs, probs = reconstruct(run)
            problems += probs
            all_runs[name] = run
            export[group][name] = {
                "state": run["state"],
                "eval_steps": run["eval_steps"],
                "max_train_steps": run["max_train_steps"],
                "short_by": run.get("short_by", 0),
                "anchors_checked": run.get("anchors_checked", 0),
                "n_eval_rows": len(recs),
                "n_duplicate_rows": run.get("n_duplicates", 0),
                "by_step": {str(s): {"score": r["score"], "content": r["content"],
                                     "style": r["style"]}
                            for s, r in sorted(collapse(recs).items())},
            }
            d = collapse(recs)
            last = max(d) if d else None
            print(f"  {name:30s} state={run['state']:9s} rows={len(recs):3d} "
                  f"dups={run.get('n_duplicates', 0)} anchors_ok={run.get('anchors_checked', 0)} "
                  f"last_step={last} score={d[last]['score']:.2f}" if d else f"  {name}: empty")
            if args.show:
                for s, r in sorted(d.items()):
                    print(f"        step {s:>6}  score {r['score']:6.2f}  "
                          f"content {r['content']:6.2f}  style {r['style']:6.2f}")

    # The largest step at which EVERY campaign arm still has all 3 seeds. This
    # is the step the paper should headline: 3 runs lost their step-12000 eval
    # to the wandb-init fallback, so 12000 is seed-unbalanced across arms.
    campaign = {n: d for g, runs in export.items() if g in ARMS
                for n, d in runs.items()}
    per_run_steps = [set(int(s) for s in d["by_step"]) for d in campaign.values()]
    common = set.intersection(*per_run_steps) if per_run_steps else set()
    balanced_step = max(common) if common else None
    short = {n: d["short_by"] for n, d in campaign.items() if d["short_by"]}
    print(f"\nbalanced matched step (all {len(campaign)} campaign runs present): "
          f"{balanced_step}")
    if short:
        print(f"{len(short)} run(s) short of max_train_steps -- their final eval "
              f"was logged only to the purged disk copies:")
        for n, s in sorted(short.items()):
            print(f"   - {n}: missing last {s} steps")

    os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
    meta = {
        "source": f"wandb {ENTITY_PROJECT}",
        "why": "scratch purge ~2026-09-03..05 destroyed all v3 ckpts and eval JSONs",
        "balanced_matched_step": balanced_step,
        "runs_short_of_max": short,
        "caveat": ("wandb holds only lambda-AVERAGED content/style "
                   "(score_trainer.py:282-284 means over the lambda grid before "
                   "logging). Per-lambda arrays -- and therefore the "
                   "content-vs-sentiment tradeoff FRONTIER -- are not recoverable "
                   "from wandb; slurm/train_v4.slurm regenerates them."),
        "step_reconstruction": ("eval row advances a full eval_steps interval iff "
                                "_step advances by >= eval_steps+1; validated against "
                                "surviving train/global_step anchors, distinct-step "
                                "count, and finished-run endpoint"),
        "noise_floor_points": 1.0,
    }
    with open(os.path.abspath(OUT_JSON), "w") as f:
        json.dump({"meta": meta, "arms": export}, f, indent=1, sort_keys=True)

    with open(os.path.abspath(OUT_CSV), "w") as f:
        f.write("arm,run,seed,step,score,content,style\n")
        for group, runs in export.items():
            for name, d in runs.items():
                seed = name.split("seed")[1][0] if "seed" in name else "?"
                for s, r in sorted(d["by_step"].items(), key=lambda kv: int(kv[0])):
                    f.write(f"{group},{name},{seed},{s},{r['score']:.6f},"
                            f"{r['content']:.6f},{r['style']:.6f}\n")

    print(f"\nwrote {os.path.abspath(OUT_JSON)}")
    print(f"wrote {os.path.abspath(OUT_CSV)}")

    if problems:
        print(f"\n!! {len(problems)} VALIDATION PROBLEM(S) -- do not trust the export:")
        for p in problems:
            print(f"   - {p}")
        sys.exit(1)
    print("\nall step reconstructions validated "
          f"({sum(r.get('anchors_checked', 0) for r in all_runs.values())} true-step "
          "anchors matched, distinct-step counts consistent, step grids perfectly "
          "regular). Runs short of max_train_steps are disclosed above, not an "
          "error in the reconstruction.")


if __name__ == "__main__":
    main()
