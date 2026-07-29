"""Health gate for the fixed-KL-anchor GRPO relaunch (2026-07-29).

Supersedes gate_grpo_500.py, whose verdict keyed on distinct-prompt count
alone. That was the wrong primary criterion: at step 500 the fixed anchor gave
seed0 score 34.1 (vs R-REBEL 33.1, vs the old rolling-anchor run's 23.9) with
only 6/100 distinct prompts. Distinct count conflates two different states --

  pathological : few prompts, LOW score, flat trajectory   <- the real failure
  committed    : few prompts, GOOD score                   <- possibly fine

-- so it cannot be the deciding signal. The question this gate answers is the
one that actually matters: is GRPO's score near R-REBEL's at the matched step,
and if not, is it still climbing? Distinct count is kept as a diagnostic.

Usage: python analysis/gate_grpo.py [step]     (default: latest common step)
Exit 0 = pass/watch, 1 = pathological, 2 = not enough data yet.
"""
import glob
import json
import os
import sys
from collections import defaultdict

V3 = "/scratch/ad11/prompt_opt/outputs/v3"
SEEDS = (0, 1, 2)
# how far below the R-REBEL band GRPO may sit before we call it a problem
SCORE_TOLERANCE = 4.0
# score gain over the last MIN_TRAJ evals that counts as "still climbing"
CLIMB_EPS = 0.5
MIN_TRAJ = 3


def summarize(path):
    d = json.load(open(path))
    by = defaultdict(list)
    for lam, s, c, y in zip(d["lmbdas"], d["mean_scores"],
                            d["mean_contents"], d["mean_styles"]):
        by[round(lam, 2)].append((s, c, y))
    ls = sorted(by)

    def m(l, i):
        return sum(x[i] for x in by[l]) / len(by[l])

    toks = d.get("output_tokens", [])
    return {
        "score": sum(m(l, 0) for l in ls) / len(ls),
        "style_lo": m(ls[0], 2),
        "content_hi": m(ls[-1], 1),
        "distinct": len({" ".join(t) for t in toks}),
        "total": len(toks),
    }


def curve(run):
    out = {}
    for p in glob.glob(f"{V3}/{run}/eval/outputs.step.*.json"):
        try:
            out[int(p.split(".")[-2])] = summarize(p)
        except Exception:
            pass
    return dict(sorted(out.items()))


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else None
    runs = sorted(os.listdir(V3))
    rr = {r: curve(r) for r in runs if "rrebel" in r}
    gr = {f"v3_grpo_seed{s}": curve(f"v3_grpo_seed{s}") for s in SEEDS}

    have = [max(c) for c in gr.values() if c]
    if not have:
        print("no GRPO evals yet")
        return 2
    step = want if want is not None else min(have)

    band = [c[step]["score"] for c in rr.values() if step in c]
    if band:
        lo, hi = min(band), max(band)
        band_txt = f"{lo:.1f}-{hi:.1f} (n={len(band)})"
    else:
        lo = hi = None
        band_txt = "unavailable at this step"

    print(f"=== GRPO health gate @ step {step} ===")
    print(f"R-REBEL score band: {band_txt}")
    if lo is not None:
        print(f"pass if GRPO score >= {lo - SCORE_TOLERANCE:.1f} "
              f"(band low - {SCORE_TOLERANCE})")
    print()

    worst = 0
    for run, c in gr.items():
        if step not in c:
            near = [s for s in c if s <= step]
            print(f"  {run}: no eval at {step} "
                  f"(latest: {max(near) if near else 'none'})")
            continue
        s = c[step]
        pts = [k for k in c if k <= step][-MIN_TRAJ:]
        traj = [c[k]["score"] for k in pts]
        climbing = len(traj) >= 2 and (traj[-1] - traj[0]) > CLIMB_EPS
        trend = " -> ".join(f"{v:.1f}" for v in traj)

        if lo is None:
            verdict = "INFO"
        elif s["score"] >= lo - SCORE_TOLERANCE:
            verdict = "PASS"
        elif len(traj) < MIN_TRAJ:
            verdict = "EARLY"
        elif climbing:
            verdict = "WATCH"
        else:
            verdict = "FAIL"; worst = 1

        print(f"  {run}: {verdict:5s} score={s['score']:5.1f}  "
              f"style@0={s['style_lo']:5.1f}  "
              f"distinct={s['distinct']:3d}/{s['total']}  "
              f"trend[{trend}]")

    print()
    if worst:
        print("PATHOLOGICAL: score below the R-REBEL band AND not climbing.")
        print("Escalation ladder (slurm/train_v3.slurm ALGO_ARGS):")
        print("  1) grpo_beta=0.04 -> 0.1      stronger anchor")
        print("  2) grpo_ent_coef=0.0 -> 0.01  entropy bonus "
              "(the lever rrebel_l1_ent already has)")
    else:
        print("No pathology: GRPO is at or near the R-REBEL band, or still "
              "climbing toward it.")
        print("NOTE: a low distinct count with a healthy score means early "
              "commitment, not collapse -- judge it on the trend, and only "
              "escalate if the score stalls below the band.")
    return worst


if __name__ == "__main__":
    sys.exit(main())
