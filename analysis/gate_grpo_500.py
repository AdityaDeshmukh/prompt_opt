"""Step-500 collapse gate for the fixed-KL-anchor GRPO relaunch (2026-07-29).

The previous "beta=0.04 VERIFIED" call was a false positive: it compared GRPO's
19/100 distinct prompts against the *collapsed* KL-free run's 1/100, instead of
against R-REBEL's ~97/100 at the same step. This gate uses the R-REBEL
reference explicitly so that mistake cannot repeat.

Usage: python analysis/gate_grpo_500.py [step]
Exit code 0 = all evaluated seeds PASS, 1 = at least one FAIL/WARN.
"""
import glob
import json
import os
import sys
from collections import defaultdict

V3 = "/scratch/ad11/prompt_opt/outputs/v3"
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 500

# R-REBEL's observed step-500 distinct-prompt counts were 95-99/100 across all
# 9 runs. GRPO need not match that, but anything in the low tens is the
# collapse signature we are trying to eliminate.
PASS_MIN = 60   # >= this: healthy exploration, comparable to R-REBEL
WARN_MIN = 35   # in between: partial collapse, escalate beta
# < WARN_MIN: same failure mode as before, escalate harder


def summarize(path):
    d = json.load(open(path))
    toks = d.get("output_tokens", [])
    distinct = len({" ".join(t) for t in toks})
    by = defaultdict(list)
    for lam, s, c, y in zip(d["lmbdas"], d["mean_scores"],
                            d["mean_contents"], d["mean_styles"]):
        by[round(lam, 2)].append((s, c, y))
    ls = sorted(by)

    def m(l, i):
        return sum(x[i] for x in by[l]) / len(by[l])

    return {
        "distinct": distinct,
        "total": len(toks),
        "score": sum(m(l, 0) for l in ls) / len(ls),
        "content_lo": m(ls[0], 1), "content_hi": m(ls[-1], 1),
        "style_lo": m(ls[0], 2), "style_hi": m(ls[-1], 2),
    }


def main():
    # R-REBEL reference at the same step, for an apples-to-apples baseline
    ref = []
    for run in sorted(os.listdir(V3)):
        if "rrebel" not in run:
            continue
        p = f"{V3}/{run}/eval/outputs.step.{STEP}.json"
        if os.path.exists(p):
            ref.append(summarize(p)["distinct"])
    ref_txt = (f"{min(ref)}-{max(ref)}/100 (n={len(ref)})" if ref
               else "unavailable")

    print(f"=== GRPO step-{STEP} collapse gate ===")
    print(f"R-REBEL reference distinct @ step {STEP}: {ref_txt}")
    print(f"thresholds: PASS >= {PASS_MIN}, WARN >= {WARN_MIN}, "
          f"else FAIL\n")

    rows, bad = [], 0
    for seed in (0, 1, 2):
        run = f"v3_grpo_seed{seed}"
        p = f"{V3}/{run}/eval/outputs.step.{STEP}.json"
        if not os.path.exists(p):
            have = sorted(int(f.split(".")[-2]) for f in
                          glob.glob(f"{V3}/{run}/eval/outputs.step.*.json"))
            print(f"  {run}: no step-{STEP} eval yet "
                  f"(have: {have[-3:] if have else 'none'})")
            continue
        s = summarize(p)
        if s["distinct"] >= PASS_MIN:
            verdict = "PASS"
        elif s["distinct"] >= WARN_MIN:
            verdict = "WARN"; bad += 1
        else:
            verdict = "FAIL"; bad += 1
        rows.append(verdict)
        print(f"  {run}: {verdict}  distinct={s['distinct']}/{s['total']}  "
              f"score={s['score']:.1f}  "
              f"content {s['content_lo']:.0f}->{s['content_hi']:.0f}  "
              f"style {s['style_lo']:.0f}->{s['style_hi']:.0f}")

    if not rows:
        print("\nno GRPO evals at this step yet")
        return 1
    if bad:
        print(f"\n{bad}/{len(rows)} seed(s) below PASS. Escalation ladder:")
        print("  1) grpo_beta=0.04 -> 0.1   (stronger anchor)")
        print("  2) grpo_ent_coef=0.0 -> 0.01 (entropy bonus, the lever "
              "rrebel_l1_ent already has)")
        print("  both are args in slurm/train_v3.slurm ALGO_ARGS")
        return 1
    print(f"\nall {len(rows)} seed(s) PASS - fixed anchor is holding "
          f"exploration open")
    return 0


if __name__ == "__main__":
    sys.exit(main())
