"""Final v3 comparison table: R-REBEL variants vs both GRPO arms.

Reports (a) the 12000-step result per arm and (b) a matched-step table, since
the GRPO arms were relaunched later and lag on steps. Matched-step is the
honest comparison while anything is unfinished.

Usage: python analysis/final_table.py
"""
import json
import os
from collections import defaultdict

V3 = "/scratch/ad11/prompt_opt/outputs/v3"
ARMS = ["rrebel_l1_std", "rrebel_huber_std", "rrebel_l1_ent",
        "grpo", "grpo_ent"]
SEEDS = (0, 1, 2)
# ~1 point: vllm_seed=null makes task-LM sampling nondeterministic; a re-run of
# one step-500 eval moved 34.1 -> 33.0. Differences under this are not real.
NOISE = 1.0


def summ(run, st):
    p = f"{V3}/{run}/eval/outputs.step.{st}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    by = defaultdict(list)
    for lam, s, c, y in zip(d["lmbdas"], d["mean_scores"],
                            d["mean_contents"], d["mean_styles"]):
        by[round(lam, 2)].append((s, c, y))
    ls = sorted(by)

    def m(l, i):
        return sum(x[i] for x in by[l]) / len(by[l])

    return (sum(m(l, 0) for l in ls) / len(ls), m(ls[0], 2),
            len({" ".join(t) for t in d.get("output_tokens", [])}))


def arm_at(a, st):
    vs = [summ(f"v3_{a}_seed{s}", st) for s in SEEDS]
    return [v for v in vs if v]


def main():
    print("=" * 72)
    print("FINAL @ step 12000")
    print("=" * 72)
    finals = {}
    for a in ARMS:
        vs = arm_at(a, 12000)
        if not vs:
            print(f"  {a:18s} -- no seed finished")
            continue
        mean = sum(v[0] for v in vs) / len(vs)
        finals[a] = (mean, len(vs))
        print(f"  {a:18s} {mean:6.2f}   seeds=" +
              " ".join(f"{v[0]:.1f}" for v in vs) +
              f"   style@0={sum(v[1] for v in vs)/len(vs):5.1f}"
              f"   distinct={sum(v[2] for v in vs)/len(vs):4.1f}"
              f"   n={len(vs)}/3")

    complete = {a: v for a, v in finals.items() if v[1] == 3}
    if complete:
        print("\n  ranking (complete arms only, n=3):")
        ranked = sorted(complete.items(), key=lambda kv: -kv[1][0])
        for i, (a, (m, _)) in enumerate(ranked, 1):
            tie = ""
            if i > 1 and abs(ranked[i - 2][1][0] - m) < NOISE:
                tie = f"  <- within noise of {ranked[i - 2][0]}"
            print(f"    {i}. {a:18s} {m:6.2f}{tie}")

    print()
    print("=" * 72)
    print("MATCHED-STEP (mean over seeds; '*' = fewer than 3 seeds)")
    print("=" * 72)
    hdr = f"{'step':>6} | " + " | ".join(f"{a[:16]:>16}" for a in ARMS)
    print(hdr)
    print("-" * len(hdr))
    for st in list(range(1000, 12001, 1000)):
        cells = []
        for a in ARMS:
            vs = arm_at(a, st)
            if not vs:
                cells.append(f"{'-':>16}")
            else:
                mark = "" if len(vs) == 3 else "*"
                cells.append(f"{sum(v[0] for v in vs)/len(vs):15.2f}{mark}")
        print(f"{st:>6} | " + " | ".join(cells))
    print(f"\nnoise floor ~{NOISE} pt (vllm_seed=null); "
          f"differences below it are not meaningful.")


if __name__ == "__main__":
    main()
