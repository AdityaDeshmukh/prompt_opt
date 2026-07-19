"""Rank R-REBEL variants from the training-time dev evals.

Each run wrote eval/outputs.step.<S>.json every 500 steps:
  lmbdas [10], mean_scores [10], mean_contents [10], mean_styles [10]
(dev split, 10 sources, lambda sweep 0..0.9).

For every eval step S available, compare variants over the runs that reached S
(fair: same step, same protocol). Metrics per run@S:
  - mean_score:    mean over lambda of the constrained score (what training maximizes)
  - hypervolume:   AUC-style dominated area of the (content, style) points wrt (0,0)
  - monotonicity:  Spearman-ish sign agreement of lambda vs content (controllability)

Usage: python analysis/rank_variants.py [outputs_root]
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '/u/ad11/prompt_opt/outputs/v2')
ALGOS = ['rrebel_l2', 'rrebel_l1_std', 'rrebel_l1_ent', 'rrebel_huber_std', 'grpo']


def run_algo(run_name: str):
    for a in ALGOS:
        if f'_{a}_seed' in run_name:
            arm = 'lora' if run_name.startswith('v2lora') else 'mlp'
            return a, arm
    return None, None


def hypervolume(contents, styles):
    """Dominated area of the per-lambda (content, style) points wrt origin.
    Scores are on 0-100 scales; normalize to [0,1]."""
    pts = sorted((c / 100.0, s / 100.0) for c, s in zip(contents, styles))
    # keep the Pareto staircase (max style for increasing content)
    best, prev_c, area = [], -1, 0.0
    for c, s in pts:
        while best and best[-1][1] <= s:
            best.pop()
        best.append((c, s))
    prev = 0.0
    for c, s in reversed(best):   # descending style, ascending content
        pass
    # simpler: integrate max style achievable at content >= c over the c-grid
    best_sorted = sorted(best)
    area, prev_c = 0.0, 0.0
    # sweep from high content to low, tracking max style
    max_s, segs = 0.0, []
    for c, s in sorted(best_sorted, reverse=True):
        max_s = max(max_s, s)
        segs.append((c, max_s))
    segs.reverse()  # ascending content, style = max achievable at >= that content
    prev_c = 0.0
    for c, s in segs:
        area += (c - prev_c) * s
        prev_c = c
    return area


def monotonicity(lmbdas, contents):
    """Fraction of adjacent lambda pairs where content moves the right way
    (higher lambda -> content should not decrease)."""
    ok = tot = 0
    for i in range(len(lmbdas) - 1):
        d = contents[i + 1] - contents[i]
        tot += 1
        ok += 1 if d >= -1.0 else 0   # 1-point tolerance
    return ok / max(tot, 1)


def main():
    # gather: data[algo][arm][step] -> list of (run, mean_score, hv, mono)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    steps_seen = set()
    for run_dir in sorted(ROOT.iterdir()):
        algo, arm = run_algo(run_dir.name)
        if algo is None:
            continue
        for f in sorted((run_dir / 'eval').glob('outputs.step.*.json')):
            step = int(f.stem.split('.')[-1])
            try:
                d = json.load(open(f))
            except json.JSONDecodeError:
                print(f"  [warn] corrupt eval json skipped: {f}")
                continue
            ms = sum(d['mean_scores']) / len(d['mean_scores'])
            hv = hypervolume(d['mean_contents'], d['mean_styles'])
            mono = monotonicity(d['lmbdas'], d['mean_contents'])
            data[algo][arm][step].append((run_dir.name, ms, hv, mono))
            steps_seen.add(step)

    for step in sorted(steps_seen):
        rows = []
        for algo in ALGOS:
            for arm in ('mlp', 'lora'):
                entries = data[algo][arm].get(step, [])
                if not entries:
                    continue
                n = len(entries)
                ms = sum(e[1] for e in entries) / n
                hv = sum(e[2] for e in entries) / n
                mono = sum(e[3] for e in entries) / n
                rows.append((algo, arm, n, ms, hv, mono))
        if not rows:
            continue
        print(f"\n=== step {step} ===")
        print(f"{'algo':18s} {'arm':5s} {'n':>2s} {'score':>8s} {'hv':>7s} {'mono':>6s}")
        for algo, arm, n, ms, hv, mono in sorted(rows, key=lambda r: -r[3]):
            print(f"{algo:18s} {arm:5s} {n:2d} {ms:8.2f} {hv:7.3f} {mono:6.2f}")

    # aggregate rank across steps (mlp arm, rrebel only): mean rank by score
    print("\n=== aggregate: mean rank by score across steps (both arms pooled) ===")
    ranksum, rankcnt = defaultdict(float), defaultdict(int)
    for step in sorted(steps_seen):
        scores = []
        for algo in ALGOS:
            if algo == 'grpo':
                continue
            entries = data[algo]['mlp'].get(step, []) + data[algo]['lora'].get(step, [])
            if entries:
                scores.append((algo, sum(e[1] for e in entries) / len(entries), len(entries)))
        if len(scores) < 2:
            continue
        scores.sort(key=lambda x: -x[1])
        for rank, (algo, ms, n) in enumerate(scores, 1):
            ranksum[algo] += rank
            rankcnt[algo] += 1
    for algo in sorted(ranksum, key=lambda a: ranksum[a] / rankcnt[a]):
        print(f"  {algo:18s} mean-rank {ranksum[algo]/rankcnt[algo]:.2f} over {rankcnt[algo]} steps")


if __name__ == '__main__':
    main()
