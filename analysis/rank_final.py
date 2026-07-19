"""Final variant ranking from the matched-step (1200) test-split evals.

test/output.step.1200.json entries are ordered [batch][lambda] (10 test
batches x 10 lambdas; each mean_* value is already the mean over that
batch's 50 sources), so aggregate per lambda across batches first.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path('/u/ad11/prompt_opt/outputs/v2')
ALGOS = ['rrebel_l1_ent', 'rrebel_l1_std', 'rrebel_huber_std', 'rrebel_l2', 'grpo']


def per_lambda(d):
    """-> {lmbda: (mean_score, mean_content, mean_style)} averaged over batches."""
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for lm, s, c, st in zip(d['lmbdas'], d['mean_scores'],
                            d['mean_contents'], d['mean_styles']):
        k = round(lm, 1)
        acc[k][0] += s; acc[k][1] += c; acc[k][2] += st; acc[k][3] += 1
    return {k: (v[0]/v[3], v[1]/v[3], v[2]/v[3]) for k, v in sorted(acc.items())}


def hypervolume(pts):
    """Dominated area wrt (0,0) of (content, style)/100 points."""
    best, max_s, segs = sorted(pts), 0.0, []
    for c, s in sorted(best, reverse=True):
        max_s = max(max_s, s)
        segs.append((c, max_s))
    segs.reverse()
    area = prev = 0.0
    for c, s in segs:
        area += (c - prev) * s
        prev = c
    return area


rows = []
for f in sorted(ROOT.glob('*/test/output.step.1200.json')):
    run = f.parts[-3]
    algo = next((a for a in ALGOS if f'_{a}_seed' in run), None)
    arm = 'lora' if run.startswith('v2lora') else 'mlp'
    d = json.load(open(f))
    pl = per_lambda(d)
    ms = sum(v[0] for v in pl.values()) / len(pl)
    hv = hypervolume([(v[1]/100, v[2]/100) for v in pl.values()])
    rows.append((algo, arm, run, ms, hv))

print(f"{'algo':18s} {'arm':5s} {'run':34s} {'score':>8s} {'hv':>7s}")
for algo, arm, run, ms, hv in sorted(rows, key=lambda r: -r[3]):
    print(f"{algo:18s} {arm:5s} {run:34s} {ms:8.2f} {hv:7.3f}")

print("\n=== per-algo aggregate (all arms pooled; test split, step 1200) ===")
agg = defaultdict(list)
for algo, arm, run, ms, hv in rows:
    agg[algo].append((ms, hv))
for algo in sorted(agg, key=lambda a: -sum(x[0] for x in agg[a])/len(agg[a])):
    e = agg[algo]
    n = len(e)
    print(f"  {algo:18s} n={n}  score={sum(x[0] for x in e)/n:7.2f}  "
          f"hv={sum(x[1] for x in e)/n:6.3f}")

print("\n=== MLP arm only ===")
aggm = defaultdict(list)
for algo, arm, run, ms, hv in rows:
    if arm == 'mlp':
        aggm[algo].append((ms, hv))
for algo in sorted(aggm, key=lambda a: -sum(x[0] for x in aggm[a])/len(aggm[a])):
    e = aggm[algo]
    n = len(e)
    print(f"  {algo:18s} n={n}  score={sum(x[0] for x in e)/n:7.2f}  "
          f"hv={sum(x[1] for x in e)/n:6.3f}")
