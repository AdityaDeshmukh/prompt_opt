"""Analysis of the v4 FULL test-split evaluation (slurm/eval_full.slurm).

Inputs : results/v4/v4_<arm>_seed0/test_full/output.step.<N>.seed<S>.json
         (per-(sentence, lambda) rows; 500 test sentences x 20 lambdas)
Outputs: paper/figures/frontier.pdf      expected sentiment vs expected content
         paper/figures/control.pdf       content / sentiment / feasibility vs tau
         paper/figures/test_learning.pdf reward & hypervolume vs training step
         paper/tables/full_results.tex   main table with 95% bootstrap CIs
         paper/tables/prompts.tex        the greedy prompt each policy emits per tau
         paper/tables/pairwise.tex       paired differences vs the best GRPO arm
         results/v4/full_eval_summary.json  every number the paper quotes

Statistics. Point estimates average the task-LM seeds first (Monte-Carlo
noise), then the 500 test sentences. Confidence intervals are percentile
bootstraps over SENTENCES (B=2000), with the same resampled sentence indices
used for every arm, so arm differences are PAIRED. Resampling sentences of
seed-averaged values captures both sentence-to-sentence variation and the
residual Monte-Carlo noise, since the latter is independent per sentence.

    python analysis/full_eval.py            # figures + tables
    python analysis/full_eval.py --status   # just report which files exist
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
TAB = os.path.join(ROOT, "paper", "tables")

# Palette: the reference instance's first three categorical slots, which are
# documented to pass the all-pairs CVD/normal-vision floors (overlapping
# curves are an all-pairs form, so slot 4 is not allowed). The two R-REBEL arms
# share the family hue and are separated by linestyle + marker + direct label.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARMS = [
    # key,              label,                         color,  ls,   marker
    ("rrebel_l1_std",    r"R-REBEL ($\ell_1$-std)",     BLUE,   "-",  "o"),
    ("rrebel_huber_std", r"R-REBEL (Huber-std)",        BLUE,   "--", "s"),
    ("grpo_ent",         r"GRPO + entropy",             ORANGE, "-",  "^"),
    ("grpo_baseref",     r"GRPO, base-LM reference",    AQUA,   "-.", "D"),
]
LABEL = {a[0]: a[1] for a in ARMS}
FINAL = 12000
B = 2000


def path(arm, step, seed):
    return os.path.join(ROOT, "results", "v4", f"v4_{arm}_seed0", "test_full",
                        f"output.step.{step}.seed{seed}.json")


def inventory():
    inv = defaultdict(dict)
    for p in glob.glob(os.path.join(ROOT, "results/v4/v4_*_seed0/test_full/output.step.*.seed*.json")):
        m = re.search(r"v4_(.+?)_seed0/test_full/output\.step\.(\d+)\.seed(\d+)\.json$", p)
        inv[(m.group(1), int(m.group(2)))][int(m.group(3))] = p
    return inv


def load(p):
    """-> dict of [n_lam, n_src] arrays + lambda grid + prompts[l][s]."""
    d = json.load(open(p))
    r = d["rows"]
    lams = sorted(set(round(x, 4) for x in r["lmbda"]))
    li = {l: i for i, l in enumerate(lams)}
    n_src = max(r["src"]) + 1
    out = {k: np.full((len(lams), n_src), np.nan) for k in ("content", "style", "score", "feasible")}
    for idx in range(len(r["src"])):
        l, s = li[round(r["lmbda"][idx], 4)], r["src"][idx]
        for k in out:
            out[k][l, s] = r[k][idx]
    # prompts: output_tokens are appended in the same (batch, lambda, sentence) order
    prompts = [[None] * n_src for _ in lams]
    for idx, toks in enumerate(d["output_tokens"]):
        prompts[li[round(r["lmbda"][idx], 4)]][r["src"][idx]] = toks
    assert not any(np.isnan(v).any() for v in out.values()), f"holes in {p}"
    out["lams"] = np.array(lams)
    out["prompts"] = prompts
    return out


def seed_avg(runs):
    keys = ("content", "style", "score", "feasible")
    return {k: np.mean([r[k] for r in runs], axis=0) for k in keys} | {"lams": runs[0]["lams"]}


def hypervolume(c, s):
    """Area of the union of [0,c_k]x[0,s_k], as a percentage of the 100x100 box."""
    pts = sorted(zip(c, s), key=lambda t: -t[0])
    area, best_s = 0.0, 0.0
    for i, (ci, si) in enumerate(pts):
        best_s = max(best_s, si)
        c_next = pts[i + 1][0] if i + 1 < len(pts) else 0.0
        area += (ci - c_next) * best_s
    return 100.0 * area / (100.0 * 100.0)


def upper_hull(c, s):
    """Upper concave envelope of the (content, sentiment) points: what a policy
    can achieve IN EXPECTATION by randomizing between its prompts, since
    expected content and sentiment are linear in the mixture weights."""
    up = []
    for p in sorted(set(zip(c, s))):
        while len(up) >= 2 and ((up[-1][0] - up[-2][0]) * (p[1] - up[-2][1])
                                - (up[-1][1] - up[-2][1]) * (p[0] - up[-2][0])) >= 0:
            up.pop()
        up.append(p)
    return np.array(up)


def attain_mix(H, x):
    """Best expected sentiment at expected content >= x, mixing allowed."""
    xs, ys = H[:, 0], H[:, 1]
    if x > xs.max():
        return -np.inf
    return max(np.interp(x, xs, ys), ys[xs >= x].max()) if x > xs.min() else ys.max()


def attain_step(c, s, x):
    """Best sentiment among the evaluated points with content >= x (no mixing)."""
    m = np.asarray(c) >= x
    return np.asarray(s)[m].max() if m.any() else -np.inf


def metrics(A, idx=None):
    """A: seed-averaged arrays; idx: bootstrap sentence indices (None = all)."""
    sl = (slice(None), idx) if idx is not None else (slice(None), slice(None))
    C, S, R, F = (A[k][sl] for k in ("content", "style", "score", "feasible"))
    ec, es = C.mean(1), S.mean(1)
    tau = 100 * A["lams"]
    return {
        "reward": R.mean(), "content": C.mean(), "sentiment": S.mean(),
        "hv": hypervolume(ec, es),
        "met": float((ec >= tau - 1e-9).sum()),        # of len(tau) floors met in expectation
        "feasible": F.mean(),                           # P(content >= tau), per sample
        "ec": ec, "es": es, "ef": F.mean(1),
    }


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def distinct_prompts(run):
    return [len({" ".join(p) for p in row}) for row in run["prompts"]]


_TOK = None


def detok(toks):
    """Byte-level GPT-2 decoding via the policy's own tokenizer."""
    global _TOK
    if _TOK is None:
        os.environ.setdefault("HF_HOME", "/scratch/ad11/hf_cache")
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("distilgpt2")
    return _TOK.convert_tokens_to_string(list(toks)).strip(" ")


def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}",
           '"': "{\\textquotedbl}", "<": r"\textless{}", ">": r"\textgreater{}",
           "\n": r"{\textbackslash}n", "\u2212": r"$-$", "\u00d7": r"$\times$",
           "\u202e": r"\textnormal{[RLO]}"}
    out = "".join(rep.get(ch, ch) for ch in s)
    bad = [ch for ch in out if ord(ch) > 127]
    assert not bad, f"unmapped non-ASCII in prompt: {[hex(ord(c)) for c in bad]}"
    return out


def ci(samples, point):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def fmt(c, nd=1):
    return f"${c['point']:.{nd}f}$\\,{{\\scriptsize$[{c['lo']:.{nd}f},{c['hi']:.{nd}f}]$}}"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="build outputs even if some arms have <5 seeds (drafting only)")
    a = ap.parse_args()
    inv = inventory()
    print("inventory (arm, step): seeds")
    for (arm, step), seeds in sorted(inv.items()):
        print(f"  {arm:18s} {step:6d}: {sorted(seeds)}")
    if a.status:
        return
    have = {arm: sorted(inv.get((arm, FINAL), {})) for arm, *_ in ARMS}
    if not all(have.values()):
        sys.exit(f"missing step-{FINAL} evals for: {[k for k, v in have.items() if not v]}")
    if not a.allow_partial and any(len(v) < 5 for v in have.values()):
        sys.exit(f"not all arms have 5 seeds yet: {have} (use --allow-partial to draft)")

    runs = {arm: [load(inv[(arm, FINAL)][s]) for s in have[arm]] for arm, *_ in ARMS}
    lams = runs[ARMS[0][0]][0]["lams"]
    for arm in runs:
        assert all(np.allclose(r["lams"], lams) for r in runs[arm]), "lambda grids differ"
        # greedy prompts must not depend on the task-LM seed
        base = [" ".join(p) for row in runs[arm][0]["prompts"] for p in row]
        for r in runs[arm][1:]:
            assert base == [" ".join(p) for row in r["prompts"] for p in row], \
                f"{arm}: prompts differ across seeds (policy decoding is not greedy?)"
    avg = {arm: seed_avg(runs[arm]) for arm in runs}
    n_src = avg[ARMS[0][0]]["content"].shape[1]
    tau = 100 * lams

    point = {arm: metrics(avg[arm]) for arm in avg}
    rng = np.random.default_rng(0)
    boot = defaultdict(lambda: defaultdict(list))
    for _ in range(B):
        idx = rng.integers(0, n_src, n_src)
        for arm in avg:
            m = metrics(avg[arm], idx)
            for k in ("reward", "content", "sentiment", "hv", "met", "feasible"):
                boot[arm][k].append(m[k])
            boot[arm]["ec"].append(m["ec"]); boot[arm]["es"].append(m["es"]); boot[arm]["ef"].append(m["ef"])

    summary = {"n_sentences": int(n_src), "lambdas": lams.tolist(), "seeds": have, "arms": {}}
    for arm in avg:
        p, bs = point[arm], boot[arm]
        dp = distinct_prompts(runs[arm][0])
        # Monte-Carlo noise floor: seed-to-seed SD of the full-test-set reward
        seed_rewards = [r["score"].mean() for r in runs[arm]]
        summary["arms"][arm] = {
            **{k: ci(bs[k], p[k]) for k in ("reward", "content", "sentiment", "hv", "met", "feasible")},
            "frontier": {"tau": tau.tolist(), "content": p["ec"].tolist(), "sentiment": p["es"].tolist(),
                         "feasible": p["ef"].tolist(),
                         "content_lo": np.percentile(bs["ec"], 2.5, 0).tolist(),
                         "content_hi": np.percentile(bs["ec"], 97.5, 0).tolist(),
                         "sentiment_lo": np.percentile(bs["es"], 2.5, 0).tolist(),
                         "sentiment_hi": np.percentile(bs["es"], 97.5, 0).tolist(),
                         "feasible_lo": np.percentile(bs["ef"], 2.5, 0).tolist(),
                         "feasible_hi": np.percentile(bs["ef"], 97.5, 0).tolist()},
            "spearman_tau_content": spearman(tau, p["ec"]),
            "monotone_violations": int((np.diff(p["ec"]) < 0).sum()),
            "distinct_prompts": dp, "distinct_prompts_median": float(np.median(dp)),
            "seed_rewards": seed_rewards,
            "seed_sd": float(np.std(seed_rewards, ddof=1)) if len(seed_rewards) > 1 else None,
            "example_prompts": {f"{t:g}": detok(runs[arm][0]["prompts"][i][0]) for i, t in enumerate(tau)},
        }
    # paired differences against the best GRPO arm and within families
    best_grpo = max(("grpo_ent", "grpo_baseref"), key=lambda k: point[k]["reward"])
    best_rr = max(("rrebel_l1_std", "rrebel_huber_std"), key=lambda k: point[k]["reward"])
    pairs = [(best_rr, best_grpo), ("rrebel_l1_std", "rrebel_huber_std"), ("grpo_ent", "grpo_baseref"),
             ("rrebel_huber_std", best_grpo), ("rrebel_l1_std", best_grpo)]
    pairs = list(dict.fromkeys(pairs))
    summary["pairwise"] = []
    for x, y in pairs:
        row = {"a": x, "b": y}
        for k in ("reward", "content", "sentiment", "hv", "feasible"):
            d = np.array(boot[x][k]) - np.array(boot[y][k])
            row[k] = ci(d, point[x][k] - point[y][k])
            row[k]["p_le_0"] = float((d <= 0).mean())
        # per-lambda sentiment gap: where on the curve does the gap live?
        dd = np.array(boot[x]["es"]) - np.array(boot[y]["es"])
        row["sentiment_gap_by_tau"] = {"point": (point[x]["es"] - point[y]["es"]).tolist(),
                                       "lo": np.percentile(dd, 2.5, 0).tolist(),
                                       "hi": np.percentile(dd, 97.5, 0).tolist()}
        summary["pairwise"].append(row)
    summary["best_grpo"], summary["best_rrebel"] = best_grpo, best_rr

    # frontier dominance of the best R-REBEL arm over every other arm, over the
    # content range the other arm actually covers
    fr = {a: (np.array(summary["arms"][a]["frontier"]["content"]),
              np.array(summary["arms"][a]["frontier"]["sentiment"])) for a in summary["arms"]}
    Hb = upper_hull(*fr[best_rr])
    dom = {}
    for other in summary["arms"]:
        if other == best_rr:
            continue
        c, s_ = fr[other]
        grid = np.linspace(c.min(), c.max(), 500)
        Ho = upper_hull(c, s_)
        mix = np.array([attain_mix(Hb, x) - attain_mix(Ho, x) for x in grid])
        stp = np.array([attain_step(*fr[best_rr], x) - attain_step(c, s_, x) for x in grid])
        dom[other] = {"mix_min": float(mix.min()), "mix_min_at": float(grid[mix.argmin()]),
                      "mix_frac": float((mix >= -1e-9).mean()), "step_frac": float((stp >= -1e-9).mean()),
                      "step_min": float(stp.min()), "step_min_at": float(grid[stp.argmin()])}
    cb = np.sort(fr[best_rr][0]); k = int(np.argmax(np.diff(cb)))
    summary["dominance"] = {"of": best_rr, "vs": dom, "largest_content_gap": [float(cb[k]), float(cb[k + 1])]}
    for a in summary["arms"]:
        H = upper_hull(*fr[a]); xs = np.linspace(0, 100, 2001)
        ys = [max(0.0, attain_mix(H, x)) if np.isfinite(attain_mix(H, x)) else 0.0 for x in xs]
        summary["arms"][a]["hv_mix"] = float(np.trapezoid(ys, xs) / 100.0)

    # learning curves on the test split (seed 0 only, matched steps)
    steps = sorted({s for (arm, s) in inv if all((a2, s) in inv and 0 in inv[(a2, s)] for a2, *_ in ARMS)})
    lc = {arm: {} for arm, *_ in ARMS}
    for s in steps:
        for arm, *_ in ARMS:
            m = metrics(seed_avg([load(inv[(arm, s)][0])]))
            lc[arm][s] = {"reward": float(m["reward"]), "hv": float(m["hv"]),
                          "content": float(m["content"]), "sentiment": float(m["sentiment"])}
    summary["learning_curve_seed0"] = lc

    # Early stopping a practitioner could actually run: pick each arm's
    # checkpoint by its 10-sentence DEV score (logged during training every 500
    # steps), among steps that also have a checkpoint and a test eval, and
    # report that checkpoint's TEST metrics. Selection never sees test data.
    dev = {}
    for arm, *_ in ARMS:
        scores = {}
        for f in glob.glob(os.path.join(ROOT, f"results/v4/v4_{arm}_seed0/eval/outputs.step.*.json")):
            st = int(re.search(r"step\.(\d+)", f).group(1))
            try:
                scores[st] = float(np.mean(json.load(open(f))["mean_scores"]))
            except Exception:
                pass
        cand = [st for st in lc[arm] if st in scores]
        if not cand:
            continue
        best = max(cand, key=lambda st: scores[st])
        dev[arm] = {"step": int(best), "dev_score": scores[best],
                    "dev_curve": {int(k): v for k, v in sorted(scores.items())},
                    "test": lc[arm][best], "candidates": sorted(int(c) for c in cand)}
    summary["dev_selected"] = dev

    json.dump(summary, open(os.path.join(ROOT, "results", "v4", "full_eval_summary.json"), "w"), indent=1)
    print_summary(summary)
    figures(summary, steps)
    tables(summary)
    macros(summary)


def print_summary(S):
    print(f"\n=== step {FINAL}, {S['n_sentences']} test sentences, seeds {S['seeds']} ===")
    for arm, *_ in ARMS:
        m = S["arms"][arm]
        print(f"{arm:18s} reward {m['reward']['point']:6.2f} [{m['reward']['lo']:.2f},{m['reward']['hi']:.2f}]"
              f"  HV {m['hv']['point']:5.2f}  content {m['content']['point']:5.2f}  sent {m['sentiment']['point']:5.2f}"
              f"  met {m['met']['point']:.0f}/{len(S['lambdas'])}  feas {m['feasible']['point']:.3f}"
              f"  rho {m['spearman_tau_content']:.3f}  viol {m['monotone_violations']}"
              f"  distinct(med) {m['distinct_prompts_median']:.0f}  seedSD {m['seed_sd']}")
    D = S["dominance"]
    print(f"\ndominance of {D['of']}: largest content gap in its frontier {D['largest_content_gap']}")
    for o, v in D["vs"].items():
        print(f"  vs {o:18s} mixing: min {v['mix_min']:+.2f} @c={v['mix_min_at']:.1f}, >= at {v['mix_frac']:.1%}"
              f" | no mixing: >= at {v['step_frac']:.1%}, min {v['step_min']:+.2f} @c={v['step_min_at']:.1f}")
    print("  HV with mixing:", {a: round(S['arms'][a]['hv_mix'], 2) for a in S['arms']})
    print("\npaired differences (a - b), 95% CI, P(diff<=0):")
    for r in S["pairwise"]:
        print(f"  {r['a']:18s} - {r['b']:18s}: reward {r['reward']['point']:+.2f} [{r['reward']['lo']:+.2f},{r['reward']['hi']:+.2f}] p={r['reward']['p_le_0']:.4f}"
              f" | HV {r['hv']['point']:+.2f} [{r['hv']['lo']:+.2f},{r['hv']['hi']:+.2f}]"
              f" | sent {r['sentiment']['point']:+.2f} | content {r['content']['point']:+.2f}")
    print("\ntest-split learning curve (seed 0):")
    for arm, *_ in ARMS:
        print(f"  {arm:18s} " + "  ".join(f"{s}:{v['reward']:.2f}/HV{v['hv']:.1f}" for s, v in sorted(S['learning_curve_seed0'][arm].items())))


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def figures(S, steps):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})

    # ---- frontier ----------------------------------------------------------
    # drawn at print size (0.72 textwidth ~ 4.7in) so fonts are not scaled down
    fig, ax = plt.subplots(figsize=(4.8, 3.75))
    for arm, label, color, ls, mk in ARMS:
        f = S["arms"][arm]["frontier"]
        c, s = np.array(f["content"]), np.array(f["sentiment"])
        ax.errorbar(c, s, xerr=[c - f["content_lo"], np.array(f["content_hi"]) - c],
                    yerr=[s - f["sentiment_lo"], np.array(f["sentiment_hi"]) - s],
                    fmt="none", ecolor=color, elinewidth=0.7, alpha=0.55, zorder=2)
        ax.plot(c, s, ls=ls, color=color, lw=1.8, marker=mk, ms=5, mec="white", mew=0.7,
                label=label, zorder=3)
    # tau labels on the best arm only (selective direct labels)
    # label multiples of 10, skipping any point too close to one already
    # labelled (plateaus stack several tau on nearly the same point)
    ref = S["best_rrebel"]
    f = S["arms"][ref]["frontier"]
    placed = []
    for t, c, s in zip(f["tau"], f["content"], f["sentiment"]):
        if int(round(t)) % 10 or any((c - c0) ** 2 + (s - s0) ** 2 < 5.0 ** 2 for c0, s0 in placed):
            continue
        placed.append((c, s))
        ax.annotate(f"$\\tau$={t:g}", (c, s), xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, color=INK2)
    ax.set_xlabel("expected content score  $\\mathbb{E}[c]$")
    ax.set_ylabel("expected sentiment score  $\\mathbb{E}[s]$")
    ax.set_xlim(15, 100); ax.set_ylim(0, 100)
    _style(ax); ax.legend(frameon=False, fontsize=7.5, loc="upper right", handlelength=2.6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "frontier.pdf")); fig.savefig(os.path.join(FIG, "frontier.png"), dpi=120)
    plt.close(fig)

    # ---- control: three small multiples vs tau ----------------------------
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.7))  # print size: full text width
    specs = [("content", "expected content  $\\mathbb{E}[c]$"),
             ("sentiment", "expected sentiment  $\\mathbb{E}[s]$"),
             ("feasible", "$P(c\\geq\\tau)$ per sample")]
    for ax, (k, yl) in zip(axs, specs):
        for arm, label, color, ls, mk in ARMS:
            f = S["arms"][arm]["frontier"]
            t = np.array(f["tau"]); y = np.array(f[k])
            ax.fill_between(t, f[f"{k}_lo"], f[f"{k}_hi"], color=color, alpha=0.15, lw=0)
            ax.plot(t, y, ls=ls, color=color, lw=1.4, marker=mk, ms=3.2, mec="white", mew=0.5, label=label)
        if k == "content":
            ax.plot([0, 95], [0, 95], color=INK2, lw=0.9, ls=":", label="floor  $\\mathbb{E}[c]=\\tau$")
        ax.set_xlabel("floor $\\tau = 100\\lambda$"); ax.set_ylabel(yl)
        _style(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=6.5, handlelength=2.4,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(FIG, "control.pdf")); fig.savefig(os.path.join(FIG, "control.png"), dpi=120)
    plt.close(fig)

    # ---- learning curves on test -----------------------------------------
    if len(steps) >= 2:
        fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.75))  # print size: 0.9 text width
        for ax, (k, yl) in zip(axs, [("reward", "mean constrained reward"), ("hv", "hypervolume (% of box)")]):
            for arm, label, color, ls, mk in ARMS:
                lc = S["learning_curve_seed0"][arm]
                xs = sorted(int(s) for s in lc)
                ax.plot(xs, [lc[s][k] for s in xs], ls=ls, color=color, lw=1.8, marker=mk, ms=5, mec="white", label=label)
            ax.set_xlabel("training step"); ax.set_ylabel(yl); _style(ax)
        # ring the checkpoint each arm's DEV score would select (early stopping)
        for arm, label, color, ls, mk in ARMS:
            d = S.get("dev_selected", {}).get(arm)
            if d:
                axs[0].plot([d["step"]], [d["test"]["reward"]], "o", ms=11, mfc="none", mec=color, mew=1.2, zorder=1)
        axs[0].plot([], [], "o", ms=8, mfc="none", mec=INK2, mew=1.1, label="dev-selected checkpoint")
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.5, handlelength=2.4,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(os.path.join(FIG, "test_learning.pdf")); fig.savefig(os.path.join(FIG, "test_learning.png"), dpi=120); plt.close(fig)


def tables(S):
    n = len(S["lambdas"])
    rows = []
    for arm, label, *_ in ARMS:
        m = S["arms"][arm]
        rows.append(f"{label} & {fmt(m['reward'])} & {fmt(m['hv'])} & {fmt(m['content'])} & "
                    f"{fmt(m['sentiment'])} & ${m['met']['point']:.0f}/{n}$ & "
                    f"${m['feasible']['point']:.2f}$ & ${m['distinct_prompts_median']:.0f}$ \\\\")
    with open(os.path.join(TAB, "full_results.tex"), "w") as fh:
        fh.write("% generated by analysis/full_eval.py -- do not edit by hand\n")
        fh.write("\\begin{tabular}{lccccccc}\n\\toprule\n")
        fh.write("Arm & Reward & Hypervolume & Content & Sentiment & Floors met & "
                 "$P(c\\geq\\tau)$ & Prompts/$\\tau$ \\\\\n\\midrule\n")
        fh.write("\n".join(rows[:2]) + "\n\\midrule\n" + "\n".join(rows[2:]) + "\n\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(TAB, "pairwise.tex"), "w") as fh:
        fh.write("% generated by analysis/full_eval.py -- do not edit by hand\n")
        fh.write("\\begin{tabular}{llccc}\n\\toprule\n$a$ & $b$ & $\\Delta$ reward & $\\Delta$ hypervolume & $\\Delta$ sentiment \\\\\n\\midrule\n")
        for r in S["pairwise"]:
            f2 = lambda c: f"${c['point']:+.2f}$\\,{{\\scriptsize$[{c['lo']:+.2f},{c['hi']:+.2f}]$}}"
            fh.write(f"{LABEL[r['a']]} & {LABEL[r['b']]} & {f2(r['reward'])} & {f2(r['hv'])} & {f2(r['sentiment'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    taus = ["0", "30", "50", "70", "90"]
    short = {"rrebel_l1_std": r"R-REBEL $\ell_1$-std", "rrebel_huber_std": "R-REBEL Huber-std",
             "grpo_ent": "GRPO + entropy", "grpo_baseref": "GRPO, base-LM ref."}
    with open(os.path.join(TAB, "prompts.tex"), "w") as fh:
        fh.write("% generated by analysis/full_eval.py -- do not edit by hand\n")
        fh.write("\\begin{tabular}{r" + ">{\\raggedright\\arraybackslash\\ttfamily\\footnotesize}p{0.205\\textwidth}" * len(ARMS) + "}\n\\toprule\n")
        fh.write("$\\tau$ & " + " & ".join("\\normalfont\\small " + short[a] for a, *_ in ARMS) + " \\\\\n\\midrule\n")
        for t in taus:
            cells = [tex_escape(S["arms"][a]["example_prompts"][t]) for a, *_ in ARMS]
            fh.write(f"${t}$ & " + " & ".join(cells) + " \\\\[2pt]\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nwrote tables -> {TAB}/{{full_results,pairwise,prompts}}.tex")


MACRO_ARM = {"rrebel_l1_std": "Lone", "rrebel_huber_std": "Huber", "grpo_ent": "Gent", "grpo_baseref": "Gbase"}
MACRO_STEP = {1500: "Sa", 3000: "Sb", 4500: "Sc", 6000: "Sd", 7500: "Se", 9000: "Sf", 10500: "Sg", 12000: "Sh"}


def macros(S):
    """paper/tables/v4_numbers.tex: one \\newcommand per number the text quotes."""
    L = ["% generated by analysis/full_eval.py -- do not edit by hand"]
    def m(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    m("vNsent", S["n_sentences"]); m("vNlam", len(S["lambdas"]))
    m("vNseeds", min(len(v) for v in S["seeds"].values()))
    for arm, A in MACRO_ARM.items():
        a = S["arms"][arm]
        for k, K in (("reward", "Rew"), ("hv", "HV"), ("content", "Con"), ("sentiment", "Sen")):
            m(f"v{A}{K}", f"{a[k]['point']:.1f}"); m(f"v{A}{K}Lo", f"{a[k]['lo']:.1f}"); m(f"v{A}{K}Hi", f"{a[k]['hi']:.1f}")
        m(f"v{A}Met", f"{a['met']['point']:.0f}"); m(f"v{A}Feas", f"{a['feasible']['point']:.2f}")
        m(f"v{A}Prompts", f"{a['distinct_prompts_median']:.0f}")
        m(f"v{A}PromptsMax", f"{max(a['distinct_prompts'])}")
        m(f"v{A}Rho", f"{a['spearman_tau_content']:.2f}")
        m(f"v{A}SeedSD", f"{a['seed_sd']:.2f}" if a['seed_sd'] is not None else "--")
        f = a["frontier"]
        m(f"v{A}ConMax", f"{max(f['content']):.1f}"); m(f"v{A}SenMax", f"{max(f['sentiment']):.1f}")
        for st, T in MACRO_STEP.items():
            lc = S["learning_curve_seed0"][arm].get(st) or S["learning_curve_seed0"][arm].get(str(st))
            if lc:
                m(f"v{A}Rew{T}", f"{lc['reward']:.1f}"); m(f"v{A}HV{T}", f"{lc['hv']:.1f}")
        d = S.get("dev_selected", {}).get(arm)
        if d:
            m(f"v{A}DevStep", f"{d['step']:,}".replace(",", "{,}")); m(f"v{A}DevRew", f"{d['test']['reward']:.1f}")
            m(f"v{A}DevHV", f"{d['test']['hv']:.1f}")
    for a, A in MACRO_ARM.items():
        m(f"v{A}HVmix", f"{S['arms'][a]['hv_mix']:.1f}")
    # learning-curve derived quantities (robust to which checkpoints exist)
    LC = {a: {int(k): v for k, v in S["learning_curve_seed0"][a].items()} for a in MACRO_ARM}
    common = sorted(set.intersection(*[set(v) for v in LC.values()]))
    m("vLcNsteps", len(common))
    m("vLcFirst", f"{common[0]:,}".replace(",", "{,}")); m("vLcLast", f"{common[-1]:,}".replace(",", "{,}"))
    gHG = [LC["rrebel_huber_std"][t]["reward"] - LC["grpo_ent"][t]["reward"] for t in common]
    m("vLcHuberGentMin", f"{min(gHG):.1f}"); m("vLcHuberGentMax", f"{max(gHG):.1f}")
    gLG = [LC["rrebel_l1_std"][t]["reward"] - LC["grpo_ent"][t]["reward"] for t in common]
    m("vLcLoneGentPosSteps", sum(g > 0 for g in gLG))
    pk = max(common, key=lambda t: LC["rrebel_l1_std"][t]["reward"])
    m("vLonePeakStep", f"{pk:,}".replace(",", "{,}")); m("vLonePeakRew", f"{LC['rrebel_l1_std'][pk]['reward']:.1f}")
    m("vLoneDrop", f"{LC['rrebel_l1_std'][pk]['reward'] - LC['rrebel_l1_std'][common[-1]]['reward']:.1f}")
    hvs = [LC["rrebel_l1_std"][t]["hv"] for t in common]
    m("vLoneHVmin", f"{min(hvs):.1f}"); m("vLoneHVmax", f"{max(hvs):.1f}")
    for a, A in MACRO_ARM.items():
        r = [LC[a][t]["reward"] for t in common]
        m(f"v{A}LcMin", f"{min(r):.1f}"); m(f"v{A}LcMax", f"{max(r):.1f}")
        m(f"v{A}HVfirst", f"{LC[a][common[0]]['hv']:.1f}")
    dsel = S.get("dev_selected", {})
    if dsel:
        rr = min(dsel[a]["test"]["reward"] for a in ("rrebel_l1_std", "rrebel_huber_std"))
        gg = max(dsel[a]["test"]["reward"] for a in ("grpo_ent", "grpo_baseref"))
        m("vDevFamilyGap", f"{rr - gg:.1f}")
    D = S["dominance"]
    for other, v in D["vs"].items():
        O = MACRO_ARM[other]
        m(f"vDom{O}MixMin", f"{v['mix_min']:+.1f}"); m(f"vDom{O}MixFrac", f"{100 * v['mix_frac']:.0f}")
        m(f"vDom{O}StepFrac", f"{100 * v['step_frac']:.0f}")
    m("vDomGapLo", f"{D['largest_content_gap'][0]:.0f}"); m("vDomGapHi", f"{D['largest_content_gap'][1]:.0f}")
    for r in S["pairwise"]:
        n = f"vGap{MACRO_ARM[r['a']]}{MACRO_ARM[r['b']]}"
        for k, K in (("reward", "Rew"), ("hv", "HV"), ("sentiment", "Sen"), ("content", "Con")):
            m(f"{n}{K}", f"{r[k]['point']:.1f}"); m(f"{n}{K}Lo", f"{r[k]['lo']:.1f}"); m(f"{n}{K}Hi", f"{r[k]['hi']:.1f}")
    open(os.path.join(TAB, "v4_numbers.tex"), "w").write("\n".join(L) + "\n")
    print(f"wrote {len(L) - 1} macros -> {TAB}/v4_numbers.tex")


if __name__ == "__main__":
    main()
