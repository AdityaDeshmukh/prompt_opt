"""Paper tables + figures for the GRPO vs R-REBEL comparison.

Reads results/v3_wandb_export.json (produced by analysis/recover_from_wandb.py)
and, when present, the per-lambda v4 test evals under results/v4/*/test/.

WHAT IS AND IS NOT A FRONTIER
-----------------------------
The v3 campaign's per-lambda eval arrays were destroyed with the /scratch purge
(see recover_from_wandb.py). wandb kept only the lambda-AVERAGED content and
style, so from v3 each run yields ONE (content, sentiment) operating point, not
a curve. Those points are plotted as operating points and labelled as such.
A true lambda-frontier requires the v4 rerun (slurm/train_v4.slurm +
eval_v4.slurm); this script draws it automatically once those JSONs exist, and
otherwise says so on the figure rather than implying a frontier exists.

Usage:
    python analysis/paper_results.py              # tables to stdout + figures
    python analysis/paper_results.py --tex        # also emit LaTeX tables
"""
import argparse
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXPORT = os.path.join(ROOT, "results", "v3_wandb_export.json")
V4_GLOB = os.path.join(ROOT, "results", "v4", "*", "test", "output*.json")
FIGDIR = os.path.join(ROOT, "paper", "figures")
TEXDIR = os.path.join(ROOT, "paper", "tables")

# vllm_seed=null makes task-LM sampling nondeterministic; a repeated step-500
# eval moved 34.1 -> 33.0. Differences below this are not meaningful.
NOISE = 1.0

ARM_ORDER = ["rrebel_l1_std", "rrebel_huber_std", "rrebel_l1_ent",
             "grpo", "grpo_ent"]
LABEL = {
    "rrebel_l1_std":    r"R-REBEL ($\ell_1$-std)",
    "rrebel_huber_std": r"R-REBEL (Huber-std)",
    "rrebel_l1_ent":    r"R-REBEL ($\ell_1$-ent)",
    "grpo":             "GRPO",
    "grpo_ent":         "GRPO + entropy",
    "grpo_rollingref":  "GRPO (rolling anchor, buggy)",
}
PLAIN = {k: (v.replace(r"$\ell_1$", "l1").replace("-std", "-std")
             .replace("R-REBEL", "R-REBEL")) for k, v in LABEL.items()}

# Categorical slots 1-3 of the validated reference palette. palette.md
# certifies the first THREE slots for all-pairs use (scatter); family identity
# carries the hue and variant identity carries marker/linestyle, so no chart
# here needs a 4th hue.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e3e2de"
STYLE = {  # arm -> (color, linestyle, marker)
    "rrebel_l1_std":    (BLUE,   "-",  "o"),
    "rrebel_huber_std": (BLUE,   "--", "s"),
    "rrebel_l1_ent":    (BLUE,   ":",  "^"),
    "grpo":             (ORANGE, "-",  "D"),
    "grpo_ent":         (ORANGE, "--", "v"),
    "grpo_rollingref":  (MUTED,  "-.", "x"),
}


def load():
    with open(EXPORT) as f:
        return json.load(f)


def at_step(runs, step):
    """[(run, metrics)] for runs having an eval at `step`."""
    out = []
    for name, d in sorted(runs.items()):
        m = d["by_step"].get(str(step))
        if m:
            out.append((name, m))
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def table_at(data, step, arms=ARM_ORDER):
    rows = []
    for arm in arms:
        got = at_step(data["arms"][arm], step)
        if not got:
            continue
        rows.append({
            "arm": arm,
            "n": len(got),
            "score": mean([m["score"] for m in (g[1] for g in got)]),
            "content": mean([m["content"] for m in (g[1] for g in got)]),
            "style": mean([m["style"] for m in (g[1] for g in got)]),
            "seeds": [m["score"] for m in (g[1] for g in got)],
        })
    rows.sort(key=lambda r: -r["score"])
    return rows


def print_table(rows, title, total_seeds=3):
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"  {'arm':22s} {'score':>7} {'content':>8} {'sentiment':>10} "
          f"{'n':>4}   per-seed score")
    print("  " + "-" * 74)
    for r in rows:
        flag = "" if r["n"] == total_seeds else "  <- INCOMPLETE"
        print(f"  {PLAIN[r['arm']]:22s} {r['score']:7.2f} {r['content']:8.2f} "
              f"{r['style']:10.2f} {r['n']:>2}/{total_seeds}   "
              + " ".join(f"{s:.1f}" for s in sorted(r["seeds"], reverse=True))
              + flag)
    if len(rows) > 1:
        print("\n  pairwise gaps vs the top arm:")
        top = rows[0]
        for r in rows[1:]:
            gap = top["score"] - r["score"]
            verdict = "within noise -- report as TIED" if gap < NOISE else "real"
            print(f"    {PLAIN[top['arm']]} - {PLAIN[r['arm']]:22s} = "
                  f"{gap:5.2f}  ({verdict})")
    print()


def latex_table(rows, step, total_seeds=3):
    lines = [
        r"\begin{tabular}{lrrrc}", r"\toprule",
        r"Algorithm & Reward & Content & Sentiment & Seeds \\", r"\midrule",
    ]
    for r in rows:
        lines.append(f"{LABEL[r['arm']]} & {r['score']:.2f} & {r['content']:.2f} "
                     f"& {r['style']:.2f} & {r['n']}/{total_seeds} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def trajectory(runs, key="score"):
    """step -> mean over seeds, plus (min, max) band."""
    acc = defaultdict(list)
    for d in runs.values():
        for s, m in d["by_step"].items():
            acc[int(s)].append(m[key])
    steps = sorted(acc)
    return (steps,
            [mean(acc[s]) for s in steps],
            [min(acc[s]) for s in steps],
            [max(acc[s]) for s in steps],
            [len(acc[s]) for s in steps])


def load_v4_frontiers():
    """arm -> (lambdas, contents, styles) from the v4 per-lambda test evals."""
    out = {}
    for path in sorted(glob.glob(V4_GLOB)):
        run = os.path.basename(os.path.dirname(os.path.dirname(path)))
        arm = run.replace("v4_", "").rsplit("_seed", 1)[0]
        with open(path) as f:
            d = json.load(f)
        acc = defaultdict(lambda: ([], []))
        for lam, mc, ms in zip(d["lmbdas"], d["mean_contents"], d["mean_styles"]):
            k = round(float(lam[0] if isinstance(lam, list) else lam), 3)
            acc[k][0].append(mc)
            acc[k][1].append(ms)
        lams = sorted(acc)
        out[arm] = (lams,
                    [mean(acc[l][0]) for l in lams],
                    [mean(acc[l][1]) for l in lams])
    return out


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.8, "figure.facecolor": "white",
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    return plt


def fig_tradeoff(data, plt, balanced_step):
    """Content vs sentiment. Real lambda-frontiers if v4 exists, else the
    lambda-averaged operating points -- never a fake curve."""
    v4 = load_v4_frontiers()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))

    if v4:
        for arm in ARM_ORDER:
            if arm not in v4:
                continue
            lams, c, s = v4[arm]
            col, ls, mk = STYLE[arm]
            ax.plot(c, s, ls, color=col, marker=mk, markersize=4.5,
                    markerfacecolor="white", markeredgecolor=col,
                    markeredgewidth=1.4, label=LABEL[arm])
        ax.set_title("Content-sentiment tradeoff curve\n"
                     r"(step 12000, $\lambda$ swept $0 \to 0.9$)")
    else:
        for arm in ARM_ORDER:
            got = at_step(data["arms"][arm], balanced_step)
            if not got:
                continue
            col, _, mk = STYLE[arm]
            cs = [m["content"] for _, m in got]
            ss = [m["style"] for _, m in got]
            # hollow = individual seeds, filled = the arm's mean
            ax.scatter(cs, ss, s=42, marker=mk, facecolor="white",
                       edgecolor=col, linewidth=1.4, zorder=3)
            ax.scatter([mean(cs)], [mean(ss)], s=110, marker=mk, color=col,
                       edgecolor="white", linewidth=1.6, zorder=4,
                       label=LABEL[arm])
        ax.set_title(f"Operating points at step {balanced_step}\n"
                     r"(mean over the $\lambda$ grid, not a frontier)")
        # Both axes are better-is-higher, so up-and-right is unambiguously
        # better on BOTH objectives -- the only claim this scatter supports.
        # diagonal, so it reads as "better on BOTH axes" rather than "better on
        # content" -- a horizontal arrow here would misstate the claim
        ax.annotate("", xy=(0.99, 0.26), xycoords="axes fraction",
                    xytext=(0.80, 0.04), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
        ax.text(0.795, 0.035, "better on\nboth", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=MUTED,
                linespacing=1.25)

    ax.set_xlabel("Content score")
    ax.set_ylabel("Sentiment score")
    # Legend below the axes: with a zoomed scatter there is no empty corner
    # inside the frame, and a legend over the marks hides data.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
              handletextpad=0.4, columnspacing=1.2, borderaxespad=0)
    out = os.path.join(FIGDIR, "tradeoff.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}" + ("  [true lambda-frontier from v4]" if v4 else
                              "  [lambda-averaged operating points; v4 pending]"))
    return bool(v4)


def fig_learning(data, plt):
    """Reward vs step, mean over seeds with a min-max band."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for arm in ARM_ORDER:
        steps, mu, lo, hi, _ = trajectory(data["arms"][arm])
        col, ls, _ = STYLE[arm]
        ax.fill_between(steps, lo, hi, color=col, alpha=0.10, linewidth=0)
        ax.plot(steps, mu, ls, color=col, label=LABEL[arm])
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Reward (mean over the $\lambda$ grid)")
    ax.set_title("Learning curves (mean of 3 seeds, band = min-max)")
    ax.legend(loc="lower right", ncol=1)
    out = os.path.join(FIGDIR, "learning_curves.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_components(data, plt):
    """Content and sentiment separately -- shows HOW the reward is earned.

    No min-max bands here: five overlapping bands across two panels is
    unreadable, and the seed spread is already shown in fig_learning. These
    panels exist to make one point -- the arms are close on content and
    separate on sentiment -- so they carry means only, on a shared y-scale so
    the two panels' spreads are directly comparable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True, sharey=True)
    for key, ax, name in (("content", axes[0], "Content score"),
                          ("style", axes[1], "Sentiment score")):
        for arm in ARM_ORDER:
            steps, mu, _, _, _ = trajectory(data["arms"][arm], key)
            col, ls, _ = STYLE[arm]
            ax.plot(steps, mu, ls, color=col, label=LABEL[arm])
        ax.set_xlabel("Training step")
        ax.set_title(name)
    axes[0].set_ylabel("Score (mean of 3 seeds)")
    # legend goes in the LEFT panel: with a shared y-scale its lower band is
    # empty, whereas the right panel's lower band is where GRPO's line sits
    axes[0].legend(loc="lower left", fontsize=7.5, borderaxespad=0.6)
    out = os.path.join(FIGDIR, "components.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_baseline_debug(data, plt):
    """The reviewer-defense figure: the buggy baseline, the fixed baseline, the
    tuned baseline, and the R-REBEL band it still does not reach."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    # R-REBEL envelope across all three variants and all seeds
    acc = defaultdict(list)
    for arm in ("rrebel_l1_std", "rrebel_huber_std", "rrebel_l1_ent"):
        for d in data["arms"][arm].values():
            for s, m in d["by_step"].items():
                acc[int(s)].append(m["score"])
    steps = sorted(acc)
    ax.fill_between(steps, [min(acc[s]) for s in steps],
                    [max(acc[s]) for s in steps], color=BLUE, alpha=0.16,
                    linewidth=0, label="R-REBEL (all variants, all seeds)")

    for arm in ("grpo_rollingref", "grpo", "grpo_ent"):
        runs = data["arms"].get(arm)
        if not runs:
            continue
        st, mu, _, _, _ = trajectory(runs)
        col, ls, _ = STYLE[arm]
        ax.plot(st, mu, ls, color=col, label=LABEL[arm])

    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Reward (mean over the $\lambda$ grid)")
    ax.set_title("The baseline was debugged and tuned, and still trails")
    ax.legend(loc="lower right")
    out = os.path.join(FIGDIR, "baseline_debug.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", action="store_true", help="emit LaTeX tables")
    args = ap.parse_args()

    data = load()
    bal = data["meta"].get("balanced_matched_step") or 11500
    short = data["meta"].get("runs_short_of_max") or {}

    print()
    rows_bal = table_at(data, bal)
    print_table(rows_bal, f"HEADLINE: matched step {bal} "
                          f"(every arm has all 3 seeds here)")
    rows_max = table_at(data, 12000)
    print_table(rows_max, "AT STEP 12000 (seed-unbalanced -- see note)")
    if short:
        print("  NOTE: these runs have no step-12000 eval. Their last cycle ran")
        print("  under the trainer's wandb-init fallback, so its evals went to")
        print("  disk only and the disk copies were purged with /scratch:")
        for n, s in sorted(short.items()):
            print(f"    - {n}: short by {s} steps")
        print(f"  Hence step {bal} is the balanced comparison point.\n")

    print("=" * 78)
    print("FAMILY SUMMARY")
    print("=" * 78)
    rr = [r for r in rows_bal if r["arm"].startswith("rrebel")]
    gp = [r for r in rows_bal if r["arm"].startswith("grpo")]
    if rr and gp:
        best_rr, best_gp = rr[0], gp[0]
        print(f"  best R-REBEL : {PLAIN[best_rr['arm']]:22s} {best_rr['score']:6.2f}")
        print(f"  best GRPO    : {PLAIN[best_gp['arm']]:22s} {best_gp['score']:6.2f}")
        print(f"  gap          : {best_rr['score'] - best_gp['score']:6.2f} points "
              f"({(best_rr['score'] - best_gp['score']) / NOISE:.0f}x the "
              f"{NOISE}-point noise floor)")
        print(f"  every R-REBEL seed beats every GRPO seed: "
              f"{min(min(r['seeds']) for r in rr) > max(max(r['seeds']) for r in gp)}")
    print()

    os.makedirs(FIGDIR, exist_ok=True)
    plt = setup_mpl()
    print("figures:")
    have_frontier = fig_tradeoff(data, plt, bal)
    fig_learning(data, plt)
    fig_components(data, plt)
    fig_baseline_debug(data, plt)
    if not have_frontier:
        print("\n  the tradeoff FRONTIER figure is still a placeholder: it needs")
        print("  the per-lambda v4 evals (slurm/train_v4.slurm -> eval_v4.slurm).")

    if args.tex:
        os.makedirs(TEXDIR, exist_ok=True)
        for rows, step, fn in ((rows_bal, bal, "main_results.tex"),
                               (rows_max, 12000, "results_12000.tex")):
            p = os.path.join(TEXDIR, fn)
            with open(p, "w") as f:
                f.write(latex_table(rows, step) + "\n")
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()
