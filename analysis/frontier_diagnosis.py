"""Why several floors tau land on one point of the tradeoff curve, and why a
point's expected content E[c] is not its tau.

Inputs: results/v4/frontier_probe/{arm}.{policy,menu}.npz (analysis/frontier_probe.py)
and the step-12000 test evals. Every prompt the evals emitted >=25 times over
5 seeds (48 prompts, >=98% of every arm's rows) has PER-SAMPLE content and
sentiment on all 500 test sentences, so any reward can be recomputed offline.

    python analysis/frontier_diagnosis.py          # prints + JSON + figures
"""
import os, sys, json, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import full_eval as fe  # noqa: E402

ROOT = fe.ROOT
PROBE = os.path.join(ROOT, "results", "v4", "frontier_probe")
OUT = os.path.join(ROOT, "results", "v4", "frontier_diagnosis")
ARM_KEYS = [a[0] for a in fe.ARMS]
EVAL_TAU = np.array([5.0 * i for i in range(20)])
FINE_TAU = np.arange(0, 100, 0.5)
PEN = 0.01            # the training reward's penalty slope


# ---------------------------------------------------------------- loading
def _tok():
    fe.detok(["a"])                                        # initializes fe._TOK
    return fe._TOK


def load_probe():
    per, pol = {}, {}
    for arm in ARM_KEYS:
        p = np.load(f"{PROBE}/{arm}.policy.npz", allow_pickle=True)
        m = np.load(f"{PROBE}/{arm}.menu.npz", allow_pickle=True)
        menu = [tuple(s.split("\x1f")) for s in p["menu"]]
        ids = p["fine_ids"]                                # [200, n_src, 5] token ids
        uniq, inv = np.unique(ids.reshape(-1, 5), axis=0, return_inverse=True)
        toks = [tuple(_tok().convert_ids_to_tokens(u.tolist())) for u in uniq]
        fine_prompts = [[toks[i] for i in row] for row in inv.reshape(ids.shape[:2])]
        pol[arm] = dict(menu=menu, lp3=p["menu_lp3"], fine_prompts=fine_prompts,
                        fine_lp3=p["fine_lp3"], fine_margin=p["fine_margin"],
                        fine_grid=p["fine_grid"], n_shared=int(p["n_shared"]))
        for s, C, S in zip(m["menu"], m["content"], m["style"]):
            per[tuple(s.split("\x1f"))] = (C.astype(np.float32), S.astype(np.float32))
    return per, pol


def reward(C, S, tau, pen=PEN):
    """Per-sentence mean of the per-sample constrained reward. C,S [n_src,N]."""
    return np.where(C >= tau, S, pen * (C - tau)).mean(-1)


def summarize_prompt(C, S, taus):
    return dict(ec=C.mean(), es=S.mean(),
                R=np.array([reward(C, S, t).mean() for t in taus]),
                P=np.array([(C >= t).mean() for t in taus]))


def main():
    os.makedirs(OUT, exist_ok=True)
    per, pol = load_probe()
    evals = {a: fe.load(fe.path(a, 12000, 0)) for a in ARM_KEYS}
    res = {"arms": {}}
    n_src = 500
    stats = {p: summarize_prompt(C, S, FINE_TAU) for p, (C, S) in per.items()}
    print(f"menu with per-sample data: {len(per)} prompts")

    for arm in ARM_KEYS:
        ev, P = evals[arm], pol[arm]
        r = res["arms"][arm] = {}
        prompts = [[tuple(x) for x in row] for row in ev["prompts"]]
        # -------- V1: the menu cross-evaluation reproduces the eval ----------
        cov = np.array([[p in per for p in row] for row in prompts])
        r["menu_coverage_rows"] = float(cov.mean())
        pred = {k: np.full((20, n_src), np.nan) for k in ("content", "style", "score", "feasible")}
        for a in range(20):
            for b in range(n_src):
                p = prompts[a][b]
                if p in per:
                    C, S = per[p][0][b], per[p][1][b]
                    pred["content"][a, b], pred["style"][a, b] = C.mean(), S.mean()
                    pred["score"][a, b] = reward(C[None], S[None], EVAL_TAU[a])[0]
                    pred["feasible"][a, b] = (C >= EVAL_TAU[a]).mean()
        r["v1_max_abs_diff_per_tau"] = {
            k: float(np.nanmax(np.abs(np.nanmean(pred[k] - ev[k], axis=1)) * (1 if k != "feasible" else 100)))
            for k in pred}
        # -------- V2: eval prompts are draws from the top-3-truncated policy --
        idx = {p: j for j, p in enumerate(P["menu"])}
        lp = np.array([[P["lp3"][idx[prompts[a][b]], a, b] if prompts[a][b] in idx else np.nan
                        for b in range(n_src)] for a in range(20)])
        r["v2_frac_rows_outside_top3_support"] = float(np.mean(np.isneginf(lp)))
        q = np.exp(P["lp3"])                              # [M, 20, n_src]
        r["menu_mass_mean"] = float(q.sum(0).mean())
        r["menu_mass_min"] = float(q.sum(0).min())
        # expected vs observed prompt shares per tau (total variation)
        tv = []
        for a in range(20):
            obs = collections.Counter(prompts[a][b] for b in range(n_src))
            exp_share = {p: q[j, a].mean() for j, p in enumerate(P["menu"])}
            keys = set(obs) | {p for p, v in exp_share.items() if v > 1e-4}
            tv.append(0.5 * sum(abs(obs.get(p, 0) / n_src - exp_share.get(p, 0.0)) for p in keys))
        r["v2_tv_expected_vs_observed_share"] = [round(float(x), 4) for x in tv]
        # -------- D6: flips between two independent top-3 draws ---------------
        agree_lo = (q ** 2).sum(0)
        agree_hi = agree_lo + (1 - q.sum(0)).clip(0) ** 2
        r["expected_flip_independent_draws"] = [float(1 - agree_hi.mean()), float(1 - agree_lo.mean())]
        # -------- D1: clustering = few prompts, piecewise-constant in lambda --
        amax = P["fine_prompts"]                           # [200][n_src] token tuples
        modal = [collections.Counter(row).most_common(1)[0] for row in amax]
        r["argmax_modal_share_min"] = float(min(n for _, n in modal) / n_src)
        r["argmax_modal_share_mean"] = float(np.mean([n for _, n in modal]) / n_src)
        modal_p = [p for p, _ in modal]
        switches = [float(P["fine_grid"][a]) for a in range(1, len(modal_p)) if modal_p[a] != modal_p[a - 1]]
        r["argmax_modal_switch_lambdas"] = switches
        r["argmax_distinct_modal_prompts"] = len(set(modal_p))
        # per-sentence argmax maps: how many switch points, and same as modal?
        sw = [sum(amax[a][b] != amax[a - 1][b] for a in range(1, len(amax))) for b in range(n_src)]
        r["argmax_switches_per_sentence_median"] = float(np.median(sw))
        r["argmax_frac_sentences_equal_modal_map"] = float(np.mean(
            [all(amax[a][b] == modal_p[a] for a in range(len(amax))) for b in range(n_src)]))
        r["argmax_min_margin_median"] = float(np.median(P["fine_margin"].min(-1)))
        r["top1_prob_top3_trunc_median"] = float(np.median(np.exp(P["fine_lp3"]).min(-1)))
        # effective number of prompts per eval tau (eval draws) and distinct locations
        eff = []
        for a in range(20):
            cnt = np.array(list(collections.Counter(prompts[a]).values())) / n_src
            eff.append(float(np.exp(-(cnt * np.log(cnt)).sum())))
        r["eff_prompts_per_tau"] = [round(x, 2) for x in eff]
        # -------- D2: the reward's own optimum over the arm's menu -----------
        used = [p for j, p in enumerate(P["menu"]) if q[j].mean(1).max() > 0.01 and p in per]
        r["menu_used"] = [dict(prompt=fe.detok(list(p)), ec=float(stats[p]["ec"]), es=float(stats[p]["es"]))
                          for p in used]
        Rm = np.array([stats[p]["R"] for p in used])       # [K, len(FINE_TAU)]
        best = Rm.argmax(0)
        r["opt_switch_taus"] = [float(FINE_TAU[i]) for i in range(1, len(best)) if best[i] != best[i - 1]]
        r["opt_n_distinct"] = int(len(set(best)))
        # policy's expected reward (mixture under q) vs the menu optimum, at eval taus
        R_pol, R_opt, ec_pol, ec_opt = [], [], [], []
        for a, t in enumerate(EVAL_TAU):
            fi = int(np.searchsorted(FINE_TAU, t))
            w = np.array([q[P["menu"].index(p), a].mean() for p in used]); w = w / w.sum()
            R_pol.append(float((w * Rm[:, fi]).sum())); R_opt.append(float(Rm[:, fi].max()))
            ec_pol.append(float((w * [stats[p]["ec"] for p in used]).sum()))
            ec_opt.append(float(stats[used[best[fi]]]["ec"]))
        r["regret_vs_own_menu"] = [round(o - p_, 3) for o, p_ in zip(R_opt, R_pol)]
        r["opt_ec_minus_tau"] = [round(e - t, 1) for e, t in zip(ec_opt, EVAL_TAU)]
        r["pol_ec_minus_tau"] = [round(e - t, 1) for e, t in zip(ec_pol, EVAL_TAU)]
        # -------- D7: are the per-sentence switch points meaningful? ----------
        # per-sentence reward of each used prompt on the fine lambda grid, split
        # into two independent halves of the N=50 samples: half A picks the
        # per-sentence best prompt, half B scores it (no winner's curse)
        fg = P["fine_grid"]
        Cm = np.stack([per[p][0] for p in used]); Sm = np.stack([per[p][1] for p in used])  # [K,n,N]
        h = Cm.shape[-1] // 2
        agree = {"policy_vs_sentence_opt": [], "policy_vs_global_opt": [], "global_vs_sentence_opt": []}
        gain = {"sentence_opt": [], "global_opt": [], "policy": []}
        for a in range(0, len(fg), 5):
            t = 100 * fg[a]
            RA = np.where(Cm[..., :h] >= t, Sm[..., :h], PEN * (Cm[..., :h] - t)).mean(-1)   # [K,n]
            RB = np.where(Cm[..., h:] >= t, Sm[..., h:], PEN * (Cm[..., h:] - t)).mean(-1)
            sent = RA.argmax(0); glob = int(RA.mean(1).argmax())
            poli = np.array([used.index(p) if p in used else -1 for p in amax[a]])
            ok = poli >= 0
            agree["policy_vs_sentence_opt"].append(float(np.mean(poli[ok] == sent[ok])))
            agree["policy_vs_global_opt"].append(float(np.mean(poli[ok] == glob)))
            agree["global_vs_sentence_opt"].append(float(np.mean(sent == glob)))
            idx_b = np.arange(n_src)
            gain["sentence_opt"].append(float(RB[sent, idx_b].mean()))
            gain["global_opt"].append(float(RB[glob].mean()))
            gain["policy"].append(float(RB[poli[ok], idx_b[ok]].mean()))
        r["d7_agreement_mean"] = {k: float(np.mean(v)) for k, v in agree.items()}
        r["d7_heldout_reward_mean"] = {k: float(np.mean(v)) for k, v in gain.items()}

        # -------- D5: true argmax decoding vs the top-3-sampled eval -----------
        am_eval = [amax[int(np.argmin(np.abs(P["fine_grid"] - t / 100)))] for t in EVAL_TAU]
        am_cov = np.mean([[p in per for p in row] for row in am_eval])
        r["argmax_menu_coverage"] = float(am_cov)
        if am_cov > 0.97:
            fr = {k: [] for k in ("ec", "es", "R")}
            for a, t in enumerate(EVAL_TAU):
                vals = [(per[p][0][b], per[p][1][b]) for b, p in enumerate(am_eval[a]) if p in per]
                fr["ec"].append(np.mean([C.mean() for C, _ in vals]))
                fr["es"].append(np.mean([S.mean() for _, S in vals]))
                fr["R"].append(np.mean([reward(C[None], S[None], t)[0] for C, S in vals]))
            samp = {"ec": np.nanmean(pred["content"], 1), "es": np.nanmean(pred["style"], 1),
                    "R": np.nanmean(pred["score"], 1)}
            r["argmax_vs_sampled"] = {
                "argmax_reward": float(np.mean(fr["R"])), "sampled_reward": float(np.mean(samp["R"])),
                "argmax_hv": fe.hypervolume(fr["ec"], fr["es"]),
                "sampled_hv": fe.hypervolume(samp["ec"], samp["es"]),
                "argmax_ec": [round(float(x), 1) for x in fr["ec"]],
                "argmax_es": [round(float(x), 1) for x in fr["es"]]}
        # -------- D4: per-sample content spread of the used prompts ----------
        r["content_spread"] = []
        for p in used:
            C = per[p][0]
            r["content_spread"].append(dict(
                prompt=fe.detok(list(p)), ec=float(C.mean()),
                within_sd=float(C.std(1).mean()), between_sd=float(C.mean(1).std()),
                frac_ge99=float((C >= 99).mean()),
                q10=float(np.quantile(C, .1)), q50=float(np.quantile(C, .5)), q90=float(np.quantile(C, .9))))

    # -------- D3: counterfactual objectives on the UNION menu ------------------
    allp = list(per)
    res["counterfactual"] = {}
    for name, kind, pen in (("per-sample, pen 0.01 (training)", "sample", 0.01),
                            ("per-sample, pen 1", "sample", 1.0),
                            ("per-sentence mean floor", "mean", 0.01)):
        rows = []
        for t in EVAL_TAU:
            if kind == "sample":
                Rb = np.array([reward(per[p][0], per[p][1], t, pen) for p in allp])   # [M, n_src]
            else:
                ecb = np.array([per[p][0].mean(1) for p in allp]); esb = np.array([per[p][1].mean(1) for p in allp])
                Rb = np.where(ecb >= t, esb, pen * (ecb - t))
            j_ind = int(Rb.mean(1).argmax())                                   # one prompt for all x
            j_x = Rb.argmax(0)                                                 # per-sentence best
            ec_ind = float(per[allp[j_ind]][0].mean())
            ec_x = float(np.mean([per[allp[j]][0][b].mean() for b, j in enumerate(j_x)]))
            es_x = float(np.mean([per[allp[j]][1][b].mean() for b, j in enumerate(j_x)]))
            P_x = float(np.mean([(per[allp[j]][0][b] >= t).mean() for b, j in enumerate(j_x)]))
            rows.append(dict(tau=float(t), ind_prompt=fe.detok(list(allp[j_ind])), ind_ec=ec_ind,
                             ind_es=float(per[allp[j_ind]][1].mean()),
                             x_ec=ec_x, x_es=es_x, x_P=P_x, x_nprompts=int(len(set(j_x)))))
        res["counterfactual"][name] = rows

    json.dump(res, open(f"{OUT}/frontier_diagnosis.json", "w"), indent=1, default=float)
    report(res)
    return res, per, pol, evals, stats


def report(res):
    for arm, r in res["arms"].items():
        print(f"\n=========== {arm}")
        for k, v in r.items():
            if k in ("content_spread", "menu_used"):
                print(f"  {k}:")
                for row in v:
                    print("     ", {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in row.items()})
            else:
                print(f"  {k}: {v}")
    for name, rows in res["counterfactual"].items():
        print(f"\n--- counterfactual: {name}")
        print("   tau | one-prompt-for-all: E[c]  E[s]  prompt | per-sentence best: E[c]  E[s]  P(c>=t) #prompts")
        for w in rows:
            print(f"  {w['tau']:4.0f} | {w['ind_ec']:6.1f} {w['ind_es']:5.1f}  {w['ind_prompt'][:28]!r:30s} |"
                  f" {w['x_ec']:6.1f} {w['x_es']:5.1f} {w['x_P']:5.2f} {w['x_nprompts']:3d}")


# ---------------------------------------------------------------- figures
def locations(used, stats, radius=3.0):
    """Group prompts whose pooled (E[c], E[s]) are within `radius` points:
    on the tradeoff plot they are one location. Ordered by content."""
    order = sorted(range(len(used)), key=lambda j: stats[used[j]]["ec"])
    groups = []
    for j in order:
        c, s = stats[used[j]]["ec"], stats[used[j]]["es"]
        for g in groups:
            c0, s0 = stats[used[g[0]]]["ec"], stats[used[g[0]]]["es"]
            if abs(c - c0) < radius and abs(s - s0) < radius:
                g.append(j); break
        else:
            groups.append([j])
    return groups


def figures(res, per, pol, evals, stats):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.labelcolor": fe.INK,
                         "text.color": fe.INK, "legend.fontsize": 7, "pdf.fonttype": 42})
    # Prompt locations are identities to match across panels -> categorical
    # slots in their validated order, assigned in content order; at most five
    # named locations per arm, the rest pooled as grey "other".
    SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    OTHER = "#b5b4ae"

    def arm_groups(key):
        P = pol[key]; q = np.exp(P["lp3"])
        used = [p for j, p in enumerate(P["menu"]) if q[j].mean(1).max() > 0.01 and p in per]
        groups = locations(used, stats)
        share = [sum(q[P["menu"].index(used[j])].sum() for j in g) for g in groups]
        keep = sorted(sorted(range(len(groups)), key=lambda i: -share[i])[:len(SLOTS) - 1]
                      if len(groups) > len(SLOTS) else range(len(groups)))
        named = [groups[i] for i in keep]
        rest = [j for i, g in enumerate(groups) if i not in keep for j in g]
        cols = SLOTS[:len(named)]
        return P, q, used, named, rest, cols

    def loc_label(used, g):
        c, s_ = stats[used[g[0]]]["ec"], stats[used[g[0]]]["es"]
        return f"c={c:.0f}, s={s_:.0f}" + (f" ({len(g)} prompts)" if len(g) > 1 else "")

    # ---- Fig 1: calibration of E[c] and P(c>=tau) against tau --------------
    seeds = {a: fe.seed_avg([fe.load(fe.path(a, 12000, s)) for s in range(5)]) for a in ARM_KEYS}
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.75))
    for key, lab, col, ls, mk in fe.ARMS:
        A = seeds[key]
        axs[0].plot(EVAL_TAU, A["content"].mean(1), color=col, ls=ls, marker=mk, ms=3.2, lw=1.3, label=lab)
        axs[1].plot(EVAL_TAU, 100 * A["feasible"].mean(1), color=col, ls=ls, marker=mk, ms=3.2, lw=1.3)
    axs[0].plot([0, 100], [0, 100], color=fe.INK2, ls=":", lw=0.9)
    axs[0].annotate(r"$\mathbb{E}[c]=\tau$", (12, 12), xytext=(15, 4), textcoords="data",
                    color=fe.INK2, fontsize=7)
    axs[0].set(xlabel=r"Content floor $\tau$", ylabel=r"Expected content $\mathbb{E}[c]$",
               xlim=(-2, 97), ylim=(0, 100))
    axs[1].set(xlabel=r"Content floor $\tau$", ylabel=r"Samples meeting the floor (%)",
               xlim=(-2, 97), ylim=(0, 102))
    axs[0].set_title("(a) where each floor's point sits", fontsize=8)
    axs[1].set_title(r"(b) what the reward scores: $P(c\geq\tau)$", fontsize=8)
    for ax in axs: fe._style(ax)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{OUT}/tau_calibration.pdf"); fig.savefig(f"{OUT}/tau_calibration.png", dpi=200)
    plt.close(fig)

    # ---- Fig 2: reward of each menu prompt vs tau, and the policy's choice --
    fig, axs = plt.subplots(2, 4, figsize=(6.9, 4.1), sharex=True,
                            gridspec_kw={"height_ratios": [1.5, 1]})
    for col, (key, lab, _, _, _) in enumerate(fe.ARMS):
        P, q, used, named, rest, cols = arm_groups(key)
        ax, bx = axs[0, col], axs[1, col]
        Rm = np.array([stats[p]["R"] for p in used])
        for j in rest:
            ax.plot(FINE_TAU, Rm[j], color=OTHER, lw=0.8)
        for g, c in zip(named, cols):
            for j in g:
                ax.plot(FINE_TAU, Rm[j], color=c, lw=1.3)
        ax.plot(FINE_TAU, Rm.max(0), color=fe.INK, lw=0.7, ls=(0, (1, 1.5)))
        # where the best LOCATION changes (switches between near-identical
        # prompts of one location do not move the point)
        gid = {j: gi for gi, g in enumerate(named) for j in g} | {j: -1 for j in rest}
        bg = [gid[j] for j in Rm.argmax(0)]
        for i in range(1, len(bg)):
            if bg[i] != bg[i - 1]:
                bx.axvline(FINE_TAU[i], color=fe.INK, lw=0.7, ls=":", zorder=5)
        # policy's prompt shares under the eval decoder, stacked in content order
        layers, lcols = [], []
        for g, c in zip(named, cols):
            layers.append(sum(q[P["menu"].index(used[j])].mean(1) for j in g)); lcols.append(c)
        if rest:
            layers.append(sum(q[P["menu"].index(used[j])].mean(1) for j in rest)); lcols.append(OTHER)
        L = np.array(layers); L = L / L.sum(0, keepdims=True)
        bx.stackplot(EVAL_TAU, L, colors=lcols, edgecolor="#fcfcfb", linewidth=0.8)
        ax.set_title(lab, fontsize=7.5)
        ax.set_ylim(-3, 95); bx.set_ylim(0, 1); bx.set_xlim(0, 95)
        fe._style(ax); fe._style(bx); bx.grid(False)
        if col == 0:
            ax.set_ylabel(r"Reward of each prompt, $\bar R_\tau(p)$")
            bx.set_ylabel("Policy's prompt share")
        bx.set_xlabel(r"Content floor $\tau$")
        for g, c in zip(named, cols):
            ax.plot([], [], color=c, lw=2, label=loc_label(used, g))
        if rest:
            ax.plot([], [], color=OTHER, lw=2, label=f"other ({len(rest)} prompts)")
        ax.legend(loc="upper right", fontsize=5.5, frameon=False, handlelength=1.1, borderaxespad=0.1)
    axs[0, 0].plot([], [], color=fe.INK, lw=0.7, ls=(0, (1, 1.5)), label="best prompt")
    axs[0, 0].legend(loc="upper right", fontsize=5.5, frameon=False, handlelength=1.1, borderaxespad=0.1)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    fig.savefig(f"{OUT}/menu_mechanism.pdf"); fig.savefig(f"{OUT}/menu_mechanism.png", dpi=200)
    plt.close(fig)

    # ---- Fig 3: per-sample content distribution of each location -----------
    fig, axs = plt.subplots(1, 4, figsize=(6.9, 2.35), sharey=True)
    t = np.linspace(0, 100, 201)
    for col, (key, lab, _, _, _) in enumerate(fe.ARMS):
        P, q, used, named, rest, cols = arm_groups(key)
        ax = axs[col]
        for g, c in zip(named, cols):
            j = max(g, key=lambda j: q[P["menu"].index(used[j])].sum())
            C = per[used[j]][0].ravel()
            ax.plot(t, [100 * (C >= x).mean() for x in t], color=c, lw=1.3)
            m = C.mean()
            ax.plot([m], [100 * (C >= m).mean()], "o", color=c, ms=4, mec="#fcfcfb", mew=0.7, zorder=4)
        ax.set_title(lab, fontsize=7.5); ax.set_xlim(0, 100); ax.set_ylim(0, 102)
        ax.set_xlabel("Content threshold $t$")
        fe._style(ax)
    axs[0].set_ylabel(r"Samples with $c \geq t$ (%)")
    axs[0].plot([], [], "o", color=fe.INK2, ms=4, label="at the prompt's own $\\mathbb{E}[c]$")
    axs[0].legend(loc="lower left", fontsize=6, frameon=False, handlelength=0.8)
    fig.tight_layout(w_pad=0.5)
    fig.savefig(f"{OUT}/content_survival.pdf"); fig.savefig(f"{OUT}/content_survival.png", dpi=200)
    plt.close(fig)

    # ---- Fig 4: where the floors WOULD sit under other constraint forms ------
    CF = res["counterfactual"]
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    ax.plot([0, 100], [0, 100], color=fe.INK2, ls=":", lw=0.9)
    rows = CF["per-sample, pen 0.01 (training)"]
    ax.step([w["tau"] for w in rows], [w["ind_ec"] for w in rows], where="mid", color=fe.INK2,
            lw=1.0, ls="--", label="training reward, one prompt for all sentences")
    for (name, lab_), c, mk in zip(
            (("per-sample, pen 0.01 (training)", "training reward (per-sample floor, penalty 0.01)"),
             ("per-sample, pen 1", "per-sample floor, penalty 1"),
             ("per-sentence mean floor", r"floor on the mean, $\mathbb{E}_x[c]\geq\tau$")),
            SLOTS[:3], ("o", "s", "^")):
        rows = CF[name]
        ax.plot([w["tau"] for w in rows], [w["x_ec"] for w in rows], color=c, marker=mk, ms=3.2, lw=1.3,
                label=lab_)
    ax.set(xlabel=r"Content floor $\tau$", ylabel=r"Expected content $\mathbb{E}[c]$ of the best choice",
           xlim=(-2, 97), ylim=(0, 100))
    ax.annotate(r"$\mathbb{E}[c]=\tau$", (12, 12), xytext=(15, 4), color=fe.INK2, fontsize=7)
    ax.set_title("Best choice among the 48 prompts, per sentence", fontsize=7.5)
    fe._style(ax)
    ax.legend(loc="upper left", fontsize=5.8, frameon=False, handlelength=2.2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/tau_counterfactual.pdf"); fig.savefig(f"{OUT}/tau_counterfactual.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    figures(*main())
