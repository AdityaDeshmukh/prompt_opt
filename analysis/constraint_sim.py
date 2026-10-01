"""How should the per-group multiplier be set so that R-REBEL-std converges to
the eps-constraint solution E_pi[c | x, tau] = tau?  Simulated on the real
prompt statistics of the 48-prompt probe (per-sentence means and per-sample
spreads), one context at a time, for the update rules considered:

  exact / sampled : per-group exact projection -- the smallest content weight
                    a_g in r = (1-a) S + a C whose own R-REBEL target meets the
                    floor (with exact expectations / with G=16 sampled prompts)
  dual            : one multiplier per context, dual ascent on realized content
  P<nu>           : proportional multiplier mu_g = nu (tau - group-mean C)_+,
                    the gradient of the quadratic penalty (nu/2)(tau - E[c])_+^2.
                    THIS is the rule implemented in tst_score.py
                    (reward_constraint=expectation, constraint_kappa = nu).

Each window moves log pi a fraction `step` of the way to the R-REBEL target
pi exp(z/T), z the within-group standardized reward, T = beta = 0.5 (v4's
measured logit growth corresponds to step ~ 0.05).

    python analysis/constraint_sim.py   # -> results/v4/reward_design/constraint_sim.json,
                                        #    paper/tables/{constraint_sim,sim_numbers}.tex
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frontier_diagnosis as fd  # noqa: E402
import full_eval as fe  # noqa: E402

T = 0.5


def solve_alpha(S, C, tau, w=None, iters=30):
    """Smallest a in [0,1] with tilted mean content >= tau (vectorized over rows).
    S, C [B, G]; w optional sampling weights [B, G] (exact population version)."""
    if w is None:
        w = np.ones_like(S) / S.shape[1]
    def tilted_c(a):
        r = (1 - a)[:, None] * S + a[:, None] * C
        mu = (w * r).sum(1, keepdims=True)
        sd = np.sqrt((w * (r - mu) ** 2).sum(1, keepdims=True)) + 1e-8
        z = (r - mu) / sd
        q = w * np.exp((z - z.max(1, keepdims=True)) / T)
        q = q / q.sum(1, keepdims=True)
        return (q * C).sum(1), z
    B = S.shape[0]
    lo, hi = np.zeros(B), np.ones(B)
    f0, _ = tilted_c(lo)
    ok0 = f0 >= tau
    for _ in range(iters):
        mid = (lo + hi) / 2
        f, _ = tilted_c(mid)
        hi = np.where(f >= tau, mid, hi); lo = np.where(f >= tau, lo, mid)
    a = np.where(ok0, 0.0, hi)
    _, z = tilted_c(a)
    return a, z


def run(Cbar, Sbar, sdC, sdS, tau, windows=80, mode="exact", rng=None, G=16, steps=150, N=50,
        eta=2e-3, trace=None, step=1.0):
    """mode: exact | sampled (alpha solved on the same samples) | split (alpha
    solved with one half of each prompt's task-LM samples, the floor checked on
    the other half) | dual (one multiplier per context, dual ascent on the
    groups' realized mean content)."""
    M = len(Cbar)
    logpi = np.zeros(M)
    mu = 0.0
    for _ in range(windows):
        pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
        if mode == "exact":
            _, z = solve_alpha(Sbar[None], Cbar[None], tau, w=pi[None])
            logpi = logpi + step * z[0] / T
            if trace is not None:
                pi2 = np.exp(logpi - logpi.max()); pi2 /= pi2.sum(); trace.append((pi2 * Cbar).sum())
            continue
        idx = rng.choice(M, size=(steps, G), p=pi)
        e1, e2 = rng.normal(0, 1, (2,) + idx.shape)
        C1 = Cbar[idx] + e1 * sdC[idx] / np.sqrt(N / 2)          # two independent halves
        C2 = Cbar[idx] + e2 * sdC[idx] / np.sqrt(N / 2)
        Cs = (C1 + C2) / 2
        Ss = Sbar[idx] + rng.normal(0, 1, idx.shape) * sdS[idx] / np.sqrt(N)
        if mode == "sampled":
            _, z = solve_alpha(Ss, Cs, tau)
        elif mode == "split":
            a = solve_alpha_split(Ss, C1, C2, tau)
            r = (1 - a)[:, None] * Ss + a[:, None] * Cs
            z = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-8)
        elif mode.startswith("P"):
            # proportional multiplier: gradient of (kappa/2)(tau - E_pi[C|x])_+^2,
            # E_pi[C|x] estimated by the group's mean content
            kappa = float(mode[1:])
            mu_g = kappa * np.clip(tau - Cs.mean(1), 0, None)
            r = Ss + mu_g[:, None] * Cs
            z = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-8)
        elif mode == "dual":
            z = np.zeros(idx.shape)
            for k in range(steps):                                 # one group per step
                r = Ss[k] + mu * Cs[k]
                z[k] = (r - r.mean()) / (r.std() + 1e-8)
                mu = max(0.0, mu + eta * (tau - Cs[k].mean()))
        tot = np.zeros(M); cnt = np.zeros(M)
        np.add.at(tot, idx.ravel(), z.ravel()); np.add.at(cnt, idx.ravel(), 1)
        seen = cnt > 0
        upd = np.zeros(M); upd[seen] = tot[seen] / cnt[seen]
        logpi = logpi + step * upd / T
        if trace is not None:
            pi2 = np.exp(logpi - logpi.max()); pi2 /= pi2.sum(); trace.append((pi2 * Cbar).sum())
    pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
    return pi


def solve_alpha_split(S, C1, C2, tau, iters=30):
    """alpha from the tilt of (S, C1) but the floor checked with C2: the tilt
    cannot select prompts for a lucky C2, so no winner's curse."""
    def tilted_c2(a):
        r = (1 - a)[:, None] * S + a[:, None] * (C1 + C2) / 2
        z = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-8)
        # tilt uses C1 only through r; the checked content is C2
        q = np.exp((z - z.max(1, keepdims=True)) / T); q /= q.sum(1, keepdims=True)
        return (q * C2).sum(1)
    B = S.shape[0]
    lo, hi = np.zeros(B), np.ones(B)
    ok0 = tilted_c2(lo) >= tau
    for _ in range(iters):
        mid = (lo + hi) / 2
        f = tilted_c2(mid)
        hi = np.where(f >= tau, mid, hi); lo = np.where(f >= tau, lo, mid)
    return np.where(ok0, 0.0, hi)


MODES = (("exact", 1.0, 80), ("exact", 0.05, 1600), ("sampled", 0.05, 1600), ("dual", 0.05, 1600),
         ("P1", 0.05, 1600), ("P5", 0.05, 1600), ("P5", 0.2, 400))
LABELS = {"exact": "per-group projection, exact", "sampled": "per-group projection, sampled",
          "dual": "dual ascent (one multiplier)", "P1": "proportional, $\\nu=1$", "P5": "proportional, $\\nu=5$"}
MACRO = {("exact", 1.0): "ExactFull", ("exact", 0.05): "Exact", ("sampled", 0.05): "Proj",
         ("dual", 0.05): "Dual", ("P1", 0.05): "Pone", ("P5", 0.05): "Pfive", ("P5", 0.2): "PfiveFast"}


def main():
    import json
    per, _ = fd.load_probe()
    P = list(per)
    C = np.stack([per[p][0] for p in P]); S = np.stack([per[p][1] for p in P])   # [M, n, N]
    rng = np.random.default_rng(0)
    taus = fd.EVAL_TAU
    bind = (taus >= 30) & (taus <= 90)
    sents = list(rng.choice(500, 5, replace=False))
    out = {"sentences": [int(b) for b in sents], "tau": taus.tolist(), "rules": []}
    for mode, stp, W in MODES:
        ec_all, es_all, wob = [], [], []
        for t in taus:
            ecs, ess = [], []
            for b in sents:
                Cb, Sb = C[:, b].mean(-1), S[:, b].mean(-1)
                sdC, sdS = C[:, b].std(-1), S[:, b].std(-1)
                tr = []
                pi = run(Cb, Sb, sdC, sdS, t, mode=mode, rng=rng, trace=tr, step=stp, windows=W,
                         steps=150 if mode == "exact" else 30)
                ecs.append((pi * Cb).sum()); ess.append((pi * Sb).sum())
                if tr: wob.append(np.std(tr[-20:]))
            ec_all.append(np.mean(ecs)); es_all.append(np.mean(ess))
        ec, es = np.array(ec_all), np.array(es_all)
        g = ec - taus
        row = dict(mode=mode, step=stp, label=LABELS[mode], macro=MACRO[(mode, stp)],
                   calib=float(np.abs(g[bind]).mean()), worst=float(np.abs(g[bind]).max()),
                   distinct=int(sum(1 for k in range(20) if k == 0 or np.hypot(ec[k]-ec[k-1], es[k]-es[k-1]) >= 2)),
                   hv=float(fe.hypervolume(ec, es)), wobble=float(np.max(wob)) if wob else 0.0,
                   gap=np.round(g, 2).tolist(), es=np.round(es, 2).tolist())
        out["rules"].append(row)
        print(f"{mode+'@'+str(stp):12s} mean|gap|={row['calib']:.2f} worst={row['worst']:.1f} "
              f"distinct={row['distinct']} HV={row['hv']:.1f} wobble={row['wobble']:.2f}")
    os.makedirs(os.path.join(fe.ROOT, "results", "v4", "reward_design"), exist_ok=True)
    json.dump(out, open(os.path.join(fe.ROOT, "results", "v4", "reward_design", "constraint_sim.json"), "w"), indent=1)
    with open(os.path.join(fe.TAB, "constraint_sim.tex"), "w") as fh:
        fh.write("% generated by analysis/constraint_sim.py -- do not edit by hand\n")
        fh.write("\\begin{tabular}{lcccccc}\n\\toprule\n"
                 "Multiplier rule & Step & mean $|\\mathbb{E}[c]-\\tau|$ & worst & Distinct points & "
                 "Hypervolume & Late wobble \\\\\n\\midrule\n")
        for r in out["rules"]:
            fh.write(f"{r['label']} & ${r['step']:g}$ & ${r['calib']:.1f}$ & ${r['worst']:.1f}$ & "
                     f"${r['distinct']}/20$ & ${r['hv']:.1f}$ & ${r['wobble']:.1f}$ \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(fe.TAB, "sim_numbers.tex"), "w") as fh:
        fh.write("% generated by analysis/constraint_sim.py -- do not edit by hand\n")
        for r in out["rules"]:
            for k, K in (("calib", "Calib"), ("worst", "Worst"), ("hv", "HV"), ("wobble", "Wobble")):
                fh.write(f"\\newcommand{{\\sim{r['macro']}{K}}}{{{r[k]:.1f}}}\n")
            fh.write(f"\\newcommand{{\\sim{r['macro']}Distinct}}{{{r['distinct']}}}\n")
    print("wrote", os.path.join(fe.TAB, "constraint_sim.tex"), "and sim_numbers.tex")


if __name__ == "__main__":
    main()
