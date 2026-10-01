"""Offline comparison of constraint forms for the lambda-indexed reward, on the
per-sample data of the 48-prompt probe (analysis/frontier_probe.py).

Candidates, for prompt z on sentence x at floor tau (C, S = per-sample content,
sentiment; bars = means over the N task-LM samples):

  sample  (current)  mean_n[ s * 1{c >= tau} + 0.01 (c - tau) 1{c < tau} ]
  mean    (B)        Sbar * 1{Cbar >= tau} + 0.01 (Cbar - tau) 1{Cbar < tau}
  mix     (A)        policy-level: max E_pi[Sbar] s.t. E_pi[Cbar] >= tau over
                     MIXTURES of prompts (the eps-constraint on the plotted
                     expectations, i.e. the upper concave hull per sentence)

For each: the best choice per sentence (selected on half the N samples,
evaluated on the other half, so no winner's curse) and the best single prompt
for all sentences. Reports the curve's distinct locations, E[c]-tau, hypervolume,
and how much per-sentence choice is worth under that reward.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frontier_diagnosis as fd  # noqa: E402
import full_eval as fe  # noqa: E402

TAU = fd.EVAL_TAU
OUT = os.path.join(fd.ROOT, "results", "v4", "reward_design")


def distinct(c, s, r=2.0):
    locs = []
    for k in range(len(c)):
        if not locs or np.hypot(c[k] - locs[-1][0], s[k] - locs[-1][1]) >= r:
            locs.append((c[k], s[k]))
    return len(locs)


def hull_mix(Cb, Sb, tau):
    """Per sentence: best mixture of prompts with mean content >= tau.
    Cb, Sb [M] -> (weights [M])."""
    M = len(Cb)
    w = np.zeros(M)
    j0 = int(np.argmax(Sb))
    if Cb[j0] >= tau:
        w[j0] = 1; return w
    if Cb.max() < tau:
        w[int(np.argmax(Cb))] = 1; return w
    # best pair (i below tau, j above tau) on the chord through tau
    best, arg = -np.inf, None
    lo = np.where(Cb < tau)[0]; hi = np.where(Cb >= tau)[0]
    for j in hi:
        a = (Cb[j] - tau) / np.maximum(Cb[j] - Cb[lo], 1e-9)       # weight on lo point
        val = a * Sb[lo] + (1 - a) * Sb[j]
        k = int(np.argmax(val))
        if val[k] > best: best, arg = val[k], (lo[k], j, a[k])
        if Sb[j] > best: best, arg = Sb[j], (j, j, 0.0)
    i, j, a = arg
    w[i] += a; w[j] += 1 - a
    return w


def main():
    os.makedirs(OUT, exist_ok=True)
    per, pol = fd.load_probe()
    P = list(per)
    C = np.stack([per[p][0] for p in P]); S = np.stack([per[p][1] for p in P])   # [M, n, N]
    M, n, N = C.shape
    h = N // 2
    halves = [(slice(0, h), slice(h, N)), (slice(h, N), slice(0, h))]
    out = {}

    def rew(kind, Cs, Ss, t):
        if kind == "sample":
            return np.where(Cs >= t, Ss, 0.01 * (Cs - t)).mean(-1)
        Cb, Sb = Cs.mean(-1), Ss.mean(-1)
        return np.where(Cb >= t, Sb, 0.01 * (Cb - t))

    for kind in ("sample", "mean", "mix"):
        curves = {"x": {"ec": [], "es": []}, "one": {"ec": [], "es": []}}
        gain = []
        for t in TAU:
            ecx = esx = ec1 = es1 = gx = g1 = 0.0
            for sel, ev in halves:
                CA, SA, CB, SB = C[..., sel], S[..., sel], C[..., ev], S[..., ev]
                CBm, SBm = CB.mean(-1), SB.mean(-1)                          # [M, n]
                if kind == "mix":
                    CAm, SAm = CA.mean(-1), SA.mean(-1)
                    W = np.stack([hull_mix(CAm[:, b], SAm[:, b], t) for b in range(n)], 1)  # [M, n]
                    ecx += (W * CBm).sum(0).mean() / 2; esx += (W * SBm).sum(0).mean() / 2
                    # one mixture for all sentences: hull of the pooled means
                    w1 = hull_mix(CAm.mean(1), SAm.mean(1), t)
                    ec1 += (w1[:, None] * CBm).sum(0).mean() / 2; es1 += (w1[:, None] * SBm).sum(0).mean() / 2
                    # value under the policy-level constraint = E[S] when E[C] >= tau
                    vx = np.where((W * CBm).sum(0) >= t - 1e-6, (W * SBm).sum(0), 0).mean()
                    v1 = np.where((w1[:, None] * CBm).sum(0) >= t - 1e-6, (w1[:, None] * SBm).sum(0), 0).mean()
                    gx += vx / 2; g1 += v1 / 2
                    continue
                RA, RB = rew(kind, CA, SA, t), rew(kind, CB, SB, t)          # [M, n]
                jx = RA.argmax(0); j1 = int(RA.mean(1).argmax())
                idx = np.arange(n)
                ecx += CBm[jx, idx].mean() / 2; esx += SBm[jx, idx].mean() / 2
                ec1 += CBm[j1].mean() / 2; es1 += SBm[j1].mean() / 2
                gx += RB[jx, idx].mean() / 2; g1 += RB[j1].mean() / 2
            curves["x"]["ec"].append(ecx); curves["x"]["es"].append(esx)
            curves["one"]["ec"].append(ec1); curves["one"]["es"].append(es1)
            gain.append(gx - g1)
        res = {}
        for which, cv in curves.items():
            ec, es = np.array(cv["ec"]), np.array(cv["es"])
            res[which] = dict(distinct=distinct(ec, es), hv=float(fe.hypervolume(ec, es)),
                              gap=[round(float(v), 1) for v in ec - TAU],
                              ec=[round(float(v), 1) for v in ec], es=[round(float(v), 1) for v in es])
        res["per_sentence_gain_mean"] = float(np.mean(gain))
        res["per_sentence_gain_max"] = float(np.max(gain))
        out[kind] = res

    # ---- which prompts does the policy-level (mix) optimum use? -------------
    Cm, Sm = C.mean(-1), S.mean(-1)
    mixuse = {}
    for t in (40, 50, 60, 70):
        W = np.stack([hull_mix(Cm[:, b], Sm[:, b], t) for b in range(n)], 1)
        top = np.argsort(-W.sum(1))[:3]
        mixuse[t] = [(fe.detok(list(P[j])), round(float(W[j].mean()), 3),
                      round(float(Cm[j].mean()), 1), round(float(Sm[j].mean()), 1)) for j in top]
    out["mix_prompts_used"] = mixuse

    # ---- per-sentence switch points: sentiment prompt -> mid prompt ---------
    A = [p for p in P if fe.detok(list(p)) == 'Awesome!!" terrific." impressed'][0]
    B = [p for p in P if fe.detok(list(p)) == 'Positive praise (% contrasts (−'][0]
    ia, ib = P.index(A), P.index(B)
    grid = np.arange(0, 100, 0.5)
    sw = {}
    for kind in ("sample", "mean"):
        first = []
        for b in range(n):
            ra = np.array([rew(kind, C[ia, b][None], S[ia, b][None], t)[0] for t in grid])
            rb = np.array([rew(kind, C[ib, b][None], S[ib, b][None], t)[0] for t in grid])
            k = np.argmax(rb > ra) if (rb > ra).any() else len(grid) - 1
            first.append(grid[k])
        first = np.array(first)
        sw[kind] = dict(median=float(np.median(first)), iqr=[float(np.quantile(first, .25)), float(np.quantile(first, .75))],
                        sd=float(first.std()), p10_p90=[float(np.quantile(first, .1)), float(np.quantile(first, .9))])
    out["switch_sentiment_to_mid"] = sw
    json.dump(out, open(f"{OUT}/reward_design.json", "w"), indent=1)

    for kind in ("sample", "mean", "mix"):
        r = out[kind]
        print(f"\n=== {kind}: per-sentence choice worth {r['per_sentence_gain_mean']:+.2f} (max {r['per_sentence_gain_max']:+.2f})")
        for which in ("x", "one"):
            w = r[which]
            print(f"  {'per-sentence' if which == 'x' else 'one-for-all ':12s} distinct={w['distinct']:2d} HV={w['hv']:.1f}")
            print(f"     E[c]  {w['ec']}")
            print(f"     E[s]  {w['es']}")
            print(f"     gap   {w['gap']}")
    print("\nmix optimum's prompts (weight, c, s):")
    for t, v in mixuse.items():
        print(f"  tau={t}: {v}")
    print("\nswitch tau, sentiment prompt -> mid prompt, across sentences:", json.dumps(sw))


if __name__ == "__main__":
    main()
