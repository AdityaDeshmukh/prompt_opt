"""Numerical sanity checks for the note rrebel_std_theory.tex.

Every quantitative claim in the note is checked on small tabular problems,
exactly (full enumeration of groups) wherever possible. Standalone: numpy +
matplotlib only, no repository code. Prints PASS/FAIL per check, writes the
note's figures to figures/, and exits nonzero if any check fails.

    /u/ad11/miniconda3/envs/prompt_opt/bin/python checks.py
"""
import itertools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
FAILS = []


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def zscores(r):
    """In-group z-scores with the G-1 (sample) normalization; 0 for a flat group."""
    r = np.asarray(r, float)
    s = r.std(ddof=1)
    return np.zeros_like(r) if s < 1e-12 * (1 + np.abs(r).max()) else (r - r.mean()) / s


def compositions(n, k):
    """All length-k nonnegative integer vectors summing to n (stars and bars)."""
    out = []
    for bars in itertools.combinations(range(n + k - 1), k - 1):
        prev, v = -1, []
        for b in bars:
            v.append(b - prev - 1)
            prev = b
        v.append(n + k - 2 - prev)
        out.append(v)
    return np.array(out, dtype=np.int64)


class Zeta:
    """Exact expected in-group z-score  zeta_q(y) = E[z_1 | y_1 = y]  for a group
    of size G whose other G-1 members are iid from q (finite output set with
    rewards r). Enumerates the multiset of the other members, so it is exact and
    cheap for |Y| <= 8, G <= 16."""

    def __init__(self, r, G):
        r = np.asarray(r, float)
        self.G, self.K = G, len(r)
        rn = (r - r.min()) / max(r.max() - r.min(), 1e-300)  # z is affine invariant
        C = compositions(G - 1, self.K)
        self.C = C
        self.logcoef = math.lgamma(G) - np.array(
            [sum(math.lgamma(c + 1) for c in row) for row in C])
        cnt = C[:, None, :] + np.eye(self.K, dtype=np.int64)[None]     # [M, K(member), K]
        mean = (cnt * rn).sum(-1) / G                                   # [M, K]
        ss = (cnt * (rn[None, None, :] - mean[..., None]) ** 2).sum(-1)
        dev = rn[None, :] - mean
        flat = ss < 1e-18
        self.phi = np.where(flat, 0.0, dev / np.sqrt(np.where(flat, 1.0, ss) / (G - 1)))

    def probs(self, q):
        with np.errstate(divide="ignore"):
            lq = np.log(np.asarray(q, float))
        lp = self.logcoef + np.where(self.C > 0, self.C * lq[None, :], 0.0).sum(1)
        return np.exp(lp)

    def __call__(self, q):
        return self.probs(q) @ self.phi


# =============================================================================
# PART I
# =============================================================================
def check_finite_G_target():
    """Thm (finite-G target): the l2 tabular minimizer is h = c + G/(G-1) zeta_q.
    Brute force over all ordered groups; includes G = 2 (win-rate form)."""
    rng = np.random.default_rng(0)
    for G in (2, 3, 4, 5):
        Y = 5
        r = rng.normal(size=Y) * 7 + 3
        q = rng.dirichlet(np.ones(Y))
        rows, tgts = [], []
        for idx in itertools.product(range(Y), repeat=G):
            idx = np.array(idx)
            w = math.sqrt(np.prod(q[idx]))
            u = zscores(r[idx])
            for i, j in zip(*np.triu_indices(G, 1)):
                a = np.zeros(Y); a[idx[i]] += 1; a[idx[j]] -= 1
                rows.append(w * a); tgts.append(w * (u[i] - u[j]))
        h = np.linalg.lstsq(np.array(rows), np.array(tgts), rcond=None)[0]
        zeta = Zeta(r, G)(q)
        pred = G / (G - 1) * zeta
        err = np.abs((h - h.mean()) - (pred - pred.mean())).max()
        report(f"finite-G l2 target = G/(G-1)*zeta (G={G})", err < 1e-9, f"max err {err:.1e}")
        report(f"  E_q zeta = 0 (G={G})", abs(q @ zeta) < 1e-12, f"{q @ zeta:.1e}")
        if G == 2:
            lq = np.array([(q * np.sign(r[y] - r)).sum() for y in range(Y)])
            e2 = np.abs(pred - np.sqrt(2) * lq).max()
            report("  G=2: target = sqrt(2) * win-rate margin l_q(y)", e2 < 1e-12, f"err {e2:.1e}")


def check_phi_formula_and_bounds():
    """Lemma (z-score sigmoid) + Samuelson bounds + rank preservation + osc <= 2 sqrt(G)."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(2000):
        G = rng.integers(3, 30)
        others = rng.standard_t(2, size=G - 1) * 10 ** rng.uniform(-3, 3)
        x = rng.standard_t(2) * 10 ** rng.uniform(-3, 3)
        z1 = zscores(np.r_[x, others])[0]
        mp, V, c = others.mean(), ((others - others.mean()) ** 2).sum(), (G - 1) / G
        t = x - mp
        phi = math.sqrt(G - 1) * c * t / math.sqrt(V + c * t * t)
        worst = max(worst, abs(z1 - phi) / (1 + abs(z1)))
    report("z-score of a member = sqrt(G-1) c t / sqrt(V' + c t^2)", worst < 1e-10, f"max rel err {worst:.1e}")
    # Samuelson: |z_i| <= (G-1)/sqrt(G), |z_i - z_j| <= sqrt(2(G-1)); both tight
    ok1 = ok2 = True
    for G in (2, 3, 8, 16, 40):
        for _ in range(3000):
            z = zscores(rng.standard_t(1.5, size=G) * 10 ** rng.uniform(-4, 4))
            ok1 &= np.abs(z).max() <= (G - 1) / math.sqrt(G) + 1e-9
            ok2 &= np.ptp(z) <= math.sqrt(2 * (G - 1)) + 1e-9
        e = np.zeros(G); e[0] = 1
        ok1 &= abs(zscores(e)[0] - (G - 1) / math.sqrt(G)) < 1e-12
        e2 = np.zeros(G); e2[0], e2[1] = 1, -1
        ok2 &= abs(np.ptp(zscores(e2)) - math.sqrt(2 * (G - 1))) < 1e-12
    report("Samuelson |z| <= (G-1)/sqrt(G) (tight)", ok1)
    report("pair bound |z_i - z_j| <= sqrt(2(G-1)) (tight)", ok2)
    # rank preservation + oscillation bound for G = 16 with exact zeta
    ok_rank, ok_osc, max_ratio = True, True, 0.0
    for trial in range(40):
        K = 6
        r = rng.standard_t(2, size=K) * 10 ** rng.uniform(-2, 2)
        q = rng.dirichlet(np.ones(K) * rng.uniform(0.05, 2))
        q = np.maximum(q, 1e-12); q /= q.sum()
        zeta = Zeta(r, 16)(q)
        o = np.argsort(r)
        ok_rank &= np.all(np.diff(zeta[o]) >= -1e-12)
        h = 16 / 15 * zeta
        ok_osc &= np.ptp(h) <= 2 * 4 + 1e-9
        max_ratio = max(max_ratio, np.ptp(h) / 8)
    report("zeta is nondecreasing in reward (G=16, 40 random q)", ok_rank)
    report("osc of beta*log-ratio <= 2 sqrt(G) (G=16)", ok_osc, f"(max osc / bound = {max_ratio:.2f})")


def binary_gap(G, qb):
    """G/(G-1) * (zeta(good) - zeta(bad)) for binary rewards, q(good) = qb."""
    tot_g = tot_b = 0.0
    for K in range(G):                       # K = number of goods among the other G-1
        p = math.comb(G - 1, K) * qb ** K * (1 - qb) ** (G - 1 - K)
        kg = K + 1                           # goods in group if member is good
        if kg < G:
            tot_g += p * math.sqrt((G - 1) * (G - kg) / (G * kg))
        if K > 0:                            # member bad, K goods in group
            tot_b -= p * math.sqrt((G - 1) * K / (G * (G - K)))
    return G / (G - 1) * (tot_g - tot_b)


def check_late_phase_and_mG():
    """Cor (late phase): per-window log-odds gain -> sqrt(G)/beta; m_G = min over q."""
    for G in (2, 4, 16, 64):
        g = binary_gap(G, 1 - 1e-7)
        report(f"late-phase gap -> sqrt(G) (G={G})", abs(g - math.sqrt(G)) < 1e-3 * math.sqrt(G),
               f"gap(q=1-1e-7) = {g:.4f}, sqrt(G) = {math.sqrt(G):.4f}")
    qs = np.linspace(1e-4, 1 - 1e-4, 4001)
    out, ok = {}, True
    for G in (2, 3, 4, 8, 16, 32, 64):
        gaps = np.array([binary_gap(G, q) for q in qs])
        out[G] = (gaps.min(), qs[gaps.argmin()])
        ok &= gaps.min() >= math.sqrt(2) - 1e-12 and gaps.max() <= math.sqrt(G) + 1e-12
        if G >= 4:
            ok &= abs(qs[gaps.argmin()] - 0.5) < 1e-3
    msg = ", ".join(f"G={G}: m_G={m:.3f}" for G, (m, qq) in out.items())
    report("binary per-window gain * beta lies in [sqrt(2), sqrt(G)]; min at q=1/2 for G>=4", ok, msg)
    for G in (2, 3):
        report(f"  G={G}: gain is exactly sqrt({G}) for all q",
               max(abs(binary_gap(G, q) - math.sqrt(G)) for q in qs) < 1e-12)
    return out


def check_global_convergence():
    """Thm (fixed-beta global convergence): pi_t(y*) is nondecreasing and every
    suboptimal log-odds grows by >= pi_t(y*)^(G-1) sqrt(G)/beta per window."""
    rng = np.random.default_rng(11)
    ok = True; worst_slack = np.inf
    for _ in range(6):
        G, K, beta = 8, 6, 0.5
        r = rng.normal(size=K) * 10 ** rng.uniform(-2, 2)
        Z = Zeta(r, G); eta = G / ((G - 1) * beta); ys = r.argmax()
        logq = np.log(rng.dirichlet(np.ones(K)))
        prev_best = 0.0
        for t in range(60):
            q = np.exp(logq - logq.max()); q /= q.sum()
            ok &= q[ys] >= prev_best - 1e-12; prev_best = q[ys]
            new = logq + eta * Z(q)
            inc = (new[ys] - new) - (logq[ys] - logq)
            lb = q[ys] ** (G - 1) * math.sqrt(G) / beta
            m = np.delete(inc, ys).min() - lb
            ok &= m >= -1e-9; worst_slack = min(worst_slack, m)
            logq = new
    report("fixed beta: pi_t(y*) nondecreasing, log-odds gain >= pi_t(y*)^(G-1) sqrt(G)/beta", ok)


def check_balancing():
    """Prop (balancing): at a window start the mean pair loss of EVERY non-flat group is
    exactly 2 under l2-std (2 * sample variance without std)."""
    rng = np.random.default_rng(12)
    ok = True
    for _ in range(500):
        G = rng.integers(2, 40)
        x = rng.standard_t(2, size=G) * 10 ** rng.uniform(-4, 4)
        z = zscores(x)
        i, j = np.triu_indices(G, 1)
        ok &= abs(((z[i] - z[j]) ** 2).mean() - 2) < 1e-9
        ok &= abs(((x[i] - x[j]) ** 2).mean() - 2 * x.var(ddof=1)) < 1e-9 * (1 + x.var())
    report("window-start l2 loss = 2 for every non-flat group (vs 2*sample var)", ok)


def check_onpolicy_threshold():
    """Prop (on-policy, G = inf, binary): fixed point iff beta >= min_y cosh(y)/y."""
    lo, hi = 0.5, 2.0
    for _ in range(200):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if m * math.tanh(m) < 1 else (lo, m)
    thr = math.cosh(lo) / lo
    report("infinite-G on-policy binary threshold beta* = 1.5089", abs(thr - 1.5089) < 1e-4, f"{thr:.5f}")
    # finite G = 16, beta = 0.5: damped iteration x <- gap(q(x))/beta, x = logit(q), pi_ref = 1/2
    G, beta, x = 16, 0.5, 0.0
    for _ in range(4000):
        qb = 1 / (1 + math.exp(-x))
        x = 0.9 * x + 0.1 * binary_gap(G, qb) / beta
    resid = abs(x - binary_gap(G, 1 / (1 + math.exp(-x))) / beta)
    report("finite-G (G=16, beta=0.5) on-policy fixed point exists, inside 2 sqrt(G)/beta",
           resid < 1e-9 and x <= 2 * math.sqrt(G) / beta,
           f"log-odds at fixed point = {x:.3f} (bound {2 * math.sqrt(G) / beta:.0f})")


def tilt(p0, v, t):
    lw = np.log(p0) + t * v
    lp = lw - lw.max(); lp -= math.log(np.exp(lp).sum())
    p = np.exp(lp)
    return p, float((p * (lp - np.log(p0))).sum())


def check_cgf_and_fisher():
    """Prop (fixed KL budget): KL = t L'(t) - L(t), gain = sigma L'(t); Gaussian shape -> 1/(2 beta^2).
    Prop (normalized NPG): Fisher norm of the natural gradient = sigma_pi; TRPO step = std window."""
    rng = np.random.default_rng(2)
    K, beta = 300, 0.5
    p0 = rng.dirichlet(np.ones(K))
    r = rng.gamma(2.0, size=K) * 37.0 - 5
    mu = p0 @ r; sig = math.sqrt(p0 @ (r - mu) ** 2)
    u = (r - mu) / sig
    t = 1 / beta
    p, kl = tilt(p0, r, 1 / (beta * sig))
    Lam = math.log(p0 @ np.exp(t * u)); dLam = (p0 * u * np.exp(t * u)).sum() / (p0 @ np.exp(t * u))
    report("KL(pi_std || pi_ref) = t L'(t) - L(t)", abs(kl - (t * dLam - Lam)) < 1e-9, f"KL = {kl:.4f}")
    report("gain = sigma * L'(1/beta)", abs((p @ r - mu) - sig * dLam) < 1e-8 * sig)
    xs = np.linspace(-12, 12, 40001); pg = np.exp(-xs ** 2 / 2); pg /= pg.sum()
    _, klg = tilt(pg, xs, 1 / beta)
    report("Gaussian-shaped reward: KL = 1/(2 beta^2) exactly", abs(klg - 1 / (2 * beta ** 2)) < 1e-6, f"{klg:.6f}")
    # scale invariance of the std target vs sigma^2 scaling of the plain target
    kls, klp = [], []
    for s in 10.0 ** np.arange(-3, 3.1, 1.0):
        rr = 4.0 + s * u
        kls.append(tilt(p0, rr, 1 / (beta * s))[1]); klp.append(tilt(p0, rr, 1 / beta)[1])
    report("std-target KL independent of reward scale (6 decades)", np.ptp(kls) < 1e-9,
           f"plain KL ranges {min(klp):.1e} .. {max(klp):.2f}")
    # Fisher norm of the tabular natural gradient
    pi = rng.dirichlet(np.ones(20)); rr = rng.normal(size=20) * 3
    A = rr - pi @ rr; F = np.diag(pi) - np.outer(pi, pi)
    report("Fisher norm of natural gradient = sigma_pi", abs(math.sqrt(A @ F @ A) - math.sqrt(pi @ A ** 2)) < 1e-12)
    # TRPO step with radius delta = 1/(2 beta^2) equals the exact std window
    g = pi * A
    ng = np.linalg.pinv(F) @ g
    step = math.sqrt(2 * (1 / (2 * beta ** 2)) / (g @ ng)) * ng
    p_trpo = pi * np.exp(step); p_trpo /= p_trpo.sum()
    p_std = pi * np.exp(A / (beta * math.sqrt(pi @ A ** 2))); p_std /= p_std.sum()
    report("TRPO step (radius 1/(2 beta^2)) = exact R-REBEL-std window (tabular)",
           np.abs(p_trpo - p_std).max() < 1e-12)


# =============================================================================
# PART II
# =============================================================================
def check_gauss_newton():
    """Prop (GN, population): beta^2 d'F d = explained fraction R^2 <= 1, = 1 iff compatible.
    Prop (GN, finite sample): interpolation => within-group std of Delta h = 1."""
    rng = np.random.default_rng(3)
    Y, beta = 30, 0.5
    r = rng.normal(size=Y) * 50
    for k in (4, 12, Y - 1):
        phi = rng.normal(size=(Y, k))
        th = rng.normal(size=k) * 0.3
        lp = phi @ th; pi = np.exp(lp - lp.max()); pi /= pi.sum()
        g = phi - pi @ phi                                   # centered score under pi (= nu)
        uc = (r - pi @ r) / math.sqrt(pi @ (r - pi @ r) ** 2)
        F = (pi[:, None] * g).T @ g
        d = np.linalg.pinv(F) @ ((pi * uc) @ g) / beta
        quad = beta ** 2 * d @ F @ d
        fit = beta * g @ d
        R2 = pi @ fit ** 2                                   # ||Pi u_c||^2 (u_c has unit norm)
        ok = abs(quad - R2) < 1e-9 and quad <= 1 + 1e-9 and (k < Y - 1 or abs(quad - 1) < 1e-8)
        report(f"GN step: beta^2 d'Fd = R^2_compat <= 1 (feature dim {k}/{Y - 1})", ok, f"= {quad:.4f}")
    G, D = 16, 40
    gs = rng.normal(size=(G, D)); rw = rng.lognormal(size=G) * 1e3
    z = zscores(rw)
    Cc = (gs - gs.mean(0)).T @ (gs - gs.mean(0))
    d = np.linalg.pinv(Cc) @ (z @ gs) / beta
    Dh = beta * gs @ d
    report("finite-sample GN: fit interpolates z-scores", np.abs((Dh - Dh.mean()) - z).max() < 1e-9)
    report("  within-group sample std of the change in beta*log pi = 1", abs(Dh.std(ddof=1) - 1) < 1e-9)


def check_zeta_lemma():
    """Lemma (zeta properties): E_q zeta^2 <= (G-1)/G;  zeta(y*) >= kappa_G (r* - E_q r)/R."""
    rng = np.random.default_rng(4)
    G = 16; kap = 2 * ((G - 1) / G) ** 1.5
    ok2 = ok4 = True; tight = 0.0
    for _ in range(40):
        K = 6
        r = rng.normal(size=K) * 10 ** rng.uniform(-2, 2)
        q = rng.dirichlet(np.ones(K) * rng.uniform(0.05, 3)); q = np.maximum(q, 1e-14); q /= q.sum()
        zeta = Zeta(r, G)(q)
        ok2 &= q @ zeta ** 2 <= (G - 1) / G + 1e-12
        lhs = zeta[r.argmax()]; rhs = kap * (r.max() - q @ r) / np.ptp(r)
        ok4 &= lhs >= rhs - 1e-12
        tight = max(tight, rhs / max(lhs, 1e-300))
    report("E_q[zeta^2] <= (G-1)/G", ok2)
    report("zeta(y*) >= kappa_G (J* - J_q)/R", ok4, f"(max rhs/lhs = {tight:.2f})")


def simulate(r, T, kind, beta, G=16):
    K = len(r)
    logq = np.full(K, -math.log(K))
    Z = Zeta(r, G) if kind == "std" else None
    eta = G / ((G - 1) * beta)
    out = np.empty(T)
    for t in range(T):
        q = np.exp(logq - logq.max()); q /= q.sum()
        out[t] = (r.max() - q @ r) / np.ptp(r)
        logq = logq + (eta * Z(q) if kind == "std" else r / beta)
    return out


def check_scale_free_rate():
    """Thm (scale-free rate) and Prop (separation)."""
    rng = np.random.default_rng(5)
    G, K, T = 16, 6, 3000
    shape = rng.uniform(size=K)
    scales = [1e-2, 1.0, 1e2]
    bound = G / (G - 1) * math.sqrt(math.log(K) / T)
    beta_std = math.sqrt(G * T / ((G - 1) * math.log(K)))
    beta_plain = max(scales) * math.sqrt(T / (4 * math.log(K)))
    rs, rp = [], []
    for s in scales:
        r = s * shape
        rs.append(simulate(r, T, "std", beta_std, G).mean())
        rp.append(simulate(r, T, "plain", beta_plain).mean())
    report("scale-free rate: avg range-normalized regret <= (G/(G-1)) sqrt(ln|Y|/T) in EVERY context",
           max(rs) <= bound, f"std: {[f'{v:.4f}' for v in rs]} vs bound {bound:.4f}; "
           f"plain: {[f'{v:.4f}' for v in rp]}")
    # random shapes per context too
    worst = 0.0
    for _ in range(5):
        for s in 10.0 ** rng.uniform(-3, 3, size=3):
            worst = max(worst, simulate(s * rng.standard_t(3, size=K), T, "std", beta_std, G).mean())
    report("  ... and for random reward shapes/scales", worst <= bound, f"worst {worst:.4f}")


def fig_separation():
    """Common per-window D_inf budget B: plain beta = R_max/B, std beta = 2 sqrt(G)/B."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(6)
    G, K, T, B = 16, 6, 400, 1.0
    shape = rng.uniform(size=K)
    scales = [1e2, 1.0, 1e-2]
    colors = ["#1f5fa8", "#d17a00", "#b8322a"]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for s, c in zip(scales, colors):
        r = s * shape
        ax.semilogy(simulate(r, T, "plain", max(scales) / B), color=c, lw=1.8,
                    label=f"plain, range $R={s:g}$ ($\\kappa={max(scales) / s:g}$)")
    ax.semilogy(simulate(shape, T, "std", 2 * math.sqrt(G) / B, G), color="k", lw=1.4, ls="--",
                label="std, all three ranges (curves coincide)")
    ax.set_xlabel("window $t$ (reference refreshes)")
    ax.set_ylabel(r"$(J^\star - J_t)/R$")
    ax.set_ylim(1e-6, 2.5)
    ax.legend(fontsize=7, frameon=True, framealpha=0.9, loc="center right")
    ax.set_title("Same per-window trust region ($D_\\infty \\leq 1$ nat) in every context", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "separation.pdf")); plt.close(fig)
    # the three std curves coincide exactly (affine invariance)
    c = [simulate(s * shape, T, "std", 2 * math.sqrt(G) / B, G) for s in scales]
    report("std trajectories identical across reward scales (exact invariance)",
           max(np.abs(c[0] - c[1]).max(), np.abs(c[0] - c[2]).max()) < 1e-10)


def fig_squashing(mG):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    t = np.linspace(-12, 12, 1201)
    for G, col in zip((2, 4, 16, 64), ("#b8322a", "#d17a00", "#1f5fa8", "#2e7d32")):
        c = (G - 1) / G
        V = max(G - 2, 0)                     # others' sum of squares, in units of their std^2
        phi = np.sign(t) * math.sqrt(G - 1) * math.sqrt(c) if V == 0 else \
            math.sqrt(G - 1) * c * t / np.sqrt(V + c * t ** 2)
        axs[0].plot(t, phi, color=col, lw=1.6, label=f"$G={G}$")
        axs[0].axhline((G - 1) / math.sqrt(G), color=col, lw=0.6, ls=":")
    axs[0].plot(t, t, color="k", lw=1.0, ls="--", label=r"$G\to\infty$ ($z=t$)")
    axs[0].set_ylim(-9, 9); axs[0].set_xlabel(r"reward offset $t=(r-\bar r_{-})/s_{-}$")
    axs[0].set_ylabel("in-group z-score")
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    axs[0].set_title(r"(a) soft clip at $\pm(G-1)/\sqrt{G}$", fontsize=9)
    qs = np.linspace(0.002, 0.998, 400)
    for G, col in zip((2, 4, 16, 64), ("#b8322a", "#d17a00", "#1f5fa8", "#2e7d32")):
        axs[1].plot(qs, [binary_gap(G, q) for q in qs], color=col, lw=1.6, label=f"$G={G}$")
        axs[1].axhline(math.sqrt(G), color=col, lw=0.6, ls=":")
    axs[1].set_xlabel(r"$q(\mathrm{better\ output})$")
    axs[1].set_ylabel(r"$\beta\times$ log-odds gain per window")
    axs[1].set_title(r"(b) binary rewards: gain $\in[\sqrt{2},\sqrt{G}]$", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "squashing.pdf")); plt.close(fig)


def check_reliability():
    """Prop (noise): target temperature beta*sigma_tot => KL ~ rho^2 / (2 beta^2)."""
    rng = np.random.default_rng(7)
    K = 400; q = rng.dirichlet(np.ones(K))
    r = rng.normal(size=K) * 3.0
    for nv in (0.0, 1.0, 9.0, 81.0):
        v = nv * rng.uniform(0.5, 1.5, size=K)
        var_r = q @ (r - q @ r) ** 2; tot = var_r + q @ v
        rel = var_r / tot
        for beta in (10.0,):
            _, kl = tilt(q, r, 1 / (beta * math.sqrt(tot)))
            pred = rel / (2 * beta ** 2)
            report(f"reliability-weighted KL (noise var ~{nv:g}, rho^2={rel:.3f})",
                   abs(kl - pred) / pred < 2e-2, f"KL={kl:.3e} pred={pred:.3e}")


def check_monotone():
    """Prop (monotone improvement): one exact std window => first-order stochastic dominance."""
    rng = np.random.default_rng(8)
    ok = True
    for _ in range(30):
        K = 6; r = rng.normal(size=K) * 10 ** rng.uniform(-2, 2)
        q = rng.dirichlet(np.ones(K)); Z = Zeta(r, 8)
        q2 = q * np.exp(8 / 7 / 0.5 * Z(q)); q2 /= q2.sum()
        o = np.argsort(r)
        ok &= np.all(np.cumsum(q2[o]) <= np.cumsum(q[o]) + 1e-12)
    report("exact std window => reward law first-order stochastically dominates", ok)


def check_branch_invariance():
    """Prop (branch-wise lambda invariance): all-infeasible groups -> targets of s2 alone."""
    rng = np.random.default_rng(9)
    tau = 70.0
    s2 = rng.uniform(20, 69, size=16)          # every member infeasible
    s1 = rng.uniform(0, 100, size=16)
    ref = zscores(s2)
    errs = [np.abs(zscores(lam * (s2 - tau)) - ref).max() for lam in (1e-4, 1e-2, 1.0, 50.0)]
    report("all-infeasible group: std targets independent of lambda, = those of s2", max(errs) < 1e-10)
    report("all-feasible group: std targets = those of s1", np.abs(zscores(s1) - zscores(3 * s1 + 7)).max() < 1e-12)


def check_robust_d_finite_G():
    """Observation: l1 / Huber finite-G tabular minimizers are also rank-preserving
    and inside osc <= 2 sqrt(G). IRLS on the exact (enumerated) population loss."""
    rng = np.random.default_rng(10)
    G, Y = 4, 5
    res = {"l1": [True, True, 0.0], "huber": [True, True, 0.0]}
    for _ in range(12):
        r = rng.standard_t(2, size=Y) * 10 ** rng.uniform(-2, 2)
        q = rng.dirichlet(np.ones(Y))
        A, t, w0 = [], [], []
        for idx in itertools.product(range(Y), repeat=G):
            idx = np.array(idx); p = np.prod(q[idx]); u = zscores(r[idx])
            for i, j in zip(*np.triu_indices(G, 1)):
                if idx[i] == idx[j]:
                    continue
                a = np.zeros(Y); a[idx[i]] += 1; a[idx[j]] -= 1
                A.append(a); t.append(u[i] - u[j]); w0.append(p)
        A, t, w0 = np.array(A), np.array(t), np.array(w0)
        for kind in ("l1", "huber"):
            h = np.linalg.lstsq(A * np.sqrt(w0)[:, None], t * np.sqrt(w0), rcond=None)[0]
            for _ in range(3000):
                e = A @ h - t
                wt = 1 / np.maximum(np.abs(e), 1e-10) if kind == "l1" else \
                    np.where(np.abs(e) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(e), 1e-12))
                W = np.sqrt(w0 * wt)
                h_new = np.linalg.lstsq(A * W[:, None], t * W, rcond=None)[0]
                if np.abs(h_new - h).max() < 1e-12:
                    h = h_new; break
                h = h_new
            o = np.argsort(r)
            res[kind][0] &= bool(np.all(np.diff(h[o]) >= -1e-6))
            res[kind][1] &= bool(np.ptp(h) <= 2 * math.sqrt(G) + 1e-9)
            res[kind][2] = max(res[kind][2], np.ptp(h) / (2 * math.sqrt(G)))
    for kind, (rk, osc, mx) in res.items():
        report(f"observation: {kind} finite-G minimizer rank-preserving & osc <= 2 sqrt(G)",
               rk and osc, f"(max osc/bound {mx:.2f})")


if __name__ == "__main__":
    print("--- Part I ---")
    check_finite_G_target()
    check_phi_formula_and_bounds()
    mG = check_late_phase_and_mG()
    check_onpolicy_threshold()
    check_cgf_and_fisher()
    print("--- Part II ---")
    check_balancing()
    check_gauss_newton()
    check_zeta_lemma()
    check_global_convergence()
    check_scale_free_rate()
    check_reliability()
    check_monotone()
    check_branch_invariance()
    check_robust_d_finite_G()
    fig_squashing(mG)
    fig_separation()
    print(f"\n{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)
