"""Numerical checks of every claim in the paper's theory section.

Each check prints PASS/FAIL with the measured error; the script exits nonzero
if any check fails. Checks 1-3 run the repository's actual loss code
(losses/loss_functions.py) on a tabular single-token policy, so they verify
the IMPLEMENTATION, not just the algebra. CPU only, ~1 minute.

    python analysis/theory_checks.py [--fig paper/figures/theory_toy.pdf]
"""
import argparse, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from losses.loss_functions import _rrebel_core, grpo_v2_loss

torch.set_default_dtype(torch.float64)
FAILS = []


def report(name, err, tol):
    ok = err <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: err={err:.3e} (tol {tol:.0e})")
    if not ok:
        FAILS.append(name)


def psi(kind, e, delta=1.0):
    return {'l2': 2 * e, 'l1': torch.sign(e),
            'huber': e.clamp(-delta, delta)}[kind]


# --------------------------------------------------------------------------
# Check 1 (Prop. "first-order equivalence"): at pi_theta = pi_ref the R-REBEL
# gradient equals a policy gradient with advantage u_i = sum_j psi(rt_i - rt_j).
# --------------------------------------------------------------------------
def check_first_order(S=6, G=16, V=40, beta=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(S, V, generator=g)
    acts = torch.stack([torch.multinomial(F.softmax(base[s], -1), G, True, generator=g)
                        for s in range(S)])                       # [S, G]
    scores = (torch.rand(S, G, generator=g) * 10 ** torch.randn(S, 1, generator=g))
    lam = torch.rand(S, generator=g)
    K = G * (G - 1) // 2
    for kind in ('l2', 'l1', 'huber'):
        theta = base.clone().requires_grad_(True)
        logits = theta.repeat_interleave(G, 0).unsqueeze(1)       # [S*G, 1, V]
        loss, _ = _rrebel_core(lmbda=lam, logits=logits, logits_=logits.detach(),
                               actions=acts.view(-1, 1), scores_tensor=scores.view(-1),
                               num_src=S, d_kind=kind, beta=beta, score_scale=0.1,
                               reward_std_scale=True, ent_coef=0.0, huber_delta=1.0)
        loss.backward()
        # analytic: -(beta/(S K)) sum_i u_i grad log pi(z_i)
        from losses.loss_functions import score_scaler_fnc
        sc = 0.1 * score_scaler_fnc(lam.view(S, 1)) * scores
        rt = sc / sc.std(-1, keepdim=True)
        diff = rt.unsqueeze(-1) - rt.unsqueeze(-2)                 # [S, G, G] rt_i - rt_j
        u = psi(kind, diff).sum(-1)                                # diagonal psi(0)=0
        p = F.softmax(base, -1)
        glog = F.one_hot(acts, V).double() - p.unsqueeze(1)        # [S, G, V]
        pred = -(beta / (S * K)) * (u.unsqueeze(-1) * glog).sum(1)
        rel = (theta.grad - pred).norm() / pred.norm()
        report(f"first-order gradient identity, d={kind}", rel.item(), 1e-8)
        if kind == 'l1':   # centered ranks, and invariance to monotone maps
            ranks = rt.argsort(-1).argsort(-1).double() + 1
            report("  l1 advantage = 2*rank - (G+1)",
                   (u - (2 * ranks - (G + 1))).abs().max().item(), 1e-12)
        if kind == 'l2':   # direction equals GRPO's
            th2 = base.clone().requires_grad_(True)
            lg2 = th2.repeat_interleave(G, 0).unsqueeze(1)
            l2, _ = grpo_v2_loss(lmbda=lam, logits=lg2, logits_=lg2.detach(),
                                 actions=acts.view(-1, 1), scores_tensor=scores.view(-1),
                                 content_tensor=None, style_tensor=None, num_src=S, beta=0.0)
            l2.backward()
            cos = F.cosine_similarity(theta.grad.flatten(), th2.grad.flatten(), 0)
            report("  l2-std gradient parallel to GRPO gradient (1-cos)",
                   1 - cos.item(), 1e-10)
            ratio = (theta.grad.norm() / th2.grad.norm()).item()
            want = 4 * beta * G / (G - 1)   # = 2*beta*G/K * G ... per-token mean, T=1
            report(f"  magnitude ratio = 4*beta*G/(G-1) = {want:.4f}",
                   abs(ratio - want) / want, 1e-8)


# --------------------------------------------------------------------------
# Check 2 (Lemma "invariance"): the std loss is invariant to r -> a r + b per
# group (a > 0); the unstandardized loss is not.
# --------------------------------------------------------------------------
def check_invariance(S=5, G=16, V=30, seed=1):
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(S, V, generator=g)
    ref = theta + 0.3 * torch.randn(S, V, generator=g)
    acts = torch.randint(0, V, (S * G, 1), generator=g)
    r = torch.randn(S, G, generator=g)
    a = torch.exp(3 * torch.randn(S, 1, generator=g)); b = 5 * torch.randn(S, 1, generator=g)
    lam = torch.rand(S, generator=g)
    lg, lg_ = theta.repeat_interleave(G, 0).unsqueeze(1), ref.repeat_interleave(G, 0).unsqueeze(1)
    for std in (True, False):
        L = [_rrebel_core(lmbda=lam, logits=lg, logits_=lg_, actions=acts,
                          scores_tensor=x.view(-1), num_src=S, d_kind='l1', beta=0.5,
                          score_scale=0.1, reward_std_scale=std, ent_coef=0.0)[0].item()
             for x in (r, a * r + b)]
        err = abs(L[0] - L[1]) / abs(L[0])
        if std:
            report("std loss invariant to per-group affine reward maps", err, 1e-9)
        else:
            ok = err > 1e-3
            print(f"[{'PASS' if ok else 'FAIL'}] unstandardized loss is NOT invariant: rel change={err:.3f}")
            if not ok:
                FAILS.append("non-invariance")


# --------------------------------------------------------------------------
# Check 3 (Lemma "bounded targets"): |rt_i - rt_j| <= sqrt(2(G-1)), tight.
# --------------------------------------------------------------------------
def check_bounded(G=16, trials=20000, seed=2):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(trials, G, generator=g) ** 3 * 10 ** (4 * torch.rand(trials, 1, generator=g))
    rt = x / x.std(-1, keepdim=True)
    worst = (rt.max(-1).values - rt.min(-1).values).max().item()
    bound = math.sqrt(2 * (G - 1))
    extremal = torch.zeros(G); extremal[0], extremal[1] = 1.0, -1.0
    tight = ((extremal[0] - extremal[1]) / extremal.std()).item()
    report(f"bounded targets: max over {trials} heavy-tailed groups <= {bound:.3f}",
           max(0.0, worst - bound), 1e-9)
    report("  bound attained by the two-point group", abs(tight - bound), 1e-12)


# --------------------------------------------------------------------------
# Check 4 (Theorem "consistency"): for ANY full-support sampler q and ANY d,
# the population minimizer is pi_ref * exp(r / (beta * s)), s = 1 or sigma_q.
# --------------------------------------------------------------------------
def pop_loss(logit, ref_logp, r, q, beta, s, kind):
    h = beta * (F.log_softmax(logit, -1) - ref_logp)
    res = (h.unsqueeze(1) - h.unsqueeze(0)) - (r.unsqueeze(1) - r.unsqueeze(0)) / s
    rho = {'l2': res ** 2, 'l1': res.abs(), 'huber': F.huber_loss(res, torch.zeros_like(res), reduction='none'),
           'cauchy': torch.log1p(res ** 2)}[kind]
    return (q.unsqueeze(1) * q.unsqueeze(0) * rho).sum()


def check_consistency(V=12, beta=0.5, seed=3):
    g = torch.Generator().manual_seed(seed)
    r = 30 * torch.rand(V, generator=g)
    ref_logp = F.log_softmax(torch.randn(V, generator=g), -1)
    # a perturbed sampler: ref with its top-3 prompts suppressed, mixed with uniform
    q = ref_logp.exp().clone(); q[q.argsort()[-3:]] *= 0.05; q = 0.8 * q / q.sum() + 0.2 / V
    sig = torch.sqrt((q * (r - (q * r).sum()) ** 2).sum())
    for std in (False, True):
        s = sig if std else torch.tensor(1.0)
        target = F.log_softmax(ref_logp + r / (beta * s), -1)
        for kind in ('l2', 'l1', 'huber', 'cauchy'):
            logit = torch.zeros(V, requires_grad=True)
            opt = torch.optim.LBFGS([logit], lr=1, max_iter=2000, tolerance_grad=1e-14,
                                    tolerance_change=1e-16, line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad(); L = pop_loss(logit, ref_logp, r, q, beta, s, kind); L.backward(); return L
            for _ in range(5):
                opt.step(closure)
            got = F.log_softmax(logit.detach(), -1)
            kl = (got.exp() * (got - target)).sum().item()
            report(f"consistency ({'std' if std else 'plain'}, d={kind}, perturbed q): KL(min || predicted)",
                   abs(kl), 1e-6 if kind != 'l1' else 1e-4)


# --------------------------------------------------------------------------
# Check 5 (Prop. "uniform trust region"): KL of the std target depends only on
# the SHAPE of the reward distribution; the plain target's KL scales ~sigma^2.
# --------------------------------------------------------------------------
def tilt_stats(r, p0, t):
    lw = torch.log(p0) + t * r; logp = F.log_softmax(lw, -1); p = logp.exp()
    return (p * (logp - torch.log(p0))).sum().item(), p   # log-space: no 0*log0


def check_trust_region(beta=0.5, V=400, seed=4):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(V, generator=g); u = (u - u.mean()) / u.std(unbiased=False)
    p0 = torch.full((V,), 1.0 / V)
    kls_std, kls_plain = [], []
    for scale in 10.0 ** np.arange(-3, 2.5, 0.5):
        r = 7.0 + scale * u
        sig = torch.sqrt((p0 * (r - (p0 * r).sum()) ** 2).sum())
        kls_std.append(tilt_stats(r, p0, 1 / (beta * sig))[0])
        kls_plain.append(tilt_stats(r, p0, 1 / beta)[0])
    spread = max(kls_std) - min(kls_std)
    report(f"std-target KL constant across 5.5 decades of reward scale (KL={kls_std[0]:.4f})",
           spread, 1e-10)
    # small-scale regime: plain KL ~ sigma^2/(2 beta^2)
    small = 1e-3
    pred = small ** 2 / (2 * beta ** 2)
    report("plain-target KL ~ sigma^2/(2 beta^2) at small sigma", abs(kls_plain[0] - pred) / pred, 1e-2)
    # large-beta expansion of the std KL: 1/(2b^2) + skew/(3 b^3)
    for b in (5.0, 10.0):
        kl, _ = tilt_stats(u, p0, 1 / b)
        skew = (p0 * u ** 3).sum().item()
        pred = 1 / (2 * b ** 2) + skew / (3 * b ** 3)
        report(f"  KL expansion 1/(2b^2)+skew/(3b^3) at beta={b:g}", abs(kl - pred) / pred, 5e-3 if b == 5 else 1e-3)
    # exact value at beta = 0.5 for Gaussian-shaped rewards is 1/(2 beta^2) = 2
    xs = torch.linspace(-9, 9, 20001); pg = torch.exp(-xs ** 2 / 2); pg = pg / pg.sum()
    report("Gaussian shape: exact KL = 1/(2 beta^2) = 2 at beta=0.5",
           abs(tilt_stats(xs, pg, 1 / beta)[0] - 2.0), 1e-4)
    return kls_std, kls_plain


# --------------------------------------------------------------------------
# Check 6 (Prop. "pairwise differencing symmetrizes noise"): with iid skewed,
# heavy-tailed reward noise, the pairwise-l1 fit recovers the true reward
# difference; a pointwise-l1 fit of r itself is biased by the noise median.
# --------------------------------------------------------------------------
def check_symmetrization(n=400000, seed=5):
    rng = np.random.default_rng(seed)
    d_true = 1.7
    eps_i = rng.lognormal(0, 1.2, n) - math.exp(0.72)    # zero-mean, skewed, heavy
    eps_j = rng.lognormal(0, 1.2, n) - math.exp(0.72)
    pair_median = np.median(d_true + eps_i - eps_j)      # l1 pairwise minimizer
    point_median = np.median(eps_i)                      # l1 pointwise bias
    report("pairwise l1 recovers true reward difference under skewed iid noise",
           abs(pair_median - d_true), 2e-2)
    ok = abs(point_median) > 0.5
    print(f"[{'PASS' if ok else 'FAIL'}] pointwise l1 is biased under the same noise: median offset={point_median:.3f}")
    if not ok:
        FAILS.append("pointwise bias")
    # heteroscedastic but SYMMETRIC noise: still unbiased
    s_i, s_j = 0.3, 4.0
    hs = np.median(d_true + rng.standard_t(2, n) * s_i - rng.standard_t(2, n) * s_j)
    report("pairwise l1 unbiased under heteroscedastic symmetric (t_2) noise", abs(hs - d_true), 3e-2)


# --------------------------------------------------------------------------
# Check 7 (Prop. "mirror descent"): windowed updates with a lagged reference.
# Contexts share a reward SHAPE but differ in scale by 10^4. Std: same KL per
# window and same progress in every context. Plain: small-scale contexts stall.
# --------------------------------------------------------------------------
def check_mirror_descent(beta=0.5, V=200, windows=40, seed=6):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(V, generator=g)
    scales = [1e-2, 1e0, 1e2]
    out = {}
    for mode in ('std', 'plain'):
        curves = []
        for sc in scales:
            r = sc * u; p = torch.full((V,), 1.0 / V); traj = [p[u.argmax()].item()]; kls = []
            for _ in range(windows):
                sig = torch.sqrt((p * (r - (p * r).sum()) ** 2).sum())
                # a point-mass policy has sigma=0: every reward difference in a
                # group is 0, so the code's targets are 0 and the policy stays put
                t = (1 / (beta * sig) if sig > 1e-12 else 0.0) if mode == 'std' else 1 / beta
                kl, p = tilt_stats(r, p, t); kls.append(kl); traj.append(p[u.argmax()].item())
            curves.append((sc, traj, kls))
        out[mode] = curves
    first_kl = [c[2][0] for c in out['std']]
    report("mirror descent (std): first-window KL identical across scales", max(first_kl) - min(first_kl), 1e-9)
    trajs = [c[1] for c in out['std']]
    report("mirror descent (std): identical progress across scales",
           max(abs(a - b) for a, b in zip(trajs[0], trajs[-1])), 1e-9)
    stall = out['plain'][0][1][-1]; fast = out['plain'][-1][1][-1]
    ok = stall < 0.05 and fast > 0.95
    print(f"[{'PASS' if ok else 'FAIL'}] mirror descent (plain): small-scale context stalls "
          f"(p*={stall:.3f}) while large-scale context saturates (p*={fast:.3f})")
    if not ok:
        FAILS.append("plain stall")
    return out


# --------------------------------------------------------------------------
# Check 8 (paragraph "does the choice between l1 and Huber matter for
# settling?"). Deterministic population loss + constant-step Adam: Huber
# converges, l1 stalls at a small floor. Sampled groups + in-group std + noisy
# rewards: the two have indistinguishable stationary jitter (noise dominates).
# --------------------------------------------------------------------------
def _stoch_jitter(kind, seed, V=30, G=16, beta=0.5, lr=1e-3, steps=8000, noise=0.3, S=15):
    g = torch.Generator().manual_seed(seed)
    r = 3 * torch.randn(V, generator=g); ref = F.log_softmax(torch.randn(V, generator=g), -1)
    logit = ref.clone().requires_grad_(True); opt = torch.optim.Adam([logit], lr=lr)
    q = F.softmax(ref, -1); i, j = torch.triu_indices(G, G, 1); hist = []
    for t in range(steps):
        opt.zero_grad()
        z = torch.multinomial(q.expand(S, V), G, replacement=True, generator=g)
        rh = r[z] + noise * r.std() * torch.randn(S, G, generator=g)
        rt = rh / rh.std(-1, keepdim=True)
        h = beta * (F.log_softmax(logit, -1) - ref)[z]
        res = (h[:, i] - h[:, j]) - (rt[:, i] - rt[:, j])
        (res.abs().mean() if kind == 'l1' else F.huber_loss(res, torch.zeros_like(res))).backward()
        opt.step()
        if t >= steps - 2000:
            hist.append(F.log_softmax(logit.detach(), -1))
    H = torch.stack(hist); m = torch.logsumexp(H, 0) - math.log(len(H)); m = m - torch.logsumexp(m, 0)
    return float(np.mean([(h.exp() * (h - m)).sum().item() for h in H]))


def check_self_limiting(V=12, beta=0.5, steps=6000, lr=3e-3, seed=7):
    g = torch.Generator().manual_seed(seed)
    r = 3 * torch.randn(V, generator=g)
    ref_logp = F.log_softmax(torch.randn(V, generator=g), -1)
    q = torch.full((V,), 1.0 / V)
    s = torch.sqrt((q * (r - (q * r).sum()) ** 2).sum())
    target = F.log_softmax(ref_logp + r / (beta * s), -1)
    err = {}
    for kind in ('huber', 'l1'):
        logit = ref_logp.clone().requires_grad_(True)
        opt = torch.optim.Adam([logit], lr=lr)
        tail = []
        for t in range(steps):
            opt.zero_grad(); pop_loss(logit, ref_logp, r, q, beta, s, kind).backward(); opt.step()
            if t >= steps - 500:
                got = F.log_softmax(logit.detach(), -1)
                tail.append((got.exp() * (got - target)).sum().item())
        err[kind] = float(np.mean(tail))
    ok = err['huber'] < err['l1']
    print(f"[{'PASS' if ok else 'FAIL'}] deterministic, constant-step Adam: Huber KL-to-target {err['huber']:.1e} "
          f"< l1 {err['l1']:.1e}")
    if not ok:
        FAILS.append("self-limiting (deterministic)")
    jh = np.mean([_stoch_jitter('huber', sd) for sd in range(4)])
    jl = np.mean([_stoch_jitter('l1', sd) for sd in range(4)])
    ok = 0.5 < jl / jh < 2.0
    print(f"[{'PASS' if ok else 'FAIL'}] stochastic groups: stationary jitter l1 {jl:.1e} vs Huber {jh:.1e} "
          f"(ratio {jl / jh:.2f}, within 2x = indistinguishable)")
    if not ok:
        FAILS.append("self-limiting (stochastic)")
    return err, jh, jl

# --------------------------------------------------------------------------
# Constraint theory (Section "Which constraint should the reward impose?")
# Check C1 (Prop. "eps-constraint over distributions"): on a finite menu the
# best expected sentiment subject to E[c] >= tau is attained by mixing <= 2
# prompts, meets the floor with EQUALITY when it binds, and is certified
# optimal by an LP dual (mu*, v): s_z + mu*(c_z - tau) <= v for all z, with
# equality on the support -- an independent optimality proof, not a re-run of
# the construction.
# --------------------------------------------------------------------------
def eps_constraint_opt(c, s, tau):
    """Best mixture by enumeration of single prompts and pairs bracketing tau."""
    best, sol = -np.inf, None
    for i in range(len(c)):
        if c[i] >= tau and s[i] > best:
            best, sol = s[i], {i: 1.0}
    lo, hi = np.where(c < tau)[0], np.where(c >= tau)[0]
    for i in lo:
        for j in hi:
            a = (c[j] - tau) / (c[j] - c[i])
            v = a * s[i] + (1 - a) * s[j]
            if v > best:
                best, sol = v, {i: a, j: 1 - a}
    return best, sol


def check_eps_constraint(trials=2000, K=12, seed=11):
    rng = np.random.default_rng(seed)
    worst_dual, worst_eq, max_supp = 0.0, 0.0, 0
    for _ in range(trials):
        c = rng.uniform(10, 95, K); s = np.clip(100 - c + rng.normal(0, 15, K), 0, 100)
        c0 = c[s == s.max()].max()        # largest content among the most positive prompts
        tau = rng.uniform(c0, c.max())      # a binding floor
        v, sol = eps_constraint_opt(c, s, tau)
        supp = list(sol); max_supp = max(max_supp, len(supp))
        cm = sum(w * c[i] for i, w in sol.items())
        worst_eq = max(worst_eq, abs(cm - tau))
        # dual certificate: the multiplier is the slope of the chord used
        mu = 0.0 if len(supp) == 1 else (s[supp[0]] - s[supp[1]]) / (c[supp[1]] - c[supp[0]])
        lag = s + mu * (c - tau)
        worst_dual = max(worst_dual, max(0.0, lag.max() - v), abs(lag[supp].max() - v), max(0.0, -mu))
    report("eps-constraint optimum certified by an LP dual (max violation)", worst_dual, 1e-9)
    report("eps-constraint optimum meets a binding floor with equality, |E[c]-tau|", worst_eq, 1e-9)
    ok = max_supp <= 2
    print(f"[{'PASS' if ok else 'FAIL'}] eps-constraint optimum mixes at most 2 prompts (max support {max_supp})")
    if not ok:
        FAILS.append("support<=2")


# --------------------------------------------------------------------------
# Check C2 (Prop. "penalized fixed point"): the maximizer of
#   J(pi) = E_pi[s] - (nu/2)(tau - E_pi[c])_+^2 - beta KL(pi || pi0)
# is the Gibbs policy pi0 exp((s + mu c)/beta) with mu = nu (tau - E_pi[c])_+,
# found by 1-D bisection; it matches direct autograd maximization of J; its
# multiplier never exceeds mu*, the multiplier of the KL-regularized constrained
# problem, so the shortfall is <= mu*/nu.
# --------------------------------------------------------------------------
def _gibbs(p0, s, c, mu, beta):
    lw = np.log(p0) + (s + mu * c) / beta
    w = np.exp(lw - lw.max()); return w / w.sum()


def fixed_point(p0, s, c, tau, nu, beta):
    lo, hi = 0.0, 1e4               # g(mu) = mu - nu (tau - C(pi_mu))_+ is increasing
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid - nu * max(0.0, tau - _gibbs(p0, s, c, mid, beta) @ c) < 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def check_penalized_fixed_point(trials=30, K=10, beta=0.5, seed=12):
    rng = np.random.default_rng(seed)
    kl_max, rel_max, bound_viol, kkt_max, conv = 0.0, 0.0, 0.0, 0.0, []
    for t in range(trials):
        c = rng.uniform(10, 95, K); s = np.clip(100 - c + rng.normal(0, 15, K), 0, 100)
        p0 = rng.dirichlet(np.ones(K)); tau = rng.uniform(40, 85)
        # mu*: KL-regularized constrained problem, C(pi_mu*) = tau (bisection)
        lo, hi = 0.0, 1e4
        for _ in range(200):
            mid = (lo + hi) / 2
            if _gibbs(p0, s, c, mid, beta) @ c < tau: lo = mid
            else: hi = mid
        mu_star = (lo + hi) / 2
        for nu in (1.0, 5.0, 50.0):
            mu = fixed_point(p0, s, c, tau, nu, beta); pi = _gibbs(p0, s, c, mu, beta)
            short = tau - pi @ c
            rel_max = max(rel_max, abs(max(0.0, short) - mu / nu) / max(1.0, mu / nu))
            bound_viol = max(bound_viol, max(0.0, mu - mu_star))
            if t < 10 and nu in (1.0, 50.0):
                conv.append((nu, short, mu_star / nu))
            # J is concave, so the KKT condition is a sufficient optimality
            # certificate: grad J must be constant across prompts (pi > 0 on all)
            g_ = s + nu * max(0.0, tau - pi @ c) * c - beta * (np.log(pi) - np.log(p0))
            kkt_max = max(kkt_max, float(g_.max() - g_.min()))
            if t < 6 and nu == 1.0:
                # independent: exponentiated-gradient ascent on J from pi0 (only
                # grad J, never the closed form); its step is stable at nu = 1
                q = p0.copy()
                for it in range(400000):
                    grad = s + nu * max(0.0, tau - q @ c) * c - beta * (np.log(q) - np.log(p0))
                    lq = np.log(q) + 2e-3 * grad
                    q_new = np.exp(lq - lq.max()); q_new /= q_new.sum()
                    if np.abs(q_new - q).max() < 1e-15:
                        q = q_new; break
                    q = q_new
                kl_max = max(kl_max, float((pi * (np.log(pi) - np.log(q))).sum()))
    report("penalized objective: fixed point satisfies KKT (spread of grad J)", kkt_max, 1e-6)
    report("penalized objective: fixed point = exponentiated-gradient maximizer (nu=1), KL", abs(kl_max), 1e-8)
    report("penalized objective: shortfall = mu/nu (rel.)", rel_max, 1e-8)
    report("penalized objective: mu_nu <= mu* (max excess)", bound_viol, 1e-8)
    ok = all(sh <= b + 1e-9 for _, sh, b in conv)
    print(f"[{'PASS' if ok else 'FAIL'}] shortfall <= mu*/nu on every instance; "
          f"median shortfall nu=1: {np.median([x[1] for x in conv if x[0] == 1.0]):.3f}, "
          f"nu=50: {np.median([x[1] for x in conv if x[0] == 50.0]):.4f}")
    if not ok:
        FAILS.append("shortfall bound")


# --------------------------------------------------------------------------
# Check C3 (Lemma "invariance" applied to the v5 reward): the std loss is the
# same for r = s + mu_g (c - tau) and for its per-group rescaling by 1/(1+mu_g)
# used in tst_score.py, so only the content weight mu_g matters.
# --------------------------------------------------------------------------
def check_v5_reward_invariance(S=5, G=16, V=30, seed=13):
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(S, V, generator=g); ref = theta + 0.3 * torch.randn(S, V, generator=g)
    acts = torch.randint(0, V, (S * G, 1), generator=g)
    cc = 100 * torch.rand(S, G, generator=g); ss = 100 * torch.rand(S, G, generator=g)
    tau = 100 * torch.rand(S, 1, generator=g); mu = 5 * torch.clamp(tau - cc.mean(1, keepdim=True), min=0)
    lam = tau.view(-1) / 100
    lg, lg_ = theta.repeat_interleave(G, 0).unsqueeze(1), ref.repeat_interleave(G, 0).unsqueeze(1)
    L = [_rrebel_core(lmbda=lam, logits=lg, logits_=lg_, actions=acts, scores_tensor=x.reshape(-1),
                      num_src=S, d_kind='huber', beta=0.5, score_scale=0.1, reward_std_scale=True,
                      ent_coef=0.0)[0].item()
         for x in (ss + mu * cc, (ss + mu * (cc - tau)) / (1 + mu))]
    report("v5 reward: std loss unchanged by the (1+mu) rescaling and tau shift", abs(L[0] - L[1]) / abs(L[0]), 1e-9)


def make_figure(path, kl_std, kl_plain, md):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.55))  # print size
    sc = 10.0 ** np.arange(-3, 2.5, 0.5)
    ax[0].loglog(sc, kl_plain, "o-", color=ORANGE, label="plain: $\\pi_{\\rm ref}e^{r/\\beta}$")
    ax[0].loglog(sc, kl_std, "s-", color=BLUE, label="std: $\\pi_{\\rm ref}e^{r/(\\beta\\sigma)}$")
    ax[0].set_xlabel("reward scale $\\sigma$ (same shape)"); ax[0].set_ylabel("KL(target $\\|$ ref)  [nats]")
    ax[0].set_title("one refresh window", fontsize=8.5); ax[0].legend(frameon=False, fontsize=6.5)
    for (s, traj, _), ls in zip(md['plain'], (":", "--", "-")):
        ax[1].plot(traj, ls, color=ORANGE, lw=1.6, label=f"plain, $\\sigma$={s:g}")
    # the three std trajectories coincide exactly (Prop. mirror descent)
    ax[1].plot(md['std'][0][1], "-", color=BLUE, lw=3.2, alpha=0.8,
               label="std, every $\\sigma$ (identical)")
    ax[1].set_xlabel("refresh window $k$"); ax[1].set_ylabel("mass on best prompt")
    ax[1].set_title("lagged reference = mirror descent", fontsize=8.5)
    ax[1].legend(frameon=False, fontsize=6.5, loc='center right')
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(path); fig.savefig(os.path.splitext(path)[0] + ".png", dpi=110); print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fig", default=None); a = ap.parse_args()
    # The propositions are exact for the idealized std (no epsilon guard). The
    # code adds STD_EPS=1e-8 to the denominator, which perturbs scale
    # invariance by O(eps/sigma) -- ~1e-7 on these toy scales. Run the exact
    # checks with the guard removed; the guard itself is a numerical safety
    # net for all-tied groups (sigma=0), where the target is 0 either way.
    import losses.loss_functions as lf
    lf.STD_EPS = 0.0
    check_first_order(); check_invariance(); check_bounded(); check_consistency()
    kls_std, kls_plain = check_trust_region(); check_symmetrization(); md = check_mirror_descent()
    check_self_limiting()
    check_eps_constraint(); check_penalized_fixed_point(); check_v5_reward_invariance()
    if a.fig:
        make_figure(a.fig, kls_std, kls_plain, md)
    print("\nALL CHECKS PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
