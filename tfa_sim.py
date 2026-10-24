"""Temporal Fairness-Attribution (TFA): synthetic experiments.
Features per window: x1 prior-antibiotic exposure, x2 culture trend,
x3 genotype marker, x4 vital-sign trend. Groups a (drifting) / b (reference).
"""
import numpy as np, json, sys, time
from itertools import combinations
from math import factorial

W, D, N = 5, 4, 6000
B0 = -0.4
KAPPA = 0.5
BASE_BETA = np.array([0.8, 0.6, 1.0, 0.4])
# ---- Shapley machinery (exact, d=4) ----
MASKS = np.array([[(s >> j) & 1 for j in range(D)] for s in range(2 ** D)], float)  # 16x4
SIZES = MASKS.sum(1).astype(int)
def shap_weights():
    # w[i][S] weight for S not containing i, index of S|{i}
    out = []
    for i in range(D):
        lst = []
        for s in range(2 ** D):
            if (s >> i) & 1: continue
            k = bin(s).count("1")
            lst.append((s, s | (1 << i), factorial(k) * factorial(D - k - 1) / factorial(D)))
        out.append(lst)
    return out
SW = shap_weights()

def shapley_from_values(V):
    """V: (..., 16) coalition values -> (..., 4) Shapley values"""
    phi = np.zeros(V.shape[:-1] + (D,))
    for i in range(D):
        for s, s2, w in SW[i]:
            phi[..., i] += w * (V[..., s2] - V[..., s])
    return phi

def sigmoid(z): return 1 / (1 + np.exp(-z))

def fit_logit(X, y, iters=25):
    Xa = np.hstack([np.ones((len(X), 1)), X]); b = np.zeros(Xa.shape[1])
    for _ in range(iters):
        p = sigmoid(Xa @ b); g = Xa.T @ (p - y)
        H = (Xa * (p * (1 - p))[:, None]).T @ Xa + 1e-8 * np.eye(len(b))
        step = np.linalg.solve(H, g); b -= step
        if np.abs(step).max() < 1e-8: break
    return b[0], b[1:]

def coalition_probs(X, b0, beta, xbar):
    """P[n,16]: p_hat with features outside S set to baseline xbar (single mean point)."""
    C = X * beta; cb = xbar * beta
    L = b0 + C @ MASKS.T + cb @ (1 - MASKS).T
    return sigmoid(L)

def coalition_probs_bg(X, b0, beta, Bg):
    """marginal (background-sample) masking: average p over K background rows."""
    C = X * beta; Cb = Bg * beta  # (n,4),(K,4)
    out = np.zeros((len(X), 16))
    for s in range(16):
        m = MASKS[s]
        Ls = b0 + (C @ m)[:, None] + (Cb @ (1 - m))[None, :]
        out[:, s] = sigmoid(Ls).mean(1)
    return out

def disparity_and_phi(P, y, g):
    pa = (y == 1) & (g == 1); pb = (y == 1) & (g == 0)
    V = P[pa].mean(0) - P[pb].mean(0)          # 16 coalition values, v(empty)=0
    return V[-1], shapley_from_values(V), V

def pred_shap_diff(P, y, g):
    """group-differential per-prediction Shapley (same masking, Y=1 conditioning)"""
    Phi = shapley_from_values(P)               # n x 4 per-prediction Shapley values
    pa = (y == 1) & (g == 1); pb = (y == 1) & (g == 0)
    return Phi[pa].mean(0) - Phi[pb].mean(0)

# ---- data generation ----
def gen(seed, scen, n=N):
    r = np.random.default_rng(seed)
    g = (r.random(n) < 0.5).astype(int)       # 1 = group a (drifting), 0 = group b (reference)
    Xs, Ys, Bs = [], [], []
    rho = 0.6 if scen == "S5" else 0.0
    for w in range(W):
        Z = r.standard_normal((n, D))
        if rho > 0:                            # corr(x2, x3) = rho within window
            Z[:, 2] = rho * Z[:, 1] + np.sqrt(1 - rho ** 2) * Z[:, 2]
        if scen == "S3":                       # covariate shift only (group a, x1 mean grows)
            Z[:, 0] += g * 0.30 * w
        beta = np.tile(BASE_BETA, (n, 1))
        if scen in ("S1", "S5"):
            beta[:, 2] = np.where(g == 1, 1.0 + KAPPA * w, 1.0)
        if scen == "S2":
            beta[:, 2] = np.where(g == 1, 1.0 + 0.4 * w, 1.0)
            beta[:, 1] = np.where((g == 1) & (w >= 2), 0.6 + 1.0, 0.6)
        logit = B0 + (Z * beta).sum(1)
        y = (r.random(n) < sigmoid(logit)).astype(int)
        if scen == "S4":                       # label noise in group b? -> group a labels flipped
            flip = (g == 1) & (r.random(n) < 0.06 * w)
            y = np.where(flip, 1 - y, y)
        Xs.append(Z); Ys.append(y)
    return np.stack(Xs), np.stack(Ys), g       # X: W x n x D, Y: W x n

def split(n, seed):
    r = np.random.default_rng(10_000 + seed); idx = r.permutation(n)
    return idx[: n // 2], idx[n // 2:]

# ---- per-window TFA (+ bootstrap) ----
def window_stats(Xtr, ytr, Xte, yte, gte, mode="mean", Bg=None):
    b0, beta = fit_logit(Xtr, ytr)
    xbar = Xtr.mean(0)
    P = coalition_probs(Xte, b0, beta, xbar) if mode == "mean" else coalition_probs_bg(Xte, b0, beta, Bg)
    delta, phi, V = disparity_and_phi(P, yte, gte)
    # logit-output group-differential (per-prediction SHAP on log-odds, exact for linear logit)
    pa = (yte == 1) & (gte == 1); pb = (yte == 1) & (gte == 0)
    psi_logit = beta * (Xte[pa].mean(0) - Xte[pb].mean(0))
    psi_prob = pred_shap_diff(P, yte, gte)
    return dict(delta=delta, phi=phi, psi_logit=psi_logit, psi_prob=psi_prob, beta=beta)

def run_scenario(scen, seeds, B=100, do_boot=True):
    res = []
    for sd in seeds:
        X, Y, g = gen(sd, scen); tr, te = split(N, sd)
        r = np.random.default_rng(777 + sd)
        rec = dict(seed=sd, delta=[], phi=[], psi_logit=[], psi_prob=[], ci=[])
        for w in range(W):
            s = window_stats(X[w][tr], Y[w][tr], X[w][te], Y[w][te], g[te])
            for k in ("delta", "phi", "psi_logit", "psi_prob"):
                rec[k].append(np.asarray(s[k]).tolist())
            if do_boot:
                bd, bp, bl = [], [], []
                for _ in range(B):
                    itr = r.integers(0, len(tr), len(tr)); ite = r.integers(0, len(te), len(te))
                    sb = window_stats(X[w][tr][itr], Y[w][tr][itr], X[w][te][ite], Y[w][te][ite], g[te][ite])
                    bd.append(sb["delta"]); bp.append(sb["phi"]); bl.append(sb["psi_logit"])
                bd, bp, bl = np.array(bd), np.array(bp), np.array(bl)
                rec["ci"].append(dict(delta=np.percentile(bd, [2.5, 97.5]).tolist(),
                                      phi=np.percentile(bp, [2.5, 97.5], axis=0).T.tolist(),
                                      psi_logit=np.percentile(bl, [2.5, 97.5], axis=0).T.tolist()))
        res.append(rec)
    return res

# ---- static baselines ----
def baselines(scen, seeds):
    out = dict(collapsed=[], pooled=[], equiv=[], bgdiff=[], base_sens=[])
    for sd in seeds:
        X, Y, g = gen(sd, scen); tr, te = split(N, sd)
        # B1: collapse windows: features averaged, outcome = resistance in any window
        Xc = X.mean(0); yc = (Y.max(0))
        b0, beta = fit_logit(Xc[tr], yc[tr]); P = coalition_probs(Xc[te], b0, beta, Xc[tr].mean(0))
        d, phi, _ = disparity_and_phi(P, yc[te], g[te]); out["collapsed"].append(dict(delta=d, phi=phi.tolist()))
        # B2: pooled long format (one model for all windows, no window index)
        Xl = np.concatenate([X[w][tr] for w in range(W)]); yl = np.concatenate([Y[w][tr] for w in range(W)])
        Xt = np.concatenate([X[w][te] for w in range(W)]); yt = np.concatenate([Y[w][te] for w in range(W)])
        gt = np.tile(g[te], W)
        b0, beta = fit_logit(Xl, yl); P = coalition_probs(Xt, b0, beta, Xl.mean(0))
        d, phi, _ = disparity_and_phi(P, yt, gt); out["pooled"].append(dict(delta=d, phi=phi.tolist()))
        # equivalence TFA == group-differential per-prediction Shapley (same masking, Y=1 conditioning)
        mx = 0.0
        for w in range(W):
            b0, beta = fit_logit(X[w][tr], Y[w][tr]); P = coalition_probs(X[w][te], b0, beta, X[w][tr].mean(0))
            d, phi, _ = disparity_and_phi(P, Y[w][te], g[te]); psi = pred_shap_diff(P, Y[w][te], g[te])
            mx = max(mx, np.abs(phi - psi).max(), abs(phi.sum() - d))
        out["equiv"].append(mx)
        # baseline-sensitivity: marginal background (K=40) vs single mean point, last window
        w = W - 1; b0, beta = fit_logit(X[w][tr], Y[w][tr])
        rr = np.random.default_rng(sd); Bg = X[w][tr][rr.choice(len(tr), 40, replace=False)]
        Pm = coalition_probs(X[w][te], b0, beta, X[w][tr].mean(0)); Pb = coalition_probs_bg(X[w][te], b0, beta, Bg)
        d1, p1, _ = disparity_and_phi(Pm, Y[w][te], g[te]); d2, p2, _ = disparity_and_phi(Pb, Y[w][te], g[te])
        # reference-group (b) mean baseline
        xb = X[w][tr][g[tr] == 0].mean(0); Pr = coalition_probs(X[w][te], b0, beta, xb)
        d3, p3, _ = disparity_and_phi(Pr, Y[w][te], g[te])
        out["base_sens"].append(dict(mean=p1.tolist(), bg=p2.tolist(), refmean=p3.tolist(), dm=d1, db=d2, dr=d3))
    return out

if __name__ == "__main__":
    scen = sys.argv[1]; B = int(sys.argv[2]); seeds = list(range(15))
    t = time.time()
    main = run_scenario(scen, seeds, B=B, do_boot=True)
    base = baselines(scen, seeds)
    json.dump(dict(main=main, base=base), open(f"/home/claude/work/res_{scen}.json", "w"))
    print(scen, "done", round(time.time() - t, 1), "s")
