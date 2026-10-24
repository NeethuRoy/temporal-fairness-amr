"""Extra checks reported in the paper: drift-slope scan, Corollary 1 verification, metric variants."""
import numpy as np, json
import tfa_sim as T
np.set_printoptions(precision=4, suppress=True)
# 1) drift-slope scan (S1), no bootstrap
for k in (0.15, 0.25, 0.35, 0.5):
    T.KAPPA = k
    res = T.run_scenario("S1", list(range(15)), do_boot=False)
    d = np.array([r["delta"] for r in res]); print("kappa", k, "mean delta by window", np.round(d.mean(0), 3))
T.KAPPA = 0.5
# 2) Corollary 1: brute-force log-odds Shapley == beta_i * (class-conditional mean difference)
X, Y, g = T.gen(3, "S1"); idx = np.random.default_rng(10003).permutation(T.N); tr, te = idx[:3000], idx[3000:]
w = 4; b0, beta = T.fit_logit(X[w][tr], Y[w][tr]); xbar = X[w][tr].mean(0)
C = X[w][te] * beta; cb = xbar * beta
L = b0 + C @ T.MASKS.T + cb @ (1 - T.MASKS).T
pa = (Y[w][te] == 1) & (g[te] == 1); pb = (Y[w][te] == 1) & (g[te] == 0)
phi_bf = T.shapley_from_values(L[pa].mean(0) - L[pb].mean(0))
phi_cf = beta * (X[w][te][pa].mean(0) - X[w][te][pb].mean(0))
print("Corollary 1 max abs diff", np.abs(phi_bf - phi_cf).max())
# 3) metric variants at window 5
def variants(scen):
    out = {k: [] for k in ["prob_TPR", "prob_FPR", "thr_TPR", "thr_FPR"]}
    for sd in range(15):
        X, Y, g = T.gen(sd, scen); tr, te = T.split(T.N, sd); w = 4
        b0, beta = T.fit_logit(X[w][tr], Y[w][tr]); P = T.coalition_probs(X[w][te], b0, beta, X[w][tr].mean(0))
        y = Y[w][te]; gg = g[te]
        for name, (Pm, cond) in {"prob_TPR": (P, 1), "prob_FPR": (P, 0), "thr_TPR": ((P >= .5).astype(float), 1), "thr_FPR": ((P >= .5).astype(float), 0)}.items():
            a = (y == cond) & (gg == 1); b = (y == cond) & (gg == 0); V = Pm[a].mean(0) - Pm[b].mean(0)
            out[name].append([V[-1]] + T.shapley_from_values(V).tolist())
    return {k: (np.mean(v, 0).tolist(), np.std(v, 0, ddof=1).tolist()) for k, v in out.items()}
for sc in ("S1", "S3"):
    for k, (m, s) in variants(sc).items(): print(sc, k, "delta %.3f+-%.3f" % (m[0], s[0]), "phi", np.round(m[1:], 3))
