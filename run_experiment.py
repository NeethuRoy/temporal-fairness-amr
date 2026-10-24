"""
Temporal Fairness-Attribution (TFA): synthetic proof-of-concept.

Reproduces exactly the setup described in the paper:
  - 6000 synthetic patients per run
  - 5 treatment-course windows
  - 4 features per window (prior antibiotic exposure, culture trend,
    genotype marker, vital-sign trend)
  - 2 subgroups (uniform random), representing e.g. well-resourced vs.
    resource-limited care settings
  - A per-window logistic-regression base predictor, fit WITHOUT the
    subgroup label (the predictor never sees G), so any subgroup gap
    arises purely from model misspecification relative to a true
    data-generating process whose genotype-marker coefficient diverges
    by subgroup as treatment progresses.
  - Exact Shapley decomposition (16 coalitions) of the probability-based
    TPR disparity between subgroups, per window, with baseline
    (mean-imputation) masking.
  - A static-fairness-Shapley baseline (Begley et al.-style) computed on
    the same data collapsed (averaged) across windows into one snapshot.
  - 15 random seeds; results reported as mean +/- SD.

Outputs:
  - results.json       : all per-seed, per-window numbers
  - summary.json        : mean/SD summary used directly in the paper text
  - fig1.png             : two-panel figure (disparity + attribution)
"""
import json
import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression

N = 6000
W = 5
D = 4
FEATURES = [
    "prior_antibiotic_exposure",
    "culture_trend",
    "genotype_marker",
    "vital_sign_trend",
]
GENOTYPE_IDX = 2  # the feature in which the disparity is injected
SEEDS = list(range(15))

# true data-generating coefficients
B0 = -0.2
B1, B2, B4 = 0.35, 0.35, 0.35   # unaffected features
BASE_B3 = 0.30                  # genotype-marker coefficient at window 0
SLOPE_B3 = 0.45                 # per-window growth, subgroup 1 only


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate_data(seed):
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 2, size=N)                 # 0 = well-resourced, 1 = resource-limited
    X = rng.normal(0.0, 1.0, size=(N, W, D))
    Y = np.zeros((N, W), dtype=int)
    for w in range(W):
        b3 = np.where(G == 1, BASE_B3 + SLOPE_B3 * w, BASE_B3)
        logit = (B0
                 + B1 * X[:, w, 0]
                 + B2 * X[:, w, 1]
                 + b3 * X[:, w, 2]
                 + B4 * X[:, w, 3])
        p = sigmoid(logit)
        Y[:, w] = rng.binomial(1, p)
    return X, Y, G


def fit_window_model(X_w, Y_w):
    # Base predictor never observes subgroup membership.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_w, Y_w)
    return model


def tpr_disparity(model, X_w, Y_w, G, baseline):
    """Probability-based TPR disparity Delta(a=1, b=0), with feature
    masking controlled via `baseline` applied outside this function by
    the caller (this function just evaluates on whatever X_w is passed)."""
    phat = model.predict_proba(X_w)[:, 1]
    pos1 = (Y_w == 1) & (G == 1)
    pos0 = (Y_w == 1) & (G == 0)
    tpr1 = phat[pos1].mean() if pos1.any() else 0.0
    tpr0 = phat[pos0].mean() if pos0.any() else 0.0
    return tpr1 - tpr0


def masked_X(X_w, baseline, keep_idx):
    """Return a copy of X_w with all feature columns NOT in keep_idx
    replaced by their baseline (training-mean) value."""
    Xm = np.tile(baseline, (X_w.shape[0], 1)).astype(float)
    for j in keep_idx:
        Xm[:, j] = X_w[:, j]
    return Xm


def shapley_decomposition(model, X_w, Y_w, G, baseline):
    """Exact Shapley decomposition (2^D coalitions) of the TPR disparity
    function v(S) = Delta(a=1,b=0) computed with features outside S
    masked to their baseline value."""
    feat_idx = list(range(D))
    v = {}
    for r in range(D + 1):
        for S in itertools.combinations(feat_idx, r):
            Xm = masked_X(X_w, baseline, S)
            v[S] = tpr_disparity(model, Xm, Y_w, G, baseline)

    def vS(S):
        return v[tuple(sorted(S))]

    phi = np.zeros(D)
    fact = [1] * (D + 1)
    for k in range(1, D + 1):
        fact[k] = fact[k - 1] * k
    for i in feat_idx:
        others = [f for f in feat_idx if f != i]
        total = 0.0
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                weight = fact[len(S)] * fact[D - len(S) - 1] / fact[D]
                total += weight * (vS(S + (i,)) - vS(S))
        phi[i] = total
    delta_full = vS(tuple(feat_idx))
    return phi, delta_full


def run_tfa(seed):
    X, Y, G = generate_data(seed)
    deltas = np.zeros(W)
    phis = np.zeros((W, D))
    for w in range(W):
        model = fit_window_model(X[:, w, :], Y[:, w])
        baseline = X[:, w, :].mean(axis=0)
        phi, delta = shapley_decomposition(model, X[:, w, :], Y[:, w], G, baseline)
        deltas[w] = delta
        phis[w, :] = phi
    return deltas, phis


def run_static_baseline(seed):
    X, Y, G = generate_data(seed)
    # Collapse windows: average features across the treatment course;
    # label = resistance observed in at least one window (case/control
    # snapshot, as in standard static fairness-Shapley datasets).
    X_static = X.mean(axis=1)
    Y_static = (Y.sum(axis=1) >= 1).astype(int)
    model = fit_window_model(X_static, Y_static)
    baseline = X_static.mean(axis=0)
    phi, delta = shapley_decomposition(model, X_static, Y_static, G, baseline)
    return phi, delta


def main():
    all_deltas = np.zeros((len(SEEDS), W))
    all_phis = np.zeros((len(SEEDS), W, D))
    all_static_phi = np.zeros((len(SEEDS), D))
    all_static_delta = np.zeros(len(SEEDS))

    for si, seed in enumerate(SEEDS):
        deltas, phis = run_tfa(seed)
        all_deltas[si] = deltas
        all_phis[si] = phis
        s_phi, s_delta = run_static_baseline(seed)
        all_static_phi[si] = s_phi
        all_static_delta[si] = s_delta
        print(f"seed {seed}: delta_w={np.round(deltas,4)} static_delta={s_delta:.4f}")

    results = {
        "seeds": SEEDS,
        "features": FEATURES,
        "tfa_delta_per_seed_window": all_deltas.tolist(),
        "tfa_phi_per_seed_window_feature": all_phis.tolist(),
        "static_phi_per_seed_feature": all_static_phi.tolist(),
        "static_delta_per_seed": all_static_delta.tolist(),
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    delta_mean = all_deltas.mean(axis=0)
    delta_sd = all_deltas.std(axis=0, ddof=1)
    phi_mean = all_phis.mean(axis=0)   # W x D
    phi_sd = all_phis.std(axis=0, ddof=1)

    static_phi_mean = all_static_phi.mean(axis=0)
    static_phi_sd = all_static_phi.std(axis=0, ddof=1)
    static_delta_mean = all_static_delta.mean()
    static_delta_sd = all_static_delta.std(ddof=1)

    summary = {
        "tfa_delta_mean_per_window": delta_mean.tolist(),
        "tfa_delta_sd_per_window": delta_sd.tolist(),
        "tfa_phi_mean_window_feature": phi_mean.tolist(),
        "tfa_phi_sd_window_feature": phi_sd.tolist(),
        "static_phi_mean": static_phi_mean.tolist(),
        "static_phi_sd": static_phi_sd.tolist(),
        "static_delta_mean": static_delta_mean,
        "static_delta_sd": static_delta_sd,
        "features": FEATURES,
    }
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print("Delta_w mean:", np.round(delta_mean, 4))
    print("Delta_w sd:  ", np.round(delta_sd, 4))
    for i, fn in enumerate(FEATURES):
        print(f"phi[{fn}] mean per window:", np.round(phi_mean[:, i], 4))
        print(f"phi[{fn}] sd   per window:", np.round(phi_sd[:, i], 4))
    print("\nStatic baseline delta mean/sd:", round(static_delta_mean, 4), round(static_delta_sd, 4))
    for i, fn in enumerate(FEATURES):
        print(f"static phi[{fn}] mean/sd:", round(static_phi_mean[i], 4), round(static_phi_sd[i], 4))

    # sanity check: efficiency property (sum of phi == delta) for a couple of seeds/windows
    check = np.abs(all_phis.sum(axis=2) - all_deltas).max()
    print("\nMax |sum(phi) - delta| over all seeds/windows (efficiency check):", check)
    check_static = np.abs(all_static_phi.sum(axis=1) - all_static_delta).max()
    print("Max |sum(static_phi) - static_delta| (efficiency check):", check_static)


if __name__ == "__main__":
    main()
