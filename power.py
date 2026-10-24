import numpy as np, json, time
import tfa_sim as T
def gen(seed, n, pa):
    r = np.random.default_rng(seed)
    g = (r.random(n) < pa).astype(int)
    Xs, Ys = [], []
    for w in range(T.W):
        Z = r.standard_normal((n, T.D)); beta = np.tile(T.BASE_BETA, (n, 1))
        beta[:, 2] = np.where(g == 1, 1.0 + 0.5 * w, 1.0)
        y = (r.random(n) < T.sigmoid(T.B0 + (Z * beta).sum(1))).astype(int); Xs.append(Z); Ys.append(y)
    return np.stack(Xs), np.stack(Ys), g
def det(lo, hi): return (lo > 0) | (hi < 0)
out = {}
configs = [(1500, .5), (3000, .5), (6000, .5), (12000, .5), (6000, .15)]
for n, pa in configs:
    t = time.time(); res = {w: dict(d=[], p3=[]) for w in (2, 4)}
    for sd in range(15):
        X, Y, g = gen(sd, n, pa); r = np.random.default_rng(1000 + sd)
        idx = np.random.default_rng(10000 + sd).permutation(n); tr, te = idx[: n // 2], idx[n // 2:]
        for w in (2, 4):                      # windows 3 and 5 (0-indexed 2, 4)
            bd, bp = [], []
            for _ in range(100):
                a = r.integers(0, len(tr), len(tr)); b = r.integers(0, len(te), len(te))
                try:
                    s = T.window_stats(X[w][tr][a], Y[w][tr][a], X[w][te][b], Y[w][te][b], g[te][b])
                    bd.append(s["delta"]); bp.append(s["phi"][2])
                except Exception: pass
            lo, hi = np.percentile(bd, [2.5, 97.5]); res[w]["d"].append(bool(det(lo, hi)))
            lo, hi = np.percentile(bp, [2.5, 97.5]); res[w]["p3"].append(bool(det(lo, hi)))
    out[f"{n}_{pa}"] = {f"w{w+1}": dict(delta=float(np.mean(res[w]["d"])), phi3=float(np.mean(res[w]["p3"]))) for w in (2, 4)}
    print(n, pa, out[f"{n}_{pa}"], round(time.time() - t, 1), "s", flush=True)
json.dump(out, open("power.json", "w"))
