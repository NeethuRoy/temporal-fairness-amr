import numpy as np, time, json
sig = lambda z: 1/(1+np.exp(-z))
D = 12; n = 6000
rng = np.random.default_rng(0)
beta_base = np.array([0.8,0.6,1.0,0.4] + [0.3]*8); b0 = -0.6
def make(seed):
    r = np.random.default_rng(seed)
    g = (r.random(n) < .5).astype(int)
    X = r.standard_normal((n, D)); beta = np.tile(beta_base, (n,1)); beta[:,2] = np.where(g==1, 1+0.5*4, 1.0)  # window-5 setting
    y = (r.random(n) < sig(b0 + (X*beta).sum(1))).astype(int); return X,y,g
def fit(X,y,it=25):
    Xa=np.hstack([np.ones((len(X),1)),X]); b=np.zeros(Xa.shape[1])
    for _ in range(it):
        p=sig(Xa@b); gr=Xa.T@(p-y); H=(Xa*(p*(1-p))[:,None]).T@Xa+1e-8*np.eye(len(b)); st=np.linalg.solve(H,gr); b-=st
        if abs(st).max()<1e-9: break
    return b[0], b[1:]
X,y,g = make(1); idx=np.random.default_rng(5).permutation(n); tr,te=idx[:n//2],idx[n//2:]
b0h, bh = fit(X[tr],y[tr]); xbar = X[tr].mean(0); Xt, yt, gt = X[te], y[te], g[te]
pa=(yt==1)&(gt==1); pb=(yt==1)&(gt==0)
C = Xt*bh; cb = xbar*bh
# ---- exact: 2^12 coalitions on positives only ----
t=time.time()
Xp = np.where(pa|pb)[0]; Cp=C[Xp]; pap=pa[Xp]; pbp=pb[Xp]
M = ((np.arange(2**D)[:,None]>>np.arange(D))&1).astype(float)   # 4096 x 12
L = b0h + Cp@M.T + cb@(1-M).T                                     # npos x 4096
P = sig(L); V = P[pap].mean(0)-P[pbp].mean(0)
from math import factorial
phi = np.zeros(D); pc = M.sum(1).astype(int)
wt = np.array([factorial(k)*factorial(D-k-1)/factorial(D) for k in range(D)])
for i in range(D):
    wo = M[:,i]==0; s = np.where(wo)[0]; s2 = s | (1<<i)
    phi[i] = (wt[pc[s]]*(V[s2]-V[s])).sum()
t_exact = time.time()-t; delta = V[-1]
print("exact: delta",round(delta,4),"phi3",round(phi[2],4),"sum",round(phi.sum(),6),"time",round(t_exact,2),"s")
# ---- permutation sampling of per-prediction Shapley, aggregated by group (Proposition 1) ----
def mc(m, seed):
    r=np.random.default_rng(seed); acc_a=np.zeros(D); acc_b=np.zeros(D)
    Cpp = Cp
    for _ in range(m):
        perm = r.permutation(D); cur = np.full(len(Cpp), b0h + cb.sum()); prev = sig(cur)
        for j in perm:
            cur = cur + Cpp[:,j] - cb[j]; new = sig(cur); d = new-prev
            acc_a[j]+= d[pap].mean(); acc_b[j]+= d[pbp].mean(); prev = new
    return (acc_a-acc_b)/m
out={}
for m in (10,50,200,1000):
    errs=[]; t=time.time(); ests=[]
    for rep in range(20):
        e = mc(m, 100+rep); ests.append(e); errs.append(np.abs(e-phi).max())
    tt=(time.time()-t)/20; ests=np.array(ests)
    out[m]=dict(maxerr=float(np.mean(errs)), err3=float(np.mean(np.abs(ests[:,2]-phi[2]))), sd3=float(ests[:,2].std(ddof=1)), time=float(tt), effgap=float(np.abs(ests.sum(1)-delta).max()))
    print(m, out[m])
# bootstrap CI width scale of phi3 for reference (exact-style on 4 main features is ~0.05); compute via MC at m=200? report from earlier: 0.053
json.dump(dict(exact=dict(delta=float(delta),phi=phi.tolist(),time=t_exact),mc=out), open("scale.json","w"))
