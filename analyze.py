import json, numpy as np
np.set_printoptions(precision=3, suppress=True)
R={s:json.load(open(f"res_{s}.json")) for s in ["S0","S1","S2","S3","S4","S5"]}
def arr(s,k): return np.array([r[k] for r in R[s]["main"]])   # seeds x W (x D)
def ci(s,k):  # seeds x W x (2) or x D x 2
    return np.array([[c[k] for c in r["ci"]] for r in R[s]["main"]])
def det(lo_hi):  # CI excludes 0
    lo,hi=lo_hi[...,0],lo_hi[...,1]; return (lo>0)|(hi<0)
out={}
for s in R:
    d=arr(s,"delta"); phi=arr(s,"phi"); pl=arr(s,"psi_logit")
    cd=ci(s,"delta"); cp=ci(s,"phi"); cl=ci(s,"psi_logit")
    print("=====",s)
    print("delta mean",d.mean(0),"sd",d.std(0,ddof=1))
    print("phi mean (W x D)\n",phi.mean(0))
    print("phi sd\n",phi.std(0,ddof=1))
    print("det rate delta by window",det(cd).mean(0))
    print("det rate phi (W x D)\n",det(cp).mean(0))
    print("psi_logit mean\n",pl.mean(0))
    print("det rate psi_logit (W x D)\n",det(cl).mean(0))
    print("sum of phi3 others (w5)",(phi[:,:,[0,1,3]].sum(2)).mean(0))
    b=R[s]["base"]
    print("collapsed delta",np.mean([x["delta"] for x in b["collapsed"]]),np.std([x["delta"] for x in b["collapsed"]],ddof=1),"phi",np.mean([x["phi"] for x in b["collapsed"]],0), np.std([x["phi"] for x in b["collapsed"]],0,ddof=1))
    print("pooled delta",np.mean([x["delta"] for x in b["pooled"]]),np.std([x["delta"] for x in b["pooled"]],ddof=1),"phi",np.mean([x["phi"] for x in b["pooled"]],0))
    print("equiv max",np.max(b["equiv"]))
    bs=b["base_sens"]
    for k in("mean","bg","refmean"):
        print("last-window phi",k,np.mean([x[k] for x in bs],0))
    print("delta mean/bg/ref",np.mean([x["dm"] for x in bs]),np.mean([x["db"] for x in bs]),np.mean([x["dr"] for x in bs]))
