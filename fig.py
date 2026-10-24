import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"Liberation Serif","font.size":8,"axes.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False})
R={s:json.load(open(f"res_{s}.json"))["main"] for s in ["S0","S1","S2","S3"]}
w=np.arange(1,6)
def A(s,k): return np.array([r[k] for r in R[s]])
fig,ax=plt.subplots(1,3,figsize=(7.1,2.25),dpi=300)
# (a) delta
for s,c,l,m in [("S0","#7f7f7f","S0 null","o"),("S1","#c0392b","S1 concept drift ($x_3$)","s"),("S3","#2c6fbb","S3 covariate shift ($x_1$)","^")]:
    d=A(s,"delta"); ax[0].errorbar(w,d.mean(0),d.std(0,ddof=1),color=c,marker=m,ms=3.5,lw=1,capsize=2,label=l)
ax[0].axhline(0,color="k",lw=0.4); ax[0].set_xlabel("Treatment window"); ax[0].set_ylabel(r"TPR disparity $\Delta_w$"); ax[0].set_xticks(w)
ax[0].legend(frameon=False,fontsize=6.3,loc="upper left"); ax[0].set_title("(a) Disparity",fontsize=8,loc="left")
# (b) S1 phi
p=A("S1","phi"); names=["$x_1$ exposure","$x_2$ culture trend","$x_3$ genotype marker","$x_4$ vital trend"]
cols=["#8c8c8c","#b0b0b0","#c0392b","#d0d0d0"]; mk=["o","v","s","D"]
for i in [0,1,3,2]:
    ax[1].errorbar(w+(i-1.5)*0.05,p[:,:,i].mean(0),p[:,:,i].std(0,ddof=1),color=cols[i],marker=mk[i],ms=3.2,lw=1,capsize=1.8,label=names[i],zorder=3 if i==2 else 2)
ax[1].axhline(0,color="k",lw=0.4); ax[1].set_xlabel("Treatment window"); ax[1].set_ylabel(r"Attribution $\varphi_{i,w}$"); ax[1].set_xticks(w)
ax[1].legend(frameon=False,fontsize=6.0,loc="upper left"); ax[1].set_title("(b) S1: attribution",fontsize=8,loc="left")
# (c) S2
p=A("S2","phi")
for i,c,m,l in [(2,"#c0392b","s","$x_3$ (linear drift from w2)"),(1,"#e08e0b","o","$x_2$ (step at w3)")]:
    ax[2].errorbar(w,p[:,:,i].mean(0),p[:,:,i].std(0,ddof=1),color=c,marker=m,ms=3.5,lw=1,capsize=2,label=l)
for i in (0,3): ax[2].plot(w,p[:,:,i].mean(0),color="#b0b0b0",lw=0.9,marker="o",ms=2.5,label="$x_1$, $x_4$" if i==0 else None)
ax[2].set_ylim(-0.03,0.09); ax[2].axhline(0,color="k",lw=0.4); ax[2].set_xlabel("Treatment window"); ax[2].set_ylabel(r"Attribution $\varphi_{i,w}$"); ax[2].set_xticks(w)
ax[2].legend(frameon=False,fontsize=6.0,loc="upper left"); ax[2].set_title("(c) S2: two drivers, staggered onset",fontsize=8,loc="left")
plt.tight_layout(pad=0.5,w_pad=1.0); plt.savefig("fig1.png"); print("ok")
