import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"Liberation Serif","font.size":7})
R={s:json.load(open(f"res_{s}.json"))["main"] for s in ["S1","S2"]}
fig,ax=plt.subplots(1,2,figsize=(3.45,2.0),dpi=300,sharey=True)
for a,s,t in zip(ax,["S1","S2"],["(a) S1","(b) S2"]):
    phi=np.array([r["phi"] for r in R[s]]); m=phi.mean(0).T            # d x W
    ci=np.array([[c["phi"] for c in r["ci"]] for r in R[s]])           # seeds x W x d x 2
    det=(((ci[...,0]>0)|(ci[...,1]<0)).mean(0)).T                      # d x W
    im=a.imshow(m,cmap="RdBu_r",vmin=-0.1,vmax=0.1,aspect="auto")
    for i in range(4):
        for j in range(5):
            a.text(j,i,f"{m[i,j]:.2f}"+("*" if det[i,j]>=0.8 else ""),ha="center",va="center",fontsize=5.6,color="k" if abs(m[i,j])<0.07 else "w")
    a.set_xticks(range(5)); a.set_xticklabels([1,2,3,4,5]); a.set_xlabel("Window"); a.set_title(t,fontsize=7,loc="left")
ax[0].set_yticks(range(4)); ax[0].set_yticklabels(["$x_1$","$x_2$","$x_3$","$x_4$"])
cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02); cb.ax.tick_params(labelsize=5.5); cb.set_label("φ",fontsize=6)
plt.savefig("fig2.png",bbox_inches="tight",pad_inches=0.03); 
from PIL import Image; print(Image.open("fig2.png").size)
