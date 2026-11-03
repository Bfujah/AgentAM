"""Process map panel for Figure 1 (fig_overview): regimes and spatter contours from the deployed surrogates."""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server")); import server as s
r = s.get_material_ranges("SS316L"); print(r)
P = np.linspace(*r["P"], 120); V = np.linspace(*r["V"], 160)
cls = np.zeros((len(P), len(V))); sp = np.zeros_like(cls); rob = np.zeros_like(cls)
code = {"keyhole":0, "lack of fusion":1, "lack_of_fusion":1, "balling":2, "good":3}
for i,p in enumerate(P):
    for j,v in enumerate(V):
        m = s.predict_all_metrics("SS316L", p, v)
        lab = str(m["defect_type"]).lower().replace("-", " ")
        cls[i,j] = next((c for k,c in code.items() if k.replace("_"," ") in lab), np.nan)
        sp[i,j] = m["spatter_volume"]
        mg = s.threshold_margins(p, v, m["depth"], m["width"], m["length"], r)
        rob[i,j] = float(m["defect_type"]=="Good" and mg["depth_um"]>=15 and mg["dw"]>=0.10 and mg["lw"]>=0.15 and mg["range"]>=0.05)
print(np.unique(cls, return_counts=True))
plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":8,"axes.linewidth":0.5,"hatch.linewidth":0.35,"hatch.color":"#127A55"})
fig, ax = plt.subplots(figsize=(1.68,1.45), dpi=300)
cm = ListedColormap(["#F6C9C8","#C9DCF3","#FBE5B0","#C7EBDC"])
ax.pcolormesh(V, P, cls, cmap=cm, vmin=-0.5, vmax=3.5, shading="auto", rasterized=True)
ax.contour(V, P, cls, levels=[0.5,1.5,2.5], colors="#52514E", linewidths=0.4)
spg = np.where(cls==3, sp/1e3, np.nan)
ax.contourf(V, P, rob, levels=[0.5,1.5], colors="none", hatches=["////"])
ax.contour(V, P, rob, levels=[0.5], colors="#127A55", linewidths=0.6, linestyles=":")
ax.plot(348.9, 160, marker="o", ms=5, mfc="white", mec="#0B0B0B", mew=0.7, clip_on=False, zorder=5)
ax.plot(411.1, 184, marker="*", ms=9, mfc="#F5C400", mec="#0B0B0B", mew=0.6, zorder=6)
for txt,(x,y),c in [("keyhole",(300,395),"#B3302F"),("lack of\nfusion",(1180,215),"#1E5AA3"),("balling",(1030,405),"#9A6A00"),("good",(470,215),"#127A55")]:
    ax.text(x,y,txt,fontsize=6.5,color=c,ha="center",va="center",style="italic",zorder=7,bbox=dict(fc="white",ec="none",pad=0.6,alpha=0.85) if txt=="good" else None)
ax.set_xlabel("$V$ (mm/s)", labelpad=1); ax.set_ylabel("$P$ (W)", labelpad=1)
ax.tick_params(length=2, width=0.4, pad=1, labelsize=6.5)
fig.tight_layout(pad=0.2); fig.savefig("overview_map_m.pdf"); fig.savefig("overview_map_m.png")
