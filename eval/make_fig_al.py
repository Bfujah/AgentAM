"""Figure: pool-based active learning on the FLOW-3D runs (fig_active.pdf). Uses results/al_curves_test20.csv."""
import sys, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
src = sys.argv[1] if len(sys.argv) > 1 else "results/al_curves_test20.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "../figs/fig_active"
d = pd.read_csv(src)
full = d.groupby("rep").full_acc.first()
ST = [("random", "random", dict(color=MUTED, ls=":", lw=1.2)),
      ("space", "space-filling", dict(color=INK2, ls="--", lw=1.1)),
      ("qbc", "query by committee", dict(color=INK, ls="-", lw=1.3)),
      ("exploit", "exploit (lowest spatter)", dict(color="#b5651d", ls="-.", lw=1.1))]

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5, zorder=0); ax.tick_params(length=3)

fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.55), gridspec_kw=dict(wspace=0.3))
ax = axs[0]; style(ax)
for s, lab, kw in ST:
    g = d[d.strategy == s].groupby("n").acc; m, se = 100 * g.mean(), 100 * 1.96 * g.std() / np.sqrt(g.count())
    m, se = m[m.index <= 150], se[se.index <= 150]
    ax.plot(m.index, m.values, label=lab, zorder=3, **kw)
    if s in ("qbc", "space", "random"): ax.fill_between(m.index, m - se, m + se, color=kw["color"], alpha=0.12, lw=0, zorder=2)
ax.axhline(100 * full.mean(), color="#2a78d6", lw=0.8, zorder=1, label="all available runs")
ax.set_xlim(15, 152); ax.set_ylim(84, 96)
ax.set_xlabel("simulations acquired"); ax.set_ylabel("test accuracy of defect map (%)")
ax.text(-0.14, 1.03, "(a)", transform=ax.transAxes, fontsize=9)
ax.legend(frameon=False, loc="lower right", handlelength=2.4)

ax = axs[1]; style(ax)
for s, lab, kw in ST:
    g = d[d.strategy == s].groupby("n").regret; m = 100 * g.median(); m = m[m.index <= 150]
    ax.plot(m.index, m.values, label=lab, zorder=3, **kw)
ax.axhline(5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
ax.text(150, 5.6, "5%", fontsize=7, color=INK2, ha="right")
ax.set_xlim(15, 152); ax.set_ylim(-1, 45)
ax.set_xlabel("simulations acquired"); ax.set_ylabel("spatter gap to best Good run (%)")
ax.text(-0.14, 1.03, "(b)", transform=ax.transAxes, fontsize=9)
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
print("saved", out)
