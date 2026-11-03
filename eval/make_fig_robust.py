"""Figure: plain vs margin-aware optima (fig_robust.pdf). Uses results/robust_*.csv and ../data/process_map.csv."""
import csv, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
CLS = {"Good": "#1baf7a", "Keyhole": "#e34948", "Balling": "#eda100", "Lack of Fusion": "#2a78d6"}
MK = {"Good": "o", "Keyhole": "s", "Balling": "D", "Lack of Fusion": "^"}
pm = list(csv.DictReader(open("../data/process_map.csv")))
cases = {r["case"]: r for r in csv.DictReader(open("results/robust_cases.csv"))}
tr = list(csv.DictReader(open("results/robust_tradeoff.csv")))
def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5, zorder=0); ax.tick_params(length=3)

fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.45), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.42))
ax = axs[0]; style(ax)
for k in CLS:
    d = [r for r in pm if r["defect"] == k]
    ax.scatter([float(r["V"]) for r in d], [float(r["P"]) for r in d], s=10, marker=MK[k], c=CLS[k], alpha=0.28, edgecolors="none", zorder=2)
groups = [("Q1 grid scan", "Q1"), ("Q2 depth + spatter", "Q2, Q3, unconstr."), ("Q4 depth + length + spatter", "Q4")]
for name, lab in groups:
    r = cases[name]; p0 = (float(r["plain_V"]), float(r["plain_P"])); p1 = (float(r["robust_V"]), float(r["robust_P"]))
    ax.scatter(*p0, s=34, facecolors="white", edgecolors=INK, linewidths=0.9, zorder=5)
    ax.scatter(*p1, s=70, marker="*", c=INK, zorder=6)
    if p0 != p1: ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="->", color=INK, lw=0.8), zorder=4)
    off = {"Q1": (6, 5), "Q2, Q3, unconstr.": (9, -6), "Q4": (6, 4)}[lab]
    ax.annotate(lab, xy=p1, xytext=off, textcoords="offset points", fontsize=7, color=INK, ha="left" if off[0] > 0 else "right")
ax.set_xlabel("$V$ (mm/s)"); ax.set_ylabel("$P$ (W)"); ax.set_xlim(100, 1100); ax.set_ylim(150, 440)
h = [Line2D([], [], ls="", marker="o", mfc="white", mec=INK, ms=5, label="plain optimum"),
     Line2D([], [], ls="", marker="*", color=INK, ms=7, label="margin-aware")]
ax.legend(handles=h, frameon=False, loc="upper right", handletextpad=0.3)
ax.text(-0.2, 1.03, "(a)", transform=ax.transAxes, fontsize=9)
for j, (key, ylab) in enumerate((("spatter", r"spatter of optimum ($10^4$ µm$^3$)"), ("pgood_B", r"probability of staying Good"))):
    ax = axs[j + 1]; style(ax)
    for rm, ls, lab in (("0.0", "-", "no range margin"), ("0.05", "--", "5% range margin")):
        d = [r for r in tr if r["range_margin"] == rm]
        x = [float(r["depth_margin"]) for r in d]; y = [float(r[key]) / (1e4 if key == "spatter" else 1) for r in d]
        ax.plot(x, y, ls=ls, marker="o", ms=3.2, color=INK if rm == "0.0" else INK2, lw=0.9, label=lab)
    ax.set_xlabel(r"depth margin (µm)"); ax.set_ylabel(ylab)
    if key == "pgood_B": ax.set_ylim(0.7, 1.01)
    ax.text(-0.25, 1.03, "(b)" if j == 0 else "(c)", transform=ax.transAxes, fontsize=9)
axs[1].legend(frameon=False, loc="lower right")
fig.savefig("../figs/fig_robust.pdf"); fig.savefig("../figs/fig_robust.png", dpi=300); print("saved")
