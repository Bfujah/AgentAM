"""Redesigned figures for the LPBF agent paper.
Data were transcribed from the agent's screenshots (see data/*.csv); the confusion matrix is the
in-sample comparison with the 279 FLOW-3D tracks in data/Agentic_defect_Dataset.csv (eval/cv_validation.py)."""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
D = _os.path.join(_HERE, "..", "data") + _os.sep
O = _os.path.join(_HERE, "pdf") + _os.sep
import os; os.makedirs(O, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
# Defect classes: colour + marker (secondary encoding; validated palette, CVD-safe with shapes)
CLS = {
    "Good":           dict(c="#1baf7a", m="o", lab="Good (stable)"),
    "Keyhole":        dict(c="#e34948", m="s", lab="Keyhole"),
    "Balling":        dict(c="#eda100", m="D", lab="Balling"),
    "Lack of Fusion": dict(c="#2a78d6", m="^", lab="Lack of fusion"),
}
ORDER = ["Good", "Keyhole", "Balling", "Lack of Fusion"]

def style(ax, grid=True):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if grid: ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(length=3)

def iso_J(ax, Js, vmin=100, vmax=1500, pmin=150, pmax=440):
    V = np.linspace(vmin, vmax, 300)
    for J in Js:
        P = J * V
        ok = (P >= pmin) & (P <= pmax)
        ax.plot(V[ok], P[ok], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        i = np.where(ok)[0]
        if len(i):
            k = i[-1]
            ax.text(V[k], P[k], f"$J$={J:g} J/mm", color=INK2, fontsize=6.8, va="bottom", ha="center",
                    zorder=6, bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))

def class_legend(ax, **kw):
    h = [Line2D([], [], ls="", marker=CLS[k]["m"], ms=5.5, mfc=CLS[k]["c"], mec="white", mew=0.5, label=CLS[k]["lab"]) for k in ORDER]
    return ax.legend(handles=h, frameon=False, **kw)

pm = pd.read_csv(D + "process_map.csv")

# ---------------------------------------------------------------- Fig: validation
def fig_validation():
    fig = plt.figure(figsize=(7.1, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.32)
    ax = fig.add_subplot(gs[0]); style(ax)
    for k in ORDER:
        d = pm[pm.defect == k]
        ax.scatter(d.V, d.P, s=22, marker=CLS[k]["m"], c=CLS[k]["c"], edgecolors="white", linewidths=0.5, zorder=3)
    iso_J(ax, [0.25, 0.5, 1.0], pmax=446)
    ex = pd.read_csv(D + "single_points.csv")
    lab = {"Keyhole": "K", "Good": "G", "Balling": "B", "Lack of Fusion": "L"}
    for _, r in ex.iterrows():
        ax.scatter(r.V, r.P, s=70, facecolors="none", edgecolors=INK, linewidths=1.0, zorder=4)
        ax.annotate(lab[r.regime], (r.V, r.P), xytext=(6, 6), textcoords="offset points", fontsize=8, weight="bold", color=INK, zorder=5)
    ax.set_xlim(50, 1550); ax.set_ylim(145, 470); ax.set_yticks(range(150, 451, 50))
    ax.set_xlabel("Scan velocity $V$ (mm/s)"); ax.set_ylabel("Laser power $P$ (W)")
    ax.set_title("(a) Defect regimes predicted across the $P$–$V$ space", loc="left")
    class_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, handletextpad=0.2, columnspacing=1.0)

    # confusion matrix with counts
    labels = ["Good", "Keyhole", "Lack of\nfusion", "Balling"]
    cm = np.array([[45, 5, 0, 3], [4, 58, 0, 0], [1, 0, 135, 2], [0, 0, 1, 25]])
    pct = cm / cm.sum(1, keepdims=True)
    ax2 = fig.add_subplot(gs[1])
    cmap = mpl.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    ax2.imshow(pct, cmap=cmap, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            v = pct[i, j]
            ax2.text(j, i, f"{v*100:.1f}%\n({cm[i,j]})", ha="center", va="center", fontsize=7.2,
                     color="white" if v > 0.55 else INK)
    ax2.set_xticks(range(4)); ax2.set_yticks(range(4))
    ax2.set_xticklabels(labels, fontsize=7.5); ax2.set_yticklabels(labels, fontsize=7.5)
    ax2.set_xlabel("Predicted (surrogate)"); ax2.set_ylabel("Simulated (FLOW-3D)")
    ax2.tick_params(length=0)
    for s in ax2.spines.values(): s.set_visible(False)
    ax2.set_xticks(np.arange(-.5, 4), minor=True); ax2.set_yticks(np.arange(-.5, 4), minor=True)
    ax2.grid(which="minor", color="white", lw=2); ax2.tick_params(which="minor", length=0)
    ax2.set_title("(b) Classification vs. 279 simulations", loc="left")
    ax2.text(1.5, 5.05, "Accuracy 94.3% (in-sample), 93.4% (5-fold cross-validated)", ha="center", va="top", fontsize=7.5, color=INK2)
    fig.savefig(O + "fig_validation.pdf"); fig.savefig(O + "fig_validation.png")
    plt.close(fig)

# ---------------------------------------------------------------- Fig: line scans
def band_shading(ax, x, cls):
    # shade contiguous regime bands behind a 1-D scan; boundaries at midpoints
    x = np.asarray(x); edges = np.r_[x[0], (x[1:] + x[:-1]) / 2, x[-1]]
    i = 0
    while i < len(cls):
        j = i
        while j + 1 < len(cls) and cls[j + 1] == cls[i]: j += 1
        ax.axvspan(edges[i], edges[j + 1], color=CLS[cls[i]]["c"], alpha=0.10, lw=0, zorder=0)
        xm = (edges[i] + edges[j + 1]) / 2
        ax.text(xm, 1.0, CLS[cls[i]]["lab"].replace(" (stable)", ""), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.2, color=INK2)
        i = j + 1

def scan_panel(ax, df, x, xlabel, title, best):
    style(ax, grid=False); ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    cls = list(df.defect)
    band_shading(ax, df[x], cls)
    ax.plot(df[x], df.spatter / 1e3, color=INK2, lw=1.2, zorder=2)
    for k in ORDER:
        d = df[df.defect == k]
        ax.scatter(d[x], d.spatter / 1e3, s=26, marker=CLS[k]["m"], c=CLS[k]["c"], edgecolors="white", linewidths=0.6, zorder=3)
    b = df.loc[df[x] == best].iloc[0]
    ax.scatter(b[x], b.spatter / 1e3, s=95, facecolors="none", edgecolors=INK, linewidths=1.1, zorder=4)
    ax.annotate(f"min. spatter in Good window\n{b.spatter/1e3:.1f}×10$^3$ µm$^3$ at ${xlabel.split()[0]}$ = {best} {xlabel.split()[1].strip('()')}",
                (b[x], b.spatter / 1e3), xytext=(0, -30), textcoords="offset points", ha="center", va="top", fontsize=7.2,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xlabel(xlabel); ax.set_ylabel("Spatter volume (×10$^3$ µm$^3$)")
    ax.set_title(title, loc="left", pad=14)

def fig_linescans():
    a = pd.read_csv(D + "linescan_V_at_P300.csv"); b = pd.read_csv(D + "linescan_P_at_V600.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw=dict(wspace=0.28))
    scan_panel(axs[0], a, "V", "V (mm/s)", "(a) Velocity sweep at $P$ = 300 W", 533)
    axs[0].set_xlabel("Scan velocity $V$ (mm/s)"); axs[0].set_ylim(0, 215)
    scan_panel(axs[1], b, "P", "P (W)", "(b) Power sweep at $V$ = 600 mm/s", 330)
    axs[1].set_xlabel("Laser power $P$ (W)"); axs[1].set_ylim(0, 215)
    class_legend(axs[0], loc="upper center", bbox_to_anchor=(1.12, -0.2), ncol=4, handletextpad=0.2, columnspacing=1.0)
    fig.savefig(O + "fig_linescans.pdf"); fig.savefig(O + "fig_linescans.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig: optimisation
def fig_optimization():
    g = pd.read_csv(D + "gridscan_top10.csv"); d = pd.read_csv(D + "opt_depth.csv"); m = pd.read_csv(D + "opt_multi.csv")
    Q = [("Q1", "Good only, spatter < 100k µm³ (top 10 of 15×15 grid)", g, "#6250d6", "o"),
         ("Q2/Q3", "depth 120–180 µm, spatter < 90k µm³; or 0.4 ≤ J ≤ 0.6 J/mm", d, INK, "P"),
         ("Q4", "depth 150–250 µm, length 350–500 µm, spatter < 120k µm³", m, "#eb6834", "X")]
    fig = plt.figure(figsize=(7.1, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0]); style(ax)
    for k in ORDER:
        s_ = pm[pm.defect == k]
        ax.scatter(s_.V, s_.P, s=9, marker=CLS[k]["m"], c=CLS[k]["c"], alpha=0.22, edgecolors="none", zorder=2)
    ax.axhline(160, color=INK2, lw=0.8, zorder=1)
    ax.text(985, 156, "lower bound of trained range (160 W)", fontsize=6.8, color=INK2, ha="right", va="top")
    ax.add_patch(Rectangle((300, 200), 700, 200, fill=False, ec=MUTED, lw=0.7, ls=(0, (4, 2)), zorder=2))
    ax.text(990, 396, "Q1 search region", fontsize=6.8, color=MUTED, va="top", ha="right")
    for tag, lab, df, c, mk in Q:
        ax.scatter(df.V.iloc[1:], df.P.iloc[1:], s=22, marker=mk, c=c, edgecolors="white", linewidths=0.4, zorder=4)
        b = df.iloc[0]
        ax.scatter(b.V, b.P, s=110, marker="*", c=c, edgecolors="white", linewidths=0.6, zorder=5)
    for (tag, lab, df, c, mk), off, ha in zip(Q, [(14, -10), (-10, 10), (16, 14)], ["left", "right", "left"]):
        b = df.iloc[0]
        ax.annotate(f"{tag}", (b.V, b.P), xytext=off, textcoords="offset points", fontsize=7.5, weight="bold", color=c, ha=ha,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6, shrinkA=0, shrinkB=4))
    ax.set_xlim(150, 1000); ax.set_ylim(140, 445)
    ax.set_xlabel("Scan velocity $V$ (mm/s)"); ax.set_ylabel("Laser power $P$ (W)")
    ax.set_title("(a) Recommended points in the $P$–$V$ plane", loc="left")
    h = [Line2D([], [], ls="", marker=mk, ms=5.5, mfc=c, mec="white", mew=0.4, label=f"{tag}: {lab}") for tag, lab, _, c, mk in Q]
    h.append(Line2D([], [], ls="", marker="*", ms=9, mfc=INK2, mec="white", label="best point returned by the agent"))
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(-0.17, -0.18), frameon=False, fontsize=7.2, ncol=2,
              handletextpad=0.3, columnspacing=1.2)
    # (b) ranked spatter of returned candidates
    ax2 = fig.add_subplot(gs[1]); style(ax2, grid=False); ax2.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
    y = 0; yt = []; yl = []
    for tag, lab, df, c, mk in Q:
        n = len(df)
        ys = y + np.arange(n) * 0.32
        ax2.scatter(df.spatter / 1e3, ys, marker=mk, s=22, c=c, edgecolors="white", linewidths=0.4, zorder=3)
        ax2.scatter(df.spatter.iloc[0] / 1e3, ys[0], marker="*", s=110, c=c, edgecolors="white", linewidths=0.6, zorder=4)
        yt.append(ys.mean()); yl.append(tag); y = ys[-1] + 0.9
    ax2.set_yticks(yt); ax2.set_yticklabels(yl); ax2.invert_yaxis(); ax2.tick_params(axis="y", length=0)
    ax2.set_xlim(35, 72); ax2.set_xlabel("Predicted spatter volume (×10$^3$ µm$^3$)")
    ax2.set_title("(b) Spatter of ranked candidates", loc="left")
    fig.savefig(O + "fig_optimization.pdf"); fig.savefig(O + "fig_optimization.png"); plt.close(fig)

fig_validation(); fig_linescans(); fig_optimization()
print("done")
