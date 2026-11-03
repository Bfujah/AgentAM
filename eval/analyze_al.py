"""Summarise results/al_curves.csv: learning curves, simulations needed to reach a target accuracy,
paired comparison against random sampling, and optimum-finding regret."""
import sys, numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/al_curves.csv")
S = list(dict.fromkeys(d.strategy))
full = d.groupby("rep").full_acc.first()
print(f"full-data test accuracy: mean {100*full.mean():.1f}%  sd {100*full.std():.1f}%  (n_avail {d.n_avail.iloc[0]})")

piv = d.pivot_table(index=["rep", "n"], columns="strategy", values="acc")
print("\nmean test accuracy (%) [95% CI half-width] at n:")
for n in [15, 25, 35, 50, 75, 100, 125, 150, 180]:
    if n not in piv.index.get_level_values(1): continue
    x = piv.xs(n, level=1)
    print(f"  n={n:3d}  " + "  ".join(f"{s}={100*x[s].mean():5.1f}[{196*x[s].std()/np.sqrt(len(x)):.1f}]" for s in S))

def n_to(rep_df, thr, sustain=True):
    """smallest n at which accuracy >= thr and stays >= thr for all larger n (if sustain)"""
    r = rep_df.sort_values("n"); ok = (r.acc >= thr).values; ns = r.n.values
    for i in range(len(ns)):
        if (ok[i:].all() if sustain else ok[i]): return ns[i]
    return np.nan

out = []
for gap in [0.03, 0.02, 0.01]:
    print(f"\nsimulations to reach (full-data accuracy - {100*gap:.0f} pts) and stay there, per repeat:")
    res = {}
    for s in S:
        v = []
        for rep, g in d[d.strategy == s].groupby("rep"):
            v.append(n_to(g, full[rep] - gap - 1e-9))
        res[s] = np.array(v, float)
    for s in S:
        v = res[s]; reached = np.isfinite(v)
        print(f"  {s:8s} median {np.nanmedian(v):6.1f}  mean {np.nanmean(v):6.1f}  IQR {np.nanpercentile(v,25):.0f}-{np.nanpercentile(v,75):.0f}  reached {reached.sum()}/{len(v)}")
        if s != "random":
            a, b = res["random"], v; m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 5:
                w = stats.wilcoxon(a[m], b[m]); red = 1 - np.median(b[m]) / np.median(a[m])
                print(f"           vs random: median saving {np.median(a[m]-b[m]):.0f} sims ({100*red:.0f}%), wins {np.sum(b[m]<a[m])}/{m.sum()}, Wilcoxon p={w.pvalue:.2g}")
        out.append(dict(gap=gap, strategy=s, median_n=np.nanmedian(v), mean_n=np.nanmean(v), reached=int(np.isfinite(v).sum())))
pd.DataFrame(out).to_csv("results/al_summary.csv", index=False)

# area under learning curve (mean accuracy over n in 15..100), paired vs random
print("\nmean accuracy over n = 15..100 (area under the learning curve), paired vs random:")
a = d[(d.n <= 100)].groupby(["strategy", "rep"]).acc.mean().unstack(0)
for s in S:
    if s == "random": print(f"  random   {100*a[s].mean():.1f}%"); continue
    w = stats.wilcoxon(a[s], a["random"])
    print(f"  {s:8s} {100*a[s].mean():.1f}%  diff {100*(a[s]-a['random']).mean():+.1f} pts  wins {np.sum(a[s]>a['random'])}/{len(a)}  p={w.pvalue:.2g}")

print("\noptimum finding: regret of best acquired Good simulation vs best Good in pool (%), median:")
for n in [25, 50, 75, 100]:
    x = d[d.n == n].groupby("strategy").regret.median()
    print(f"  n={n:3d}  " + "  ".join(f"{s}={100*x[s]:.1f}" for s in S))
print("\nshare of acquired runs within 10% of a threshold, n=100:")
print("  " + "  ".join(f"{s}={100*v:.0f}%" for s, v in d[d.n == 100].groupby("strategy").frac_near.mean()[S].items()))
