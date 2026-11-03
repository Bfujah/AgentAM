"""Extra checks on the active-learning study.
1. paired comparison of each strategy with space-filling and with the full-data model at fixed n
2. fixed coarse-grid designs (what an engineer would plan without active learning), on the same test splits
3. joint goal: n at which the map is within one test error of full data AND the best Good run is within 5 % of the pool optimum
usage: python extra_al.py results_dir data.csv test_frac"""
import sys, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score
import active_learning as al

rd, data, tf = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = pd.read_csv(f"{rd}/al_curves.csv"); S = list(dict.fromkeys(d.strategy))
full = d.groupby("rep").full_acc.first(); R = d.rep.nunique()
piv = d.pivot_table(index=["rep", "n"], columns="strategy", values="acc")

print("1. paired differences in test accuracy (pts), mean [95% CI], Wilcoxon p")
for n in [35, 50, 75, 100]:
    x = piv.xs(n, level=1)
    for s in S:
        if s in ("space",): continue
        dd = 100 * (x[s] - x["space"]); p = stats.wilcoxon(x[s], x["space"]).pvalue if (dd != 0).any() else 1
        df = 100 * (x[s] - full.loc[x.index]); pf = stats.wilcoxon(x[s], full.loc[x.index]).pvalue if (df != 0).any() else 1
        print(f"  n={n:3d} {s:8s} vs space {dd.mean():+5.1f} [{1.96*dd.std()/np.sqrt(R):.1f}] p={p:.2g}   vs full data {df.mean():+5.1f} [{1.96*df.std()/np.sqrt(R):.1f}] p={pf:.2g}")

# 2. fixed coarse designs, same splits as active_learning.py
rows = list(__import__("csv").DictReader(open(data)))
X = np.array([[float(r["Power"]), float(r["Velocity"])] for r in rows])
Y = {k: np.array([float(r[k]) for r in rows]) for k in ("Depth", "Width", "Length", "Spatter_Volume")}
y = al.label(Y["Depth"], Y["Width"], Y["Length"]); N = len(rows)
Ps, Vs = sorted(set(X[:, 0])), sorted(set(X[:, 1]))
designs = {"P all x V every 2nd": (Ps, Vs[::2]), "P every 2nd x V every 2nd": (Ps[::2], Vs[::2]),
           "P every 2nd x V every 3rd": (Ps[::2], Vs[::3]), "P every 3rd x V every 3rd": (Ps[::3], Vs[::3]),
           "P every 3rd x V every 4th": (Ps[::3], Vs[::4])}
res = {k: [] for k in designs}; sizes = {k: [] for k in designs}
for rep in range(R):
    rng0 = np.random.default_rng(1000 + rep); test = []
    for c in al.CLASSES:
        idx = np.where(y == c)[0]; rng0.shuffle(idx); test += list(idx[:int(round(tf * len(idx)))])
    test = np.array(sorted(test)); avail = np.setdiff1d(np.arange(N), test)
    for k, (pp, vv) in designs.items():
        tr = np.array([i for i in avail if X[i, 0] in pp and X[i, 1] in vv])
        p = al.fit_predict(X, Y, tr, X[test]); yh = al.label(p["Depth"], p["Width"], p["Length"])
        res[k].append(accuracy_score(y[test], yh)); sizes[k].append(len(tr))
print("\n2. fixed coarse-grid designs (test accuracy, same splits)")
for k in designs:
    a = np.array(res[k]); n = int(np.mean(sizes[k]))
    near = min(piv.index.get_level_values(1).unique(), key=lambda m: abs(m - n))
    x = piv.xs(near, level=1)
    print(f"  {k:28s} n~{n:3d}  acc {100*a.mean():5.1f}%   at n={near}: " + "  ".join(f"{s}={100*x[s].mean():.1f}" for s in S)
          + f"   qbc-grid {100*(x['qbc'].values-a).mean():+.1f} pts p={stats.wilcoxon(x['qbc'].values, a).pvalue:.2g}")

print("\n3. joint goal: map within one test error of full data AND best Good run within 5% of pool optimum")
ntest = int(round(tf * N)); tol = 1.0 / ntest
for s in S:
    v = []
    for rep, g in d[d.strategy == s].groupby("rep"):
        g = g.sort_values("n"); ok = ((g.acc >= full[rep] - tol - 1e-9) & (g.regret <= 0.05)).values; ns = g.n.values
        v.append(next((ns[i] for i in range(len(ns)) if ok[i:].all()), np.nan))
    v = np.array(v, float)
    print(f"  {s:8s} median {np.nanmedian(v):6.1f}  IQR {np.nanpercentile(v,25):.0f}-{np.nanpercentile(v,75):.0f}  reached {np.isfinite(v).sum()}/{len(v)}")
