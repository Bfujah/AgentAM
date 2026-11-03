"""Threshold and rule-order sensitivity of the geometry-derived defect regimes (reviewer points 1-2).
Surrogate form as deployed: PolynomialFeatures(4, no bias) + Ridge(alpha=10) on (P, V) for D, W, L.
For each criterion setting: in-sample accuracy (full fit), 5-fold CV accuracy (5 repeats), and the
Good-window area fraction on a 120x120 grid over P=160-430 W, V=100-1500 mm/s. Writes sensitivity.csv."""
import numpy as np, pandas as pd, random, warnings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
d = pd.read_csv("../data/Agentic_defect_Dataset.csv")
X = d[["Power","Velocity"]].values.astype(float); G = {k: d[k].values.astype(float) for k in ("Depth","Width","Length")}
mk = lambda: make_pipeline(PolynomialFeatures(4, include_bias=False), Ridge(alpha=10.0))
import joblib
full = {k: joblib.load(f"../../spatter_mcp/{k}_poly_model.pkl") for k in G}  # deployed models (cross-validation still refits)
P, V = np.meshgrid(np.linspace(160,430,120), np.linspace(100,1500,120)); XG = np.c_[P.ravel(), V.ravel()]
grid = {k: full[k].predict(XG) for k in G}; ins = {k: full[k].predict(X) for k in G}
folds_cache = []
for rep in range(5):
    idx = list(range(len(d))); random.Random(rep).shuffle(idx); fs = [idx[i::5] for i in range(5)]
    gp = {k: np.zeros(len(d)) for k in G}
    for f in fs:
        tr = [i for i in idx if i not in set(f)]
        for k in G: gp[k][f] = mk().fit(X[tr], G[k][tr]).predict(X[f])
    folds_cache.append(gp)
def lab(D, W, L, kh=1.25, t=137.0, bl=2.35, bu=3.7, order=("K","L","B")):
    dw, lw = D/W, L/W
    tests = {"K": dw > kh, "L": D < t, "B": (lw > bl) & (lw < bu)}
    names = {"K": "Keyhole", "L": "Lack of Fusion", "B": "Balling"}
    out = np.full(D.shape, "Good", dtype=object); done = np.zeros(D.shape, bool)
    for o in order:
        m = tests[o] & ~done; out[m] = names[o]; done |= m
    return out
def evaluate(**kw):
    yt = lab(G["Depth"], G["Width"], G["Length"], **kw)
    yi = lab(ins["Depth"], ins["Width"], ins["Length"], **kw)
    cv = [np.mean(lab(gp["Depth"], gp["Width"], gp["Length"], **kw) == yt) for gp in folds_cache]
    yg = lab(grid["Depth"], grid["Width"], grid["Length"], **kw)
    cnt = {c: int((yt == c).sum()) for c in ("Good","Keyhole","Lack of Fusion","Balling")}
    return dict(acc_in=round(100*np.mean(yi == yt),1), acc_cv=round(100*np.mean(cv),1), good_area=round(100*np.mean(yg=="Good"),1), **{"n_"+k.split()[0]: v for k, v in cnt.items()})
rows = []
base = dict(kh=1.25, t=137.0, bl=2.35, bu=3.7)
rows.append(dict(param="baseline", value="", **evaluate(**base)))
for v in [1.0,1.1,1.2,1.3,1.4,1.5]: rows.append(dict(param="keyhole D/W", value=v, **evaluate(**{**base, "kh": v})))
for v in [110,120,130,145,155,165]: rows.append(dict(param="layer t (um)", value=v, **evaluate(**{**base, "t": float(v)})))
for v in [2.1,2.2,2.3,2.4,2.5,2.6]: rows.append(dict(param="balling lower L/W", value=v, **evaluate(**{**base, "bl": v})))
for v in [3.1,3.4,4.0,np.inf]: rows.append(dict(param="balling upper L/W", value=("none" if np.isinf(v) else v), **evaluate(**{**base, "bu": v})))
for o in [("K","B","L"),("L","K","B"),("B","K","L"),("L","B","K")]: rows.append(dict(param="rule order", value="-".join(o), **evaluate(**{**base, "order": o})))
r = pd.DataFrame(rows); r.to_csv("sensitivity.csv", index=False); print(r.to_string(index=False))
