"""Robustness of plain vs margin-aware optima under alternative perturbation models (reviewer point 4).
Optima: the 17 queries with both optima in ../eval/results/robust_cases.csv. Surrogate: deployed form
(PolynomialFeatures(4, no bias) + Ridge(alpha=10)) refitted on the 279 tracks; P and V not clamped inside the Monte Carlo, as in robust_eval.py (reproduces the published medians).
Scenarios (1000 Monte Carlo samples each, same seed):
  A      independent N(0, s) relative error on P and V, s = 2% (as in the paper); also s = 1%, 3%
  A-rho  P and V errors correlated, rho = +0.8 (energy-preserving drift) and -0.8 (energy-changing drift), s = 2%
  B      A(2%) + independent N(0, 5%) relative error on D, W, L (as in the paper); also 3%, 10%
  B-res  A(2%) + geometry errors bootstrapped jointly from 5-fold cross-validated relative residuals (D, W, L),
         either from all 279 tracks or from the 30 tracks nearest the point in normalized P-V
Writes robustness_variants.csv (per query) and robustness_summary.csv (median / min over queries)."""
import numpy as np, pandas as pd, random, warnings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
d = pd.read_csv("../data/Agentic_defect_Dataset.csv"); X = d[["Power","Velocity"]].values.astype(float)
K = ("Depth","Width","Length"); Y = np.c_[[d[k].values for k in K]].T.astype(float)
mk = lambda: make_pipeline(PolynomialFeatures(4, include_bias=False), Ridge(alpha=10.0))
import joblib
M = [joblib.load(f"../../spatter_mcp/{k}_poly_model.pkl") for k in K]  # deployed models (residuals still from refits)
# cross-validated relative residuals (pooled over 5 fold assignments)
res = np.zeros((5, len(d), 3))
for rep in range(5):
    idx = list(range(len(d))); random.Random(rep).shuffle(idx)
    for f in [idx[i::5] for i in range(5)]:
        tr = [i for i in idx if i not in set(f)]
        for j in range(3): res[rep, f, j] = Y[f, j] / mk().fit(X[tr], Y[tr, j]).predict(X[f]) - 1
RES = res.reshape(-1, 3); RES_track = res  # (rep, track, 3)
print("CV relative residual sd  D %.3f  W %.3f  L %.3f" % tuple(RES.std(0)))
CLAMP = False
Xn = (X - [160, 100]) / [270, 1400]
def geom(P, V):
    Z = np.c_[np.clip(P, 160, 430), np.clip(V, 100, 1500)] if CLAMP else np.c_[P, V]
    return np.c_[[m.predict(Z) for m in M]].T
def good(G):
    D, W, L = G[:, 0], G[:, 1], G[:, 2]
    return (D / W <= 1.25) & (D >= 137) & ~((L / W > 2.35) & (L / W < 3.7))
def p_good(P, V, s=0.02, rho=0.0, geo=None, sg=0.0, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n) * s
    G = geom(P * (1 + z[:, 0]), V * (1 + z[:, 1]))
    if geo == "normal": G = G * (1 + rng.normal(0, sg, G.shape))
    elif geo == "res_all": G = G * (1 + RES[rng.integers(0, len(RES), n)])
    elif geo == "res_local":
        near = np.argsort(((Xn - ((P - 160) / 270, (V - 100) / 1400)) ** 2).sum(1))[:30]
        pool = RES_track[:, near, :].reshape(-1, 3); G = G * (1 + pool[rng.integers(0, len(pool), n)])
    return good(G).mean()
SC = {"A 2% (paper)": dict(), "A 1%": dict(s=0.01), "A 3%": dict(s=0.03),
      "A 2%, rho=+0.8": dict(rho=0.8), "A 2%, rho=-0.8": dict(rho=-0.8),
      "B 5% (paper)": dict(geo="normal", sg=0.05), "B 3%": dict(geo="normal", sg=0.03), "B 10%": dict(geo="normal", sg=0.10),
      "B CV residuals (all tracks)": dict(geo="res_all"), "B CV residuals (30 nearest)": dict(geo="res_local")}
cases = pd.read_csv("../eval/results/robust_cases.csv").dropna(subset=["plain_P", "robust_P"])
rows = []
for _, c in cases.iterrows():
    for tag in ("plain", "robust"):
        r = dict(case=c.case, optimum=tag, P=c[f"{tag}_P"], V=c[f"{tag}_V"])
        for name, kw in SC.items(): r[name] = round(p_good(c[f"{tag}_P"], c[f"{tag}_V"], **kw), 3)
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv("robustness_variants.csv", index=False)
summ = []
for name in SC:
    s = {"scenario": name}
    for tag in ("plain", "robust"):
        v = R[R.optimum == tag][name]; s[f"{tag}_median"] = round(v.median(), 2); s[f"{tag}_min"] = round(v.min(), 2)
    summ.append(s)
S = pd.DataFrame(summ); S.to_csv("robustness_summary.csv", index=False)
print("queries:", len(cases)); print(S.to_string(index=False))
