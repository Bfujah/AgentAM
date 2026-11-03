"""Pool-based active-learning study on the 279 FLOW-3D single-track simulations (SS316L).

Each FLOW-3D run in the dataset is treated as a simulation that can be "requested" from a pool; its
melt pool geometry and spatter are revealed only once it is acquired. Starting from a small random
set, each strategy chooses the next batch of simulations, the surrogates are refitted (same model
form as the deployed server: PolynomialFeatures(4) + Ridge(alpha=10) on (P, V) for D, W, L), and the
defect map is scored on a held-out test set that is never acquired.

Strategies
  random     uniform random from the pool
  space      greedy maximin (farthest point in normalised P-V), non-adaptive
  qbc        query by committee: bootstrap committee of surrogates, maximum vote entropy of the class
  margin     smallest normalised distance of the predicted geometry to a regime threshold
             (D/W = 1.25, D = 137 um, L/W = 2.35 / 3.7), with batch diversity
  exploit    lowest predicted spatter among points predicted Good (what mcp_suggest_next does)
  hybrid     2 of every 5 points from exploit, 3 from qbc

  python active_learning.py --data ../data/Agentic_defect_Dataset.csv --repeats 30
Writes results/al_curves.csv (per repeat, strategy, n) and results/al_summary.csv.
"""
import argparse, csv, os, warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")

CLASSES = ["Good", "Keyhole", "Lack of Fusion", "Balling"]
P90, V90 = 400.0, 1400.0


def label(D, W, L):
    D, W, L = np.asarray(D, float), np.asarray(W, float), np.asarray(L, float)
    dw, lw = D / W, L / W
    out = np.full(D.shape, "Good", dtype=object)
    ball = (lw > 2.35) & (lw < 3.7)
    out[ball] = "Balling"
    out[D < 137] = "Lack of Fusion"
    out[dw > 1.25] = "Keyhole"
    return out


def geo_model():
    return make_pipeline(PolynomialFeatures(degree=4, include_bias=False), Ridge(alpha=10.0))


def spat_feats(X):
    tP, tV = X[:, 0] / P90, X[:, 1] / V90
    return np.column_stack([tV / tP, tV ** 3, tV ** 4, tP ** 3])


def spat_model():
    return make_pipeline(PolynomialFeatures(degree=4), LinearRegression())


def fit_predict(X, Y, tr, Xq):
    """fit D, W, L on rows tr, predict at Xq"""
    return {k: geo_model().fit(X[tr], Y[k][tr]).predict(Xq) for k in ("Depth", "Width", "Length")}


def margins(pred):
    """normalised distance of predicted geometry to the nearest regime threshold"""
    D, W, L = pred["Depth"], pred["Width"], pred["Length"]
    dw, lw = D / W, L / W
    m = np.column_stack([np.abs(dw - 1.25) / 1.25, np.abs(D - 137) / 137,
                         np.abs(lw - 2.35) / 2.35, np.abs(lw - 3.7) / 3.7])
    return m.min(axis=1)


def norm(X):
    return np.column_stack([(X[:, 0] - 160) / 270, (X[:, 1] - 100) / 1400])


def pick_diverse(order, Xn, k, min_d=0.06):
    """take candidates in order, skipping ones closer than min_d to an already chosen one in this batch"""
    chosen = []
    for i in order:
        if all(np.hypot(*(Xn[i] - Xn[j])) >= min_d for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    for i in order:  # fill if diversity left the batch short
        if len(chosen) == k: break
        if i not in chosen: chosen.append(i)
    return chosen


def acquire(strategy, X, Xn, Y, lab, pool, k, rng):
    pool = np.array(sorted(pool))
    if strategy == "random":
        return list(rng.choice(pool, size=k, replace=False))
    if strategy == "space":
        chosen, ref = [], list(lab)
        for _ in range(k):
            d = np.min(np.linalg.norm(Xn[pool][:, None, :] - Xn[ref][None, :, :], axis=2), axis=1)
            d[np.isin(pool, chosen)] = -1
            i = pool[int(np.argmax(d))]; chosen.append(i); ref.append(i)
        return chosen
    lab = np.array(sorted(lab))
    if strategy == "margin":
        m = margins(fit_predict(X, Y, lab, X[pool]))
        order = pool[np.argsort(m + 1e-9 * rng.random(len(m)))]
        return pick_diverse(order, Xn, k)
    if strategy == "qbc":
        votes = []
        for b in range(15):
            bs = rng.choice(lab, size=len(lab), replace=True)
            p = fit_predict(X, Y, bs, X[pool]); votes.append(label(p["Depth"], p["Width"], p["Length"]))
        votes = np.array(votes)
        ent = np.zeros(len(pool))
        for c in CLASSES:
            f = (votes == c).mean(axis=0); ent -= np.where(f > 0, f * np.log(f + 1e-12), 0)
        m = margins(fit_predict(X, Y, lab, X[pool]))
        order = pool[np.lexsort((m, -ent))]  # max entropy, ties broken by smallest margin
        return pick_diverse(order, Xn, k)
    if strategy == "exploit":
        p = fit_predict(X, Y, lab, X[pool]); cls = label(p["Depth"], p["Width"], p["Length"])
        s = spat_model().fit(spat_feats(X[lab]), Y["Spatter_Volume"][lab]).predict(spat_feats(X[pool]))
        s = np.where(cls == "Good", s, np.inf)
        order = pool[np.argsort(s + 1e-9 * rng.random(len(s)))]
        return pick_diverse(order, Xn, k)
    if strategy == "hybrid":
        # goal-directed: part of each batch exploits (lowest predicted spatter, Good), the rest is QBC
        ke = max(1, k // 2)
        a = acquire("exploit", X, Xn, Y, lab, pool, ke, rng)
        b = acquire("qbc", X, Xn, Y, lab, set(pool) - set(a), k - ke, rng)
        return list(a) + list(b)
    raise ValueError(strategy)


def main(a):
    rows = list(csv.DictReader(open(a.data)))
    X = np.array([[float(r["Power"]), float(r["Velocity"])] for r in rows]); Xn = norm(X)
    Y = {k: np.array([float(r[k]) for r in rows]) for k in ("Depth", "Width", "Length", "Spatter_Volume")}
    y = label(Y["Depth"], Y["Width"], Y["Length"])
    N = len(rows)
    print("n =", N, {c: int((y == c).sum()) for c in CLASSES})
    strategies = a.strategies.split(",")
    curves = []
    for rep in range(a.repeats):
        rng0 = np.random.default_rng(1000 + rep)
        # stratified 20 % test split
        test = []
        for c in CLASSES:
            idx = np.where(y == c)[0]; rng0.shuffle(idx); test += list(idx[:int(round(a.test_frac * len(idx)))])
        test = np.array(sorted(test)); avail = np.setdiff1d(np.arange(N), test)
        init = list(rng0.choice(avail, size=a.n0, replace=False))
        # reference: all available simulations
        pf = fit_predict(X, Y, avail, X[test]); yf = label(pf["Depth"], pf["Width"], pf["Length"])
        full_acc = accuracy_score(y[test], yf)
        good_av = avail[y[avail] == "Good"]; best_good = Y["Spatter_Volume"][good_av].min()
        for st in strategies:
            rng = np.random.default_rng(rep * 97 + 7 * strategies.index(st))
            lab = list(init); pool = set(avail) - set(lab)
            while True:
                p = fit_predict(X, Y, np.array(lab), X[test]); yh = label(p["Depth"], p["Width"], p["Length"])
                L = np.array(lab); g = L[y[L] == "Good"]
                best = Y["Spatter_Volume"][g].min() if len(g) else np.nan
                curves.append(dict(rep=rep, strategy=st, n=len(lab), acc=accuracy_score(y[test], yh),
                                   f1=f1_score(y[test], yh, labels=CLASSES, average="macro", zero_division=0),
                                   full_acc=full_acc, n_avail=len(avail), n_good_acq=len(g),
                                   best_good_spatter=best, regret=(best - best_good) / best_good if len(g) else np.nan,
                                   frac_near=float(np.mean(margins({k: Y[k][L] for k in ("Depth", "Width", "Length")}) < 0.10))))
                if len(lab) >= a.nmax or not pool: break
                new = acquire(st, X, Xn, Y, lab, pool, min(a.batch, len(pool)), rng)
                lab += [int(i) for i in new]; pool -= set(new)
        print(f"rep {rep}: full-data test accuracy {100*full_acc:.1f}%", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "al_curves.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(curves[0])); w.writeheader(); w.writerows(curves)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/Agentic_defect_Dataset.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--n0", type=int, default=15)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--nmax", type=int, default=150)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--strategies", default="random,space,qbc,margin,exploit")
    main(ap.parse_args())
