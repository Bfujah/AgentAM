"""In-sample vs cross-validated defect classification on the FLOW-3D single-track data.

The surrogate geometry models (PolynomialFeatures(4) + Ridge, from spatter_mcp) are refitted on k-1 folds
of data/Agentic_defect_Dataset.csv and used to classify the held-out fold; labels come from the simulated
geometry with the same criteria. Repeated over several random fold splits.

  python cv_validation.py --server_py ../server/server.py
Writes results/cv_validation.csv and prints in-sample and cross-validated accuracy, per-class recall and R^2.
"""
import argparse, csv, importlib.util, random, warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, r2_score, confusion_matrix
warnings.filterwarnings("ignore")
CLASSES = ["Good", "Keyhole", "Lack of Fusion", "Balling"]

def rebuild(pipe):
    """same form as the deployed model: PolynomialFeatures(degree) + Ridge(alpha), unfitted
    (rebuilt by hand because pickled pipelines from another scikit-learn version cannot be cloned)"""
    poly, ridge = pipe.steps[0][1], pipe.steps[-1][1]
    return make_pipeline(PolynomialFeatures(degree=poly.degree, include_bias=getattr(poly, "include_bias", True)),
                         Ridge(alpha=getattr(ridge, "alpha", 1.0), fit_intercept=getattr(ridge, "fit_intercept", True)))

def label(D, W, L):
    dw, lw = D / W, L / W
    if dw > 1.25: return "Keyhole"
    if D < 137: return "Lack of Fusion"
    if 2.35 < lw < 3.7: return "Balling"
    return "Good"

def report(name, y, yh):
    acc = accuracy_score(y, yh); f1 = f1_score(y, yh, average="weighted")
    rec = recall_score(y, yh, labels=CLASSES, average=None, zero_division=0)
    pre = precision_score(y, yh, labels=CLASSES, average=None, zero_division=0)
    print(f"{name}: accuracy {100*acc:.1f}%  weighted F1 {f1:.3f}  errors {sum(a!=b for a,b in zip(y,yh))}")
    print("   recall   " + "  ".join(f"{c}={r:.3f}" for c, r in zip(CLASSES, rec)))
    print("   precision" + "  ".join(f" {c}={p:.3f}" for c, p in zip(CLASSES, pre)))
    return dict(case=name, accuracy=round(100*acc, 2), f1=round(f1, 3), **{f"recall_{c}": round(r, 3) for c, r in zip(CLASSES, rec)},
                **{f"precision_{c}": round(p, 3) for c, p in zip(CLASSES, pre)})

def main(a):
    spec = importlib.util.spec_from_file_location("srv", a.server_py); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    M = m.get_models(); rows = list(csv.DictReader(open(a.data)))
    X = np.array([[float(r["Power"]), float(r["Velocity"])] for r in rows])
    Y = {k: np.array([float(r[k]) for r in rows]) for k in ("Depth", "Width", "Length")}
    y_true = [label(d, w, l) for d, w, l in zip(Y["Depth"], Y["Width"], Y["Length"])]
    print("labels from simulated geometry:", {c: y_true.count(c) for c in CLASSES}, "n =", len(rows))
    out = []
    # in-sample: the deployed server models
    pred = {k: M[k].predict(X) for k in Y}
    y_in = [label(d, w, l) for d, w, l in zip(pred["Depth"], pred["Width"], pred["Length"])]
    out.append(report("in-sample (deployed models)", y_true, y_in))
    for k in Y: print(f"   R2 {k} in-sample {r2_score(Y[k], pred[k]):.3f}")
    print("   confusion (rows true, cols pred, order", CLASSES, ")\n", confusion_matrix(y_true, y_in, labels=CLASSES))
    # k-fold cross-validation, repeated
    accs, all_t, all_p, r2s = [], [], [], {k: [] for k in Y}
    for rep in range(a.repeats):
        idx = list(range(len(rows))); random.Random(rep).shuffle(idx); folds = [idx[i::a.k] for i in range(a.k)]
        yp = [None] * len(rows); gp = {k: np.zeros(len(rows)) for k in Y}
        for f in folds:
            tr = [i for i in idx if i not in set(f)]
            for k in Y:
                mdl = rebuild(M[k]).fit(X[tr], Y[k][tr]); gp[k][f] = mdl.predict(X[f])
        yp = [label(d, w, l) for d, w, l in zip(gp["Depth"], gp["Width"], gp["Length"])]
        accs.append(accuracy_score(y_true, yp)); all_t += y_true; all_p += yp
        for k in Y: r2s[k].append(r2_score(Y[k], gp[k]))
    r = report(f"{a.k}-fold CV x{a.repeats} (pooled)", all_t, all_p)
    r["accuracy_sd"] = round(100 * float(np.std(accs)), 2); out.append(r)
    print(f"   accuracy over repeats: mean {100*np.mean(accs):.1f}%  sd {100*np.std(accs):.1f}%  min {100*min(accs):.1f}%")
    for k in Y: print(f"   R2 {k} cross-validated {np.mean(r2s[k]):.3f}")
    print("   confusion (pooled)\n", confusion_matrix(all_t, all_p, labels=CLASSES))
    with open("results/cv_validation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, sorted({k for o in out for k in o}, key=lambda k: (k != "case", k))); w.writeheader(); w.writerows(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--server_py", required=True)
    ap.add_argument("--data", default="../data/Agentic_defect_Dataset.csv"); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20); main(ap.parse_args())
