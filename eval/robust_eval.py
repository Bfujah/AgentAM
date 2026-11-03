"""Offline evaluation of margin-aware optimization (mcp_find_robust) against plain minimum-spatter optimization.

For each query: the plain optimum (no margins) and the robust optimum (default margins) on the same 46x46 grid,
their margins to every threshold, and the probability that each stays Good under
  scenario A: process variability, P and V perturbed by N(0, 2%) (1 sigma);
  scenario B: scenario A plus N(0, 5%) relative error on predicted D, W and L (surrogate error).
Also a trade-off curve: spatter of the robust optimum and its P(Good) as the depth margin grows.

  python robust_eval.py --server_py ../server/server.py
Writes results/robust_cases.csv, results/robust_tradeoff.csv and results/table_robust.tex.
"""
import argparse, csv, importlib.util, json, os, random

def load(path):
    spec = importlib.util.spec_from_file_location("srv", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def classify(m, D, W, L):
    dw = D / W if W > 0 else 9; lw = L / W if W > 0 else 0
    if dw > m.KEYHOLE_DW_THRESHOLD: return "Keyhole"
    if D < m.LOF_DEPTH_THRESHOLD: return "Lack of Fusion"
    if m.BALLING_LW_MIN < lw < m.BALLING_LW_MAX: return "Balling"
    return "Good"

def p_good(m, P, V, n, sp, sg, rng):
    ok = 0
    for _ in range(n):
        Pp, Vp = P * (1 + rng.gauss(0, sp)), V * (1 + rng.gauss(0, sp))
        g = m.predict_geometry(Pp, Vp)
        f = (lambda: 1 + rng.gauss(0, sg)) if sg else (lambda: 1.0)
        ok += classify(m, g["depth"] * f(), g["width"] * f(), g["length"] * f()) == "Good"
    return ok / n

def c7_to_args(c, rng_P, rng_V):
    a = {}
    Pb, Vb = list(rng_P), list(rng_V)
    if "P_max" in c: Pb[1] = c["P_max"]
    if "V_min" in c: Vb[0] = c["V_min"]
    if "fix_P" in c: Pb = [c["fix_P"], c["fix_P"]]
    if "fix_V" in c: Vb = [c["fix_V"], c["fix_V"]]
    a["P_bounds"], a["V_bounds"] = Pb, Vb
    for k, arg in (("depth", "depth_range"), ("width", "width_range"), ("length", "length_range"), ("J", "j_limits")):
        if k in c: a[arg] = c[k]
    if "spatter_max" in c: a["max_spatter"] = c["spatter_max"]
    return a

def main(a):
    m = load(a.server_py); rng = random.Random(0); R = m.get_material_ranges("SS316L")
    cases = [("Q1 grid scan", dict(P_bounds=[200, 400], V_bounds=[300, 1000], max_spatter=100000)),
             ("Q2 depth + spatter", dict(depth_range=[120, 180], max_spatter=90000)),
             ("Q3 energy window", dict(j_limits=[0.4, 0.6])),
             ("Q4 depth + length + spatter", dict(depth_range=[150, 250], length_range=[350, 500], max_spatter=120000)),
             ("Unconstrained minimum spatter", dict())]
    for it in map(json.loads, open(a.bench)):
        if it["category"].startswith("C7"): cases.append((it["id"], c7_to_args(it["constraints"], R["P"], R["V"])))
    zero = dict(min_depth_margin=0, min_dw_margin=0, min_lw_margin=0, min_range_margin=0)
    rows = []
    for name, args in cases:
        pl = m._find_robust_impl(grid=a.grid, **args, **zero)["data"]["best"]
        rb = m._find_robust_impl(grid=a.grid, **args)["data"]
        rbest = rb["best"]
        row = dict(case=name)
        for tag, p in (("plain", pl), ("robust", rbest)):
            if p is None: row[f"{tag}_status"] = "none"; continue
            row.update({f"{tag}_P": p["P"], f"{tag}_V": p["V"], f"{tag}_spatter": p["spatter_volume"],
                        f"{tag}_depth_margin": p["margins"]["depth_um"], f"{tag}_dw_margin": p["margins"]["dw"],
                        f"{tag}_range_margin": p["margins"]["range"], f"{tag}_nearest": p["margins"]["nearest_threshold"],
                        f"{tag}_pgood_A": p_good(m, p["P"], p["V"], a.n, 0.02, 0, rng),
                        f"{tag}_pgood_B": p_good(m, p["P"], p["V"], a.n, 0.02, 0.05, rng)})
        if pl and rbest: row["cost_pct"] = round(100 * (rbest["spatter_volume"] - pl["spatter_volume"]) / pl["spatter_volume"], 1)
        rows.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    os.makedirs("results", exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "case", k))
    with open("results/robust_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    trade = []
    for dm in (0, 5, 10, 15, 20, 25, 30, 40):
        for rm in (0.0, 0.05):
            p = m._find_robust_impl(grid=a.grid, min_depth_margin=dm, min_dw_margin=0.10 if dm else 0, min_lw_margin=0.15 if dm else 0, min_range_margin=rm)["data"]["best"]
            if p: trade.append(dict(depth_margin=dm, range_margin=rm, P=p["P"], V=p["V"], spatter=p["spatter_volume"],
                                    pgood_A=p_good(m, p["P"], p["V"], a.n, 0.02, 0, rng), pgood_B=p_good(m, p["P"], p["V"], a.n, 0.02, 0.05, rng)))
            print("trade", trade[-1] if p else (dm, rm, None))
    with open("results/robust_tradeoff.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(trade[0])); w.writeheader(); w.writerows(trade)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--server_py", required=True); ap.add_argument("--bench", default="benchmark.jsonl")
    ap.add_argument("--grid", type=int, default=46); ap.add_argument("--n", type=int, default=1000)
    main(ap.parse_args())
