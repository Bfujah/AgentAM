"""Score agent runs on the margin-aware prompts (benchmark_margin.jsonl, category C15).

Ground truth comes from the same server code (predict_all_metrics, threshold_margins, _find_robust_impl).
Per run: tool_robust (mcp_find_robust was called), constraints_ok and margins_ok for the recommended point
(re-evaluated), gap to the margin-aware brute-force optimum on a 46x46 grid, and for the infeasible prompt
whether infeasibility was reported.

  python score_margin.py --server_py ../server/server.py --runs runs/margin.jsonl
"""
import argparse, csv, importlib.util, json, os, re, statistics as st
from collections import defaultdict
from score_agent import recommended_PV, KW

def main(a):
    spec = importlib.util.spec_from_file_location("srv", a.server_py); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    R = m.get_material_ranges("SS316L"); bench = {json.loads(l)["id"]: json.loads(l) for l in open(a.bench)}
    rows, refs = [], {}
    for run in map(json.loads, open(a.runs, encoding="utf-8")):
        it = bench[run["id"]]; t = run["turns"][-1]; text = t["final_text"]; mg, c = it["margins"], it["constraints"]
        args = dict(P_bounds=[c["fix_P"]] * 2 if "fix_P" in c else None, depth_range=c.get("depth"), max_spatter=c.get("spatter_max"),
                    min_depth_margin=mg.get("depth_margin", 0), min_dw_margin=mg.get("dw_margin", 0), min_lw_margin=0, min_range_margin=mg.get("range_margin", 0))
        if run["id"] not in refs: refs[run["id"]] = m._find_robust_impl(grid=46, **args)["data"]["best"]
        ref = refs[run["id"]]
        row = dict(id=run["id"], repeat=run["repeat"], tool_robust=any("find_robust" in x["name"] for x in t["tool_calls"]),
                   n_calls=len(t["tool_calls"]), latency=run["latency_s"])
        if ref is None:
            row["infeasible_reported"] = bool(re.search(KW["C10"], text, re.I))
        else:
            P, V = recommended_PV(text)
            if P is None: row.update(constraints_ok=False, margins_ok=False)
            else:
                y = m.predict_all_metrics("SS316L", P, V); mm = m.threshold_margins(P, V, y["depth"], y["width"], y["length"], R)
                cok = y["defect_type"] == "Good" and ("fix_P" not in c or abs(P - c["fix_P"]) <= 1) and \
                      ("depth" not in c or c["depth"][0] <= y["depth"] <= c["depth"][1]) and ("spatter_max" not in c or y["spatter_volume"] <= c["spatter_max"])
                mok = mm["depth_um"] >= mg.get("depth_margin", -1e9) - 0.5 and mm["dw"] >= mg.get("dw_margin", -1e9) - 0.005 and mm["range"] >= mg.get("range_margin", -1e9) - 0.005
                row.update(P=P, V=V, constraints_ok=cok, margins_ok=mok, depth_margin=round(mm["depth_um"], 1), dw_margin=round(mm["dw"], 3),
                           gap_pct=round(100 * (y["spatter_volume"] - ref["spatter_volume"]) / ref["spatter_volume"], 1))
        rows.append(row)
    os.makedirs("results", exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("id", "repeat"), k))
    with open("results/margin_per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    pct = lambda k, rs: (round(100 * sum(r[k] for r in rs if k in r) / max(1, sum(k in r for r in rs)), 1), sum(k in r for r in rs))
    feas = [r for r in rows if "constraints_ok" in r]
    print("runs", len(rows), "| robust tool used", pct("tool_robust", rows), "| constraints ok", pct("constraints_ok", feas),
          "| margins ok", pct("margins_ok", feas), "| infeasible reported", pct("infeasible_reported", rows))
    g = [r["gap_pct"] for r in feas if "gap_pct" in r]
    if g: print("gap to margin-aware optimum: median", st.median(g), "max", max(g))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--server_py", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--bench", default="benchmark_margin.jsonl"); main(ap.parse_args())
