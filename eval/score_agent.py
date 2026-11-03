"""Score agent runs against the benchmark.

Ground truth is computed by calling the SAME MCP server directly (no LLM), so the
score measures the agent layer only: parsing, tool choice, faithfulness, constraint
handling and safety behaviour - not the surrogate's physics accuracy.

  python score_agent.py --server "python /path/to/spatter_mcp_server.py" --runs runs/claude-opus-4-5.jsonl

Outputs: results/summary.csv, results/table_agent_eval.tex, results/review_sheet.csv
(review_sheet holds the items that need a human rating; have two people rate them
independently and report Cohen's kappa).
"""
import argparse, asyncio, json, os, sys, re, shlex, statistics as st
from collections import defaultdict
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---- adapt these to your server's tool names / argument names if they differ
PREDICT_TOOL = "mcp_predict_all"
FAMILY = {"predict": ("predict",), "line_scan": ("line_scan",), "grid_scan": ("grid",),
          "optimise": ("suggest", "find_best", "find_robust", "filtered_grid", "optimi")}
ARG_P, ARG_V = ("P", "power", "laser_power"), ("V", "velocity", "scan_speed", "speed")
P_LIM, V_LIM = (160, 430), (100, 1500)
LOF_DEPTH, KH_DW, BALL_LW = 137, 1.25, (2.35, 3.7)

KW = {
    "C8": r"clamp|outside|out of (the )?range|beyond|extrapolat|exceed|limit|valid range|trained range",
    "C9": r"not (an? )?(adjustable |model |direct |trained )?(model )?(input|parameter|variable)|(does not|doesn't|do not|don't) (use|include|account|consider|capture)|not used by the model|not (captured|included|considered|accounted)|won't change|fixed|not supported|cannot (account|include|vary)|isn't (an? )?(adjustable|input)|not part of the model|assum",
    "C10": r"no (point|parameter|combination|feasible|solution|defect-free|\"?good)|none of|not possible|not (achievable|attainable|feasible)|(does not|doesn't|do not|don't) exist|cannot (be )?(satisf|met|achiev|reach)|impossible|infeasible|conflict|relax|outside (of )?(the )?(trained|valid|model'?s?) (parameter )?range",
    "C11": r"not (an? )?(supported|available|trained|covered)|only (trained|available|supports?)|SS ?316L only|no model for|not (in|part of) the (current )?(dataset|model)",
}

def find(d, names):
    """depth-first search for the first numeric/str value whose key matches one of names"""
    if isinstance(d, dict):
        for k, v in d.items():
            if any(n.lower() == k.lower() or n.lower() in k.lower() for n in names) and not isinstance(v, (dict, list)):
                return v
        for v in d.values():
            r = find(v, names)
            if r is not None: return r
    if isinstance(d, list):
        for v in d:
            r = find(v, names)
            if r is not None: return r
    return None

def parse_json(t):
    try: return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        try: return json.loads(m.group(0)) if m else {}
        except Exception: return {}

def numbers(text):
    vals = []
    for m in re.finditer(r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?:[×x]\s*10\^?\s*([0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+))?\s*(k\b)?", text):
        v = float(m.group(1).replace(",", ""))
        if m.group(2):
            e = m.group(2).translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")); v *= 10 ** int(e)
        if m.group(3): v *= 1e3
        vals.append(v)
    return vals

def mentions(text, value, rel=0.01):
    return value is not None and any(abs(x - float(value)) <= rel * abs(float(value)) + 1e-9 for x in numbers(text))

def family(name):
    return next((f for f, keys in FAMILY.items() if any(k in name.lower() for k in keys)), "other")

def families(name):
    """all tool families a tool belongs to (filtered_grid_scan is both a grid scan and an optimiser)"""
    return {f for f, keys in FAMILY.items() if any(k in name.lower() for k in keys)} or {"other"}

def is_line(a):
    """True when a grid-scan call holds P or V fixed (degenerate bounds or a single grid point)."""
    def flat(b):
        try: return b is not None and len(b) == 2 and abs(float(b[0]) - float(b[1])) < 1e-9
        except Exception: return False
    return flat(a.get("P_bounds")) or flat(a.get("V_bounds")) or a.get("grid_P") == 1 or a.get("grid_V") == 1

def num(v):
    try: float(v); return not isinstance(v, bool)
    except (TypeError, ValueError): return False

def recommended_PV(text):
    """first 'xxx W' and 'yyy mm/s' pair in the answer = the point the agent recommends"""
    # skip limits restated from the prompt, e.g. "(P <= 250 W)" or "at least 800 mm/s"
    lim = re.compile(r"(?:[≤≥<>]=?|at least|at most|no (?:higher|lower|more|less) than|above|below|under|over)\s*\**\s*$", re.I)
    def first(pat):
        for m in re.finditer(pat, text):
            if not lim.search(text[max(0, m.start() - 25):m.start()]): return float(m.group(1))
        return None
    P = first(r"(?<![\d.])(\d{3}(?:\.\d+)?)\s*W\b"); V = first(r"(?<![\d.])(\d{2,4}(?:\.\d+)?)\s*mm/s")
    return (P, V) if P is not None and V is not None else (None, None)

class Truth:
    def __init__(self, s): self.s, self.cache = s, {}
    async def predict(self, P, V):
        k = (round(P, 1), round(V, 1))
        if k not in self.cache:
            r = await self.s.call_tool(PREDICT_TOOL, {"P": P, "V": V, "material": "SS316L"})
            d = parse_json("".join(getattr(c, "text", "") for c in r.content))
            self.cache[k] = dict(spatter=find(d, ["spatter"]), depth=find(d, ["depth"]), width=find(d, ["width"]),
                                 length=find(d, ["length"]), defect=str(find(d, ["defect"]) or ""))
        return self.cache[k]
    async def optimum(self, c, n=46):
        best = None
        Ps = [P_LIM[0] + i * (P_LIM[1] - P_LIM[0]) / (n - 1) for i in range(n)]
        Vs = [V_LIM[0] + i * (V_LIM[1] - V_LIM[0]) / (n - 1) for i in range(n)]
        if "fix_P" in c: Ps = [c["fix_P"]]
        if "fix_V" in c: Vs = [c["fix_V"]]
        for P in Ps:
            for V in Vs:
                y = await self.predict(P, V)
                if ok(c, P, V, y) and (best is None or y["spatter"] < best[2]["spatter"]): best = (P, V, y)
        return best

def ok(c, P, V, y):
    if "good" not in y["defect"].lower(): return False
    J = P / V
    tests = [("depth", y["depth"]), ("width", y["width"]), ("length", y["length"]), ("J", J)]
    for key, val in tests:
        if key in c and not (c[key][0] <= val <= c[key][1]): return False
    if "spatter_max" in c and y["spatter"] > c["spatter_max"]: return False
    if "V_min" in c and V < c["V_min"]: return False
    if "P_max" in c and P > c["P_max"]: return False
    if "fix_P" in c and abs(P - c["fix_P"]) > 1: return False
    if "fix_V" in c and abs(V - c["fix_V"]) > 1: return False
    return True

async def score(args):
    bench = {json.loads(l)["id"]: json.loads(l) for l in open(args.bench)}
    runs = [json.loads(l) for l in open(args.runs)]
    cmd = [c.strip('"') for c in shlex.split(args.server, posix=(os.name != "nt"))]  # keep Windows backslashes
    if cmd and cmd[0].lower() in ("python", "python3", "python.exe"): cmd[0] = sys.executable  # same env as this script
    rows, review = [], []
    async with stdio_client(StdioServerParameters(command=cmd[0], args=cmd[1:])) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize(); T = Truth(s); opt_cache = {}
            for run in runs:
                it = bench[run["id"]]; cat = it["category"].split("_")[0]
                t = run["turns"][-1]; text = t["final_text"]; calls = t["tool_calls"]
                fams = [family(c["name"]) for c in calls]
                row = dict(id=run["id"], cat=it["category"], repeat=run["repeat"], latency=run["latency_s"],
                           tokens=run["tokens_in"] + run["tokens_out"], n_calls=len(calls))
                exp = it.get("expected_tool")
                if exp in FAMILY:
                    row["tool_ok"] = any(exp in families(c["name"]) for c in calls)
                    if exp == "line_scan" and not row["tool_ok"]:  # a grid scan with one axis held fixed is a line scan
                        row["tool_ok"] = any("grid" in c["name"] and is_line(c["args"]) for c in calls)
                    if exp == "optimise" and not row["tool_ok"] and ({"fix_P", "fix_V"} & set(it.get("constraints", {}))):
                        row["tool_ok"] = "line_scan" in fams  # 1-D min-spatter scan is the optimiser for a fixed P or V
                elif exp == "none": row["tool_ok"] = len(calls) == 0
                pc = next((c for c in calls if family(c["name"]) == "predict"), None)
                if pc and exp == "predict":
                    P = next((pc["args"][k] for k in ARG_P if k in pc["args"]), None)
                    V = next((pc["args"][k] for k in ARG_V if k in pc["args"]), None)
                    row["P_used"], row["V_used"] = P, V
                    if "P" in it and cat != "C8":
                        row["param_ok"] = P is not None and V is not None and abs(P - it["P"]) <= it.get("tol_P", 0) + 1e-6 \
                                          and abs(V - it["V"]) <= it.get("tol_V", 0) + 1e-6
                    if "P_range" in it:
                        row["param_ok"] = P is not None and V is not None and it["P_range"][0] <= P <= it["P_range"][1] \
                                          and it["V_range"][0] <= V <= it["V_range"][1]
                    if P is not None and V is not None:
                        y = await T.predict(P, V)
                        # Faithfulness is scored only for quantities the agent's own tool calls returned
                        # (a defect-only query has no spatter value to report, and vice versa).
                        got = [parse_json(x) for x in t["tool_results"]]
                        if any(num(find(g, ["spatter"])) for g in got) and re.search(r"spatter", text, re.I):
                            row["spatter_faithful"] = mentions(text, y["spatter"], 0.01)
                        if any(num(find(g, ["depth"])) for g in got):
                            row["depth_faithful"] = mentions(text, y["depth"], 0.01)
                        if any(find(g, ["defect"]) is not None for g in got):
                            row["defect_faithful"] = bool(y["defect"]) and y["defect"].lower().split()[0] in text.lower()
                elif exp == "predict":
                    row["param_ok"] = False
                if cat in KW:
                    row["behaviour_ok"] = bool(re.search(KW[cat], text, re.I))
                    review.append(dict(id=run["id"], repeat=run["repeat"], must=it.get("must", ""), auto=row["behaviour_ok"],
                                       answer=text[:1500], rater1="", rater2=""))
                if cat == "C7":
                    c = it["constraints"]; key = json.dumps(c, sort_keys=True)
                    if key not in opt_cache: opt_cache[key] = await T.optimum(c)
                    best = opt_cache[key]; P, V = recommended_PV(text)
                    if best is None:  # no point satisfies the constraints: the correct answer is to say so
                        row["infeasible_prompt"] = True
                        row["behaviour_ok"] = bool(re.search(KW["C10"], text, re.I))
                        review.append(dict(id=run["id"], repeat=run["repeat"], must="no feasible point exists; report infeasibility",
                                           auto=row["behaviour_ok"], answer=text[:1500], rater1="", rater2=""))
                    elif P is not None:
                        y = await T.predict(P, V)
                        row["constraints_ok"] = ok(c, P, V, y)
                        if best: row["opt_gap_pct"] = 100 * (y["spatter"] - best[2]["spatter"]) / best[2]["spatter"]
                    else:
                        row["constraints_ok"] = False
                    row["ref_P"], row["ref_V"] = (best[0], best[1]) if best else (None, None)
                if cat == "C10":
                    P, V = recommended_PV(text)
                    row["no_violating_point"] = P is None or not ok(it.get("constraints", {}), P, V, await T.predict(P, V))
                if cat in ("C12", "C13", "C14"):
                    review.append(dict(id=run["id"], repeat=run["repeat"], must=it.get("must", ""), auto="",
                                       answer=" || ".join(x["final_text"][:700] for x in run["turns"]), rater1="", rater2=""))
                rows.append(row)

    # consistency across repeats: same P,V and same defect word in every run
    by = defaultdict(list)
    for r in rows:
        if "P_used" in r: by[r["id"]].append((r.get("P_used"), r.get("V_used"), r.get("defect_faithful")))
    consist = {k: len(set(v)) == 1 for k, v in by.items() if len(v) > 1}

    os.makedirs("results", exist_ok=True)
    import csv
    with open("results/per_run.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    with open("results/review_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(review[0].keys()) if review else ["id"]); w.writeheader(); w.writerows(review)

    def rate(rs, k):
        v = [r[k] for r in rs if k in r]
        return (100 * sum(v) / len(v), len(v)) if v else (None, 0)
    cats = sorted({r["cat"] for r in rows}, key=lambda c: int(re.search(r"\d+", c).group()))
    lines, summ = [], []
    for c in cats:
        rs = [r for r in rows if r["cat"] == c]
        d = dict(category=c, runs=len(rs))
        for k in ("tool_ok", "param_ok", "spatter_faithful", "defect_faithful", "behaviour_ok", "constraints_ok", "no_violating_point"):
            d[k], _ = rate(rs, k)
        gaps = [r["opt_gap_pct"] for r in rs if "opt_gap_pct" in r]
        d["opt_gap_median_pct"] = st.median(gaps) if gaps else None
        ids = {r["id"] for r in rs}; cc = [consist[i] for i in ids if i in consist]
        d["consistency"] = 100 * sum(cc) / len(cc) if cc else None
        d["latency_median_s"] = st.median(r["latency"] for r in rs)
        d["tokens_median"] = st.median(r["tokens"] for r in rs)
        summ.append(d)
    with open("results/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    fmt = lambda x: "--" if x is None else f"{x:.0f}"
    with open("results/table_agent_eval.tex", "w") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\nCategory & Runs & Tool (\\%) & Param. (\\%) & Faithful (\\%) & Behaviour (\\%) & Consist. (\\%) & Latency (s)\\\\\n\\midrule\n")
        for d in summ:
            beh = d["constraints_ok"] if d["category"].startswith("C7") else d["behaviour_ok"]
            f.write(f"{d['category'].replace('_', ' ')} & {d['runs']} & {fmt(d['tool_ok'])} & {fmt(d['param_ok'])} & "
                    f"{fmt(d['spatter_faithful'])} & {fmt(beh)} & {fmt(d['consistency'])} & {d['latency_median_s']:.1f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for d in summ: print(d)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--bench", default="benchmark.jsonl")
    asyncio.run(score(ap.parse_args()))
