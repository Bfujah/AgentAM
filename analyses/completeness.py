"""Answer completeness / omission scoring (reviewer: faithfulness does not establish completeness).
For every run in C1-C7 and C13 (feasible C7 only), the quantities the request needs are derived from the
prompt, and the final answer is checked for each. Writes completeness_per_run.csv, completeness_summary.csv."""
import json, re, csv, collections
B = {json.loads(l)["id"]: json.loads(l) for l in open("../eval/benchmark.jsonl")}
runs = [json.loads(l) for l in open("../eval/runs/claude-opus-4-5.jsonl")]
INFEAS = {"C7_optimise-06", "C7_optimise-14"}
U = r"\s*(?:µm|μm|um|\\mu m|microns?)"
NUM = r"~?≈?\s*\d[\d,]*(?:\.\d+)?\s*(?:k\b)?"
CLS = re.compile(r"\b(good|keyhol\w*|lack[- ]of[- ]fusion|lof|balling|defect[- ]free|stable)\b", re.I)
def has_dim(t, name):
    return bool(re.search(name + r"[^\n]{0,60}?" + NUM + U + r"(?![\^³23])", t, re.I))
def has_spatter(t):
    return bool(re.search(r"spatter[^\n]{0,70}?" + NUM + r"\s*(?:µm|μm|um)\s*(?:³|\^3|3)", t, re.I)) or has_table_col(t, "spatter")
def has_table_col(t, name):
    lines = t.splitlines()
    for i, l in enumerate(lines):
        if l.strip().startswith("|") and re.search(name, l, re.I):
            return any(re.search(r"\|\s*\**~?\d", x) for x in lines[i+1:i+6])
    return False
INFEAS_RE = re.compile(r"no defect-free|no power setting|no (?:speed|velocity) setting|no (?:feasible|good|valid) (?:point|solution|option)|not possible|infeasible|cannot be (?:met|achieved)", re.I)
def has_point(t):
    inline = bool(re.search(r"\d+(?:\.\d+)?\s*W\b", t)) and bool(re.search(r"\d+(?:\.\d+)?\s*mm/s", t))
    return inline or (has_table_col(t, r"P \(W\)|Power") and has_table_col(t, r"V \(mm/s\)|Velocity|Speed"))
def has_J(t): return bool(re.search(r"\d\.\d+\s*J/mm", t))
def has_boundaries(t): return len({m.group(1).lower()[:4] for m in CLS.finditer(t)}) >= 2
def required(b, prompt):
    p = (prompt or "").lower(); req = []
    cat = b["category"]
    if cat in ("C6_linescan", "C6_gridscan"):
        if re.search(r"minimum|lowest|optim|best|min-spatter|minimi", p): req += ["point", "spatter"]
        if re.search(r"only good|list only|good points", p): req += ["point"] if "point" not in req else []
        elif re.search(r"defect|regime|window|boundar|change|map|transition|where", p) or cat == "C6_gridscan": req += ["boundaries"]
        if not req: req = ["point", "spatter"]
        return req
    if cat == "C7_optimise":
        req = ["point", "spatter", "defect"]; c = b.get("constraints", {})
        for k in ("depth", "width", "length"):
            if k in c: req.append(k)
        if "J" in c: req.append("J")
        return req
    if "spatter" in p: req.append("spatter")
    if re.search(r"defect|stable|safe|window|regime|expect|ok\b|good", p): req.append("defect")
    for k in ("depth", "width", "length"):
        if k in p: req.append(k)
    if re.search(r"melt pool (geometry|dimension|size|shape)", p): req += [k for k in ("depth", "width", "length") if k not in req]
    if not req: req = ["spatter", "defect"]
    return req
CHECK = {"spatter": has_spatter, "defect": lambda t: bool(CLS.search(t)), "depth": lambda t: has_dim(t, "depth"),
         "width": lambda t: has_dim(t, "width"), "length": lambda t: has_dim(t, "length"), "point": has_point,
         "J": has_J, "boundaries": has_boundaries}
CATS = ["C1_explicit","C2_units","C3_implicit","C4_arithmetic","C5_colloquial","C6_linescan","C6_gridscan","C7_optimise","C13_multiturn"]
out = []
for r in runs:
    if r["category"] not in CATS or r["id"] in INFEAS: continue
    b = B[r["id"]]; turn = r["turns"][-1]; prompt = turn["user"] if r["category"] == "C13_multiturn" else b["prompt"]
    req = required(b, prompt); txt = turn["final_text"] or ""
    infeasible_answer = bool(INFEAS_RE.search(txt))
    miss = [q for q in req if not CHECK[q](txt) and not (infeasible_answer and q in ("spatter", "point", "depth", "width", "length", "J"))]
    out.append(dict(id=r["id"], repeat=r["repeat"], category=r["category"], required="+".join(req), missing="+".join(miss), infeasible_answer=int(infeasible_answer), complete=int(not miss)))
with open("completeness_per_run.csv", "w", newline="") as f:
    w = csv.DictWriter(f, out[0].keys()); w.writeheader(); w.writerows(out)
S = collections.defaultdict(lambda: [0, 0, collections.Counter(), collections.Counter()])
for o in out:
    s = S[o["category"]]; s[0] += 1; s[1] += o["complete"]
    for q in o["required"].split("+"): s[2][q] += 1
    for q in filter(None, o["missing"].split("+")): s[3][q] += 1
rows = []
for c in CATS:
    n, k, rq, ms = S[c]; rows.append(dict(category=c, runs=n, complete_pct=round(100*k/n, 1), required_items=sum(rq.values()), omitted_items=sum(ms.values()), omissions="; ".join(f"{q} {ms[q]}/{rq[q]}" for q in ms)))
n = sum(S[c][0] for c in CATS); k = sum(S[c][1] for c in CATS); ri = sum(sum(S[c][2].values()) for c in CATS); mi = sum(sum(S[c][3].values()) for c in CATS)
rows.append(dict(category="All", runs=n, complete_pct=round(100*k/n, 1), required_items=ri, omitted_items=mi, omissions=f"item omission rate {100*mi/ri:.1f}%"))
with open("completeness_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
