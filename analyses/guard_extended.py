"""Extended answer-guard rule (d): when no tool returned a result, block an answer that states a numerical outcome
(spatter, melt-pool depth/width/length, density, porosity, roughness) unless it also states that no model is available.
Evaluated on the no-tools answers (three system prompts) and, for false alarms, on the legitimate tool-free answers
of the runs with tools."""
import json, re, importlib.util, collections
spec = importlib.util.spec_from_file_location("sn", "score_neutral.py"); sn = importlib.util.module_from_spec(spec)
import sys; sys.argv = ["x"]; src = open("score_neutral.py").read().split("ORDER = ")[0]; exec(src, sn.__dict__)
def rule_d(t): return bool(sn.OUTCOME.search(t) or sn.OUTCOME_TBL.search(t)) and not sn.DECL.search(t)
def rule_markup(t): return bool(sn.MARKUP.search(t))
F = {"original": "../eval/runs/claude-opus-4-5_notools.jsonl", "N1 neutral": "../eval/runs/claude-opus-4-5_notools_neutral.jsonl", "N2 acknowledge": "../eval/runs/claude-opus-4-5_notools_ack.jsonl"}
for k, f in F.items():
    R = [json.loads(l) for l in open(f)]
    turns = [t["final_text"] for r in R for t in r["turns"]]
    b = sum(rule_d(t) or rule_markup(t) for t in turns)
    inv = [r for r in R if r["category"].startswith(sn.SCOPE)]
    sub = [" ".join(t["final_text"] for t in r["turns"]) for r in inv]
    undisclosed = [t for t in sub if sn.cls(t) in ("fake_markup", "claimed_model", "unlabelled", "hedged")]
    caught = sum(rule_d(t) or rule_markup(t) for t in undisclosed)
    print(f"{k:16s} turns {len(turns)} blocked {b} ({100*b/len(turns):.0f}%) | undisclosed invented answers in scope: {len(undisclosed)}, caught {caught} ({100*caught/max(1,len(undisclosed)):.0f}%)")
R = [json.loads(l) for l in open("../eval/runs/claude-opus-4-5.jsonl")]
legit = [t["final_text"] for r in R for t in r["turns"] if not t["tool_calls"]]
fa = [t for t in legit if rule_d(t) or rule_markup(t)]
print(f"runs with tools: {len(legit)} tool-free answers, false alarms {len(fa)}")
for t in fa: print("  FA:", re.sub(r"\s+", " ", t)[:200])
