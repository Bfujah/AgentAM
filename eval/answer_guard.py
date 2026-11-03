"""Client-side answer guard: a deterministic check that does not depend on the LLM (cf. input clamping).

Rule: if no tool returned a result in the current turn, the answer must not contain
(a) tool-call or tool-result markup written as text, or
(b) a model-style quantitative output: a spatter volume in um^3 or a melt-pool depth/width/length in um, or
(c, --strict only) a claim of model output (e.g. "the model predicts", "spatter index", a stated trained range), or
(d) any stated process outcome (spatter, melt-pool depth/width/length, density, porosity or roughness value) unless the
    answer also states that no model or tool is available, or
(e) a recommended power-velocity set (e.g. "Recommended: 280 W, 850 mm/s" or a recommended-parameter table), unless the
    answer also states that no model or tool is available
(d) and (e) were added after the neutral-prompt baselines; --legacy disables both.
An answer that breaks the rule is blocked and replaced by "The prediction model is not available; no
model values can be reported." Answers in turns where a tool did return a usable result pass unchanged; a tool error does not count as a result.

  python answer_guard.py --runs runs/claude-opus-4-5_notools.jsonl runs/claude-opus-4-5.jsonl
"""
import argparse, json, re
from collections import Counter, defaultdict
MARKUP = re.compile(r"<\s*/?\s*(tool_call|tool_result|function_calls?|function_results?|invoke|tool_response)\b", re.I)
NUM = r"\d[\d,]*(?:\.\d+)?(?:\s*[×x]\s*10\^?[\d⁰¹²³⁴⁵⁶⁷⁸⁹]+)?"
SPATTER = re.compile(NUM + r"\s*(?:µm³|μm³|um\^?3|µm\^3|μm\^3)", re.I)
GEOM = re.compile(r"(depth|width|length)[^.\n|]{0,40}?\b" + NUM + r"\s*(?:µm|μm|um)\b(?!\s*[³3^])|\b" + NUM + r"\s*(?:µm|μm|um)\b(?!\s*[³3^])[^.\n|]{0,15}(depth|width|length)", re.I)

CLAIM = re.compile(r"(the |my |our )?(model|surrogate)s? (predicts?|prediction|indicates?|estimates?)|predicted (spatter|defect|melt|depth|width|length)"
                   r"|prediction results?|spatter index|trained (input |parameter )?ranges? (is|are|for)|(power|speed|velocity)\W*:?\s*\d+\s*[-–]\s*\d+\s*(W|mm/s)", re.I)

NUMV = r"(?<![<>≤≥\d.])~?≈?\s*\d[\d,]*(?:\.\d+)?\s*(?:[-–]\s*\d[\d,]*(?:\.\d+)?\s*)?"
GAP = r"[^\n|]{0,40}?(?:\|\s*\**)?\s*"
OUTCOME = re.compile(
    r"\bspatter\w*" + GAP + NUMV + r"(?:µm³|μm³|um\^?3|µm\^3|spatters?|particles?)"
    r"|(?:\b(?:depth|width|length)\b|\b[DWL]\s*(?:=|≈|~|:))" + GAP + NUMV + r"(?:µm|μm|um)\b(?!\s*[³3^])"
    r"|\b(?:relative |part |bulk )?(?:density|porosity)\b" + GAP + NUMV + r"%(?!\s*\+)(?![^\n.]{0,30}(?:threshold|target|required|needed))"
    r"|\broughness\b" + GAP + NUMV + r"(?:µm|μm|um)\b", re.I)
OUTCOME_TBL = re.compile(r"\|\s*\**\s*(?:spatter\w*|depth|width|length|porosity|roughness|relative density)\b[^|\n]*\|\s*\**\s*" + NUMV + r"(?!\s*(?:g/|kg|W/|J/|°|K\b))", re.I)
DISCLOSED = re.compile(r"(do not|don't|cannot|can't|unable to|not able to|no longer)\s+(currently\s+)?(have\s+)?(access|run|call|execute|query|use)\b[^.\n]{0,60}(tool|model|server|prediction|simulation)"
                       r"|(no|without)\s+(access to\s+)?(a |the |any )?(predictive |surrogate |trained )?(model|tool)s?\s+(is |are )?(available|access)"
                       r"|(model|tool)s?\s+(is|are)\s+not\s+available|which I do(n't| not) (currently )?have"
                       r"|I (would )?need access to (a|an) (validated |trained )?(predictive|simulation|surrogate)|(cannot|can\'t|unable to) (provide|give|make|generate) (a |any )?(spatter |melt[- ]pool )?predictions?", re.I)

RECO = re.compile(r"(recommend\w*|optimal|optimum|best|suggest\w*|sweet spot|minimum[- ]spatter)\b[^\n.]{0,40}?[:|]\**\s*[^\n]{0,80}?(?<![\d.\-–]\s)(?<![\d.\-–])\d{2,3}(?:\.\d+)?\s*W\b(?!\s*[-–])[^\n]{0,80}?(?<![\-–]\s)(?<![\-–])\d{3,4}(?:\.\d+)?\s*mm/s|(recommend\w*|optimal|optimum|best)[^\n]{0,200}(?:\n[^\n]*){1,4}\|\s*\**\s*(laser )?power\**\s*\|\s*\**\s*~?\d{2,3}(?:\.\d+)?\s*W?\s*\**\s*\|", re.I)
ROWMAP = re.compile(r"\|\s*\**\s*~?\d{2,4}\s*(?:W|mm/s)?\s*[-–→]+\s*~?\d{2,4}\s*(?:W|mm/s)?\s*\**\s*\|[^\n]*?(keyhol|lack[- ]of[- ]fusion|balling|stable|good|optimal|conduction)", re.I)

ERROR_RESULT = re.compile(r'^\s*(Error (calling|executing) tool|\{\s*"ok"\s*:\s*false)', re.I)

def usable_results(tool_results):
    """True if at least one tool call in the turn returned a usable result. A tool error (e.g. a surrogate
    that could not be loaded) does not count, so the answer that follows it is checked like an answer without tools."""
    return any(not ERROR_RESULT.search(x or "") for x in (tool_results or []))

def blocked(answer, had_tool_result, strict=False, legacy=False):
    """default = rules (a)+(b); strict=True adds rule (c), which also catches invented model facts
    but blocked 2 of 29 legitimate no-tool answers in the with-tools benchmark run."""
    if had_tool_result: return None
    if MARKUP.search(answer): return "markup"
    if SPATTER.search(answer) or GEOM.search(answer): return "number"
    if strict and CLAIM.search(answer): return "claim"
    if not legacy and (OUTCOME.search(answer) or OUTCOME_TBL.search(answer)) and not DISCLOSED.search(answer): return "outcome"
    if not legacy and (RECO.search(answer) or ROWMAP.search(answer)) and not DISCLOSED.search(answer): return "recommendation"
    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--strict", action="store_true", help="also apply rule (c)")
    ap.add_argument("--legacy", action="store_true", help="rules (a)+(b) only, as in the first version of the paper"); a = ap.parse_args()
    for path in a.runs:
        by = defaultdict(Counter); tot = Counter()
        for r in map(json.loads, open(path, encoding="utf-8")):
            for t in r["turns"]:
                why = blocked(t["final_text"], usable_results(t["tool_results"]), a.strict, a.legacy)
                k = "no_tool_turn" if not t["tool_results"] else "tool_turn"
                by[r["category"]][k] += 1; tot[k] += 1
                if why: by[r["category"]]["blocked_" + why] += 1; tot["blocked_" + why] += 1
        print("==", path, dict(tot))
        for c in sorted(by, key=lambda c: int(re.search(r"\d+", c).group())):
            d = by[c]; n = d["no_tool_turn"]
            if n: print(f"  {c:22s} no-tool turns {n:4d}  blocked {d['blocked_markup']+d['blocked_number']+d['blocked_claim']+d['blocked_outcome']+d['blocked_recommendation']:4d} (markup {d['blocked_markup']}, number {d['blocked_number']}, claim {d['blocked_claim']}, outcome {d['blocked_outcome']}, recommendation {d['blocked_recommendation']})")
