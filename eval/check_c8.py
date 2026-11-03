"""Automatic check of the out-of-range category (C8) after the server change.

For every C8 run in a runs file it checks, without human grading:
  clamp_disclosed  the answer says the input was outside the range / clamped
  used_reported    the answer states the clamped value that the server actually used
  range_correct    every trained range the answer states is 160-430 W and 100-1500 mm/s
  range_from_tool  the tool output contained the trained range (true after the server change)

  python check_c8.py runs/c8_rerun.jsonl [runs/claude-opus-4-5.jsonl]
Prints one line per run and a summary per file. Runs with range_correct = False are the ones to read.
"""
import json, re, sys

W = re.compile(r"(\d{2,4})\s*(?:–|—|-|to)\s*(\d{2,4})\s*W\b")
V = re.compile(r"(\d{2,4})\s*(?:–|—|-|to)\s*(\d{2,5})\s*mm/s")
RANGE_CTX = re.compile(r"[^.\n]*(trained|valid|supported|training|model's)[^.\n]*(range|bounds|data)[^.\n]*", re.I)


def check(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if not r["id"].startswith("C8"):
            continue
        t = r["turns"][0]
        ans = t.get("final_text", "")
        res = " ".join(t.get("tool_results", []))
        used = []
        for x in t.get("tool_results", []):
            try:
                d = json.loads(x).get("data", {})
                used += [d.get("P_used", d.get("P")), d.get("V_used", d.get("V"))]
            except Exception:
                pass
        used = [u for u in used if u is not None]
        wrong = []
        for m in RANGE_CTX.finditer(ans):
            s = m.group(0)
            wrong += [f"{a}-{b} W" for a, b in W.findall(s) if (int(a), int(b)) != (160, 430)]
            wrong += [f"{a}-{b} mm/s" for a, b in V.findall(s) if (int(a), int(b)) != (100, 1500)]
        rows.append(dict(
            id=r["id"], rep=r["repeat"],
            tools=",".join(c["name"] for c in t.get("tool_calls", [])),
            clamp_disclosed=bool(re.search(r"clamp|outside|beyond|below the|above the|out of range|exceeds", ans, re.I)),
            used_reported=any(re.search(rf"\b{int(u)}\s*(W|mm/s)", ans) for u in used),
            range_correct=not wrong, wrong=";".join(wrong),
            range_from_tool=("trained_range" in res or '"ranges"' in res)))
    return rows


for path in sys.argv[1:]:
    rows = check(path)
    print(f"\n== {path}: {len(rows)} C8 runs")
    for x in rows:
        flag = "" if (x["clamp_disclosed"] and x["used_reported"] and x["range_correct"]) else "   <-- read this one"
        print(f'{x["id"]} r{x["rep"]}  tools={x["tools"]}  disclosed={x["clamp_disclosed"]}  used={x["used_reported"]}  '
              f'range_ok={x["range_correct"]} {x["wrong"]}  range_in_tool={x["range_from_tool"]}{flag}')
    n = len(rows) or 1
    for k in ("clamp_disclosed", "used_reported", "range_correct", "range_from_tool"):
        print(f"  {k}: {sum(x[k] for x in rows)}/{len(rows)} ({100*sum(x[k] for x in rows)/n:.0f}%)")
