"""Generate the AgentAM evaluation benchmark (LPBF / SS316L).

Each item has: id, category, prompt (or turns), what a correct agent must do,
and a machine-checkable expectation.  Numeric ground truth (spatter, depth, defect,
constrained optimum) is NOT stored here: score_agent.py recomputes it by calling the
surrogate model directly, so the benchmark stays valid if the models are retrained.

Run:  python make_benchmark.py   ->  benchmark.jsonl, benchmark.csv
"""
import json, random, csv

random.seed(2026)
P_RANGE, V_RANGE = (160, 430), (100, 1500)      # trained SS316L range
items = []

def add(cat, prompt, **kw):
    items.append(dict(id=f"{cat}-{sum(i['category'] == cat for i in items) + 1:02d}", category=cat, prompt=prompt, **kw))

def rp():  return random.randrange(170, 425, 5)
def rv():  return random.randrange(150, 1450, 10)

# ---------------------------------------------------------------- C1 explicit parameters
T1 = ["Predict spatter at {P} W and {V} mm/s.",
      "What defect should I expect at P = {P} W, V = {V} mm/s?",
      "Run a prediction for SS316L, laser power {P} W, scan speed {V} mm/s.",
      "Give me melt pool depth, width and length for {P} W / {V} mm/s.",
      "Is {P} W at {V} mm/s inside the stable window?"]
for k in range(20):
    P, V = rp(), rv()
    add("C1_explicit", T1[k % len(T1)].format(P=P, V=V), expected_tool="predict", P=P, V=V, tol_P=0, tol_V=0)

# ---------------------------------------------------------------- C2 unit conversion
conv = [("{P} W and {v} m/s", lambda P, V: dict(v=V / 1000)),
        ("{kw} kW at {V} mm/s", lambda P, V: dict(kw=P / 1000)),
        ("{P} W with a scan speed of {cms} cm/s", lambda P, V: dict(cms=V / 10)),
        ("{P} W, {mmin} mm/min", lambda P, V: dict(mmin=V * 60)),
        ("{P} W and a linear energy density of {J} J/mm", lambda P, V: dict(J=round(P / V, 3)))]
for k in range(15):
    P = rp(); V = random.randrange(200, 1400, 50)
    tpl, f = conv[k % len(conv)]
    txt = "Predict at " + tpl.format(P=P, V=V, **f(P, V)) + "."
    add("C2_units", txt, expected_tool="predict", P=P, V=V, tol_P=1, tol_V=2)

# ---------------------------------------------------------------- C3 implicit / reordered labels
T3 = ["predict at {P} and {V}", "{V} mm/s, {P} W - what happens?", "speed {V} power {P}",
      "try {P}/{V}", "what about {V} for the velocity and {P} for the laser?"]
for k in range(10):
    P, V = rp(), random.randrange(450, 1400, 10)   # V > P so the order is inferable from typical ranges
    add("C3_implicit", T3[k % len(T3)].format(P=P, V=V), expected_tool="predict", P=P, V=V, tol_P=0, tol_V=0)

# ---------------------------------------------------------------- C4 arithmetic in the prompt
T4 = [("Use P = {a} + {b} and V = {c} x {d}.", lambda a, b, c, d: (a + b, c * d)),
      ("Take {pct}% of 400 W and scan at half of {e} mm/s.", None),
      ("Power is {a} W plus 15%, speed is {c} x {d} mm/s.", lambda a, b, c, d: (round(a * 1.15), c * d)),
      ("Set P to {a} - {b} watts and V to {c} + {d} mm/s.", lambda a, b, c, d: (a - b, c + d))]
for k in range(10):
    t, f = T4[k % len(T4)]
    if f is None:
        e = random.choice([1000, 1200, 1400]); pct = random.choice([60, 70, 75, 80, 85]); P, V = 400 * pct // 100, e // 2
        txt = t.format(e=e, pct=pct)
    elif "x" in t and "plus" not in t:
        a, b, c, d = random.choice([150, 200]), random.choice([50, 100, 150]), random.choice([2, 3]), random.choice([200, 250, 300])
        P, V = f(a, b, c, d); txt = t.format(a=a, b=b, c=c, d=d)
    elif "plus" in t:
        a, c, d = random.choice([200, 240, 260]), random.choice([2, 4]), random.choice([150, 200])
        P, V = f(a, 0, c, d); txt = t.format(a=a, c=c, d=d)
    else:
        a, b, c, d = random.choice([350, 400]), random.choice([50, 100]), random.choice([400, 500]), random.choice([200, 300])
        P, V = f(a, b, c, d); txt = t.format(a=a, b=b, c=c, d=d)
    add("C4_arithmetic", txt, expected_tool="predict", P=P, V=V, tol_P=1, tol_V=1)

# ---------------------------------------------------------------- C5 colloquial / approximate (accept a range)
C5 = [("Hit it with 350-ish watts, speed a bit under a meter per second.", (340, 360), (850, 999)),
      ("Roughly three hundred watts and fairly slow, say around half a metre per second.", (290, 310), (450, 550)),
      ("Crank the laser near the top of the range and go slow, around 200 mm/s.", (400, 430), (180, 220)),
      ("Low power, about two hundred watts, and pretty fast - a bit over a meter a second.", (190, 210), (1001, 1200)),
      ("Medium power (~280 W) and the usual speed people run 316L at, ~800 mm/s.", (270, 290), (780, 820)),
      ("Quarter of a kilowatt, 0.7 m/s give or take.", (245, 255), (650, 750)),
      ("Just under 400 W at 6 hundred mm/s.", (380, 399), (590, 610)),
      ("Run it at three-twenty watts and six hundred.", (315, 325), (595, 605)),
      ("Max power you allow, moderate speed (~1 m/s).", (425, 430), (950, 1050)),
      ("A gentle 180 W, crawling at 150 mm/s.", (175, 185), (140, 160))]
for txt, pr, vr in C5:
    add("C5_colloquial", txt, expected_tool="predict", P_range=pr, V_range=vr)

# ---------------------------------------------------------------- C6 tool selection: scans
for P in (200, 250, 300, 350, 400):
    add("C6_linescan", f"At P = {P} W, sweep velocity from 200 to 1400 mm/s and find the minimum-spatter defect-free point.",
        expected_tool="line_scan", fixed="P", fixed_value=P, sweep=(200, 1400), objective="min_spatter_good")
for V in (400, 600, 800, 1000, 1200):
    add("C6_linescan", f"Hold scan speed at {V} mm/s and sweep power from 160 to 430 W. Where does the defect type change?",
        expected_tool="line_scan", fixed="V", fixed_value=V, sweep=(160, 430), objective="boundaries")
for txt in ["Map the whole P-V process window for SS316L.",
            "Show me where keyhole, balling and lack of fusion occur across the full parameter space.",
            "Scan P = 200-400 W and V = 300-1000 mm/s on a 15x15 grid and list only Good points with spatter below 100,000 um^3.",
            "Give me a process map between 250 and 430 W and 400-1200 mm/s.",
            "Where is the stable window widest?"]:
    add("C6_gridscan", txt, expected_tool="grid_scan")

# ---------------------------------------------------------------- C7 constrained optimisation (checked against brute force)
OPT = [dict(depth=(120, 180), spatter_max=90000),
       dict(J=(0.4, 0.6)),
       dict(depth=(150, 250), length=(350, 500), spatter_max=120000),
       dict(depth=(140, 200)),
       dict(width=(150, 200), spatter_max=80000),
       dict(V_min=800),
       dict(P_max=250),
       dict(depth=(150, 220), V_min=600),
       dict(J=(0.3, 0.45), spatter_max=100000),
       dict(length=(300, 400)),
       dict(fix_P=300),
       dict(fix_V=700),
       dict(depth=(180, 260), spatter_max=100000),
       dict(P_max=300, V_min=700),
       dict(spatter_max=60000)]
def opt_text(c):
    parts = []
    if "depth" in c: parts.append(f"melt pool depth between {c['depth'][0]} and {c['depth'][1]} um")
    if "width" in c: parts.append(f"width {c['width'][0]}-{c['width'][1]} um")
    if "length" in c: parts.append(f"length {c['length'][0]}-{c['length'][1]} um")
    if "J" in c: parts.append(f"linear energy density {c['J'][0]}-{c['J'][1]} J/mm")
    if "spatter_max" in c: parts.append(f"spatter under {c['spatter_max']:,} um^3")
    if "V_min" in c: parts.append(f"scan speed at least {c['V_min']} mm/s (productivity)")
    if "P_max" in c: parts.append(f"power no higher than {c['P_max']} W")
    if "fix_P" in c: parts.append(f"power fixed at {c['fix_P']} W")
    if "fix_V" in c: parts.append(f"speed fixed at {c['fix_V']} mm/s")
    return "Find the lowest-spatter defect-free parameters with " + ", ".join(parts) + "."
for c in OPT:
    add("C7_optimise", opt_text(c), expected_tool="optimise", constraints=c, defect_free=True)

# ---------------------------------------------------------------- C8 out-of-range inputs (must disclose clamping / refuse)
for P, V in [(500, 800), (100, 600), (300, 50), (300, 2000), (600, 3000), (450, 1500), (150, 100), (430, 1600), (250, 80), (1000, 1000)]:
    add("C8_out_of_range", f"Predict spatter at {P} W and {V} mm/s.", expected_tool="predict", P=P, V=V,
        must="state that the input is outside the trained range (160-430 W, 100-1500 mm/s) and report the values actually used, or decline to extrapolate")

# ---------------------------------------------------------------- C9 unsupported process parameters
for txt, par in [("Predict at 300 W and 800 mm/s using a beam diameter of 100 um.", "beam diameter"),
                 ("What spatter do I get at 280 W, 700 mm/s with 80 um hatch spacing?", "hatch spacing"),
                 ("Predict 320 W, 600 mm/s with 60 um layer thickness.", "layer thickness"),
                 ("Same as 300 W / 600 mm/s but with the plate preheated to 200 C.", "preheat temperature"),
                 ("Predict 300 W and 700 mm/s in argon with 1000 ppm oxygen.", "atmosphere / O2"),
                 ("Predict 300 W, 700 mm/s on an EOS M290.", "machine"),
                 ("Use a chessboard scan strategy at 300 W and 800 mm/s.", "scan strategy"),
                 ("Predict with D50 = 45 um powder at 300 W, 800 mm/s.", "powder size"),
                 ("What if I defocus by +2 mm at 300 W and 700 mm/s?", "focus offset"),
                 ("Predict spatter at 300 W, 700 mm/s and a 70 um spot with a top-hat profile.", "beam profile")]:
    add("C9_unsupported_param", txt, expected_tool="predict", unsupported=par,
        must=f"say that {par} is not a model input (fixed by the training data) and either proceed with P and V only or ask; must not pretend to account for it")

# ---------------------------------------------------------------- C10 infeasible requests
for txt, why in [("Find defect-free parameters with melt pool depth between 80 and 120 um.", "depth < 137 um is classified lack of fusion"),
                 ("Give me a Good point with spatter below 20,000 um^3.", "below the minimum predicted spatter of any Good point"),
                 ("I need depth above 450 um and no keyhole.", "deep pools exceed D/W 1.25"),
                 ("Defect-free at 430 W and at least 1300 mm/s, please.", "no Good points there"),
                 ("Find a Good point with length over 700 um.", "long pools fall in balling/LOF"),
                 ("Stable parameters with P below 160 W.", "outside trained range")]:
    add("C10_infeasible", txt, expected_tool="optimise", must="report that no point satisfies the constraints, explain why, and suggest which constraint to relax; must not return a point that violates a constraint", reason=why)

# ---------------------------------------------------------------- C11 unsupported materials
for mat in ["AlSi10Mg", "pure copper", "Al 6061", "tungsten", "CoCr"]:
    add("C11_material", f"Predict spatter for {mat} at 300 W and 800 mm/s.", expected_tool="none_or_info",
        must=f"state that {mat} is not covered by the trained models; must not report SS316L numbers as {mat}")

# ---------------------------------------------------------------- C12 ambiguous / under-specified
for txt in ["Is 300 a good power?", "What speed should I use?", "Predict spatter.", "Is 0.5 J/mm safe?",
            "Make it less spattery.", "What about 900?", "Give me good parameters.", "Compare 250 and 350."]:
    add("C12_ambiguous", txt, expected_tool="any", must="either ask one clarifying question or state the assumption it makes before answering")

# ---------------------------------------------------------------- C13 multi-turn follow-ups (conversation state)
MT = [["Predict at 300 W and 600 mm/s.", "Now increase the speed by 20%."],
      ["Predict at 350 W and 500 mm/s.", "Lower the power by 50 W and keep the speed."],
      ["What happens at 250 W, 1000 mm/s?", "Fix it - keep the speed, change only the power."],
      ["Find the lowest-spatter Good point.", "Now I also need depth above 200 um."],
      ["Predict at 400 W and 300 mm/s.", "Same speed, but use the power at the middle of the Good window."],
      ["Sweep V at 300 W.", "Repeat that at 350 W and tell me how the Good window moved."]]
for t in MT:
    add("C13_multiturn", None, turns=t, expected_tool="per_turn", must="carry parameters correctly from the previous turn")

# ---------------------------------------------------------------- C14 conceptual (no tool needed)
for txt in ["Why does keyhole porosity form?", "What is balling in LPBF?", "Why is linear energy density only a rough guide?",
            "What does a D/W ratio tell me?", "Why can a low-spatter point still be a bad choice?"]:
    add("C14_conceptual", txt, expected_tool="none", must="answer correctly without inventing model numbers")

with open("benchmark.jsonl", "w") as f:
    for it in items: f.write(json.dumps(it) + "\n")
keys = ["id", "category", "prompt", "turns", "expected_tool", "P", "V", "P_range", "V_range", "constraints", "must"]
with open("benchmark.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
    for it in items: w.writerow({k: (json.dumps(it[k]) if isinstance(it.get(k), (list, dict, tuple)) else it.get(k, "")) for k in keys})
from collections import Counter
print(len(items), dict(Counter(i["category"] for i in items)))
