import json
T = []
def add(id, src, blocks, note=""): T.append(dict(id=id, source=src, blocks=blocks, note=note))
U = lambda t: dict(type="user", text=t)
TOOL = lambda name, args=None, note="": dict(type="tool", name=name, args=args or {}, note=note)
A = lambda t, cut=False: dict(type="answer", text=t, cut=cut)

add("keyhole", "Keyhole.png", [U("Predict at 400 W and 300 mm/s"), TOOL("mcp_predict_all", note="arguments not shown in the screenshot"),
A("""Here are the predictions for **SS316L at 400 W and 300 mm/s**:

| Metric | Value |
|---|---|
| Spatter Volume | 342,833 µm³ |
| Melt Pool Depth | 442 µm |
| Melt Pool Width | 251 µm |
| Melt Pool Length | 266 µm |
| Cross-Section Area | 66,215 µm² |
| Melt Pool Volume | 8.74 × 10⁶ µm³ |
| Energy Density (J) | 1.33 J/mm |
| Defect Type | **Keyhole** |

The depth-to-width ratio of 1.76 exceeds the keyhole threshold (1.25), indicating keyhole-mode melting at these parameters. This results in high spatter and potential porosity. You may want to consider reducing power or increasing scan speed to move""", cut=True)])

add("balling", "balling.png", [U("Predict at 300 W and 800 mm/s"), TOOL("mcp_predict_all", note="arguments not shown in the screenshot"),
A("""Here are the predictions for **SS316L at 300 W and 800 mm/s**:

| Parameter | Value |
|---|---|
| Energy Density (J) | 0.375 J/mm |
| Spatter Volume | 164,357 µm³ |
| Melt Pool Length | 493 µm |
| Melt Pool Width | 179 µm |
| Melt Pool Depth | 143 µm |
| Cross-Section Area | 17,483 µm² |
| Melt Pool Volume | 4,018,877 µm³ |
| Defect Type | Balling |

The model predicts a **balling defect** at these parameters. This is indicated by the length-to-width ratio (L/W = 2.76), which falls within the balling threshold range (2.35–3.7). The relatively high spatter volume also suggests this isn't an optimal operating point.

Would you like me to suggest better parameters to avoid this defect?""")])

add("lackoffusion", "lackoffusion.png", [U("Predict at 300 W and 1200 mm/s"), TOOL("mcp_predict_all", note="arguments not shown in the screenshot"),
A("""Here are the predictions for **SS316L at 300 W and 1200 mm/s**:

| Parameter | Value |
|---|---|
| Energy Density (J) | 0.25 J/mm |
| Spatter Volume | 109,574 µm³ |
| Melt Pool Length | 612 µm |
| Melt Pool Width | 164 µm |
| Melt Pool Depth | 92 µm |
| Cross-Section Area | 9,787 µm² |
| Melt Pool Volume | 2.40 × 10⁶ µm³ |
| Defect Type | Lack of Fusion |

The shallow depth (92 µm, below the 137 µm threshold) indicates insufficient melting at these parameters. You may want to increase power or reduce scan speed to achieve better fusion""", cut=True)])

add("unit1", "Unit1.png", [U("what should I expect from operating at 350 W and 1 m/s"), TOOL("mcp_predict_all", note="arguments not shown in the screenshot"),
A("""Here's what to expect for **SS316L at 350 W and 1000 mm/s**:

| Parameter | Value |
|---|---|
| Energy Density (J) | 0.35 J/mm |
| Spatter Volume | 226,437 µm³ |
| Melt Pool Length | 601 µm |
| Melt Pool Width | 189 µm |
| Melt Pool Depth | 133 µm |
| Cross-Section Area | 16,349 µm² |
| Melt Pool Volume | 4.43 × 10⁶ µm³ |
| Defect Type | Lack of Fusion |

These parameters are on the edge—the depth (133 µm) is just below the 137 µm threshold for good fusion. You're close to a stable operating regime, but still at risk for lack-of-fusion porosity.""")])

add("unit2", "Unit2.png", [U("predict at 295 and 750"), TOOL("mcp_predict_all", note="arguments not shown in the screenshot"),
A("""Here are the predictions for **SS316L at P = 295 W, V = 750 mm/s**:

| Metric | Value |
|---|---|
| Energy Density (J) | 0.393 J/mm |
| Spatter Volume | 144,675 µm³ |
| Melt Pool Length | 468 µm |
| Melt Pool Width | 182 µm |
| Melt Pool Depth | 153 µm |
| Cross-Section Area | 18,937 µm² |
| Melt Pool Volume | 4.13 × 10⁶ µm³ |
| Defect Type | Balling |

The model predicts **Balling** defects at these parameters. This is indicated by the length-to-width ratio (L/W = 2.58), which falls within the balling range (2.35–3.7). You may want to adjust parameters to move into a more stable processing window.""")])

add("unit3", "Unit3.png", [U("Use P = 150 + 0.5×500 and V = 2×400."), TOOL("mcp_predict_all", note="arguments not shown in the screenshot"),
A("""Here are the predictions for **P = 400 W** and **V = 800 mm/s** (SS316L):

| Metric | Value |
|---|---|
| Energy Density (J) | 0.50 J/mm |
| Spatter Volume | 180,078 µm³ |
| Melt Pool Length | 514 µm |
| Melt Pool Width | 205 µm |
| Melt Pool Depth | 216 µm |
| Cross-Section Area | 28,686 µm² |
| Melt Pool Volume | 6.90 × 10⁶ µm³ |
| Defect Type | Balling |

The model predicts a **Balling** defect at these parameters. The length-to-width ratio (L/W ≈ 2.51) falls within the balling regime (2.35–3.7), indicating an elongated, unstable melt pool. You may want to reduce power or increase velocity to move toward a more stable "Good" regime.""")])

add("unit4", "Unit4.png", [U("Hit it with 350-ish watts, speed a bit under a meter per second."), TOOL("mcp_predict_all", note="arguments not shown in the screenshot"),
A("""At **350 W / 950 mm/s** for SS316L:

| Metric | Value |
|---|---|
| Spatter Volume | ~230,000 µm³ |
| Melt Pool Depth | 142 µm |
| Melt Pool Width | 190 µm |
| Melt Pool Length | 581 µm |
| Energy Density (J) | 0.37 J/mm |
| Defect Type | Balling |

The length-to-width ratio of ~3.1 pushes this into **balling territory** — the melt pool is getting elongated and unstable. Spatter is also on the higher side.

If you want to stay in this power range but reduce spatter and avoid balling, bumping the speed down closer to 600–700 mm/s or dropping power to ~280–300 W would help stabilize things. Want me to find an optimized point nearby?""")])

add("beam_diameter", "beam_diameter.png", [U("I want to predict at 300 W and 800 mm/s using beam diameter of 100 micron"),
A("""Similar to hatch spacing, **beam diameter is not an adjustable input** in this model—it's another fixed assumption from the training data.

The model only accepts:
- **P** (Power in Watts)
- **V** (Velocity in mm/s)
- **Material**

Let me run the prediction with the parameters you can control, and then I'll show you how to infer what beam diameter the model likely assumes:"""),
TOOL("mcp_predict_all", {"P": 300, "V": 800, "material": "SS316L"}, note="tool request awaiting the user's approval; the screenshot ends here")],
note="earlier turns of this conversation (hatch spacing) are not shown")

add("linescan_P300", "Line_Scan_Power.png", [A("""**Query:** "At P = 300 W, sweep velocity from 200 to 1200 mm/s and find the minimum spatter operating point."

**Line Scan Results for SS316L at P = 300 W:**

| V (mm/s) | J (J/mm) | Spatter Volume (µm³) | Defect Type |
|---|---|---|---|
| 200 | 1.50 | 196,222 | Keyhole |
| 302 | 0.99 | 175,536 | Keyhole |
| 405 | 0.74 | 109,925 | Keyhole |
| 431 | 0.70 | 94,965 | Good ✓ |
| 482 | 0.62 | 73,736 | Good ✓ |
| **533** | **0.56** | **66,239** | **Good ✓** |
| 584 | 0.51 | 72,400 | Good ✓ |
| 636 | 0.47 | 90,246 | Good ✓ |
| 687 | 0.44 | 115,725 | Good ✓ |
| 713 | 0.42 | 129,351 | Balling |
| 815 | 0.37 | 167,017 | Balling |
| 841 | 0.36 | 168,512 | Lack of Fusion |
| 1,020 | 0.29 | 125,163 | Lack of Fusion |
| 1,200 | 0.25 | 109,574 | Lack of Fusion |

**Optimal Point:** V = 533 mm/s yields minimum spatter (66,239 µm³) within the stable "Good" regime.

The scan reveals clear defect transitions: Keyhole → Good → Balling → Lack of Fusion as velocity increases. Spatter volume shows a characteristic minimum in the conduction-mode regime before rising again as the melt pool becomes unstable.""")],
note="the screenshot shows the agent's answer only")

add("linescan_V600", "Line_Scan_Velocity.png", [A("""**Query:** "At V = 600 mm/s, sweep power from 200 to 420 W and show how the defect type changes."

**Line Scan Results for SS316L at V = 600 mm/s:**

| P (W) | J (J/mm) | Spatter Volume (µm³) | Defect Type |
|---|---|---|---|
| 200 | 0.33 | 51,152 | Lack of Fusion |
| 217 | 0.36 | 60,583 | Lack of Fusion |
| **228** | **0.38** | **67,561** | **Good ✓** |
| 256 | 0.43 | 80,331 | Good ✓ |
| 290 | 0.48 | 79,556 | Good ✓ |
| 324 | 0.54 | 69,810 | Good ✓ |
| **330** | **0.55** | **69,297** | **Good ✓** |
| 352 | 0.59 | 77,332 | Good ✓ |
| 358 | 0.60 | 82,732 | Good ✓ |
| 364 | 0.61 | 89,724 | Keyhole |
| 392 | 0.65 | 145,658 | Keyhole |
| 420 | 0.70 | 187,234 | Keyhole |

**Optimal Point:** P = 330 W yields minimum spatter (69,297 µm³) within the stable "Good" regime.

At this scan velocity, the stable processing window spans approximately 228–358 W. Below 228 W, insufficient energy leads to lack of fusion defects. Above 358 W, excessive penetration triggers keyhole-mode melting with significantly elevated spatter.""")],
note="the screenshot shows the agent's answer only")

add("gridscan", "Grid_Scan.png", [A("""**Query:** "Scan the parameter space from P = 200–400 W and V = 300–1000 mm/s. Show me only 'Good' points with spatter < 100,000 µm³, ranked by lowest spatter."

**Filtered Grid Scan Results for SS316L:**

| Rank | P (W) | V (mm/s) | J (J/mm) | Spatter (µm³) | Depth (µm) | Length (µm) | Defect |
|---|---|---|---|---|---|---|---|
| **1** | **200** | **500** | **0.40** | **55,600** | **142** | **305** | Good ✓ |
| 2 | 200 | 450 | 0.44 | 59,119 | 158 | 290 | Good ✓ |
| 3 | 214 | 550 | 0.39 | 59,687 | 140 | 332 | Good ✓ |
| 4 | 214 | 500 | 0.43 | 60,777 | 156 | 315 | Good ✓ |
| 5 | 229 | 550 | 0.42 | 65,788 | 155 | 343 | Good ✓ |
| 6 | 229 | 500 | 0.46 | 65,791 | 172 | 325 | Good ✓ |
| 7 | 214 | 450 | 0.48 | 66,008 | 174 | 299 | Good ✓ |
| 8 | 314 | 550 | 0.57 | 66,110 | 243 | 378 | Good ✓ |
| 9 | 300 | 550 | 0.55 | 66,804 | 228 | 376 | Good ✓ |
| 10 | 229 | 600 | 0.38 | 67,781 | 139 | 361 | Good ✓ |

**Summary:** Scanned 225 points (15×15 grid). Found 10 points meeting all constraints.

**Best Operating Point:** P = 200 W, V = 500 mm/s produces the lowest spatter (55,600 µm³) while maintaining defect-free processing with adequate melt pool depth (142 µm).""")],
note="the screenshot shows the agent's answer only")

add("opt_energy", "Optimization_Energy.png", [A("""**Query:** "Optimize for minimum spatter within an energy density window of 0.4–0.6 J/mm."

**Optimization Results for SS316L:**

**Constraint Applied:**
- Energy density (J): 0.4–0.6 J/mm
- Defect type: Good (defect-free)

| Rank | P (W) | V (mm/s) | J (J/mm) | Spatter (µm³) | Depth (µm) |
|---|---|---|---|---|---|
| **1** | **160** | **341** | **0.47** | **40,892** | **146** |
| 2 | 169 | 390 | 0.43 | 46,540 | 142 |
| 3 | 179 | 438 | 0.41 | 50,456 | 138 |
| 4 | 188 | 438 | 0.43 | 54,623 | 148 |
| 5 | 179 | 390 | 0.46 | 54,785 | 153 |

**Recommendation:** The optimal point at J = 0.47 J/mm sits comfortably in the middle of the specified energy density window, providing margin against process drift while minimizing spatter generation.""")],
note="the screenshot shows the agent's answer only")

add("opt_depth", "Optimization_Depth.png", [A("""**Query:** "Find the best parameters with melt pool depth between 120–180 µm and spatter under 90,000 µm³."

**Optimization Results for SS316L:**

**Constraints Applied:**
- Melt pool depth: 120–180 µm
- Maximum spatter: 90,000 µm³
- Defect type: Good (defect-free)

| Rank | P (W) | V (mm/s) | Spatter (µm³) | Depth (µm) | Length (µm) | Status |
|---|---|---|---|---|---|---|
| **1** | **160** | **341** | **40,892** | **146** | **240** | **Optimal** |
| 2 | 169 | 390 | 46,540 | 142 | 256 | Alternative |
| 3 | 179 | 438 | 50,456 | 138 | 273 | Alternative |
| 4 | 188 | 438 | 54,623 | 148 | 279 | Alternative |
| 5 | 179 | 390 | 54,785 | 153 | 260 | Alternative |

**Search Summary:** Scanned 900 points (30×30 grid). Found 5 points satisfying all constraints.

**Recommendation:** P = 160 W at V = 341 mm/s delivers the lowest spatter (40,892 µm³) while maintaining a melt pool depth of 146 µm — well within the specified range for reliable inter-layer bonding.""")],
note="the screenshot shows the agent's answer only")

add("opt_multi", "Optimization_Multi.png", [A("""**Query:** "I need a deep melt pool (150–250 µm) with length between 350–500 µm, and spatter under 120,000 µm³. What are my options?"

**Optimization Results for SS316L:**

**Constraints Applied:**
- Melt pool depth: 150–250 µm
- Melt pool length: 350–500 µm
- Maximum spatter: 120,000 µm³
- Defect type: Good (defect-free)

| Rank | P (W) | V (mm/s) | Spatter (µm³) | Depth (µm) | Length (µm) | Width (µm) |
|---|---|---|---|---|---|---|
| **1** | **300** | **534** | **66,254** | **235** | **369** | **211** |
| 2 | 309 | 534 | 66,351 | 244 | 370 | 213 |
| 3 | 290 | 534 | 67,340 | 225 | 367 | 208 |
| 4 | 328 | 583 | 67,649 | 242 | 395 | 211 |
| 5 | 318 | 583 | 67,768 | 233 | 394 | 209 |

**Search Summary:** Scanned 900 points. Found 5 viable parameter sets meeting all constraints.

**Recommendation:** P = 300 W at V = 534 mm/s provides the best balance — achieving a deep melt pool (235 µm) with sufficient length (369 µm) for good track overlap while maintaining the lowest spatter in this constrained region.""")],
note="the screenshot shows the agent's answer only")

json.dump(T, open("screenshot_transcripts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(T), "transcripts written")
