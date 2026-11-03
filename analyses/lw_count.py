"""Count simulated tracks by L/W and D/W band (reviewer point 2)."""
import pandas as pd, numpy as np
d = pd.read_csv("../data/Agentic_defect_Dataset.csv")
d["LW"] = d.Length/d.Width; d["DW"] = d.Depth/d.Width
def label(D,W,L):
    if D/W > 1.25: return "Keyhole"
    if D < 137: return "Lack of Fusion"
    if 2.35 < L/W < 3.7: return "Balling"
    return "Good"
d["lab"] = [label(*r) for r in d[["Depth","Width","Length"]].values]
print("n =", len(d)); print(d.lab.value_counts().to_string())
print("\nL/W range %.2f-%.2f" % (d.LW.min(), d.LW.max()))
hi = d[d.LW > 3.7]
print("tracks with L/W > 3.7:", len(hi))
print(hi[["Power","Velocity","Depth","Width","Length","LW","DW","lab"]].round(2).to_string(index=False))
nk = d[(d.DW<=1.25)]
print("\namong non-keyhole tracks: L/W>3.7 & D>=137 (would be Good->Balling if cap removed):", ((nk.LW>3.7)&(nk.Depth>=137)).sum())
print("L/W>3.7 & D<137 (LoF either way):", ((nk.LW>3.7)&(nk.Depth<137)).sum())
