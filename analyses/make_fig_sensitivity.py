import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = pd.read_csv("sensitivity.csv"); b = r[r.param=="baseline"].iloc[0]
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
specs = [("keyhole D/W", 1.25, "Keyhole threshold $D/W$"), ("layer t (um)", 137, r"Layer thickness $t$ ($\mu$m)"),
         ("balling lower L/W", 2.35, "Balling lower bound $L/W$"), ("balling upper L/W", 3.7, "Balling upper bound $L/W$")]
fig, axs = plt.subplots(1, 4, figsize=(10, 2.6), sharey=True)
for ax, (p, bv, xl) in zip(axs, specs):
    s = r[r.param==p].copy()
    if p == "balling upper L/W":
        s["x"] = s.value.replace("none", "5.0").astype(float)
    else: s["x"] = s.value.astype(float)
    s = pd.concat([s, pd.DataFrame([{**b.to_dict(), "x": bv}])]).sort_values("x")
    ax.plot(s.x, s.acc_in, "o-", color="#1f4e79", ms=3.5, label="in-sample")
    ax.plot(s.x, s.acc_cv, "s--", color="#5b9bd5", ms=3.5, label="5-fold CV")
    ax2 = ax.twinx(); ax2.plot(s.x, s.good_area, "^:", color="#c55a11", ms=3.5, label="Good area")
    ax2.set_ylim(0, 35); ax.set_ylim(84, 100); ax.axvline(bv, color="0.6", lw=0.8, ls="-")
    ax.set_xlabel(xl)
    if p == "balling upper L/W":
        ax.set_xticks([3.1, 3.4, 3.7, 4.0, 5.0]); ax.set_xticklabels(["3.1", "3.4", "3.7", "4.0", "none"])
    if ax is not axs[-1]: ax2.set_yticklabels([])
    else: ax2.set_ylabel("Good area (%)", color="#c55a11")
axs[0].set_ylabel("Accuracy (%)")
h1, l1 = axs[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1+h2, l1+l2, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
for i, ax in enumerate(axs): ax.text(0.02, 0.04, "(%s)" % "abcd"[i], transform=ax.transAxes)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig("fig_sensitivity.pdf"); fig.savefig("fig_sensitivity.png", dpi=200)
print("saved")
