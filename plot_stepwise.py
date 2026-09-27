"""画逐 step 精炼曲线（从 eval_stepwise.py 的 json）。"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = sys.argv[1]
out = sys.argv[2]
d = json.load(open(p))
steps = d["steps"]
cum = d["cum_tokens"]
L1 = d["L1_per_reso"]
bf = d["L1_bestfit"]
psnr = d["PSNR_bestfit"]
const = d["const_baseline"]
names = ["448x252", "252x448", "224x224", "448x448"]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for i, nm in enumerate(names):
    ax[0].plot(range(1, len(steps) + 1), L1[i], marker="o", ms=3, lw=1.3, label=nm)
ax[0].plot(range(1, len(steps) + 1), bf, "k--", marker="s", ms=3, lw=2, label="best-fit")
ax[0].axhline(sum(const) / len(const), color="gray", ls=":", lw=1.5,
              label="mean-color baseline")
ax[0].set_xlabel("decode step (1..12)")
ax[0].set_ylabel("pixel L1 (0-255)")
ax[0].set_title(f"per-step refinement  n={d['n']} (COCO2017 val)")
ax[0].set_xticks(range(1, len(steps) + 1))
ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

ax[1].plot(cum, bf, "k-o", ms=4, lw=1.6)
for t, (c, v) in enumerate(zip(cum, bf)):
    if t in (0, 2, 5, 10, 11):
        ax[1].annotate(f"step{t+1}", (c, v), textcoords="offset points",
                       xytext=(4, 6), fontsize=7)
ax[1].axhline(sum(const) / len(const), color="gray", ls=":", lw=1.5)
ax[1].set_xlabel("cumulative z_s tokens (rate)")
ax[1].set_ylabel("pixel L1 (0-255)")
ax[1].set_title("quality vs transmitted tokens (best-fit)")
ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(out, dpi=130)
print("wrote", out)
