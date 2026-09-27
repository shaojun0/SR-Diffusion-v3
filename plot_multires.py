"""从 train_log.jsonl 出训练曲线（PNG + 文本摘要）。"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/autodl-tmp/output/multires_natural/train_log.jsonl"
out = sys.argv[2] if len(sys.argv) > 2 else \
    "/root/autodl-tmp/output/multires_natural/curve.png"

recs = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
if not recs:
    print("no records in", path)
    sys.exit(1)

steps = [r["step"] for r in recs]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(steps, [r["loss"] for r in recs], label="loss (weighted sum)")
ax[0].plot(steps, [r["recon"] for r in recs], label="recon (last step)", alpha=.7)
for i in range(4):
    ax[0].plot(steps, [r["per_decoder"][i] for r in recs],
               lw=1, alpha=.6, label=f"dec{i} " + ["448x252", "252x448",
                                                   "224x224", "448x448"][i])
ax[0].set_xlabel("step"); ax[0].set_ylabel("L1 (normalized)")
ax[0].set_title("MultiResSR training loss"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
ax[1].plot(steps, [r.get("grad_norm", 0) for r in recs], color="tab:red", lw=1)
ax[1].set_xlabel("step"); ax[1].set_ylabel("grad_norm"); ax[1].set_yscale("log")
ax[1].set_title("grad norm"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(out, dpi=130)
print("wrote", out)

last = recs[-1]
print(f"records={len(recs)} step={last['step']} loss={last['loss']} "
      f"recon={last['recon']} per_decoder={last['per_decoder']} "
      f"peak={last.get('peak_gib')}GiB {last.get('sec_per_step')}s/step")
first_loss = recs[0]["loss"]
best = min(r["loss"] for r in recs)
print(f"loss: first={first_loss} best={best} last={last['loss']}")
