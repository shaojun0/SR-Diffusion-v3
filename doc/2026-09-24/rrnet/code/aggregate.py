"""汇总 12 组实验结果 -> 表格 (markdown/csv) + 曲线图。"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT = "/root/autodl-tmp/rrnet/ckpt"
OUT = "/root/autodl-tmp/rrnet/results"
BACKBONES = ["resnet-10", "resnet-18", "resnet-34", "resnet-50", "resnet-101", "resnet-152"]
RESOLUTIONS = [224, 448]


def load():
    rows = []
    for res in RESOLUTIONS:
        for bb in BACKBONES:
            d = os.path.join(CKPT, f"{bb}_{res}")
            mp = os.path.join(d, "metrics.json")
            if not os.path.exists(mp):
                rows.append({"backbone": bb, "resolution": res, "status": "missing"})
                continue
            m = json.load(open(mp))
            hist = json.load(open(os.path.join(d, "history.json"))) if os.path.exists(os.path.join(d, "history.json")) else []
            cfg = json.load(open(os.path.join(d, "config.json"))) if os.path.exists(os.path.join(d, "config.json")) else {}
            rows.append({
                "backbone": bb, "resolution": res, "status": "ok",
                "best_val_l1": m["best_val_l1"],
                "final_val_l1": (m.get("final") or {}).get("l1"),
                "final_val_psnr": (m.get("final") or {}).get("psnr"),
                "final_val_ssim": (m.get("final") or {}).get("ssim"),
                "params_M": m.get("params_M"), "params_trainable_M": m.get("params_trainable_M"),
                "batch_size": m.get("batch_size"), "epochs_done": m.get("epochs_done"),
                "total_time_min": m.get("total_time_min"),
                "history": hist, "config": cfg,
            })
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    json.dump([{k: v for k, v in r.items() if k not in ("history", "config")} for r in rows],
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)

    # ---- csv ----
    cols = ["backbone", "resolution", "status", "best_val_l1", "final_val_l1",
            "final_val_psnr", "final_val_ssim", "params_M", "batch_size",
            "epochs_done", "total_time_min"]
    with open(os.path.join(OUT, "summary.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")

    # ---- markdown ----
    lines = ["# Recurrent ResNet: 6 backbone x 2 resolution (12 experiments)", "",
             "| backbone | res | best val L1 | final val L1 | val PSNR | val SSIM | params | bs | ep | time(min) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['backbone']} | {r['resolution']} | _missing_ | | | | | | | |")
            continue
        lines.append("| {bb} | {res} | {bl:.5f} | {fl:.5f} | {p:.3f} | {s:.4f} | {pm:.2f}M | {bs} | {ep} | {t:.1f} |".format(
            bb=r["backbone"], res=r["resolution"], bl=r["best_val_l1"],
            fl=r["final_val_l1"] or float("nan"), p=r["final_val_psnr"] or float("nan"),
            s=r["final_val_ssim"] or float("nan"), pm=r["params_M"] or 0,
            bs=r["batch_size"], ep=r["epochs_done"], t=r["total_time_min"] or 0))
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # ---- curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res in zip(axes, RESOLUTIONS):
        for bb in BACKBONES:
            r = next((x for x in rows if x["backbone"] == bb and x["resolution"] == res), None)
            if not r or r["status"] != "ok" or not r["history"]:
                continue
            ep = [h["epoch"] for h in r["history"]]
            ax.plot(ep, [h["l1"] for h in r["history"]], marker=".", label=bb)
        ax.set_title(f"val L1 @ {res}x{res}")
        ax.set_xlabel("epoch"); ax.set_ylabel("L1"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "curves_val_l1.png"), dpi=130)
    print("saved", os.path.join(OUT, "curves_val_l1.png"), flush=True)


if __name__ == "__main__":
    main()
