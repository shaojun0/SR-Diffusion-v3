"""sweep_res_report.py — 汇总分辨率扫描：逐 step 变化率 + 与 JPEG 的 RD 对比

输入：out/<size>_<arm>/result.json（sweep_res_train.py）
      out/<size>_jpeg.json（sweep_res_jpeg.py）
输出：sweep_res_report.md、sweep_res_rd.png、sweep_res_steprate.png、
      sweep_res_summary.json

用法:
  python sweep_res_report.py --out_dir /root/autodl-tmp/srres/out \
      --report /root/autodl-tmp/srres/out/sweep_res_report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np


def windows(N, steps):
    """每个采样步**实际**读到的窗口宽度（含首步的 z_cls）。

    块 k = [k², min((k+1)²−1, N)]；首步 lo=0 ⇒ 多一个 z_cls。
    K=N 时末块的右端被截断 ⇒ **末步窗口退化为 1 个 token**
    （仓库 doc/2026-09-16/ANALYSIS_v2_bptt_cpu_verify.md §2.4 已记录同一现象）。
    """
    out = []
    for i, t in enumerate(steps):
        k = math.isqrt(int(t))
        hi = min((k + 1) ** 2 - 1, N)
        lo = 0 if i == 0 else max(k * k, 1)
        out.append(hi - lo + 1)
    return out


def interp(xs, ys, x):
    xs, ys = np.asarray(xs), np.asarray(ys)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(x, xs, ys))


def load(out_dir):
    runs = {}
    for fp in sorted(glob.glob(os.path.join(out_dir, "*", "result.json"))):
        r = json.load(open(fp, encoding="utf-8"))
        runs[(r["size"], r["arm"])] = r
    base = {}
    for fp in sorted(glob.glob(os.path.join(out_dir, "*_jpeg.json"))):
        b = json.load(open(fp, encoding="utf-8"))
        base[b["size"]] = b
    return runs, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/root/autodl-tmp/srres/out")
    ap.add_argument("--report", default="/root/autodl-tmp/srres/out/sweep_res_report.md")
    ap.add_argument("--arms", default="bptt,detach")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]

    runs, base = load(args.out_dir)
    global ARMS_PRESENT, ARM_COLOR
    ARMS_PRESENT = [a for a in arms if any(k[1] == a for k in runs)] or \
                   sorted({k[1] for k in runs})
    _cols = ["tab:blue", "tab:green", "tab:purple", "tab:gray", "tab:brown"]
    ARM_COLOR = lambda k: _cols[k % len(_cols)]   # noqa: E731
    if not runs:
        raise SystemExit("没有 result.json")
    sizes = sorted({s for (s, a) in runs})
    print(f"[report] sizes={sizes} arms={sorted({a for (_,a) in runs})}")

    # ── 汇总表 ──
    rows = []
    for s in sizes:
        for a in arms:
            r = runs.get((s, a))
            if not r:
                continue
            l1 = np.asarray(r["l1_255"])
            i_first, i_best, i_last = 0, int(l1.argmin()), len(l1) - 1
            win = windows(r["num_patches"], r["steps"])
            cum = list(np.cumsum(win))
            row = {
                "win": win, "cum_read": [int(v) for v in cum],
                "bpp_actual": [c * r["dim"] / (s * s) for c in cum],
                "size": s, "arm": a, "N": r["num_patches"], "T": len(r["steps"]),
                "dim": r["dim"], "train_steps": r.get("train_steps"),
                "warm_steps": r.get("warm_steps"),
                "steps": r["steps"], "bpp": r["bpp"], "l1": r["l1_255"],
                "psnr": r["psnr"],
                "l1_first": float(l1[i_first]), "l1_best": float(l1[i_best]),
                "l1_last": float(l1[i_last]), "i_best": i_best,
                "best_t": r["steps"][i_best], "best_bpp": r["bpp"][i_best],
                "best_psnr": r["psnr"][i_best],
                "rel_gain_pct": 100 * (l1[i_first] - l1[i_best]) / l1[i_first],
                "last_vs_best_pct": 100 * (l1[i_last] - l1[i_best]) / l1[i_best],
                "train_secs": r.get("train_secs"),
                "params_M": r.get("params_M"),
            }
            # 逐 step 变化率（相邻步）
            if len(l1) > 1:
                row["d_l1"] = np.diff(l1).tolist()
                row["d_l1_rel_pct"] = (100 * np.diff(l1) / l1[:-1]).tolist()
                # 改善的累计占比（0 = 首步，100 = 最好步）——"改善集中在前几步"
                denom = l1[i_first] - l1[i_best]
                row["cum_share"] = (100 * (l1[i_first] - l1) / denom).tolist() \
                    if denom > 1e-9 else [0.0] * len(l1)
                row["share_step1"] = row["cum_share"][0]
                row["share_step2"] = row["cum_share"][1] if len(l1) > 1 else 0.0
                row["max_step_change_pct"] = float(np.min(row["d_l1_rel_pct"]))
            b = base.get(s)
            if b:
                row["jpeg"] = b["codecs"].get("JPEG", [])
                row["webp"] = b["codecs"].get("WebP", [])
                if b["codecs"].get("JPEG"):
                    jb = [x["bpp"] for x in b["codecs"]["JPEG"]]
                    jp = [x["psnr"] for x in b["codecs"]["JPEG"]]
                    row["jpeg_psnr_at_best_bpp"] = interp(jb, jp, row["best_bpp"])
                    row["d_psnr_vs_jpeg"] = row["best_psnr"] - row["jpeg_psnr_at_best_bpp"]
                if b["codecs"].get("WebP"):
                    wb = [x["bpp"] for x in b["codecs"]["WebP"]]
                    wp = [x["psnr"] for x in b["codecs"]["WebP"]]
                    row["d_psnr_vs_webp"] = row["best_psnr"] - interp(wb, wp, row["best_bpp"])
            rows.append(row)

    # ── 图 1：每分辨率的 RD（我们 vs JPEG/WebP）──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(sizes)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow),
                                 squeeze=False)
        for k, s in enumerate(sizes):
            ax = axes[k // ncol][k % ncol]
            b = base.get(s)
            if b:
                for fmt, c, m in (("JPEG", "tab:red", "s"), ("WebP", "tab:orange", "^")):
                    if b["codecs"].get(fmt):
                        ax.plot([x["bpp"] for x in b["codecs"][fmt]],
                                [x["psnr"] for x in b["codecs"][fmt]],
                                "-" + m, ms=4, color=c, label=fmt)
            for k, a in enumerate(ARMS_PRESENT):
                for row in rows:
                    if row["size"] == s and row["arm"] == a:
                        ax.plot(row["bpp"], row["psnr"], "-o", ms=4,
                                color=ARM_COLOR(k), label=f"ours ({a})")
            ax.set_title(f"{s}x{s}  N={(s//14)**2}  |T|={len(runs[(s,arms[0])]['steps'])}")
            ax.set_xlabel("bpp (beta=1, estimated)")
            ax.set_ylabel("PSNR (dB)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        for k in range(n, nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle("Resolution sweep: nested per-step RD vs JPEG/WebP (DIV2K val, DINOv2-small)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "sweep_res_rd.png"), dpi=130)
        print("-> sweep_res_rd.png")
    except Exception as e:                                   # noqa: BLE001
        print(f"[warn] 画 RD 图失败: {e}")

    # ── 图 2：逐 step 变化率 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        cmap = plt.get_cmap("viridis")
        for i, s in enumerate(sizes):
            for j, a in enumerate(ARMS_PRESENT):
                for row in rows:
                    if (row["size"] == s and row["arm"] == a and "d_l1" in row
                            and "long" not in a):
                        x = np.arange(1, len(row["l1"]) + 1)
                        rel = 100 * (row["l1"][0] - np.asarray(row["l1"])) / row["l1"][0]
                        axes[0].plot(x, rel, "-", marker="o", ms=3.5,
                                     color=cmap(i / max(1, len(sizes) - 1)),
                                     label=f"{s}px")
                        axes[1].plot(np.asarray(row["bpp"])[1:],
                                     row["d_l1_rel_pct"], "-", marker="o", ms=3.5,
                                     color=cmap(i / max(1, len(sizes) - 1)),
                                     label=f"{s}px")
        axes[0].set_xlabel("decode step index")
        axes[0].set_ylabel("cumulative L1 gain vs step 1 (%)")
        axes[0].set_title("Cumulative per-step gain")
        axes[1].set_xlabel("bpp (beta=1, nominal)")
        axes[1].set_ylabel("L1 change rate per step (%/step)")
        axes[1].set_title("Per-step L1 change rate vs bpp")
        axes[1].set_xscale("log")
        for ax in axes:
            ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "sweep_res_steprate.png"), dpi=130)
        print("-> sweep_res_steprate.png")
    except Exception as e:                                   # noqa: BLE001
        print(f"[warn] 画 steprate 图失败: {e}")

    with open(os.path.join(args.out_dir, "sweep_res_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # ── markdown ──
    L = []
    L.append("# 分辨率扫描：逐 step L1 变化率 + 与 JPEG/WebP 的 RD 对比\n")
    L.append("> 小模型 = DINOv2-small(384) + **z_s = patch token(K=N)** + "
             "OutputQueryDecoder(depth=2) + PixelHead；BPTT（carry 不 detach）；"
             "DIV2K train 800 / val 100；直接 resize 到 S×S；"
             "bpp名义=(t+1)·384/S²（β=1，与仓库同一估计口径）。\n")
    L.append("> ⚠️ **预算口径**：各分辨率训练预算**相同**（warm 1000 + nested 1500 步），"
             "但大图是更难的任务 ⇒ **分辨率越大越欠训**，"
             "跨分辨率的绝对高低不能当收敛后的 RD 结论读（见报告正文「口径与边界」）。\n")
    L.append("## 主表（每条分辨率取**最好步**）\n")
    L.append("| 分辨率 | N | \\|T\\| | 末步读到 | 首步 L1 | 最好 L1 | 末步 L1 | 最好步 | 相对首步改善 | 末步vs最好 | "
             "最好步 bpp | 我们 PSNR | 同 bpp JPEG | ΔPSNR vs JPEG | ΔPSNR vs WebP | 最好步占改善 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append("| {size}² | {N} | {T} | {win_last} | {l1_first:.2f} | **{l1_best:.2f}** | {l1_last:.2f} | "
                 "t={best_t} | **{rel_gain_pct:.1f}%** | {last_vs_best_pct:+.1f}% | {best_bpp:.3f} | "
                 "{best_psnr:.2f} | {j} | {dj} | {dw} | {cs} |".format(
                     win_last=r["win"][-1], **r,
                     j=f"{r['jpeg_psnr_at_best_bpp']:.2f}" if "jpeg_psnr_at_best_bpp" in r else "—",
                     dj=f"{r['d_psnr_vs_jpeg']:+.2f}" if "d_psnr_vs_jpeg" in r else "—",
                     dw=f"{r['d_psnr_vs_webp']:+.2f}" if "d_psnr_vs_webp" in r else "—",
                     cs=(f"第1步 {r['share_step1']:.0f}% / 第2步 {r['share_step2']:.0f}%"
                         if "cum_share" in r and r.get("rel_gain_pct", 0) > 0.05 else "—")))
    L.append("\n## 逐 step 明细\n")
    for r in rows:
        L.append(f"\n### {r['size']}×{r['size']} — N={r['N']}, |T|={r['T']}, arm={r['arm']}\n")
        L.append("| i | t | 名义token | **本步读到** | 本步边际bpp | 累计bpp实际 | L1(0-255) | ΔL1 | 变化率 | 边际效率 ΔL1/Δbpp | PSNR | 累积改善 |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        prev = None
        dim = r["dim"]; px = r["size"] ** 2
        for i, (t, l1, bb, ps) in enumerate(zip(r["steps"], r["l1"], r["bpp"], r["psnr"])):
            d = "" if prev is None else f"{l1 - prev:+.3f}"
            rel = "" if prev is None else f"{100*(l1-prev)/prev:+.2f}%"
            cum = f"{100*(r['l1'][0]-l1)/r['l1'][0]:.1f}%"
            dbpp = r["win"][i] * dim / px
            eff = "" if prev is None else f"{(prev - l1) / dbpp:+.0f}"
            L.append(f"| {i} | {t} | {t+1} | {r['win'][i]} | {dbpp:.3f} | "
                     f"{r['bpp_actual'][i]:.3f} | {l1:.3f} | {d} | {rel} | {eff} | {ps:.2f} | {cum} |")
            prev = l1
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"-> {args.report}")


if __name__ == "__main__":
    main()
