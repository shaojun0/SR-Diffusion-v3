#!/usr/bin/env python3
"""compare_rerun_A3000_10k.py — 「修正配方」重跑（A 相 3000 + B 相 7000 = 总 1 万步）的汇总对照

把新臂（`--warm_steps 3000 --steps 7000 --arm A3000`）与
  · 旧等预算臂 `{size}_bptt-patch`（warm 1000 + B 1500）
  · 旧放大预算臂 `{size}_long-patch`（warm 1000 + B 5000/6000）
  · 224² 判决臂（`doc/2026-09-23/data/result_224_A3000.json`，warm 3000 + B 5000）
  · per-patch DC 平凡解基线（`tools/per_patch_baseline.py`）
  · 同图同分辨率 JPEG/WebP（`sweep_res_summary.json` 内嵌）
逐分辨率对照，输出 markdown。

用法:
    python tools/compare_rerun_A3000_10k.py --new_dir <含 {size}_A3000-patch/result.json 的目录>
    python tools/compare_rerun_A3000_10k.py --new_dir ... --report out.md --curves --hist
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_RUNS = os.path.join(REPO, "doc/2026-09-22/res_sweep/runs")
SUMMARY = os.path.join(REPO, "doc/2026-09-22/res_sweep/sweep_res_summary.json")
VERDICT = os.path.join(REPO, "doc/2026-09-23/data/result_224_A3000.json")
SIZES = [28, 56, 112, 224, 336]


def load_runs(runs_dir, pattern):
    out = {}
    for fp in sorted(glob.glob(os.path.join(runs_dir, pattern, "result.json"))):
        r = json.load(open(fp, encoding="utf-8"))
        out[int(r["size"])] = r
    return out


def load_json(path):
    return json.load(open(path, encoding="utf-8")) if path and os.path.exists(path) else None


def interp(xs, ys, x):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(x, xs, ys))


def windows(N, steps):
    """每步实际读到的窗口宽度（K=N 时末步退化为 1，见交接 §6.5）。"""
    w = []
    for i, t in enumerate(steps):
        k = math.isqrt(int(t))
        hi = min((k + 1) ** 2 - 1, N)
        lo = 0 if i == 0 else max(k * k, 1)
        w.append(hi - lo + 1)
    return w


def stats(r):
    l1 = np.asarray(r["l1_255"], float)
    ps = np.asarray(r["psnr"], float)
    i = int(np.argmin(l1))
    return {
        "l1_first": float(l1[0]), "l1_best": float(l1[i]), "i_best": i,
        "best_t": r["steps"][i], "psnr_first": float(ps[0]), "psnr_best": float(ps[i]),
        "rel_gain": float((l1[0] - l1[i]) / l1[0] * 100.0),
        "strict_down": int(np.sum(np.diff(l1) < 0)), "n": len(l1),
        "warm_steps": r.get("warm_steps", 0), "train_steps": r.get("train_steps", 0),
        "secs": float(r.get("train_secs", 0.0)), "bpp": r.get("bpp", []),
        "l1": l1, "psnr": ps,
    }


def phase_hist(r):
    """hist 拆 A/B 两相；每 checkpoint 记 (it, best L1, best PSNR, step1 L1)。"""
    a, b = [], []
    for e in r.get("hist", []):
        l1 = np.asarray(e["l1_255"], float)
        rec = (e["it"], float(l1.min()), float(np.asarray(e["psnr"], float).max()), float(l1[0]))
        (a if e["phase"] == "A" else b).append(rec)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_dir", required=True, help="新臂所在目录（含 {size}_A3000-patch/）")
    ap.add_argument("--old_runs", default=OLD_RUNS)
    ap.add_argument("--summary", default=SUMMARY)
    ap.add_argument("--verdict", default=VERDICT)
    ap.add_argument("--baseline", default="", help="per_patch_baseline.py 的 json")
    ap.add_argument("--report", default="", help="markdown 输出路径（默认打印）")
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--hist", action="store_true")
    args = ap.parse_args()

    new = load_runs(args.new_dir, "*_A3000-patch")
    old_b = load_runs(args.old_runs, "*_bptt-patch")
    old_l = load_runs(args.old_runs, "*_long-patch")
    summ = load_json(args.summary) or {}
    summ = {int(e["size"]): e for e in summ} if isinstance(summ, list) else summ
    base = load_json(args.baseline) or {}
    verdict = load_json(args.verdict)

    # JPEG 基线：优先用 new_dir 下的 {size}_jpeg.json，其次 summary 内嵌
    jpeg = {}
    for s in SIZES:
        fp = os.path.join(args.new_dir, f"{s}_jpeg.json")
        if os.path.exists(fp):
            jpeg[s] = load_json(fp).get("jpeg", [])
        elif s in summ:
            jpeg[s] = summ[s].get("jpeg", [])

    def base_psnr(s):
        if str(s) in base:
            return float(base[str(s)]["psnr"])
        return float("nan")

    L = []
    L.append("# 「修正配方」重跑汇总 — A 相 3000 + B 相 7000（总 1 万步）\n")
    L.append("机器：AutoDL bjb1 **1× A800-80GB**｜数据：DIV2K train 800 / val 100（预 resize `.npy`）｜"
             "z_s=patch token / BPTT / depth=2 / seed 42 / TF32\n")
    L.append("> 对照：旧等预算臂（warm 1000 + B 1500）、旧放大预算臂（warm 1000 + B 5000/6000）、")
    L.append("> 224² 判决臂（warm 3000 + B 5000）、per-patch DC 平凡解基线。")
    L.append("> **与旧臂的唯一变量 = A 相 1000→3000 与 B 相预算（→7000）。**\n")

    L.append("## A. 主对照表\n")
    L.append("| 分辨率 | \\|T\\| | per-patch DC 基线 | 旧等预算 best PSNR | 旧放大 best PSNR | **新臂 best PSNR** | Δ vs 旧放大 | 新臂 best L1 | 首→最好改善 | 严格下降步 | vs 基线 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in SIZES:
        bp = base_psnr(s)
        bps = f"{bp:.2f}" if np.isfinite(bp) else "—"
        if s not in new:
            L.append(f"| {s}² | — | {bps} | — | — | *（未跑）* | — | — | — | — | — |")
            continue
        n = stats(new[s])
        ob = stats(old_b[s]) if s in old_b else None
        ol = stats(old_l[s]) if s in old_l else None
        ob_s = f"{ob['psnr_best']:.2f}" if ob else "—"
        ol_s = f"{ol['psnr_best']:.2f}" if ol else "—"
        d_s = f"{n['psnr_best'] - ol['psnr_best']:+.2f} dB" if ol else "—"
        db_s = f"{n['psnr_best'] - bp:+.2f} dB" if np.isfinite(bp) else "—"
        L.append(f"| {s}² | {n['n']} | {bps} | {ob_s} | {ol_s} | **{n['psnr_best']:.2f}** | {d_s} | "
                 f"{n['l1_best']:.2f} | {n['rel_gain']:.1f}% | {n['strict_down']}/{n['n'] - 1} | {db_s} |")

    if verdict is not None:
        v = stats(verdict)
        L.append(f"\n> **224² 判决臂参照**（warm 3000 + B 5000 = 8000 步，{v['secs']:.0f} s）："
                 f"best PSNR **{v['psnr_best']:.2f}** / best L1 **{v['l1_best']:.2f}**。"
                 f"新臂 B 相多 2000 步 ⇒ 差值即「延长 B 相」的收益。\n")

    if args.curves:
        L.append("## B. 逐 step 曲线（新臂；`F_hat` = 末步）\n")
        for s in SIZES:
            if s not in new:
                continue
            r, n = new[s], stats(new[s])
            L.append(f"### {s}²（N={r['num_patches']}，\\|T\\|={n['n']}）\n")
            L.append("```")
            L.append("t     : " + " ".join(f"{t:>6}" for t in r["steps"]))
            L.append("win   : " + " ".join(f"{v:>6}" for v in windows(r["num_patches"], r["steps"])))
            L.append("L1    : " + " ".join(f"{v:>6.2f}" for v in n["l1"]))
            L.append("PSNR  : " + " ".join(f"{v:>6.2f}" for v in n["psnr"]))
            L.append("bpp   : " + " ".join(f"{v:>6.3f}" for v in n["bpp"]))
            L.append("```")
            jl = jpeg.get(s) or []
            if jl:
                jb = interp([j["bpp"] for j in jl], [j["psnr"] for j in jl], n["bpp"][n["i_best"]])
                L.append(f"- best 点 bpp={n['bpp'][n['i_best']]:.3f} ⇒ 同 bpp JPEG {jb:.2f} dB，"
                         f"**ΔPSNR vs JPEG {n['psnr_best'] - jb:+.2f} dB**")
            bp = base_psnr(s)
            if np.isfinite(bp):
                L.append(f"- vs per-patch DC 基线 {bp:.2f} dB：**{n['psnr_best'] - bp:+.2f} dB**")
            L.append("")

    if args.hist:
        L.append("## C. 两相轨迹（A 相逃逸 vs B 相冷切换+恢复）\n")
        for s in SIZES:
            if s not in new:
                continue
            a, b = phase_hist(new[s])
            bp = base_psnr(s)
            L.append(f"### {s}²\n")
            if a:
                L.append("| A 相步 | " + " | ".join(str(x[0]) for x in a) + " |")
                L.append("|---|" + "---:|" * len(a))
                L.append("| L1（单步全读） | " + " | ".join(f"{x[1]:.2f}" for x in a) + " |")
                L.append("| PSNR | " + " | ".join(f"{x[2]:.2f}" for x in a) + " |")
            if b:
                L.append("\n| B 相步 | " + " | ".join(str(x[0]) for x in b) + " |")
                L.append("|---|" + "---:|" * len(b))
                L.append("| best L1 | " + " | ".join(f"{x[1]:.2f}" for x in b) + " |")
                L.append("| step1 L1（冷切换代价） | " + " | ".join(f"{x[3]:.2f}" for x in b) + " |")
                L.append("| best PSNR | " + " | ".join(f"{x[2]:.2f}" for x in b) + " |")
            if np.isfinite(bp):
                L.append(f"\n> per-patch DC 基线 = {bp:.2f} dB")
            L.append("")

    L.append("\n## D. 墙钟\n")
    L.append("| 分辨率 | A 相 | B 相 | 总优化步 | 秒 | 分钟 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    tot = 0.0
    for s in SIZES:
        if s not in new:
            continue
        r = new[s]
        tot += float(r.get("train_secs", 0))
        L.append(f"| {s}² | {r.get('warm_steps', 0)} | {r.get('train_steps', 0)} | "
                 f"{r.get('warm_steps', 0) + r.get('train_steps', 0)} | {r.get('train_secs', 0):.0f} | "
                 f"{r.get('train_secs', 0) / 60:.1f} |")
    L.append(f"| **合计** | | | | **{tot:.0f}** | **{tot / 60:.1f}** |")

    txt = "\n".join(L) + "\n"
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"WROTE {args.report}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
