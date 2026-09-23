#!/usr/bin/env python3
"""compare_fixw4.py — 采样计划对照：平方块（步值 k²，|T|=√N）vs 固定 4（步值自然数，|T|=N/4）

两臂除**采样计划**外逐项相同（K=N / z_mode=patch / BPTT / A 相 3000 / B 相 7000 /
lr / head_zero_init / warmup / seed；336² 为 bs8×accum2，其余 bs16×1）。

A 相（单步 + 全读，`decoder.steps=[N]`）**不受计划影响** ⇒ 两臂的 A 相曲线应当逐位相等，
本脚本把它当一致性校验打出来。

用法:
    python tools/compare_fixw4.py \
        --square doc/2026-09-23/rerun_A3000_10k/runs \
        --fixed  doc/2026-09-23/rerun_fixw4_10k/runs \
        --baseline doc/2026-09-23/rerun_A3000_10k/data/per_patch_mean_baseline.json \
        --report out.md --curves
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(REPO, "doc/2026-09-22/res_sweep/sweep_res_summary.json")
SIZES = [28, 56, 112, 224, 336]

try:                                    # 旧产物无 win/cum_read/bpp_actual 字段时兜底
    import sys
    sys.path.insert(0, REPO)
    from model_v2 import step_windows as _step_windows
except Exception:                       # noqa: BLE001 — torch 缺失时退化为不做兜底
    _step_windows = None


def enrich(r, size):
    """补齐 win / cum_read / bpp_actual（旧 result.json 没有这三项）。"""
    if "cum_read" in r and "win" in r and "bpp_actual" in r:
        return r
    if _step_windows is None:
        return r
    K = int(r.get("num_specials", r["num_patches"]))
    plan = r.get("step_plan", "square")
    block = int(r.get("block", 0))
    w = _step_windows(K, r["steps"], plan, block)
    r.setdefault("win", [hi - lo + 1 for lo, hi in w])
    r.setdefault("cum_read", [hi + 1 for _, hi in w])
    r.setdefault("bpp_actual",
                 [c * 384 / (size * size) for c in r["cum_read"]])
    return r


def load_runs(d, pattern):
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, pattern, "result.json"))):
        r = json.load(open(fp, encoding="utf-8"))
        out[int(r["size"])] = enrich(r, int(r["size"]))
    return out


def load_json(p):
    return json.load(open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def interp(xs, ys, x):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(x, xs, ys))


def st(r):
    l1 = np.asarray(r["l1_255"], float)
    ps = np.asarray(r["psnr"], float)
    i = int(np.argmin(l1))          # L1 最好步
    j = int(np.argmax(ps))          # PSNR 最好步（两者可能不同步：PSNR 来自聚合 MSE）
    return {"l1_first": float(l1[0]), "l1_best": float(l1[i]), "i_best": i,
            "psnr_best": float(ps[j]), "i_best_psnr": j,
            "rel": float((l1[0] - l1[i]) / l1[0] * 100.0),
            "strict": int(np.sum(np.diff(l1) < 0)), "n": len(l1),
            "l1": l1, "psnr": ps,
            "bpp_actual": r.get("bpp_actual", r.get("bpp", []))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--square", required=True, help="平方块臂目录（*_A3000-patch/）")
    ap.add_argument("--fixed", required=True, help="固定宽度臂目录（*_A3000-patch-fixw4/）")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--square_pat", default="*_A3000-patch")
    ap.add_argument("--fixed_pat", default="*_A3000-patch-fixw4")
    ap.add_argument("--report", default="")
    ap.add_argument("--curves", action="store_true")
    args = ap.parse_args()

    sq = load_runs(args.square, args.square_pat)
    fx = load_runs(args.fixed, args.fixed_pat)
    base = load_json(args.baseline) or {}
    summ = load_json(SUMMARY) or []
    summ = {int(e["size"]): e for e in summ} if isinstance(summ, list) else {}

    def bp(s):
        return float(base[str(s)]["psnr"]) if str(s) in base else float("nan")

    L = []
    L.append("# 采样计划对照：平方块（步值 k²，|T|=√N） vs 固定 4（步值自然数，|T|=N/4）\n")
    L.append("除**采样计划**外逐项相同：K=N / z_mode=patch / BPTT / A 相 3000 + B 相 7000 / "
             "lr 1.5e-4 / head_zero_init / warmup 100 / seed 42；336² = bs8×accum2，其余 bs16×1。\n")
    L.append("> 计划几何（`model_v2.step_windows`）：\n"
             "> · `square`：步值 1,4,9,…=k²；第 k 步读 z_s[k²:(k+1)²−1]（宽 2k+1，末步退化为 1）\n"
             "> · `fixed`：步值 1,2,3,…=k；第 k 步读固定 4 个 z_s（首步 5 个含 z_cls，无退化）\n")

    L.append("## A. 主对照表\n")
    L.append("| 分辨率 | N | \\|T\\| sq→fix | DC 基线 | **平方块 best PSNR** | **固定4 best PSNR** | **Δ(fix−sq)** | 平方块 best L1 | 固定4 best L1 | 首→最好 sq / fix | 严格降 sq / fix | fix vs 基线 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in SIZES:
        b = bp(s); bs = f"{b:.2f}" if np.isfinite(b) else "—"
        if s not in sq or s not in fx:
            L.append(f"| {s}² | — | — | {bs} | "
                     + (f"{st(sq[s])['psnr_best']:.2f}" if s in sq else "*未跑*") + " | "
                     + (f"{st(fx[s])['psnr_best']:.2f}" if s in fx else "*未跑*")
                     + " | — | — | — | — | — | — |")
            continue
        a, c = st(sq[s]), st(fx[s])
        d = c["psnr_best"] - a["psnr_best"]
        db = f"{c['psnr_best'] - b:+.2f}" if np.isfinite(b) else "—"
        L.append(f"| {s}² | {sq[s]['num_patches']} | {a['n']} → **{c['n']}** | {bs} | "
                 f"{a['psnr_best']:.2f} | **{c['psnr_best']:.2f}** | **{d:+.2f} dB** | "
                 f"{a['l1_best']:.2f} | {c['l1_best']:.2f} | "
                 f"{a['rel']:.1f}% / {c['rel']:.1f}% | "
                 f"{a['strict']}/{a['n'] - 1} / {c['strict']}/{c['n'] - 1} | {db} |")

    L.append("\n## B. 一致性校验：A 相（单步+全读，不受计划影响，两臂应逐位相等）\n")
    L.append("| 分辨率 | **平方块 A 相逃逸 PSNR** | **固定4 A 相逃逸 PSNR** | 最大 |Δ| |")
    L.append("|---|---|---|---|")
    for s in SIZES:
        if s not in sq or s not in fx:
            continue
        ha = {e["it"]: e["psnr"][0] for e in sq[s].get("hist", []) if e["phase"] == "A"}
        hc = {e["it"]: e["psnr"][0] for e in fx[s].get("hist", []) if e["phase"] == "A"}
        ks = sorted(set(ha) & set(hc))
        if not ks:
            continue
        m = max(abs(ha[k] - hc[k]) for k in ks)
        fa = " / ".join(f"{ha[k]:.2f}" for k in ks)
        fc = " / ".join(f"{hc[k]:.2f}" for k in ks)
        L.append(f"| {s}² @ {ks} | {fa} | {fc} | **{m:.3f} dB** |")

    L.append("\n## C. 逐 step 曲线（固定 4 臂；`F_hat` = 末步）\n")
    L.append("> `bpp_实际` = 累计读入列数 × 384 / S²（含 z_cls）；`bpp_名义` = (t+1)·384/S²。\n")
    for s in SIZES:
        if s not in fx:
            continue
        r = fx[s]
        L.append(f"### {s}²（N={r['num_patches']}，\\|T\\|={len(r['steps'])}）\n")
        L.append("```")
        L.append("t     : " + " ".join(f"{t:>6}" for t in r["steps"]))
        L.append("win   : " + " ".join(f"{v:>6}" for v in r["win"]))
        L.append("cum   : " + " ".join(f"{v:>6}" for v in r["cum_read"]))
        L.append("L1    : " + " ".join(f"{v:>6.2f}" for v in r["l1_255"]))
        L.append("PSNR  : " + " ".join(f"{v:>6.2f}" for v in r["psnr"]))
        L.append("bpp实际: " + " ".join(f"{v:>6.3f}" for v in r["bpp_actual"]))
        L.append("```")
        c = st(r)
        j = c["i_best_psnr"]
        jl = summ.get(s, {}).get("jpeg", [])
        if jl:
            jb = interp([j2["bpp"] for j2 in jl], [j2["psnr"] for j2 in jl],
                        c["bpp_actual"][j])
            L.append(f"- best PSNR t={r['steps'][j]}，**实际** bpp={c['bpp_actual'][j]:.3f}"
                     f" ⇒ 同 bpp JPEG {jb:.2f} dB，**ΔPSNR vs JPEG {c['psnr_best'] - jb:+.2f} dB**")
        L.append(f"- best L1 t={r['steps'][c['i_best']]}（L1={c['l1_best']:.2f}，"
                 f"该步 PSNR={c['psnr'][c['i_best']]:.2f}）")
        if np.isfinite(bp(s)):
            L.append(f"- vs per-patch DC 基线 {bp(s):.2f} dB：**{c['psnr_best'] - bp(s):+.2f} dB**")
        L.append("")

    L.append("\n## D. 墙钟\n")
    L.append("| 分辨率 | 平方块 秒 | 固定4 秒 | 倍数 |")
    L.append("|---|---:|---:|---:|")
    for s in SIZES:
        if s not in sq or s not in fx:
            continue
        a, c = float(sq[s].get("train_secs", 0)), float(fx[s].get("train_secs", 0))
        L.append(f"| {s}² | {a:.0f} | {c:.0f} | {c / a:.2f}× |")

    txt = "\n".join(L) + "\n"
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"WROTE {args.report}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
