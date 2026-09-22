#!/usr/bin/env python3
"""analyze_rd_earlystop.py — E4a/E4b：嵌套码 RD 曲线 + 逐图自适应早停（零训练）

对应 doc/2026-09-22/PLAN_paper_experiments.md 的批 0：
  · E4a 固定 t 的嵌套码 RD 曲线（一个 checkpoint = 一条曲线）
  · E4b 逐 step ΔL1 早停：扫 ε_rel → 自适应 RD 曲线，并与固定 t 对照
  · padding 平凡基线 / 内容区口径（PLAN §5）

与本目录 `analyze_earlystop.py` 的分工:
  · analyze_earlystop.py —— 老工具, 只吃**聚合均值**(step_pixel_l1_255),
    给"止损判据 + mpl + 聚合版 ε 扫描"。缺 per_image 时仍然有效。
  · 本脚本 —— 吃 infer_v2_test.py **schema_version>=2** 的产物（含 step_psnr /
    step_ms_ssim / per_image），做真正的逐图自适应与 RD 对照。缺 per_image 时
    自动降级为"退化模式"（只给固定 t 曲线 + 明确警告）。

用法:
    python tools/analyze_rd_earlystop.py doc/.../bptt_construction_site_test.json
    python tools/analyze_rd_earlystop.py '<glob>' --beta 1.0 --hysteresis 2 --min-idx 1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

D_DEFAULT = 1024
PX_DEFAULT = 448 * 252


# ─────────────────────────── 基础工具 ───────────────────────────
def bpp_of(tokens: int, d: int, px: int, beta: float) -> float:
    return tokens * d * beta / px


def psnr_from_mse(mse, maxv: float = 255.0):
    mse = np.maximum(np.asarray(mse, dtype=np.float64), 1e-12)
    return 10.0 * np.log10(maxv ** 2 / mse)


def rel_deltas(l1) -> np.ndarray:
    """rel[k] = (L1(t_k) - L1(t_{k+1})) / L1(t_k)；>0 = 再跑一步变好。"""
    l1 = np.asarray(l1, dtype=np.float64)
    return (l1[:-1] - l1[1:]) / l1[:-1]


def mono_prefix_len(l1) -> int:
    l1 = np.asarray(l1, dtype=np.float64)
    for i in range(len(l1) - 1):
        if l1[i + 1] > l1[i] + 1e-12:
            return i + 1
    return len(l1)


# ─────────────────────────── E4b 核心 ───────────────────────────
def stop_index(rel_row: np.ndarray, eps: float, hyst: int, min_idx: int) -> int:
    """第一处「连续 hyst 步相对增量都 < eps」的索引；否则最后一步。

    hyst=1 即 PLAN §11.2 的原始判据；hyst=2 = 滞回（PLAN 批 0 结论四升级为必做，
    因为尾部 rel 在噪声带里反弹）。
    """
    n = len(rel_row) + 1                       # 步数 M
    for k in range(min_idx, n):
        if k >= n - 1:
            return n - 1
        if all(k + j < len(rel_row) and rel_row[k + j] < eps for j in range(hyst)):
            return k
    return n - 1


def adaptive_sweep(per_image, T, d, px, beta, eps_grid, hyst, min_idx):
    """扫 ε_rel ⇒ 自适应 RD 曲线（逐图各停各的）。"""
    curves_l1 = np.array([p["step_l1"] for p in per_image], np.float64)      # (N,M)
    curves_mse = np.array([p["step_mse"] for p in per_image], np.float64)
    rel = (curves_l1[:, :-1] - curves_l1[:, 1:]) / curves_l1[:, :-1]         # (N,M-1)
    N, M = curves_l1.shape
    tokens = np.array([t + 1 for t in T], np.float64)

    rows = []
    idx_cache = {}
    for eps in eps_grid:
        ks = np.array([stop_index(rel[i], eps, hyst, min_idx) for i in range(N)])
        idx_cache[eps] = ks
        mse_at = curves_mse[np.arange(N), ks]
        rows.append({
            "eps_rel": float(eps),
            "mean_bpp": float((tokens[ks] * d * beta / px).mean()),
            "mean_tokens": float(tokens[ks].mean()),
            "psnr_agg": float(psnr_from_mse(mse_at.mean())),
            "l1_agg": float(curves_l1[np.arange(N), ks].mean()),
            "mean_stop_t": float(np.array(T, np.float64)[ks].mean()),
            "n_at_last": int((ks == M - 1).sum()),
            "n_at_first": int((ks == min_idx).sum()),
        })
    return rows, idx_cache, curves_l1, curves_mse, tokens


def oracle_hull(curves_mse: np.ndarray, bpps: np.ndarray, n_lam: int = 120):
    """任意「逐图选 t」方案的性能天花板（拉格朗日最优码率分配）。

    min Σ_i mse_i(k_i)  s.t. Σ_i bpp(k_i) ≤ N·B  ⇒ 扫 λ 取 k_i=argmin_k(mse_i(k)+λ·bpp_k)。
    若这条 hull 与固定 t 曲线重合 ⇒ **不存在**任何逐图自适应方案能超过固定 t，
    即 C2b 不是"判据没调好"，而是结构上不可能（逐图曲线不交叉）。
    """
    N, M = curves_mse.shape
    lam = np.logspace(-2, 5, n_lam) * (curves_mse.mean() / max(bpps[0], 1e-9))
    pts = []
    for l in lam:
        k = (curves_mse + l * bpps[None, :]).argmin(axis=1)
        pts.append((float(bpps[k].mean()),
                    float(psnr_from_mse(curves_mse[np.arange(N), k].mean()))))
    pts.sort()
    out: list = []
    for b, p in pts:
        while out and p <= out[-1][1] + 1e-12:
            out.pop()
        out.append((b, p))
    return out


def interp_at(bpp_axis: np.ndarray, val_axis: np.ndarray, x: float) -> float:
    """在固定 t 曲线上插值（bpp 单调增）。"""
    order = np.argsort(bpp_axis)
    b, v = np.asarray(bpp_axis)[order], np.asarray(val_axis)[order]
    if x <= b[0]:
        return float(v[0])
    if x >= b[-1]:
        return float(v[-1])
    return float(np.interp(x, b, v))


def make_plot(path, tag, T, bpps, fixed_psnr, rows, cl1, hist_best, oracle, hull=None):
    """论文三张图: (a) RD 曲线 fixed vs adaptive  (b) 逐图曲线是否平移  (c) 停步分布。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_pi = cl1 is not None
    ncol = 3 if has_pi else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.2))
    if ncol == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(bpps, fixed_psnr, "-o", ms=4, color="tab:blue", label="Fixed t (E4a)")
    if rows:
        ax.plot([r["mean_bpp"] for r in rows], [r["psnr_agg"] for r in rows],
                "--s", ms=4, color="tab:red", label="Adaptive early-stop (E4b)")
    if hull:
        ax.plot([b for b, _ in hull], [p for _, p in hull], "-",
                color="tab:green", lw=1.6,
                label="Oracle hull (any per-image alloc)")
    if oracle:
        ax.plot([oracle["mean_bpp"]], [oracle["psnr"]], "*", ms=14,
                color="tab:green", label="Per-image best-t")
    ax.set_xscale("log")
    ax.set_xlabel("bpp (beta=1, estimated)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"Rate-Distortion - {tag}", fontsize=9)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)

    if has_pi:
        ax = axes[1]
        norm = cl1 / np.maximum(cl1[:, :1], 1e-9)
        step = max(1, len(norm) // 200)
        for i in range(0, len(norm), step):
            ax.plot(T, norm[i], color="tab:gray", alpha=.15, lw=.6)
        ax.plot(T, norm.mean(0), color="tab:red", lw=2, label="mean")
        ax.set_xscale("log")
        ax.set_xlabel("stop step t")
        ax.set_ylabel("L1(t) / L1(1)")
        ax.set_title("Per-image curves (parallel shift => no gain)", fontsize=9)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)

        ax = axes[2]
        ax.bar(range(len(T)), hist_best, color="tab:blue")
        ax.set_xticks(range(len(T)))
        ax.set_xticklabels([str(t) for t in T], rotation=90, fontsize=6)
        ax.set_xlabel("stop step t*")
        ax.set_ylabel("#images")
        ax.set_title("Stop distribution p(t*)", fontsize=9)
        ax.grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"  [plot] {path}")


# ─────────────────────────── 报告 ───────────────────────────
def report(tag: str, d: dict, args) -> None:
    T = list(d["decoder_steps"])
    M = len(T)
    # bpp 分母/β 优先取 json 自带的（避免 448×252 的默认值污染 224×126 的结果）
    px = int(d.get("bpp_px") or args.px)
    beta = float(d.get("bpp_beta") or args.beta)
    if px != args.px:
        print(f"[note] bpp 口径取 json: px={px}, beta={beta}（CLI 传的是 px={args.px}）")
    l1 = np.array(d.get("step_pixel_l1_255") or d.get("step_l1"), np.float64)
    mse = d.get("step_mse_255")
    psnr = d.get("step_psnr")
    if psnr is None and mse is not None:
        psnr = psnr_from_mse(mse)
    ssim = d.get("step_ms_ssim")
    l1c = d.get("step_l1_content_255")
    psnrc = d.get("step_psnr_content")
    per_image = d.get("per_image")
    tokens = np.array([t + 1 for t in T], np.float64)
    bpps = tokens * args.d * beta / px

    print("=" * 92)
    print(f"[{tag}]  n={d.get('n')}  K={d.get('num_specials')}  input={d.get('input')}  "
          f"steps={M}  schema={d.get('schema_version', 1)}  per_image="
          f"{len(per_image) if per_image else 0}")
    if d.get("content_frac_mean") is not None:
        cf = d["content_frac_mean"]
        print(f"  内容区占比 = {cf * 100:.2f}%  (padding {(1 - cf) * 100:.2f}%)  |  "
              f"平凡基线(全黑画布): 全画布 PSNR={d.get('trivial_black_psnr', float('nan')):.2f} dB, "
              f"内容区 PSNR={d.get('trivial_black_psnr_content', float('nan')):.2f} dB")

    mpl = mono_prefix_len(l1)
    i_best = int(np.argmin(l1))
    print(f"\n  (0) 止损判据: L1 单调下降前缀 = {mpl}/{M} 步 "
          f"(⇒ 从 t={T[mpl] if mpl < M else T[-1]} 起过冲) | "
          f"最优 t={T[i_best]} | 跑满 t={T[-1]} 差 {l1[-1] - l1[i_best]:+.3f} px")

    # ── E4a: 固定 t 曲线 ──
    print(f"\n  (1) E4a 固定 t 的嵌套码 RD 曲线（所有图停同一 t）")
    hdr = f"      {'t':>5}{'tokens':>8}{'bpp':>8}{'L1':>9}{'PSNR':>8}"
    if ssim:
        hdr += f"{'MS-SSIM':>9}"
    if l1c:
        hdr += f"{'L1内容':>9}{'PSNR内容':>10}"
    print(hdr)
    for i, t in enumerate(T):
        line = f"      {t:>5}{t + 1:>8}{bpps[i]:>8.3f}{l1[i]:>9.3f}"
        line += f"{(psnr[i] if psnr else float('nan')):>8.2f}"
        if ssim:
            line += f"{ssim[i]:>9.4f}"
        if l1c:
            line += f"{l1c[i]:>9.3f}{(psnrc[i] if psnrc else float('nan')):>10.2f}"
        if i == i_best:
            line += "   ← 最优"
        print(line)
    print(f"      有用码率区间 = {bpps[0]:.3f} – {bpps[i_best]:.3f} bpp（最优 t 之前）")

    # ── E4b ──
    if not per_image:
        print("\n  (2) E4b ⚠️ 无 per_image ⇒ **无法**做逐图自适应对照。"
              "先让 infer_v2_test.py 落逐图逐 step（schema_version=2）再重跑。")
        print("      诚实边界: 只看聚合均值时, ε 只是把固定 t 重新参数化, 增益恒为 0。")
        if args.plot:
            make_plot(args.plot, tag, T, bpps, fixed_psnr, [], None, None, None)
        return

    eps_grid = args.eps_grid
    rows, idx_cache, cl1, cmse, tok = adaptive_sweep(
        per_image, T, args.d, px, beta, eps_grid, args.hysteresis, args.min_idx)
    fixed_psnr = psnr_from_mse(cmse.mean(axis=0))
    fixed_l1 = cl1.mean(axis=0)

    print(f"\n  (2) E4b 逐图自适应早停（hysteresis={args.hysteresis} 步, min_idx={args.min_idx}）")
    print(f"      {'eps_rel':>9}{'平均t*':>8}{'平均tok':>8}{'平均bpp':>9}"
          f"{'PSNR(agg)':>10}{'L1(agg)':>9}{'同bpp固定t':>11}{'ΔPSNR':>8}"
          f"{'停到末步':>9}{'停到首步':>9}")
    for r in rows:
        fx = interp_at(bpps, fixed_psnr, r["mean_bpp"])
        print(f"      {r['eps_rel']:>9.4f}{r['mean_stop_t']:>8.1f}{r['mean_tokens']:>8.1f}"
              f"{r['mean_bpp']:>9.3f}{r['psnr_agg']:>10.2f}{r['l1_agg']:>9.3f}"
              f"{fx:>11.2f}{r['psnr_agg'] - fx:>+8.2f}"
              f"{r['n_at_last']:>9}{r['n_at_first']:>9}")
    print("      ΔPSNR > 0 = 自适应在同一平均 bpp 下优于固定 t（C2b 成立的必要条件）")

    # ── 停步分布（取"增益最大"的 ε）──
    best = max(rows, key=lambda r: r["psnr_agg"] - interp_at(bpps, fixed_psnr, r["mean_bpp"]))
    ks = idx_cache[best["eps_rel"]]
    hist = np.bincount(ks, minlength=M)
    print(f"\n  (3) 停步分布 p(t*) @ ε_rel={best['eps_rel']:.4f}（该点 Δ 最大 "
          f"{best['psnr_agg'] - interp_at(bpps, fixed_psnr, best['mean_bpp']):+.2f} dB）")
    order = np.argsort(-hist)
    top = [(int(order[j]), int(hist[order[j]])) for j in range(min(8, M)) if hist[order[j]]]
    print("      " + "  ".join(f"t={T[i]}:{c}({c / len(ks) * 100:.0f}%)" for i, c in top))
    print(f"      停步点覆盖 {int((hist > 0).sum())}/{M} 档 ⇒ "
          f"{'各图确实停在不同位置（自适应有内容）' if (hist > 0).sum() > 1 else '所有图停同一档（自适应无内容）'}")

    # ── Oracle gap ──
    i_or = np.argmin(cl1, axis=1)
    or_bpp = float((tok[i_or] * args.d * beta / px).mean())
    or_psnr = float(psnr_from_mse(cmse[np.arange(len(i_or)), i_or].mean()))
    print(f"\n  (4) Oracle（逐图取最优 t）: 平均 bpp={or_bpp:.3f}  PSNR={or_psnr:.2f} dB | "
          f"同 bpp 下固定 t={interp_at(bpps, fixed_psnr, or_bpp):.2f} dB "
          f"⇒ 上限空间 {or_psnr - interp_at(bpps, fixed_psnr, or_bpp):+.2f} dB")
    print(f"      判据最好点 {best['psnr_agg']:.2f} dB ⇒ 距 oracle 还差 "
          f"{or_psnr - best['psnr_agg']:+.2f} dB")

    # ── Oracle 凸包：任意逐图码率分配的天花板（判定 C2b 是否结构上可能）──
    hull = oracle_hull(cmse, bpps)
    gaps = [(p - interp_at(bpps, fixed_psnr, b), b, p) for b, p in hull]
    gmax = max(gaps)
    print(f"\n  (4b) 拉格朗日 oracle 凸包 = 任意「逐图选 t」方案的天花板")
    print(f"       最大 ΔPSNR = {gmax[0]:+.3f} dB @ bpp={gmax[1]:.3f} "
          f"(hull {gmax[2]:.2f} dB vs 固定 t {interp_at(bpps, fixed_psnr, gmax[1]):.2f} dB)")
    print("       " + ("⇒ 逐图自适应**结构上无法**优于固定 t：C2b 不成立，"
                      "不是判据没调好" if gmax[0] < 0.10 else
                      f"⇒ 天花板仅 {gmax[0]:.2f} dB"
                      "（<0.3 dB ≈ BD-rate 几个百分点）⇒ 不足以当主贡献，"
                      "只能当一节 analysis/ablation" if gmax[0] < 0.30 else
                      "⇒ 逐图自适应仍有空间，值得把判据调好"))

    # ── 逐图曲线形状（自适应能不能赚的前提）──
    print(f"\n  (5) 前提检查: 逐图曲线是否互相交叉（否则自适应≈固定 t）")
    for k in (0, len(T) // 2, len(T) - 1):
        q = np.percentile(cl1[:, k], [10, 50, 90])
        print(f"      t={T[k]:>4}: L1 的 p10/p50/p90 = {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}"
              f"  (IQR={q[2] - q[0]:.2f})")
    corr = np.corrcoef(cl1[:, 0], cl1[:, -1])[0, 1]
    print(f"      首步 vs 末步 L1 的跨图相关 = {corr:.4f} "
          f"（越高 ⇒ 曲线只是整体平移, 自适应越难赚）")
    print(f"      若 ΔPSNR 全表 ≈ 0 且停步集中于一档 ⇒ **C2b 不成立**, "
          f"叙事应收缩为「一个模型覆盖整条 RD 曲线」（C2a 仍然成立）")

    if args.plot:
        make_plot(args.plot, tag, T, bpps, fixed_psnr, rows, cl1, hist,
                  {"mean_bpp": or_bpp, "psnr": or_psnr}, hull)

    if args.save_json:
        out = {
            "tag": tag, "T": T, "bpp": bpps.tolist(),
            "fixed_psnr": fixed_psnr.tolist(), "fixed_l1": fixed_l1.tolist(),
            "adaptive": rows,
            "stop_hist_best": {"eps_rel": best["eps_rel"],
                               "hist": hist.tolist()},
            "oracle": {"mean_bpp": or_bpp, "psnr": or_psnr},
            "oracle_hull": hull,
            "oracle_hull_max_gain_db": gmax[0],
        }
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  [save] {args.save_json}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="infer json（支持 glob）")
    ap.add_argument("--d", type=int, default=D_DEFAULT, help="隐层维 (默认 1024)")
    ap.add_argument("--px", type=int, default=PX_DEFAULT, help="bpp 分母像素数 (默认 448*252)")
    ap.add_argument("--beta", type=float, default=1.0, help="bits/dim (1.0 或 0.5)")
    ap.add_argument("--hysteresis", type=int, default=2, help="连续多少步低于阈值才停 (默认 2)")
    ap.add_argument("--min-idx", type=int, default=1, help="最小允许停步索引 (默认 1, 防首步噪声)")
    ap.add_argument("--eps-grid", default=None, help="逗号分隔; 默认内置网格")
    ap.add_argument("--save-json", default=None, help="把曲线写成 json（供画图）")
    ap.add_argument("--plot", default=None, help="把三张图写到这个 png 路径")
    args = ap.parse_args()
    args.eps_grid = ([float(x) for x in args.eps_grid.split(",")] if args.eps_grid else
                     [0.20, 0.10, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005,
                      0.003, 0.002, 0.001, 0.0005, 0.0002, 0.0])

    files: list = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)) or [p])
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("no input json", file=sys.stderr)
        return 1
    for f in files:
        try:
            with open(f) as fh:
                report(os.path.basename(f), json.load(fh), args)
        except KeyError as e:
            print(f"[skip] {f}: 缺字段 {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
