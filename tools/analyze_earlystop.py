#!/usr/bin/env python3
"""analyze_earlystop.py — 逐 step ΔL1 早停 / 嵌套码 RD 曲线（零训练，只读现有 eval json）

对应 doc/2026-09-22/PLAN_paper_experiments.md 的 E4a / E4b（批 0）。

用法:
    python tools/analyze_earlystop.py doc/2026-09-16/data/bptt24_bptt_infer_test.json
    python tools/analyze_earlystop.py <glob> [--beta 1.0] [--d 1024] [--px 112896]

输入: infer_v2_test.py 产出的聚合 json（需含 decoder_steps / step_pixel_l1_255）。
输出:
  1) 停步止损判据 —— L1(t) 单调前缀有多长、在哪里开始过冲（- 值 = 变差）
  2) ΔL1 与相对 ΔL1 曲线（相对阈值 ε_rel 的真实取值区间）
  3) 扫 ε_rel → (t*, token 数, bpp, L1) 表  = 自适应早停的 RD 曲线
  4) 固定 t 的 RD 曲线（同一模型，所有图都停 t）—— E4b 的必备对照

⚠️ 诚实边界（会打印在末尾）:
  - 本脚本读的是**聚合均值**，不是逐图曲线 ⇒ 算不出「自适应 vs 固定 t」的真实增益
    （只有逐图曲线不同才有的赚）。要真做 E4b 必须先让 infer 落**逐图逐 step** 的 L1。
  - `(t*+1)` 个 token 只是**估计 bpp**（β bit/dim，未做量化+熵编码）；真实 bpp 见 E3。
  - 指标是 letterbox 后的 448×252 画布，含灰底 127 padding（见 PLAN §5）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

D_DEFAULT = 1024          # DINOv2-large 隐层维
PX_DEFAULT = 448 * 252    # 112,896


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def bpp_of(tokens: int, d: int, px: int, beta: float) -> float:
    """bpp = token 数 × 维度 × bits/dim ÷ 像素数。"""
    return tokens * d * beta / px


def rel_deltas(l1: list[float]) -> list[float]:
    """rel[i] = (L1(t_i) - L1(t_{i+1})) / L1(t_i)；正 = 变好，负 = 过冲。"""
    return [(l1[i] - l1[i + 1]) / l1[i] for i in range(len(l1) - 1)]


def mono_prefix_len(l1: list[float]) -> int:
    """L1 严格不升的最长前缀长度（= 前多少个 step 之后开始过冲）。"""
    n = len(l1)
    for i in range(n - 1):
        if l1[i + 1] > l1[i] + 1e-12:
            return i + 1          # t_i 是最后一个"仍在变好"的 step
    return n


def sweep_eps(T: list[int], rel: list[float], eps_grid: list[float],
              d: int, px: int, beta: float) -> list[dict]:
    """t* = 第一个 rel < eps 的 step；所有图都停在那里。"""
    rows = []
    for eps in eps_grid:
        t_star, idx = T[-1], len(T) - 1
        for i, r in enumerate(rel):
            if r < eps:
                t_star, idx = T[i], i
                break
        tokens = t_star + 1      # t* 个 z_s + 1 个 z_cls
        rows.append({"eps_rel": eps, "t*": t_star, "tokens": tokens,
                     "bpp": bpp_of(tokens, d, px, beta), "step_idx": idx})
    return rows


def report(tag: str, d: dict, args) -> None:
    T = d["decoder_steps"]
    l1 = d["step_pixel_l1_255"]
    px = args.px
    print("=" * 78)
    print(f"[{tag}]  n={d.get('n')}  patches={d.get('num_patches')}  "
          f"specials={d.get('num_specials')}  input={d.get('input')}")
    print(f"  full_pixel_l1_255 = {d['full_pixel_l1_255']:.4f} ± "
          f"{d.get('full_pixel_std_255', float('nan')):.3f}   "
          f"full_norm_l1 = {d['full_norm_l1']:.6f}")

    rel = rel_deltas(l1)
    mpl = mono_prefix_len(l1)
    i_best = min(range(len(l1)), key=lambda i: l1[i])

    # ---- 1) 止损判据 ----
    print("\n  (1) 停步止损判据")
    print(f"      L1(t) 单调下降的前缀长度 = {mpl} / {len(T)} 步"
          f"  ⇒ 从 t={T[mpl] if mpl < len(T) else T[-1]} 起开始过冲")
    print(f"      全局最优点: t={T[i_best]}  L1={l1[i_best]:.4f}"
          f"   |  跑满 t={T[-1]}  L1={l1[-1]:.4f}"
          f"  (差 {l1[-1] - l1[i_best]:+.4f} px)")
    if mpl < len(T):
        tail = [f"t={T[i]}:{l1[i]:.3f}" for i in range(mpl, len(T))]
        print(f"      ⚠️ 过冲段（再跑变差）: {'  '.join(tail)}")
    verdict = ("**单调性成立**（可做早停）" if mpl >= 6
               else "**单调性弱/不成立**（早停前需先改训练：随机截断前缀 / 嵌套 dropout）")
    print(f"      结论: {verdict}")

    # ---- 2) ΔL1 与相对 ΔL1 ----
    print("\n  (2) 逐步增量")
    hdr = "      " + "".join(f"{t:>9}" for t in T)
    print(hdr)
    print("      L1 " + "".join(f"{v:>9.3f}" for v in l1))
    print("      dL1" + "".join(f"{v:>9.3f}" for v in rel_deltas_raw(l1)) + "     ")
    print("      rel%" + "".join(f"{v * 100:>9.3f}" for v in rel) + "     ")
    print("      tok" + "".join(f"{t + 1:>9}" for t in T))
    print("      bpp" + "".join(f"{bpp_of(t + 1, args.d, px, args.beta):>9.3f}" for t in T))

    # ---- 3) 固定 t 的 RD 曲线（E4a）----
    print("\n  (3) E4a 固定 t 的嵌套码 RD 曲线（所有图都停 t）")
    print("      " + f"{'t':>6}{'tokens':>8}{'bpp':>8}{'L1':>10}")
    for t, v in zip(T, l1):
        star = "  ← 最优" if t == T[i_best] else ""
        print(f"      {t:>6}{t + 1:>8}{bpp_of(t + 1, args.d, px, args.beta):>8.3f}{v:>10.4f}{star}")

    # ---- 4) 扫 ε_rel（E4b）----
    eps_grid = [0.20, 0.10, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007,
                0.005, 0.003, 0.002, 0.001, 0.0005, 0.0]
    rows = sweep_eps(T, rel, eps_grid, args.d, px, args.beta)
    print("\n  (4) E4b 扫 ε_rel → 自适应早停 RD 曲线（本表用均值曲线，见末尾边界）")
    print("      " + f"{'eps_rel':>9}{'t*':>7}{'tokens':>8}{'bpp':>8}{'L1':>10}{'vs 跑满576':>12}")
    for r in rows:
        v = l1[r["step_idx"]]
        gain = l1[-1] - v
        print(f"      {r['eps_rel']:>9.4f}{r['t*']:>7}{r['tokens']:>8}"
              f"{r['bpp']:>8.3f}{v:>10.4f}{gain:>+12.4f}")
    print("      （'vs 跑满576' > 0 = 早停不仅省码率，质量还更好）")

    # ---- 5) 相对增量的噪声底 ----
    head = rel[:max(1, mpl - 1)]
    tail = rel[max(0, mpl - 1):]
    if head and tail:
        print("\n  (5) 阈值可分性（决定 ε_rel 选得稳不稳）")
        print(f"      单调段 rel 范围 = [{min(head) * 100:.3f}%, {max(head) * 100:.3f}%]")
        print(f"      过冲段 rel 范围 = [{min(tail) * 100:.3f}%, {max(tail) * 100:.3f}%]")
        noisy = any(rel[i + 1] > rel[i] for i in range(len(rel) - 1))
        print(f"      相对增量是否单调递减: {not noisy}"
              + ("" if not noisy else "  ⇒ 存在反弹，单一 ε 会踩在噪声上（考虑「连续2步」滞回）"))


def rel_deltas_raw(l1: list[float]) -> list[float]:
    return [l1[i] - l1[i + 1] for i in range(len(l1) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="聚合 infer json（支持 glob）")
    ap.add_argument("--d", type=int, default=D_DEFAULT, help="隐层维 (默认 1024)")
    ap.add_argument("--px", type=int, default=PX_DEFAULT, help="bpp 分母像素数 (默认 448*252)")
    ap.add_argument("--beta", type=float, default=1.0, help="bits/dim (1.0 或 0.5)")
    args = ap.parse_args()

    files: list[str] = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)) or [p])
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("no input json", file=sys.stderr)
        return 1

    for f in files:
        try:
            report(os.path.basename(f), load(f), args)
        except KeyError as e:
            print(f"[skip] {f}: 缺字段 {e}", file=sys.stderr)

    print("\n" + "=" * 78)
    print("诚实边界")
    print("=" * 78)
    print(" 1. 本脚本读**聚合均值**曲线 ⇒ 只能给「所有图停同一 t」的 RD 曲线（E4a），")
    print("    **算不出**「逐图自适应 vs 固定 t」的真实增益（E4b）。")
    print("    自适应只有在逐图曲线**互相交叉**时才赚；若各图 ΔL1 形状相似，")
    print("    ε 阈值只是把固定 t 重新参数化一遍，增益≈0。")
    print("    ⇒ 要真做 E4b：先让 infer_v2_test.py 落**逐图逐 step** L1，再重跑推理（零训练）。")
    print(" 2. `(t*+1)` 个 token 只是**估计 bpp**（β bit/dim，无量化+熵编码）⇒ 真实 bpp 见 PLAN §4 E3。")
    print(" 3. 指标算在 letterbox 后的 448×252 画布上，**含灰底 127 padding**；")
    print("    低 t 端点可能被「把灰底填对」撑高 ⇒ 必须补平凡基线与内容区 mask（PLAN §5）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
