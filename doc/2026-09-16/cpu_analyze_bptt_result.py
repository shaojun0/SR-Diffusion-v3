"""CPU 侧独立分析：只用本目录 data/ 里已入库的推理 json，重算 BPTT 对照的读数构成。

不训练、不加载权重；纯 CPU 数秒可跑。输出 doc/2026-09-16/ANALYSIS_v2_bptt_cpu_verify.md
里 §1.3 / §2 的全部数字，并落一份 data/bptt_readout_decomposition.json。

用法:  python cpu_analyze_bptt_result.py
"""
import json
import math
import statistics
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):          # Windows 控制台默认 GBK, 防 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load(name):
    return json.load(open(DATA / name, encoding="utf-8"))


def rel(x, y):
    return 100 * (x - y) / x


def windows(steps, K):
    out = []
    for i, t in enumerate(steps):
        k = math.isqrt(int(t))
        hi = min((k + 1) ** 2 - 1, K)
        lo = 0 if i == 0 else max(k * k, 1)
        out.append((lo, hi, hi - lo + 1))
    return out


def fanout(steps, K):
    """每个被读到的键平均会出现在多少个损失项里（BPTT）; detach 恒为 1。"""
    w = windows(steps, K)
    n = len(steps)
    total = sum(sz for _, _, sz in w)
    return sum((n - i) * sz for i, (_, _, sz) in enumerate(w)) / total, w


def main():
    A = load("bptt24_detach_infer_test.json")
    B = load("bptt24_bptt_infer_test.json")
    C = load("slice05_bptt_infer_test.json")
    a, b, c = A["step_pixel_l1_255"], B["step_pixel_l1_255"], C["step_pixel_l1_255"]
    res = {}

    print("=" * 78)
    print("§2.1 headline 与其它读出点（像素 L1 0-255）")
    print("=" * 78)
    rows = [("step 1", a[0], b[0]), ("best step", min(a), min(b)),
            ("mean over steps", statistics.mean(a), statistics.mean(b)),
            ("last step (F_hat)", a[-1], b[-1])]
    for name, x, y in rows:
        print(f"  {name:>18}: A={x:8.4f}  B={y:8.4f}  rel={rel(x, y):+6.2f}%")
    res["readouts"] = {n: {"A": x, "B": y, "rel_pct": rel(x, y)} for n, x, y in rows}

    print()
    print("=" * 78)
    print("§2.2 乘法分解：headline -30.8% = (与 carry 无关的整体提升) x (轨迹精修)")
    print("=" * 78)
    uni = rel(a[0], b[0])                                   # step1 对 step1
    traj = rel(a[0] * (1 - uni / 100), b[-1])               # 剩下的尾部贡献
    prod = 100 * (1 - (1 - uni / 100) * (1 - traj / 100))
    print(f"  与 carry 无关的部分 (step1 vs step1) : {uni:+.2f}%")
    print(f"  轨迹部分 (B 的首步->末步递降)       : {traj:+.2f}%")
    print(f"  乘积                                : {prod:+.2f}%  "
          f"(headline {rel(a[-1], b[-1]):+.2f}%)")
    res["decomposition"] = {"uniform_pct": uni, "trajectory_pct": traj,
                            "product_pct": prod, "headline_pct": rel(a[-1], b[-1])}

    print()
    print("=" * 78)
    print("§2.3 轨迹形状：detach 是定点，BPTT 是几何收敛 + 尾部回升")
    print("=" * 78)
    for tag, x in (("A detach", a), ("B BPTT", b)):
        spread = max(x) - min(x)
        print(f"  {tag}: min={min(x):.4f} max={max(x):.4f} spread={spread:.4f} "
              f"({100 * spread / min(x):.2f}% of level)  "
              f"首->最好={rel(x[0], min(x)):+.2f}%  最好->末={rel(min(x), x[-1]):+.2f}%")
    tot = b[0] - min(b)
    acc = 0.0
    print(f"\n  B 的逐步增量（正=改善；总增益 {tot:.4f} L1）：")
    print(f"    {'step':>4} {'dL1':>9} {'ratio':>7} {'cum share':>10}")
    prev = None
    for i in range(1, len(b)):
        d = b[i - 1] - b[i]
        acc += max(0.0, d)
        r = "-" if (prev is None or prev <= 0) else f"{d / prev:.2f}"
        print(f"    {i + 1:>4} {d:>+9.4f} {r:>7} {100 * acc / tot:>9.1f}%")
        if d > 0:
            prev = d
    shares = {k: 100 * sum(max(0.0, b[i - 1] - b[i]) for i in range(1, k + 1)) / tot
              for k in (1, 2, 3, 4, 6, 8, 12, 16)}
    print("\n  前 k 步累计拿到多少总增益:",
          ", ".join(f"k={k}: {v:.1f}%" for k, v in shares.items()))
    res["trajectory"] = {"A_spread_pct": 100 * (max(a) - min(a)) / min(a),
                         "B_spread_pct": 100 * (max(b) - min(b)) / min(b),
                         "B_first_to_best_pct": rel(b[0], min(b)),
                         "B_best_to_last_pct": rel(min(b), b[-1]),
                         "A_best_to_last_pct": rel(min(a), a[-1]),
                         "cum_share_by_step": shares}

    print()
    print("=" * 78)
    print("§2.4 每步读窗口 / 累计读入量 / 梯度扇出（解释操作点依赖）")
    print("=" * 78)
    for label, steps, K in (("24 步全轨迹 A/B", A["decoder_steps"], A["num_specials"]),
                            ("slice[0:5] C", C["decoder_steps"], C["num_specials"])):
        fan, w = fanout(steps, K)
        total = sum(sz for _, _, sz in w)
        print(f"  {label}: K={K}, {len(steps)} 步, 总读入 {total} 个 token, "
              f"梯度扇出 1(detach) -> {fan:.2f}(BPTT) = x{fan:.1f}")
        res.setdefault("fanout", {})[label] = {"K": K, "steps": len(steps),
                                               "tokens_read": total, "fanout": fan}
    print()
    print(f"  {'step':>4} {'t':>4} {'window':>12} {'size':>5} {'cum%':>7} "
          f"{'A detach':>9} {'B BPTT':>8}")
    cum = 0
    w = windows(A["decoder_steps"], A["num_specials"])
    total = sum(sz for _, _, sz in w)
    for i, ((lo, hi, sz), t) in enumerate(zip(w, A["decoder_steps"])):
        cum += sz
        print(f"  {i + 1:>4} {t:>4} {str((lo, hi)):>12} {sz:>5} {100 * cum / total:>6.1f}% "
              f"{a[i]:>9.3f} {b[i]:>8.3f}")
    print("  注: 末步 t=576 的窗口退化为 1 个 token（K=N 时 min((k+1)^2-1,K) 被截断），")
    print("      而 B 的末步仍拿到接近最优的 L1 => BPTT 学到的是对画布迭代精修，")
    print("      不是'逐步读入更多键'。")

    print()
    print("=" * 78)
    print("§2.5 训练期 eval 曲线：不是'收敛更慢'，是形态变化")
    print("=" * 78)
    ev = {"A": [(2000, 0.528403), (4000, 0.458687), (6000, 0.427281), (8000, 0.414043)],
          "B": [(2000, 0.458352), (4000, 0.398887), (6000, 0.325998), (8000, 0.287951)]}
    fits = {}
    for tag, pts in ev.items():
        d = [pts[i][1] - pts[i - 1][1] for i in range(1, len(pts))]
        ratio = [d[i] / d[i - 1] for i in range(1, len(d))]
        asym = pts[-1][1] + d[-1] * ratio[-1] / (1 - ratio[-1]) if ratio[-1] < 1 else None
        fits[tag] = {"deltas": d, "ratios": ratio, "geometric_asymptote": asym}
        print(f"  {tag}: 区间改善 {[round(v, 4) for v in d]}  比值 "
              f"{[round(v, 3) for v in ratio]}"
              + (f"  几何外推极限 ~{asym:.4f}" if asym else "  (比值>1: 非几何)"))
    print(f"  A 的极限 ~{fits['A']['geometric_asymptote']:.3f} 仍比 B 的 0.288 差约 "
          f"{100 * (fits['A']['geometric_asymptote'] - 0.287951) / fits['A']['geometric_asymptote']:.0f}%"
          f" => 不能用'A 只是训得慢'解释")
    res["eval_curve"] = fits

    with open(DATA / "bptt_readout_decomposition.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n-> 写入 {DATA / 'bptt_readout_decomposition.json'}")


if __name__ == "__main__":
    main()
