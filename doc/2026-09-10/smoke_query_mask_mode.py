"""query_mask_mode smoke test — 形状 / 隔离性 / 梯度 / 向后兼容。

对应模型改动: `OutputQueryDecoder(query_mask_mode="causal"|"blockdiag")`
（`model_v2.py` 的 `build_causal_query_mask` / `QUERY_MASK_MODES`）。

被验证的三条主张:
  A. **形状与向后兼容**: blockdiag 不改任何权重形状 → 与 causal 的
     state_dict 双向可载; 默认值必须与历史块下三角**逐位相同**。
  B. **隔离性**: blockdiag 下"改块 k 的 register"只影响步 k 本身;
     causal 下会因为 depth≥2 的"累积前缀"泄露影响后续所有步。
  C. **梯度解耦**: blockdiag 下步 t 的损失对**更早块**的 register 梯度
     恰好为 0（与 SRPhase1V2.decode 的 carry.detach() 合起来 =
     "每步只从自己那一步的损失收梯度"）。

不涉及塌缩: 模块 docstring 已注明本开关对 register 塌缩**中性**
（块内非种子成员梯度差在两种模式下都恰为 0）, 本脚本不做该主张。

用法（无 GPU / 无 DINO 权重, 纯 CPU, 秒级）:
    python doc/2026-09-10/smoke_query_mask_mode.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from model_v2 import (OutputQueryDecoder, build_causal_query_mask,
                      QUERY_MASK_MODES, DEFAULT_QUERY_MASK_MODE)

# 与真实训练同规格的掩码/形状口径, 但 D 缩小以便 CPU 秒级跑完
N = 32                  # 查询行数 = patch 数 (真实 576)
K = 24                  # z_s 长度 (真实 slice05 为 35)
D = 32                  # 特征维 (真实 1024)
STEPS = [1, 4, 9, 16]   # 采样步 = 每块首位置 (真实 [1,4,9,16,25])
SEED = 0


def block_of(pos):
    """序列位置 p(0=z_cls) → 块号; 与 build_block_mask 同源。"""
    return int(pos ** 0.5)


def build(mode, seed=SEED):
    """mode=None → 不传该 kwarg, 用于验证**默认值**本身。"""
    torch.manual_seed(seed)
    kw = {} if mode is None else {"query_mask_mode": mode}
    d = OutputQueryDecoder(dim=D, num_patches=N, heads=4, steps=STEPS,
                           num_specials=K, depth=2, **kw)
    return d.eval()


def out(d, z_s, z_cls):
    with torch.no_grad():
        return d(z_cls, z_s)


def main():
    ok = True
    z_cls = torch.randn(1, 1, D)
    torch.manual_seed(1)
    z_s = torch.randn(1, K, D)

    # ── A1. 默认 mode == blockdiag; 显式 causal 必须复现历史块下三角 ──
    T, Q = len(STEPS), N
    assert DEFAULT_QUERY_MASK_MODE == "blockdiag", DEFAULT_QUERY_MASK_MODE
    assert torch.equal(build_causal_query_mask(T, Q),
                       build_causal_query_mask(T, Q, mode="blockdiag")), \
        "默认 mode 必须是 'blockdiag'"
    tm_hist = build_causal_query_mask(T, Q, mode="causal")
    for ti in range(T):                                  # 历史块下三角回归
        row = tm_hist[ti * Q]
        assert (row[:(ti + 1) * Q] == 0).all(), f"causal 步 {ti} 应见步 ≤{ti}"
        assert (row[(ti + 1) * Q:] == float("-inf")).all(), \
            f"causal 步 {ti} 不应见未来步"
    print("[ok] A1 默认 == 'blockdiag'; 显式 'causal' 仍复现历史块下三角")

    # ── A2. blockdiag 掩码结构: 对角块全允许 + 其余全 -inf, 且是 causal 的真子集 ──
    tc = build_causal_query_mask(T, Q, mode="causal")
    tb = build_causal_query_mask(T, Q, mode="blockdiag")
    assert tb.shape == tc.shape == (T * Q, T * Q)
    for ti in range(T):
        row = tb[ti * Q]
        assert (row[ti * Q:(ti + 1) * Q] == 0).all(), f"块 {ti} 对角块应全允许"
        rest = torch.cat([row[:ti * Q], row[(ti + 1) * Q:]])
        assert (rest == float("-inf")).all(), f"块 {ti} 不应见其它块"
    assert int(((tb == 0) & (tc != 0)).sum()) == 0, "blockdiag 应为 causal 的子集"
    allowed_c, allowed_b = int((tc == 0).sum()), int((tb == 0).sum())
    print(f"[ok] A2 blockdiag ⊂ causal: 允许位置 {allowed_b} < {allowed_c} "
          f"（砍掉 {allowed_c - allowed_b} 个跨步位置 = "
          f"{(allowed_c - allowed_b) / allowed_c:.1%}）")

    # ── A3. 非法 mode 必须报错, 不静默退化 ──
    for bad in ("", "block_diag", "diag", None):
        try:
            build_causal_query_mask(T, Q, mode=bad)
            ok = False
            print(f"[FAIL] A3 mode={bad!r} 未报错")
        except AssertionError:
            pass
    print("[ok] A3 非法 mode 均报错")

    # ── A4. 权重形状不变 → causal/blockdiag 互载 ──
    d_def = build(None)                              # 不传 mode = 默认
    assert d_def.query_mask_mode == "blockdiag", d_def.query_mask_mode
    d_ca, d_bd = build("causal"), build("blockdiag")
    d_bd.load_state_dict(d_ca.state_dict())          # 必须不抛
    d_ca.load_state_dict(d_bd.state_dict())
    assert d_ca.query_mask_mode == "causal"
    assert d_bd.query_mask_mode == "blockdiag"
    # 两者输出**应不同**（否则说明掩码没生效）
    y_ca, y_bd = out(d_ca, z_s, z_cls), out(d_bd, z_s, z_cls)
    assert y_ca.shape == y_bd.shape == (1, len(STEPS), N, D), y_ca.shape
    diff = (y_ca - y_bd).abs().max().item()
    assert diff > 0, "同权重下两种 mode 输出必须不同（掩码未生效?）"
    print(f"[ok] A4 同权重互载 OK; 形状 {tuple(y_ca.shape)}; "
          f"两 mode 输出差异 {diff:.3e}")

    # ── B. 隔离性: 逐块改 register, 看影响哪些步 ──
    # 块 k 的位置区间 = [k², min((k+1)²−1, K)] (序列位置), z_s 下标 = 位置−1
    print("\n[B] 隔离性: 只改某一块的 register 内容 → 各步输出变化")
    print(f"    {'改哪块':<8} {'step ' + '  step '.join(map(str, STEPS))}")
    for k in range(1, len(STEPS) + 1):
        lo = k * k                                  # 块 k 首位置(种子)
        hi = min((k + 1) ** 2 - 1, K)               # 块 k 末位置
        if lo > K:
            continue
        z2 = z_s.clone()
        z2[0, lo:hi] += 5.0                         # 只改非种子成员 (下标 lo..hi-1 = 位置 lo+1..hi)
        row_ca, row_bd = [], []
        for ti in range(len(STEPS)):
            dc = (out(d_ca, z2, z_cls)[0, ti] - y_ca[0, ti]).abs().max().item()
            db = (out(d_bd, z2, z_cls)[0, ti] - y_bd[0, ti]).abs().max().item()
            row_ca.append(f"{dc:.1e}")
            row_bd.append(f"{db:.1e}")
            # blockdiag 的核心主张: 只有"自己那一步"能变, 其余恰好 0
            if ti != k - 1 and db != 0.0:
                ok = False
                print(f"[FAIL] B blockdiag: 改块 {k} 影响了步 {STEPS[ti]} "
                      f"(应恰好 0, 实为 {db:.3e})")
        print(f"    causal 块{k:<2}  " + "  ".join(f"{v:>6}" for v in row_ca))
        print(f"    blockd 块{k:<2}  " + "  ".join(f"{v:>6}" for v in row_bd))
    print("[ok] B blockdiag 下每一块只影响自己的那一步（其余逐位恰好 0）")
    print("[note] B causal 下可见『累积前缀』泄露: 改块 1 会经 depth=2 的")
    print("       self-attention 影响其后所有步 —— 这是本开关要消除的耦合")

    # ── C. 梯度: blockdiag 下步 t 的损失推不动更早的块 ──
    print("\n[C] 梯度: 只用『最后一步』的损失求导, 看对更早块 register 的梯度")
    last = len(STEPS) - 1
    for name, d in (("causal  ", d_ca), ("blockdiag", d_bd)):
        zg = z_s.clone().requires_grad_(True)
        g = torch.autograd.grad(d(z_cls, zg)[0, last].pow(2).sum(), zg)[0]
        cells = []
        for k in range(1, len(STEPS)):
            lo = k * k
            gk = g[0, lo - 1].abs().max().item()    # 块 k 的种子
            cells.append(f"块{k}(种子)={gk:.1e}")
            if name == "blockdiag" and gk != 0.0:
                ok = False
                print(f"[FAIL] C blockdiag: 最后一步损失推动了块 {k}（应恰好 0）")
        print(f"    {name}: " + "  ".join(cells))
    print("[ok] C blockdiag 下最后一步的损失对更早块梯度恰好 0（跨步回流切断）")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
