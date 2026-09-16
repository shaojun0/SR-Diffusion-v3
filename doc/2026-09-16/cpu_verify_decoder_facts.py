"""CPU 侧独立验证（一）：在**仓库自己的解码器**上验证两条结构性事实。

不需要 GPU / DINO 权重 / 数据集，纯 CPU 数秒可跑。

F1  「裸加 carry + pre-LN 残差 ⇒ 读出特征随步号线性累积」
    把 OutputQueryDecoder 的 stack 全部置零 ⇒ 只剩残差恒等通路 ⇒ 第 t 步输出
    应恰为 t·query_base。这条说明 out_t = t·query_base + Σ_i Δ_i（无投影/LayerNorm
    夹在递推与 PixelHead 之间），即"沿用上一步画布"在通路上是免费解。

F2  「detach 下 ∂L_t/∂query_base = 0（t≥1）」
    现行（shipped）代码的 carry 是 (query_base + Y).detach() ⇒ t≥1 的查询含被
    detach 的项 ⇒ query_base 只被 step1（读窗口最小那一步）塑造。
    这是 REPORT_v2_bptt_vs_detach.md §5 那条判据的独立复现。

用法:  python cpu_verify_decoder_facts.py
"""
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):          # Windows 控制台默认 GBK
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]          # <repo>/doc/<date>/x.py -> <repo>
sys.path.insert(0, str(ROOT))

from model_v2 import OutputQueryDecoder, PixelHead  # noqa: E402

K, N, D, B = 576, 576, 64, 2       # K=N=576 = 24 步全轨迹那个操作点


def fact_f1():
    torch.manual_seed(0)
    dec = OutputQueryDecoder(dim=D, num_patches=N, num_specials=K,
                             heads=8, depth=2, mlp_ratio=2.0)
    for p in dec.parameters():                     # 全部权重置零
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        q = torch.randn(N, D) * 0.02               # query_base 的初始化量级
        dec.query_base.copy_(q)
        out = dec(z_cls=torch.randn(B, 1, D), z_s=torch.randn(B, K, D))

    print("F1: zeroed stack -> out_t 应等于 (t+1)*query_base")
    ok = True
    for i in (0, 1, 5, 12, 23):
        err = (out[:, i] - (i + 1) * q.unsqueeze(0)).abs().max().item()
        ok &= err < 1e-5
        print(f"   step {i + 1:2d}: max|out_t - {i + 1}*q| = {err:.3e}")
    growth = out[:, 23].abs().mean().item() / out[:, 0].abs().mean().item()
    print(f"   |out_24| / |out_1| = {growth:.1f}x   "
          f"（= 步号线性累积 query_base）")
    print(f"   => {'PASS' if ok else 'FAIL'}\n")
    return ok, growth


def fact_f2():
    torch.manual_seed(0)
    dec = OutputQueryDecoder(dim=D, num_patches=N, num_specials=K,
                             heads=8, depth=2, mlp_ratio=2.0)
    head = PixelHead(D, patch_px=12, hidden=128)
    z_cls, z_s = torch.randn(B, 1, D), torch.randn(B, K, D)
    target = torch.randn(B, N, 12)
    Ypix = head(dec(z_cls, z_s))

    print("F2: shipped(detach) 代码下 d L_t / d query_base（逐步）")
    grads = []
    for i in (0, 1, 2, 5, 23):
        dec.zero_grad(set_to_none=True)
        torch.nn.functional.l1_loss(Ypix[:, i], target).backward(retain_graph=True)
        g = 0.0 if dec.query_base.grad is None else dec.query_base.grad.abs().mean().item()
        grads.append((i, g))
        print(f"   step {i + 1:2d}: |dL_t/dquery_base| = {g:.6f}")
    ok = grads[0][1] > 0 and all(g == 0.0 for _, g in grads[1:])
    print(f"   => step1 非零、t>=1 恰为 0: {'PASS' if ok else 'FAIL'}\n")
    return ok


if __name__ == "__main__":
    print(f"repo = {ROOT}")
    print(f"torch = {torch.__version__}  cuda = {torch.cuda.is_available()}\n")
    ok1, growth = fact_f1()
    ok2 = fact_f2()
    print(f"ALL CHECKS {'PASSED' if (ok1 and ok2) else 'FAILED'}")
    sys.exit(0 if (ok1 and ok2) else 1)
