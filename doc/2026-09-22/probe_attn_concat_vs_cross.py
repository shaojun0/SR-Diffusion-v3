"""探针：拼接式自注意力 vs 交叉注意力 —— 到底谁是谁？

问题（用户 2026-09-22 提问）：
    把交叉注意力的两个向量组 (query 组 X, memory 组 Y) 拼接起来做一次
    **自注意力**，这次自注意力里"是否包含着交叉注意力的内容"？

本文档对应的探针只做纯数学验证，**不依赖 model_v2**（无 DINOv2、无权重、
纯 CPU、秒级）。所有口径在 doc/2026-09-22/ANALYSIS_attn_concat_vs_cross.md
里逐条列出。

测什么：
  A. 严格等价性：拼接触发 self-attn + 掩码(self 块置 -inf) 是否 == 标准
     cross-attn（nn.MultiheadAttention 的 q/k/v 分离调用）？
     → 逐元素 max|Δ| 应为 0（浮点 ~1e-7）。
  B. 无掩码时差多少：同参数下 CONCAT(无掩码) 与 CROSS 的输出差多大？
     → 断言"不等"，并量化相对误差。
  C. 注意力预算：无掩码时 softmax 行内 self 块与 cross 块的权重占比。
  D. 梯度耦合：∂L/∂S_xy 在 CONCAT(无掩码) 下是否受 S_xx 影响（对比 CROSS）。

用法:
    python doc/2026-09-22/probe_attn_concat_vs_cross.py
    python doc/2026-09-22/probe_attn_concat_vs_cross.py --out doc/2026-09-22/data/probe_attn_concat_vs_cross.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
SEED = 42


# ───────────────────────── 参考实现 1：标准交叉注意力 ─────────────────────────
class CrossAttn(nn.Module):
    """标准 cross-attention：Q 来自 X，K/V 来自 Y，softmax 只在 Y 的 m 个位置上。"""

    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, y):
        B, n, D = x.shape
        m = y.shape[1]
        H = self.nhead
        dh = D // H

        def split(t):
            return t.view(B, -1, H, dh).transpose(1, 2)  # (B,H,*,dh)

        q = split(self.q_proj(x))          # (B,H,n,dh)
        k = split(self.k_proj(y))          # (B,H,m,dh)
        v = split(self.v_proj(y))          # (B,H,m,dh)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)  # (B,H,n,m)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                     # (B,H,n,dh)
        out = out.transpose(1, 2).reshape(B, n, D)
        return self.out_proj(out), attn, scores


# ───────────────────────── 参考实现 2：拼接式自注意力 ─────────────────────────
class ConcatSelfAttn(nn.Module):
    """把 [X;Y] 拼成一条序列做一次自注意力。

    q_proj==k_proj==v_proj 的权重与 CrossAttn 共享（"块共享"口径），
    这样唯一变量就是 softmax 的归一化范围。
    """

    def __init__(self, d_model, nhead, cross: CrossAttn, masked: bool):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.cross = cross            # 复用同一组投影
        self.masked = masked          # True: self 块置 -inf（真交叉注意力）

    def forward(self, x, y):
        B, n, D = x.shape
        m = y.shape[1]
        H = self.nhead
        dh = D // H
        seq = torch.cat([x, y], dim=1)                 # (B,n+m,D)
        w = self.cross

        def split(t):
            return t.view(B, -1, H, dh).transpose(1, 2)

        q = split(w.q_proj(seq))
        k = split(w.k_proj(seq))
        v = split(w.v_proj(seq))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)  # (B,H,n+m,n+m)

        if self.masked:
            # 行 <n（X 行）只能看列 >=n（Y 列）；行 >=n 只能看列 <n
            idx = torch.arange(n + m, device=seq.device)
            eye_x = idx < n
            allow = eye_x[:, None] != eye_x[None, :]   # 跨块=True, 同块=False
            scores = scores.masked_fill(~allow, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, n + m, D)
        out = w.out_proj(out)
        return out[:, :n], out[:, n:], attn, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data" / "probe_attn_concat_vs_cross.json"))
    args = ap.parse_args()

    torch.manual_seed(SEED)
    D, H, n, m = 64, 4, 5, 7
    x = torch.randn(2, n, D)
    y = torch.randn(2, m, D)

    cross = CrossAttn(D, H)
    ref, attn_ref, s_ref = cross(x, y)

    # A. 掩码拼接 == 标准交叉注意力
    cm = ConcatSelfAttn(D, H, cross, masked=True)
    x_m, _, _, _ = cm(x, y)
    max_a = (x_m - ref).abs().max().item()

    # B. 无掩码拼接 != 标准交叉注意力
    cu = ConcatSelfAttn(D, H, cross, masked=False)
    x_u, _, attn_u, s_u = cu(x, y)
    max_b = (x_u - ref).abs().max().item()
    denom = ref.abs().mean().item() + 1e-12
    rel_b = (x_u - ref).abs().mean().item() / denom

    # C. 注意力预算：X 行上 cross 块占多少
    A = attn_u[0].mean(0)                      # (n+m, n+m) 头平均
    budget_x_cross = A[:n, n:].sum(-1).mean().item()
    budget_x_self = A[:n, :n].sum(-1).mean().item()
    budget_y_cross = A[n:, :n].sum(-1).mean().item()
    budget_y_self = A[n:, n:].sum(-1).mean().item()

    # D. 梯度耦合：∂L/∂S_xy 是否被 S_xx 影响
    def grad_probe(masked):
        lin = cross
        # 手工前向：拿到 scores，构造 L = <A_xy, G>，看 ∂L/∂S_xy
        seq = torch.cat([x, y], dim=1)
        B, L, _ = seq.shape
        dh = D // H

        def split(t):
            return t.view(B, -1, H, dh).transpose(1, 2)

        q, k = split(lin.q_proj(seq)), split(lin.k_proj(seq))
        s = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        s = s.detach().requires_grad_(True)
        if masked:
            idx = torch.arange(L)
            same = (idx < n)[:, None] == (idx < n)[None, :]
            s2 = s.masked_fill(same, float("-inf"))
        else:
            s2 = s
        A = F.softmax(s2, dim=-1)
        G = torch.randn_like(A[:, :, :n, n:])
        L = (A[:, :, :n, n:] * G).sum()
        g = torch.autograd.grad(L, s)[0]
        g_xy = g[:, :, :n, n:]
        g_xx = g[:, :, :n, :n]
        return g_xy.abs().mean().item(), g_xx.abs().mean().item()

    torch.manual_seed(SEED)
    g_xy_u, g_xx_u = grad_probe(masked=False)
    torch.manual_seed(SEED)
    g_xy_m, g_xx_m = grad_probe(masked=True)

    # E. 注意力预算如何随 cross 侧打分尺度 lam 变化（lam 乘在 S_xy 上）
    def budget_sweep(lam):
        seq = torch.cat([x, y], dim=1)
        B, L, _ = seq.shape
        dh = D // H

        def split(t):
            return t.view(B, -1, H, dh).transpose(1, 2)

        with torch.no_grad():
            q = split(cross.q_proj(seq))
            k = split(cross.k_proj(seq))
            s = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
            s[:, :, :n, n:] = s[:, :, :n, n:] * lam     # X 行的 cross 块
            A = F.softmax(s, dim=-1)
            return (
                A[0, :, :n, n:].sum(-1).mean().item(),   # X 行 cross 占比
                A[0, :, n:, :n].sum(-1).mean().item(),   # Y 行 cross 占比
            )

    sweep = []
    for lam in [0.25, 0.5, 1.0, 2.0, 4.0]:
        bx, by = budget_sweep(lam)
        sweep.append({"lam": lam, "X_rows_cross_frac": bx, "Y_rows_cross_frac": by})

    res = {
        "seed": SEED,
        "shape": {"B": 2, "n": n, "m": m, "d_model": D, "nhead": H},
        "A_masked_equals_cross_max_abs_diff": max_a,
        "A_masked_exact": max_a < 1e-6,
        "B_unmasked_vs_cross_max_abs_diff": max_b,
        "B_unmasked_vs_cross_rel_err": rel_b,
        "B_unmasked_not_equal": max_b > 1e-3,
        "C_attention_budget": {
            "X_rows_cross_frac": budget_x_cross,
            "X_rows_self_frac": budget_x_self,
            "Y_rows_cross_frac": budget_y_cross,
            "Y_rows_self_frac": budget_y_self,
        },
        "D_grad": {
            "unmasked_grad_xy_mean": g_xy_u,
            "unmasked_grad_xx_mean": g_xx_u,
            "unmasked_grad_xx_nonzero": g_xx_u > 0,
            "masked_grad_xy_mean": g_xy_m,
            "masked_grad_xx_mean": g_xx_m,
            "masked_grad_xx_zero": g_xx_m == 0.0,
        },
        "E_budget_sweep": sweep,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
