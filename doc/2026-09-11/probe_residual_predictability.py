"""
SR-Diffusion-v3 · CPU 零训练探针：后步不干活，是"没信息"还是"损失让零成为最优"?
================================================================================
日期: 2026-09-11   设备: CPU

逻辑
----
对已经前向的 Y_t 序列（共享参数、累积损失结构），第 t 步面对的残差是
      r_t = target − PixelHead(carry_t)          carry_t = Σ_{i<t} Y_i (detach)
把 r_t 对"该步**独有**的可读信息"（= 该步能读、而前面所有步都读不到的 z_s）
做线性回归：
      · 可预测性 > 0 且 Y_t ≈ 0  ⇒ 信息在、但损失让"零"成为最优   ⇒ H2（损失结构）
      · 可预测性 ≈ 0              ⇒ 真的没信息可用                 ⇒ H1（信息/容量）
训练用哪个损失，随后就用哪个损失再训一段，看后步是否开始动——两段合起来能
把"信息不足"与"目标结构"分离。

本脚本是自包含的（不 import 主实验），用同一套数据/掩码/损失约定。

用法: python probe_residual_predictability.py [--steps 800] [--arm cum|region]
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ── 与 cpu_repro_cumloss_capacity.py 一致的结构常量 ──
K = 25
BLOCKS = [2 * k + 1 for k in range(1, math.isqrt(K - 1) + 1)]
N_PATCH, P_FAC, P_PATCH, D = 32, 8, 32, 96
STEPS = [k * k for k in range(1, len(BLOCKS) + 1)]
W_OWN, W_GLOB = 0.9, 0.35


def block_bounds(k):
    return k * k, min((k + 1) * (k + 1) - 1, K - 1)


def make_data(seed, n):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g)
    m, per = len(BLOCKS), N_PATCH // len(BLOCKS)
    FAC = rnd(n, m, P_FAC)
    Wp = rnd(P_FAC, D) * (1.0 / math.sqrt(P_FAC))
    proj = FAC @ Wp
    zs = torch.zeros(n, K, D)
    for k in range(m):
        lo, hi = block_bounds(k)
        npos = hi + 1 - lo
        zs[:, lo:hi + 1] = proj[:, k:k + 1].expand(n, npos, D) + rnd(n, npos, D) * 0.05
    zs[:, 0] = proj[:, 0] + rnd(n, D) * 0.05
    W_own = rnd(P_FAC, P_PATCH) * W_OWN
    W_glob = rnd(m * P_FAC, P_PATCH) * (W_GLOB / math.sqrt(m * P_FAC))
    own = torch.stack([FAC[:, k // per] for k in range(N_PATCH)], 1) @ W_own
    glob = FAC.reshape(n, -1) @ W_glob
    tgt = own + glob.unsqueeze(1) + rnd(n, N_PATCH, P_PATCH) * 0.05
    return zs, tgt / (tgt.std() + 1e-6)


class Decoder(nn.Module):
    def __init__(self, dim, n_patch, K, steps, depth=2, cap=False):
        super().__init__()
        self.n_patch, self.K, self.steps = n_patch, K, steps
        self.query_base = nn.Parameter(torch.randn(n_patch, dim) * 0.02)
        self.cap = cap
        if cap:
            self.qnet = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            self.patch_emb = nn.Parameter(torch.randn(n_patch, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, K, dim) * 0.02)
        self.stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, 4, dim * 4, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True),
            num_layers=depth)
        rows = []
        for i, t in enumerate(steps):
            k = math.isqrt(t)
            hi = min((k + 1) * (k + 1) - 1, K - 1)
            row = torch.full((K,), float("-inf"))
            if i == 0:
                row[:hi + 1] = 0.0
            else:
                row[max(k * k, 1):hi + 1] = 0.0
            rows.append(row)
        self.register_buffer("mem_mask", torch.stack(rows).repeat_interleave(n_patch, 0))
        L = len(steps) * n_patch
        tm = torch.full((L, L), float("-inf"))
        for i in range(len(steps)):
            tm[i * n_patch:(i + 1) * n_patch, i * n_patch:(i + 1) * n_patch] = 0.0
        self.register_buffer("tgt_mask", tm)

    def forward(self, z_cls, z_s):
        B = z_s.shape[0]
        A = torch.cat([z_cls, z_s], 1) + self.pos_embed
        A_t = A[:, self.steps]
        if self.cap:
            pooled = self.qnet(z_s.mean(1, keepdim=True))
            base = self.query_base.unsqueeze(0) + pooled + self.patch_emb.unsqueeze(0)
            Y = (A_t.unsqueeze(2) + base.unsqueeze(1)).reshape(B, len(self.steps) * self.n_patch, -1)
        else:
            Y = (A_t.unsqueeze(2) + self.query_base).reshape(B, len(self.steps) * self.n_patch, -1)
        Y = self.stack(Y, A, memory_mask=self.mem_mask, tgt_mask=self.tgt_mask)
        return Y.reshape(B, len(self.steps), self.n_patch, -1)


class Head(nn.Module):
    def __init__(self, dim, p_patch, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, p_patch))

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, arm, cap=False, depth=2):
        super().__init__()
        self.arm = arm
        self.dec = Decoder(D, N_PATCH, K, STEPS, depth=depth, cap=cap)
        self.head = Head(D, P_PATCH)
        per = N_PATCH // len(STEPS)
        self.regions = [list(range(i * per, (i + 1) * per)) for i in range(len(STEPS))]

    def forward(self, zs, tgt):
        Y = self.dec(zs[:, :1], zs[:, 1:])
        Y_cum = torch.cat([torch.zeros_like(Y[:, :1]),
                           Y.cumsum(1)[:, :-1]], 1).detach() + Y
        Y_pix = self.head(Y_cum)
        if self.arm == "cum":
            loss = F.l1_loss(Y_pix, tgt.unsqueeze(1), reduction="none").mean()
        else:
            loss = torch.stack([F.l1_loss(Y_pix[:, i, c], tgt[:, c])
                                for i, c in enumerate(self.regions)]).mean()
        return {"loss": loss, "Y": Y, "Y_pix": Y_pix}


def own_features(z_s, i):
    """第 i 步**独有**的可读信息 = 该步能读、而前面所有步都读不到的 z_s 列。"""
    k = math.isqrt(STEPS[i])
    hi = min((k + 1) * (k + 1) - 1, K - 1)
    lo = 0 if i == 0 else max(k * k, 1)
    prev_hi = 0 if i == 0 else hi  # i=0 无前步
    if i == 0:
        cols = list(range(0, hi + 1))
    else:
        cols = list(range(lo, hi + 1))
    return z_s[:, cols, :].reshape(z_s.shape[0], -1)


@torch.no_grad()
def residual_predictability(m, zs, tgt, lam=1e-2):
    """每步的残差 r_t 对"该步独有信息"的线性可预测性（held-out R²）。

    可预测性 = 1 − MSE(r_t | own_feat) / MSE(r_t | const)
    留出法：前半训练岭回归、后半评估。
    """
    out = m(zs, tgt)
    Y, Y_pix = out["Y"], out["Y_pix"]
    T = Y.shape[1]
    res = []
    n = zs.shape[0]
    half = n // 2
    for i in range(T):
        carry = Y[:, :i].sum(1).detach() if i > 0 else torch.zeros_like(Y[:, 0])
        pix_carry = m.head(carry)                       # (B,N,P)
        r = (tgt - pix_carry).reshape(n, -1)            # 残差
        X = own_features(zs, i)
        mu, sd = X[:half].mean(0, keepdim=True), X[:half].std(0, keepdim=True) + 1e-6
        X = (X - mu) / sd
        X = torch.cat([X, torch.ones(n, 1)], 1)
        XtX = X[:half].T @ X[:half]
        W = torch.linalg.solve(XtX + lam * XtX.diag().mean() * torch.eye(XtX.shape[0]),
                               X[:half].T @ r[:half])
        pred = X[half:] @ W
        base = r[half:].mean(0, keepdim=True)
        mse_fit = ((pred - r[half:]) ** 2).mean().item()
        mse_base = ((base - r[half:]) ** 2).mean().item()
        res.append({"step": STEPS[i], "r2": 1 - mse_fit / max(mse_base, 1e-12),
                    "r_l1": r.abs().mean().item(),
                    "Y_t_scale": Y_pix[:, i].abs().mean().item(),
                    "target_scale": tgt.abs().mean().item()})
    return res


def train(m, zs, tgt, steps_n, B=128, lr=2e-3):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps_n,
                                                pct_start=0.1)
    t0 = time.time()
    m.train()
    for it in range(steps_n):
        idx = torch.randint(0, zs.shape[0], (B,))
        out = m(zs[idx], tgt[idx])
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if (it + 1) % max(1, steps_n // 5) == 0:
            print(f"    it {it+1:>5}/{steps_n} loss {out['loss'].item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--arm", default="cum", choices=["cum", "region"])
    args = ap.parse_args()

    print("=" * 78)
    print("CPU 零训练探针：残差可预测性 vs 后步输出量级")
    print(f"设备: CPU | K={K} N={N_PATCH} D={D} steps={STEPS} | arm={args.arm} "
          f"| 训练 {args.steps} 步")
    print("=" * 78)

    zs_tr, tgt_tr = make_data(0, 16384)
    zs_te, tgt_te = make_data(1, 4096)

    torch.manual_seed(42)
    m = Model(args.arm, cap=False, depth=2)
    print(f"[train] {sum(p.numel() for p in m.parameters()):,} params")
    train(m, zs_tr, tgt_tr, args.steps)
    m.eval()

    res = residual_predictability(m, zs_te, tgt_te)
    tgt_scale = res[0]["target_scale"]
    print("\n[probe] 每步: 残差可预测性 R² | 残差量级 | 该步输出量级/目标量级")
    for r in res:
        print(f"    step {r['step']:>2}  R²={r['r2']:+.3f}   "
              f"|r|={r['r_l1']:.3f}   Y_t/target={r['Y_t_scale']/tgt_scale:.3f}")

    tail = [r["Y_t_scale"] / tgt_scale for r in res[1:]]
    mid = sum(tail) / len(tail)
    r2 = [r["r2"] for r in res[1:]]
    print(f"\n后 {len(tail)} 步输出量级均值 = {mid:.3f}")
    print(f"后 {len(tail)} 步残差可预测性 R² = {[round(x,3) for x in r2]}")
    if mid < 0.15 and max(r2) > 0.05:
        print("⇒ 裁定 H2（损失结构）：残差**可预测**（信息在），但后步输出≈0 —— "
              "零是当前损失的最优解，不是能力/信息不足。")
    elif mid < 0.15 and max(r2) <= 0.05:
        print("⇒ 裁定 H1（信息不足）：残差不可预测，后步零是「没有活干」。")
    else:
        print("⇒ 后步**在干活**（输出量级显著）—— 本设定下损失结构未导致退化。")

    outdir = Path(__file__).parent / "data"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"probe_residual_{args.arm}.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {p}")


if __name__ == "__main__":
    main()
