"""toy_zmode_ab.py — CPU 微型复现台（改自 SR-Diffusion-v3 doc/2026-09-16/cpu_probe_bptt_toy.py）

唯一变量 = z_s 取编码器序列的哪一段：
  · z_mode=register : z_s = [cls; specials(K); patches(N)] 里的 **specials**（仓库主线口径）
  · z_mode=patch    : z_s = 同一序列里的 **patch token**（分辨率扫描 sweep 的 --z_mode patch 口径）

其余全部照抄 model_v2.py / sweep_res_train.py 的语义：
  平方块读窗口 + 裸加 carry（BPTT）+ mean_t L1（各步平权）+ F_hat=末步
  A 相 warm start（单步 + 全读）→ B 相多步循环
每个 z_mode 各跑一份自己的 A 相（保证单变量：只有读出段不同）。

用法: python toy_zmode_ab.py [--A 600] [--B 900] [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
D, HEADS = 32, 4
ENC_LAYERS, DEC_LAYERS = 2, 2
GRID, PATCH, CH = 8, 2, 3
PATCH_PX = CH * PATCH * PATCH
N = GRID * GRID                      # 64 patches  ↔ sweep 112×112
IMG = GRID * PATCH
SEED = 42
STEPS = [k * k for k in range(1, 9)]  # 8 步 = square_block_starts(N)


def make_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    ax = torch.linspace(0, 1, IMG)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    out = []
    for _ in range(n):
        amp = torch.rand(3, generator=g)
        ph = torch.rand(6, generator=g) * 2 * math.pi
        acc = torch.zeros(CH, IMG, IMG)
        for ch in range(CH):
            for f in range(3):
                k = f + 1
                acc[ch] += amp[f] * torch.sin(2 * math.pi * k * xx + ph[f]) \
                    * torch.cos(2 * math.pi * k * yy + ph[f + 3])
        out.append(acc)
    img = torch.stack(out)
    return (img - img.mean(dim=(2, 3), keepdim=True)) / (img.std(dim=(2, 3), keepdim=True) + 1e-6)


def to_patches(img):
    B, C, H, W = img.shape
    return img.reshape(B, C, H // PATCH, PATCH, W // PATCH, PATCH) \
              .permute(0, 2, 4, 1, 3, 5).reshape(B, N, PATCH_PX)


class Enc(nn.Module):
    """[cls; specials(K); patches(N)] 过全双向注意力；读出段由 z_mode 决定。"""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(PATCH_PX, D)
        self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.spec = nn.Parameter(torch.randn(1, N, D) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 1 + N + N, D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, HEADS, D * 2, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, ENC_LAYERS)
        self.ln = nn.LayerNorm(D)

    def forward(self, pix, z_mode="register"):
        B = pix.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), self.spec.expand(B, -1, -1),
                       self.proj(pix)], dim=1) + self.pos
        x = self.ln(self.enc(x))
        if z_mode == "patch":
            return x[:, :1], x[:, 1 + N:1 + 2 * N]     # patch token（sweep 口径）
        return x[:, :1], x[:, 1:1 + N]                 # specials/register（主线口径）


class Dec(nn.Module):
    def __init__(self, detach):
        super().__init__()
        self.detach = detach
        self.query_base = nn.Parameter(torch.randn(N, D) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, N + 1, D) * 0.02)
        layer = nn.TransformerDecoderLayer(D, HEADS, D * 2, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.stack = nn.TransformerDecoder(layer, DEC_LAYERS)

    def forward(self, z_cls, z_s, steps=None):
        B = z_cls.shape[0]
        A = torch.cat([z_cls, z_s], dim=1) + self.pos_embed
        Y = self.query_base.unsqueeze(0).expand(B, -1, -1)
        if steps is None:
            return [self.stack(Y, A)]
        outs = []
        for i, t in enumerate(steps):
            k = math.isqrt(int(t))
            hi = min((k + 1) ** 2 - 1, N)
            lo = 0 if i == 0 else max(k * k, 1)
            Y = self.stack(Y, A[:, lo:hi + 1])
            outs.append(Y)
            Y = self.query_base + Y
            if self.detach:
                Y = Y.detach()
        return outs


class Model(nn.Module):
    def __init__(self, detach, freeze_enc=False, z_mode="register"):
        super().__init__()
        self.enc, self.dec = Enc(), Dec(detach)
        self.head = nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, PATCH_PX))
        self.freeze_enc = freeze_enc
        self.z_mode = z_mode

    def encode(self, pix):
        if self.freeze_enc:
            with torch.no_grad():
                return self.enc(pix, self.z_mode)
        return self.enc(pix, self.z_mode)

    def forward(self, pix, target, steps=None):
        z_cls, z_s = self.encode(pix)
        Ypix = self.head(torch.stack(self.dec(z_cls, z_s, steps), dim=1))
        per_step = F.l1_loss(Ypix, target.unsqueeze(1).expand_as(Ypix),
                             reduction="none").mean(dim=(0, 2, 3))
        return per_step, Ypix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", type=int, default=600, help="A 相步数（单步全读 warm start）")
    ap.add_argument("--B", type=int, default=900, help="B 相步数（多步循环）")
    ap.add_argument("--out", default=str(HERE / "toy_zmode_ab_result.json"))
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads() or 8))
    tr, te = to_patches(make_data(1024, 1)), to_patches(make_data(256, 2))
    g = torch.Generator().manual_seed(7)
    trivial = te.abs().mean().item()
    print(f"[toy] N=K={N}, steps={STEPS} (|T|={len(STEPS)}), D={D}, "
          f"enc/dec={ENC_LAYERS}/{DEC_LAYERS}, A={args.A}, B={args.B}, 平凡解={trivial:.4f}",
          flush=True)

    out = []
    for z_mode in ("register", "patch"):
        # ---------- A 相（每个 z_mode 各一份，保证单变量）----------
        torch.manual_seed(SEED)
        warm = Model(detach=True, z_mode=z_mode)
        opt = torch.optim.AdamW(warm.parameters(), lr=1e-3, weight_decay=0.01)
        t0 = time.time()
        for it in range(args.A):
            idx = torch.randint(0, tr.shape[0], (32,), generator=g)
            loss = warm(tr[idx], tr[idx], steps=None)[0].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            ws = warm(te, te, steps=None)[0].mean().item()
        print(f"[A/{z_mode}] warm-start 单步全读 test L1 = {ws:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        warm_state = {k: v.clone() for k, v in warm.state_dict().items()}

        # ---------- B 相：BPTT ----------
        torch.manual_seed(SEED)
        m = Model(detach=False, freeze_enc=False, z_mode=z_mode)
        m.load_state_dict(warm_state)
        opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.01)
        t0 = time.time()
        traj = []
        for it in range(args.B):
            idx = torch.randint(0, tr.shape[0], (32,), generator=g)
            loss = m(tr[idx], tr[idx], steps=STEPS)[0].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if it % max(1, args.B // 6) == 0 or it == args.B - 1:
                with torch.no_grad():
                    c = m(te, te, steps=STEPS)[0].tolist()
                traj.append({"it": it, "curve": [round(v, 4) for v in c],
                             "best": round(min(c), 4), "step1": round(c[0], 4)})
                print(f"[B/{z_mode}] it{it} curve={[round(v,4) for v in c]}", flush=True)
        with torch.no_grad():
            curve = m(te, te, steps=STEPS)[0].tolist()
            _, Ypix = m(te, te, steps=STEPS)
            deltas = [(Ypix[:, i] - Ypix[:, i - 1]).abs().mean().item()
                      for i in range(1, Ypix.shape[1])]
        rec = {"z_mode": z_mode, "warm_start_test_l1": ws, "curve": curve,
               "step1": curve[0], "best": min(curve), "last": curve[-1],
               "traj_gain_pct": 100 * (curve[0] - min(curve)) / curve[0],
               "vs_trivial_pct": 100 * (trivial - min(curve)) / trivial,
               "carry_delta_l1": deltas, "traj_hist": traj,
               "secs": round(time.time() - t0, 1)}
        out.append(rec)
        print(f"[B/{z_mode}] step1={curve[0]:.4f} best={min(curve):.4f} "
              f"last={curve[-1]:.4f} traj_gain={rec['traj_gain_pct']:.1f}% "
              f"vs_trivial={rec['vs_trivial_pct']:.1f}% ({rec['secs']}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": {"N": N, "K": N, "steps": STEPS, "D": D,
                              "phaseA_steps": args.A, "phaseB_steps": args.B,
                              "trivial_l1": trivial}, "arms": out},
                  f, indent=2, ensure_ascii=False)
    print("-> 写入 " + args.out, flush=True)


if __name__ == "__main__":
    main()
