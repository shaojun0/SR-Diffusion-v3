"""CPU 侧微型复现台：同一套解码器/损失/读窗口语义下，detach vs BPTT 的四臂对照。

目的：**在没有 GPU / 没有 DINOv2 / 没有数据集**的前提下，独立检验
REPORT_v2_bptt_vs_detach.md 的机制主张——"去掉循环 carry 的 detach 让轨迹从
'平'变'降'"，并回答一个真实 run 没分离的问题：这个增益是**解码器侧信用分配**
还是**编码器侧（读出梯度扇出被放大）**？

镜像的语义（全部照 model_v2.py / train_v2.py 写）：
  · 编码器 = [cls; specials(K); patches(N)] 过全双向注意力 → z_cls, z_s
  · 解码器 = OutputQueryDecoder：query_base + 平方块读窗口 + carry
             out_t = q + out_{t-1} + Δ_t（pre-LN 残差 + 裸加 carry）
  · 损失   = mean_t L1(PixelHead(out_t), target)   （每步直接预测，各步平权）
  · 读数   = 末步（= 真实代码里的 F_hat）

两阶段设计（关键）：
  A 相（warm start，600 步，全读单步重建）→ 让编码器产出"可用的"特征，
     粗略对应真实 run 里"预训练 DINOv2 已经能支持单次重建"的起点。
  B 相（900 步，8 步循环 + 分块读窗口）→ **冻结编码器时只剩解码器侧变量**；
     可训编码器时复现真实 run 的完整通路。两臂都从同一份 warm start 出发。

第一版（不 warm start、直接从头训整个 pipeline）两臂都停在平凡解
（test L1 ≈ 0.81 = 预测均值），无判别力，已弃用——说明这类 toy 必须先给
"编码器可用特征"的起点，否则优化困难会淹没 detach/BPTT 的差别。

用法:
  python cpu_probe_bptt_toy.py            # 完整 4 臂（8 线程约 28 分钟）
  python cpu_probe_bptt_toy.py --quick    # 冒烟（每段 60 步，约 1 分钟）
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):          # Windows 控制台默认 GBK
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
D, HEADS = 32, 4
ENC_LAYERS, DEC_LAYERS = 2, 2
GRID, PATCH, CH = 8, 2, 3
PATCH_PX = CH * PATCH * PATCH               # 12
N = GRID * GRID                             # 64 patches
IMG = GRID * PATCH                          # 16x16
SEED = 42
STEPS = [k * k for k in range(1, 9)]        # 8 步 = square_block_starts(N)


def make_data(n, seed):
    """合成平滑图（每通道 3 个 sin×cos 分量 ⇒ 低维流形，可学）。"""
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

    def forward(self, pix):
        B = pix.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), self.spec.expand(B, -1, -1),
                       self.proj(pix)], dim=1) + self.pos
        x = self.ln(self.enc(x))
        return x[:, :1], x[:, 1:1 + N]


class Dec(nn.Module):
    """照 model_v2.OutputQueryDecoder 的读窗口/查询语义。"""

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
        if steps is None:                      # A 相：单步 + 全读
            return [self.stack(Y, A)]
        outs = []
        for i, t in enumerate(steps):
            k = math.isqrt(int(t))
            hi = min((k + 1) ** 2 - 1, N)
            lo = 0 if i == 0 else max(k * k, 1)
            Y = self.stack(Y, A[:, lo:hi + 1])
            outs.append(Y)
            Y = self.query_base + Y            # 裸加 carry
            if self.detach:                    # ← 唯一变量
                Y = Y.detach()
        return outs


class Model(nn.Module):
    def __init__(self, detach, freeze_enc=False):
        super().__init__()
        self.enc, self.dec = Enc(), Dec(detach)
        self.head = nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, PATCH_PX))
        self.freeze_enc = freeze_enc

    def encode(self, pix):
        if self.freeze_enc:
            with torch.no_grad():
                return self.enc(pix)
        return self.enc(pix)

    def forward(self, pix, target, steps=None):
        z_cls, z_s = self.encode(pix)
        Ypix = self.head(torch.stack(self.dec(z_cls, z_s, steps), dim=1))
        per_step = F.l1_loss(Ypix, target.unsqueeze(1).expand_as(Ypix),
                             reduction="none").mean(dim=(0, 2, 3))
        return per_step, Ypix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="冒烟：每段 60 步")
    ap.add_argument("--out", default=str(HERE / "data" / "toy_bptt_cpu_results.json"))
    args = ap.parse_args()
    n_a, n_b = (60, 60) if args.quick else (600, 900)

    torch.set_num_threads(min(8, torch.get_num_threads() or 8))
    tr, te = to_patches(make_data(1024, 1)), to_patches(make_data(256, 2))
    g = torch.Generator().manual_seed(7)

    print(f"[toy] N=K={N}, steps={STEPS}, D={D}, enc/dec={ENC_LAYERS}/{DEC_LAYERS} 层, "
          f"phaseA={n_a} 步, phaseB={n_b} 步")
    # ---------- A 相：warm start（单步 + 全读）----------
    torch.manual_seed(SEED)
    warm = Model(detach=True)
    opt = torch.optim.AdamW(warm.parameters(), lr=1e-3, weight_decay=0.01)
    t0 = time.time()
    for it in range(n_a):
        idx = torch.randint(0, tr.shape[0], (32,), generator=g)
        loss = warm(tr[idx], tr[idx], steps=None)[0].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        ws = warm(te, te, steps=None)[0].mean().item()
    trivial = te.abs().mean().item()
    print(f"[A] warm-start 全读 test L1 = {ws:.4f}（平凡解 ~{trivial:.4f}, "
          f"{time.time() - t0:.0f}s）", flush=True)
    warm_state = {k: v.clone() for k, v in warm.state_dict().items()}

    # ---------- B 相：四臂 ----------
    out = []
    for freeze in (True, False):
        for detach in (True, False):
            torch.manual_seed(SEED)
            m = Model(detach=detach, freeze_enc=freeze)
            m.load_state_dict(warm_state)
            if freeze:
                for p in m.enc.parameters():
                    p.requires_grad_(False)
            params = [p for p in m.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)
            tag = f"{'frozenEnc' if freeze else 'trainEnc'}-{'detach' if detach else 'BPTT'}"
            t0 = time.time()
            for it in range(n_b):
                idx = torch.randint(0, tr.shape[0], (32,), generator=g)
                loss = m(tr[idx], tr[idx], steps=STEPS)[0].mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                curve = m(te, te, steps=STEPS)[0].tolist()
                _, Ypix = m(te, te, steps=STEPS)
                deltas = [(Ypix[:, i] - Ypix[:, i - 1]).abs().mean().item()
                          for i in range(1, Ypix.shape[1])]
            rec = {"tag": tag, "freeze_enc": freeze, "detach": detach, "curve": curve,
                   "step1": curve[0], "best": min(curve), "last": curve[-1],
                   "traj_gain_pct": 100 * (curve[0] - min(curve)) / curve[0],
                   "carry_delta_l1": deltas, "secs": round(time.time() - t0, 1)}
            out.append(rec)
            print(f"[{tag}] step1={curve[0]:.4f} best={min(curve):.4f} last={curve[-1]:.4f} "
                  f"traj_gain={rec['traj_gain_pct']:.1f}% deltas="
                  f"{[round(d, 4) for d in deltas]} ({rec['secs']}s)", flush=True)

    base = {r["tag"].replace("BPTT", "detach"): r["last"]
            for r in out if r["detach"]}
    for r in out:
        r["last_vs_detach_pct"] = 100 * (base[r["tag"].replace("BPTT", "detach")]
                                         - r["last"]) / base[r["tag"].replace("BPTT", "detach")]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": {"N": N, "K": N, "D": D, "steps": STEPS,
                              "phaseA_steps": n_a, "phaseB_steps": n_b,
                              "warm_start_test_l1": ws, "trivial_l1": trivial},
                   "arms": out}, f, indent=2, ensure_ascii=False)
    print(f"-> 写入 {args.out}")


if __name__ == "__main__":
    main()
