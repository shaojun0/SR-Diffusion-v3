"""MultiResSR — 多分辨率 Phase-1 变体（2026-09-26）

与 model_v2.SRPhase1V2 的三点差异：
1. **编码器不设最大尺寸**：输入可为任意 14 倍数尺寸（N 可变，K=144 ≤ N 由分桶保证）。
   仍走 register 式：序列 = [cls; specials(K); patches(N)] → DINOv2 24 层全双向。
2. **解码总 token 固定 144**：K=num_specials=144，采样步默认 square_block_starts(144)
   = [1,4,9,…,144]，共 **12 步（0~11）**。
3. **4 个解码器**分别输出 448×252 / 252×448 / 224×224 / 448×448；
   损失按「原图最适配分辨率」加权：该档 **2/5**，其余各 **1/5**（权重和 = 1）。

损失口径与 model_v2 一致：每步直接预测、步内 patch 平权、步间 mean_t 平权：
    L_i = mean_t mean_patch L1(PixelHead(Y_i,t), target_i)
    L   = Σ_i w_i · L_i ,  w_{best}=0.4, w_other=0.2
"""
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v2 import (OutputQueryDecoder, PixelHead, SpecialTokenBank,
                      square_block_starts)

PATCH = 14
# 4 个解码目标（W, H）；448×252=576 patch, 252×448=576, 224×224=256, 448×448=1024
RESOS = [(448, 252), (252, 448), (224, 224), (448, 448)]
# 编码器输入的 patch 数分桶（用于同尺寸成批；长边上限见 data_multi）
AREA_BUCKETS = [576, 1024, 2176, 4096]


def reso_patches(reso: Sequence[int]) -> int:
    w, h = reso
    assert w % PATCH == 0 and h % PATCH == 0
    return (w // PATCH) * (h // PATCH)


def best_fit_index(W: int, H: int, resos=RESOS, tie_eps: float = 1e-3) -> int:
    """原图最适配的分辨率下标：长宽比最接近（|Δlog aspect| 最小），
    平手（差 ≤ tie_eps）时取像素数最接近的那档。"""
    la = math.log(W / H)
    scored = []
    for i, (w, h) in enumerate(resos):
        d_asp = abs(math.log(w / h) - la)
        d_area = abs(math.log((w * h) / (W * H)))
        scored.append((d_asp, d_area, i))
    best = min(s[0] for s in scored)
    cand = [s for s in scored if s[0] <= best + tie_eps]
    cand.sort(key=lambda s: (s[1], s[2]))
    return cand[0][2]


class MultiResSR(nn.Module):
    def __init__(self, dinov2: nn.Module, num_specials: int = 144,
                 dim: int = 1024, heads: int = 8, depth: int = 2,
                 mlp_ratio: float = 4.0, resos: Sequence = RESOS,
                 weight_best: float = 0.4, weight_other: float = 0.2,
                 decoder_steps: Optional[Sequence[int]] = None,
                 step_plan: str = "square", block: int = 12,
                 patch_px: int = PATCH * PATCH * 3,
                 decoder_dropout: float = 0.0,
                 decoder_ckpt: bool = False):
        super().__init__()
        self.dinov2 = dinov2
        self.num_specials = int(num_specials)
        self.dim = dim
        self.decoder_ckpt = bool(decoder_ckpt)
        self.resos = [tuple(r) for r in resos]
        self.patch_px = patch_px
        self.weight_best = float(weight_best)
        self.weight_other = float(weight_other)
        assert abs(self.weight_best + 3 * self.weight_other - 1.0) < 1e-6, \
            "4 档权重之和必须为 1（2/5 + 3×1/5）"

        self.special_bank = SpecialTokenBank(num_tokens=self.num_specials, dim=dim)
        self.decoders = nn.ModuleList([
            OutputQueryDecoder(dim=dim, num_patches=reso_patches(r),
                               mlp_ratio=mlp_ratio, heads=heads,
                               steps=decoder_steps, depth=depth,
                               num_specials=self.num_specials,
                               dropout=decoder_dropout,
                               step_plan=step_plan, block=block)
            for r in self.resos])
        # 4 档共享同一个 per-patch 像素头（patch 级映射与分辨率无关）
        self.pixel_head = PixelHead(dim=dim, patch_px=patch_px)
        self.steps = list(self.decoders[0].steps)

    # ── 编码：任意 N（N ≥ K），register 式 ──
    def encode(self, pixel_values: torch.Tensor):
        x = pixel_values
        B = x.shape[0]
        assert x.shape[-1] % PATCH == 0 and x.shape[-2] % PATCH == 0, \
            f"输入须为 14 的倍数, got {tuple(x.shape)}"
        N = (x.shape[-1] // PATCH) * (x.shape[-2] // PATCH)
        assert N >= self.num_specials, \
            f"N={N} < K={self.num_specials}（z_s 比 patch 还多）"
        emb = self.dinov2.embeddings(x)                          # (B,1+N,D)
        specials = self.special_bank(B, x.device)                # (B,K,D)
        seq = torch.cat([emb[:, :1], specials, emb[:, 1:]], dim=1)
        out = self.dinov2.encoder(seq)
        if hasattr(out, "last_hidden_state"):                    # transformers ≥5
            seq = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            seq = out[0]
        else:
            seq = out
        seq = self.dinov2.layernorm(seq)
        return seq[:, :1], seq[:, 1:1 + self.num_specials]

    # ── 逐解码器损失（逐 step 计算, 避免 materialize (B,|T|,N,588) 的差值）──
    @staticmethod
    def _decoder_loss(pix: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """pix (B,|T|,N,588), tgt (B,N,588) → (B,) 每样本的 mean_t mean_patch L1。"""
        B, T = pix.shape[0], pix.shape[1]
        acc = None
        for t in range(T):
            d = (pix[:, t] - tgt).abs().mean(dim=(1, 2))          # (B,)
            acc = d if acc is None else acc + d
        return acc / T

    def forward(self, pixel_values: torch.Tensor, targets: Sequence[torch.Tensor],
                best_idx: torch.Tensor):
        """
        pixel_values: (B,3,Hb,Wb) 编码器输入（批内同尺寸）
        targets:      长度 4 的 list, 每个 (B,N_i,588) 归一化像素 patch
        best_idx:     (B,) long, 原图最适配分辨率下标（0..3）
        """
        z_cls, z_s = self.encode(pixel_values)
        B = pixel_values.shape[0]
        total = pixel_values.new_zeros(())
        per_dec, recon = [], None
        for i, dec in enumerate(self.decoders):
            if self.decoder_ckpt and self.training:
                # 每个解码器整段 12 步做一次检查点重算：把「按步存图」压成 O(1)
                Y = torch.utils.checkpoint.checkpoint(dec, z_cls, z_s,
                                                      use_reentrant=False)
            else:
                Y = dec(z_cls, z_s)                               # (B,|T|,N_i,D)
            pix = self.pixel_head(Y)                              # (B,|T|,N_i,588)
            tgt = targets[i]
            Li_b = self._decoder_loss(pix, tgt)                   # (B,)
            w = torch.where(best_idx == i,
                            torch.full_like(Li_b, self.weight_best),
                            torch.full_like(Li_b, self.weight_other))
            total = total + (w * Li_b).mean()
            per_dec.append(Li_b.mean().detach())
            if i == int(best_idx[0]):
                recon = F.l1_loss(pix[:, -1], tgt)
        out = {"loss": total, "recon": recon,
               "per_decoder": torch.stack(per_dec), "steps": self.steps}
        return out
