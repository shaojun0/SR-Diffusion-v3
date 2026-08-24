import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from model_phase1 import SpecialTokenBank, ReEncoder, FeatureDecoder


class SelectHead(nn.Module):
    def __init__(self, in_dim: int = 768, num_patches: int = 256,
                 hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_patches),
        )

    def forward(self, cls: Tensor) -> Tensor:
        return self.mlp(cls)                       # (B,N) raw scores


# ═══════════════════════════════════════════════════════════════
# BernoulliGate — sigmoid 伯努利门控（可微 one-hot 选择）
# ═══════════════════════════════════════════════════════════════

class BernoulliGate(nn.Module):

    def __init__(self, threshold: float = 0.5, T: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.T = T

    def forward(self, scores: Tensor) -> Tuple[Tensor, Tensor]:
        B, N = scores.shape
        p = torch.sigmoid(scores / self.T)               # (B,N) 保留概率
        hard = (p > self.threshold).float()              # (B,N) 0/1 阈值二值化
        # STE: 前向用 hard（one-hot 式硬选择），反向梯度经 p 流入 SelectHead。
        # ⚠ 不能直接返回 hard——比较运算无 grad_fn，梯度路径会静默死掉。
        mask = p + (hard - p).detach()                   # (B,N) STE
        return mask, p


# ═══════════════════════════════════════════════════════════════
# SRPhase1V2 — 主模型（复用 v1 的 SpecialTokenBank / ReEncoder / FeatureDecoder）
# ═══════════════════════════════════════════════════════════════

class SRPhase1V2(nn.Module):

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        dim: int = 768,
        threshold: float = 0.5,
        T: float = 1.0,
        lambda_rate: float = 0.1,
        lambda_ent: float = 0.01,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.pad_token = nn.Parameter(torch.zeros(1, dim))
        self.beta = 0.5
        self.num_patches = num_patches
        self.dim = dim
        self.threshold = threshold
        self.T = T
        self.lambda_rate = lambda_rate
        self.lambda_ent = lambda_ent

        self.select_head = SelectHead(in_dim=dim, num_patches=num_patches)
        self.gate = BernoulliGate(threshold=threshold, T=T)
        self.special_bank = SpecialTokenBank(num_patches=num_patches, dim=dim)
        self.re_encoder = ReEncoder(dim=dim, num_patches=num_patches)
        self.decoder = FeatureDecoder(dim=dim, num_patches=num_patches)

    # ── 预算旋钮：threshold（stage-1 设 0 全保留，stage-2 退火到目标值）──
    def set_threshold(self, threshold: float):
        """设置 sigmoid 二值化阈值。[TODO] 两阶段退火调度见模块头。"""
        self.threshold = threshold
        self.gate.threshold = threshold

    def init_reencoder_from_dino(self, num_layers: int = 4):
        dino_layers = self.dinov2.encoder.layer
        depth = min(num_layers, len(self.re_encoder.layers), len(dino_layers))
        for i in range(depth):
            src, dst = dino_layers[i], self.re_encoder.layers[i]
            with torch.no_grad():
                dst.norm1.load_state_dict(src.norm1.state_dict())
                dst.norm2.load_state_dict(src.norm2.state_dict())
                dst.self_attn.in_proj_weight.data.copy_(torch.cat([
                    src.attention.attention.query.weight.data,
                    src.attention.attention.key.weight.data,
                    src.attention.attention.value.weight.data,
                ], dim=0))
                dst.self_attn.in_proj_bias.data.copy_(torch.cat([
                    src.attention.attention.query.bias.data,
                    src.attention.attention.key.bias.data,
                    src.attention.attention.value.bias.data,
                ], dim=0))
                dst.self_attn.out_proj.load_state_dict(src.attention.output.dense.state_dict())
                dst.linear1.load_state_dict(src.mlp.fc1.state_dict())
                dst.linear2.load_state_dict(src.mlp.fc2.state_dict())
        print(f"[init] ReEncoder layers 0..{depth-1} warm-started from DINO encoder")

    # ── forward ──
    def forward(self, pixel_values: Tensor) -> dict:
        x = pixel_values                                # (B,3,224,224)
        B = x.shape[0]
        N = self.num_patches

        out = self.dinov2(x)
        feats = out.last_hidden_state                   # (B,257,D)
        cls = feats[:, 0]                               # (B,D)
        patch = feats[:, 1:]                            # (B,N,D)

        # ── 步骤 1: cls (B,D) → 保留概率 p (B,N) ──
        scores = self.select_head(cls)                  # (B,N)
        soft_max_scores = nn.functional.softmax(scores,dim=1)
        # mask, p = self.gate(scores)                     # mask (B,N) STE, p (B,N) 保留概率
        positions = torch.arange(scores.shape[-1]).unsqueeze(0)
        indices = torch.argmax(soft_max_scores,dim=-1).unsqueeze(1)
        mask = (positions < indices).float()
        # ── 步骤 2: ReEncoder 出 z_s，按 mask 选择后进 decoder ──
        specials = self.special_bank(B, x.device)       # (B,N,D)
        enc_in = torch.cat([cls.unsqueeze(1), specials, patch], dim=1)  # (B,2N+1,D)
        z = self.re_encoder(enc_in)                     # (B,2N+1,D)
        z_cls = z[:, 0:1]                               # (B,1,D)
        z_s = z[:, 1:1 + N]                             # (B,N,D) ← 掩码选择对象（top-k 候选）
        dec_in = torch.cat([z_cls, torch.where(mask,z_s,self.pad_token)], dim=1)   # (B,N+1,D)
        F_hat = self.decoder(dec_in)                    # (B,N,D)

        # ── 步骤 3: 损失（全部按最小化实现）──
        # L_recon: 最小化特征重建误差（主线；驱动保留概率学会"少留也能重建"）
        recon = F.l1_loss(F_hat, patch)                 # scalar
        index_label = torch.zeros_like(scores)
        index_label[range(len(indices)), torch.where(indices-1>=0,indices-1,0)] = 1
        loss = 0.8*recon + 0.2*nn.functional.binary_cross_entropy(soft_max_scores,index_label)
