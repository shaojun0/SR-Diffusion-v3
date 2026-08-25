"""
SR-Diffusion Phase 1 v2 — 可导软前缀掩码（边界分布版）
=================================================================

沿革 / 为什么重写
----------------
旧草稿（ea8352f 思路，即本文件被替换前的版本）:
    SelectHead(cls) → softmax 256 类 → argmax 切点 k
    mask = (positions < k)            ← 前缀硬掩码
    BCE(softmax, onehot(k-1))         ← 自指监督

三个致命问题（已全部修复）:
  1. 梯度断链: mask 来自 argmax + 比较运算，无 grad_fn；重建损失
     L_recon 对 SelectHead 没有任何梯度。唯一梯度来自 BCE，而 BCE 的
     target 是"自己 argmax 减一"——自指、移动靶：softmax-CE 梯度 p−y
     会把峰值从 k 往 k−1 推，argmax 追上后 target 又变 k−2……边界一路
     左移直到 k=0 停住（argmax=0 时 target 被 clamp 到 0）。结果：无论
     重建多差，选择器最终"什么都不选"。← 本轮修复的核心。
  2. off-by-one: mask 用 `positions < k`（峰值 k 本身没被保留），target
     用 onehot(k−1)，两者差一格，自指目标内部不自洽。
  3. 永远选不满: argmax ∈ [0, N−1]，mask 不可能全 1；且预算惩罚
     lambda_rate 定义了但没接进 loss。

新设计（本文件）:
    SelectHead(cls) → (B, N+1) 边界分布 logits（k ∈ {0..N}，N 类 = 全保留）
    p = softmax(logits)                        ← 边界 k 的分布
    mask_soft[p] = P(k > p) = 1 − cumsum(p)[p] ← 可导软前缀掩码
    dec_in = z_cls + mask_soft·z_s + (1−mask_soft)·pad_token
    L = L_recon + lambda_rate·E[k]/N   (+ lambda_ent·H(p) 可选锐化)

    · 重建梯度经 cumsum∘softmax 直达 SelectHead —— 选择器真正学会
      "少留也能重建"，不再依赖任何自指监督。
    · 预算 = 简单率惩罚（lambda_rate 旋钮），与 v1/v2.5 的拉格朗日
      对偶求解器相比更直接：不需要对偶变量/阈值/STE。
    · 硬模式（推理/评估）: k = argmax(logits)，decoder 只吃 k+1 个
      token（真实剪枝，key_padding_mask 屏蔽 pad），形状训练/推理一致。

数学
----
    边界 k ∈ {0, 1, ..., N}，mask[p] = 1{k > p}（保留前 k 个特殊 token）。
    E[mask[p]] = P(k > p) = 1 − Σ_{j≤p} p_j = 1 − cumsum(p)[p]，p = 0..N−1。
    率: E[k]/N = mean(mask_soft)。
    若 p = onehot(k)，则 mask_soft 恰为硬前缀掩码 —— 软掩码是硬掩码的
    期望，训练软化、推理硬化，二者无缝衔接。

训练配方（推荐）
----------------
    stage-1 预热（~前 10% 步）:  model.set_lambda_rate(0.0)
        只训 L_recon，让 decoder/ReEncoder 先学会"用尽量多的信息重建"。
    stage-2 预算退火: 每 k 步把 lambda_rate 从 0 增大（如 ×1.5），
        直到 rate = E[k]/N 逼近目标（如 0.25）；或按
        lambda_rate += η·(rate − target) 闭环微调。lambda_rate 越大
        k 越小；recon 梯度决定"哪些位置最值得留"（前缀语义见局限）。

与 v2.5/v3 的差异（为什么不用拉格朗日）:
    v2.5/v3 用 sigmoid 逐位置门控 + STE + 拉格朗日对偶变量 λ，能精确
    打到目标率，但引入阈值 τ、对偶步、STE 噪声，且是"逐位置任意子集"
    而非"前缀"。本版要的是干净的前缀选择：一个 softmax + cumsum 就
    可导，预算用单个标量 lambda_rate 控制，无 STE、无阈值、无对偶变量。

已知局限（继承旧版，如需可后续换）:
    · 前缀语义: 低位置编号的特殊 token 永远比高位置"更优先保留"。
      如果重要性是逐位置任意分布（如人脸区域），应换 v3 的逐位置
      门控；前缀适合"前 k 个 token 携带决定性信息"的任务。
    · 纯 cls 全局选择: 看不到具体 patch 内容，只能学"区域级"重要性。
    · 硬模式剪枝收益只在 decoder cross-attention（k vs N memory）；
      ReEncoder 仍全量计算（O((2N)²)）。

块状注意力掩码（ReEncoder，causal_specials=True 默认）:
    [cls; specials; patches] 的 (2N+1)×(2N+1) 掩码:
        cls(0)           → 全局（见所有人、被所有人见）
        specials(1..N)   → 只见 cls + specials≤i + 全部 patches（前缀链）
        patches(N+1..2N) → 全局（图像无时序，全双向）
    special p 的编码 z_s[p] 不依赖"后面的 special"（前缀稳定性）——每个
    special 就是"前缀 0..p 的表示"（呼应时序前缀问题里 GPT 的位置表示）；
    配合 decoder 侧的前缀选择，k 的语义 = "这条因果链用多长"。
    注意: patches 仍可关注 specials（1_{patch×special} 块），因此
    z_s[:k] 对"encoder 输入裁剪"不变量不成立——将来若想裁 encoder
    输入省算力，需同时屏蔽 patch→special 关注（见 ReEncoder docstring）。

用法
----
    model = SRPhase1V2(dinov2=dinov2_model, num_patches=256, dim=768)
    out = model(pixel_values)                     # 训练（软掩码）
    out = model(pixel_values, hard_mode=True)     # 推理（argmax 硬剪枝）
    loss = out["loss"]; loss.backward()

    自检: python model_v2.py（形状 / 掩码结构 / 梯度流向 / 预算响应 / 硬模式）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# SpecialTokenBank — 特殊 token 池（输入相同，仅位置编码不同）
#   （内联自 v1 model_phase1.py，行为不变）
# ═══════════════════════════════════════════════════════════════

class SpecialTokenBank(nn.Module):
    """每个 patch 位置一个特殊 token：共享可学习向量 + 逐位置可学习位置编码。

    token (1,1,D) 对所有位置/所有图像完全相同；pos (1,N,D) 提供位置区分。
    它们与 patch 组合后输入 ReEncoder，编码器输出中特殊 token 的表示 z_s
    是前缀选择的唯一候选——图像的原始 patch 编码不参与选择、也不进入
    decoder（只作为编码器输入参与聚合）。
    """

    def __init__(self, num_patches: int, dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)          # 共享向量
        self.pos = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)  # 位置编码

    def forward(self, B: int, device: torch.device) -> Tensor:
        return self.token.expand(B, self.num_patches, -1) + self.pos      # (B,N,D)


# ═══════════════════════════════════════════════════════════════
# ReEncoder — pass 2, 组合编码器 [cls; specials; patches]
#   （内联自 v1 model_phase1.py，行为不变）
# ═══════════════════════════════════════════════════════════════

class ReEncoder(nn.Module):
    """对 [cls; 特殊 token; patch] 组合序列做自注意力（带块状掩码）。

    输出中特殊 token 位置的表示 z_s 是前缀选择的唯一候选；patch 位置的
    输出直接丢弃。输入始终是完整 2N+1 序列（训练/推理全量计算），预算
    收益体现在 decoder 的 cross-attention（k vs N memory）。

    块状注意力掩码（causal_specials=True，默认）:
        cls(0)              → 全局（见所有人、被所有人见）
        specials(1..N)      → 只见 cls + specials≤i + 全部 patches（前缀链）
        patches(N+1..2N)    → 全局（图像无时序，全双向）
    即 M[i,j]=1 除 special 行 i 的 special 列 j>i 之外——special p 的编码
    z_s[p] 不依赖任何"后面的 special"（前缀稳定性：直接路径上）。

    注意: 当前实现里 patches 仍会关注 specials（M 的 1_{patch×special} 块），
    因此若将来要"encoder 输入也裁前缀"以省算力，需同时把 patch→special
    关注也屏蔽（patches 只关注 patches+cls），否则 patch 编码会经 specials
    间接变化、破坏 z_s[:k] 的裁剪不变性。

    Input:  x (B, 2N+1, D) = [cls; special_1..N; patch_1..N]
    Output: z (B, 2N+1, D)
    """

    def __init__(self, dim: int = 768, num_patches: int = 256, depth: int = 4,
                 heads: int = 8, mlp_ratio: float = 4.0,
                 causal_specials: bool = True):
        super().__init__()
        self.num_patches = num_patches
        L = 2 * num_patches + 1              # cls + N specials + N patches
        self.pos_embed = nn.Parameter(torch.randn(1, L, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        # 块状注意力掩码（True=允许）: [cls; specials; patches]
        m = torch.ones(L, L, dtype=torch.bool)
        if causal_specials:
            for i in range(1, num_patches + 1):      # special 行 i
                m[i, i + 1:num_patches + 1] = False  # 不能见后面的 special
        self.register_buffer("attn_mask", m)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embed[:, :x.shape[1], :]   # (B, 2N+1, D)
        for layer in self.layers:
            x = layer(x, src_mask=self.attn_mask)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════
# FeatureDecoder — reconstruct the FULL feature map from k tokens
#   （内联自 v1 model_phase1.py，新增可选 memory_key_padding_mask 支持
#     硬模式真实剪枝）
# ═══════════════════════════════════════════════════════════════

class FeatureDecoder(nn.Module):
    """Cross-attention decoder: 256 position queries attend to the re-encoded
    semantic tokens z → F_hat (B,256,D).

    L1 in feature space is far easier to learn than pixel space and is
    semantically meaningful (DINO features carry object/part info).
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 heads: int = 8, depth: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_patches = num_patches
        # learned position queries for the full grid (never pruned)
        self.queries = nn.Parameter(torch.randn(num_patches, dim) * 0.02)
        self.pos_query = nn.Parameter(torch.randn(num_patches, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, z: Tensor,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Args:
            z: (B, Nz, D) re-encoded tokens（软模式 Nz=N+1；硬模式 Nz=max_k+1）
            key_padding_mask: (B, Nz) bool，True=pad 位置（硬模式剪枝后屏蔽）
        Returns:
            F_hat: (B, num_patches, D)
        """
        B = z.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)      # (B,256,D)
        q = q + self.pos_query.unsqueeze(0)
        for layer in self.layers:
            q = layer(q, z, memory_key_padding_mask=key_padding_mask)
        return self.norm(q)


# ═══════════════════════════════════════════════════════════════
# SelectHead — cls → 边界分布 logits（N+1 类，k ∈ {0..N}）
# ═══════════════════════════════════════════════════════════════

class SelectHead(nn.Module):
    def __init__(self, in_dim: int = 768, num_patches: int = 256,
                 hidden: int = 256):
        super().__init__()
        # N+1 类：第 N 类 = "全保留"（修掉旧版永远选不满的 bug）
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_patches + 1),
        )

    def forward(self, cls: Tensor) -> Tensor:
        return self.mlp(cls)                       # (B,N+1) 边界 logits


# ═══════════════════════════════════════════════════════════════
# PrefixMask — 可导软前缀掩码（核心修复）
# ═══════════════════════════════════════════════════════════════

class PrefixMask(nn.Module):
    """边界分布 → 可导软前缀掩码。

    p = softmax(logits) (B,N+1)；k ~ p 是保留个数（0..N）。
    mask_soft[p] = P(k > p) = 1 − cumsum(p)[p]  ← 期望硬掩码，处处可导。

    若 p 退化为 onehot(k)，mask_soft 恰为硬前缀掩码 (positions < k)。
    推理硬化: k = argmax(logits)。
    """

    def forward(self, logits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        p = F.softmax(logits, dim=1)                       # (B,N+1) 边界分布
        cum = torch.cumsum(p, dim=1)                       # (B,N+1) P(k ≤ p)
        mask_soft = 1.0 - cum[:, :-1]                      # (B,N) E[mask[p]]
        k = logits.argmax(dim=-1)                          # (B,) ∈ [0,N]
        return mask_soft, p, k


# ═══════════════════════════════════════════════════════════════
# SRPhase1V2 — 主模型（自包含，不再依赖 model_phase1.py）
# ═══════════════════════════════════════════════════════════════

class SRPhase1V2(nn.Module):

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        dim: int = 768,
        lambda_rate: float = 0.1,
        lambda_ent: float = 0.0,
        reencoder_depth: int = 4,
        decoder_depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        causal_specials: bool = True,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.dim = dim
        self.lambda_rate = lambda_rate
        self.lambda_ent = lambda_ent

        self.pad_token = nn.Parameter(torch.zeros(1, dim))
        self.select_head = SelectHead(in_dim=dim, num_patches=num_patches)
        self.prefix_mask = PrefixMask()
        self.special_bank = SpecialTokenBank(num_patches=num_patches, dim=dim)
        self.re_encoder = ReEncoder(dim=dim, num_patches=num_patches,
                                    depth=reencoder_depth, heads=heads,
                                    mlp_ratio=mlp_ratio,
                                    causal_specials=causal_specials)
        self.decoder = FeatureDecoder(dim=dim, num_patches=num_patches,
                                      depth=decoder_depth, heads=heads,
                                      mlp_ratio=mlp_ratio)

    # ── 预算旋钮：stage-1 设 0.0 全量预热，stage-2 递增退火到目标率 ──
    def set_lambda_rate(self, value: float):
        """预算惩罚权重。0 = 只训重建（k 自由），越大 k 越小。
        推荐: stage-1 预热 0.0 → stage-2 每 k 步 ×1.5，或
              lambda_rate += η·(rate − target) 闭环微调。"""
        self.lambda_rate = value

    def set_entropy(self, value: float):
        """可选边界锐化: >0 时对边界分布 p 加熵惩罚（H 变小 → 分布更尖，
        软/硬模式更一致）。建议 stage-2 后期再开。"""
        self.lambda_ent = value

    def init_reencoder_from_dino(self, num_layers: int = 4):
        """用 DINO 编码器前 num_layers 层 warm-start ReEncoder（结构对齐
        nn.TransformerEncoderLayer ↔ HF DINO layer）。"""
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
    def forward(self, pixel_values: Tensor, hard_mode: bool = False) -> dict:
        """pixel_values: (B,3,224,224) → dict{loss, recon, rate_soft, rate_hard,
        k, mask_soft, p, F_hat}

        hard_mode=True（推理）: k=argmax(logits)，decoder 只吃 k+1 个 token
        （真实剪枝 + key_padding_mask），不产生掩码梯度——仅供评估。
        """
        x = pixel_values                                # (B,3,224,224)
        B = x.shape[0]
        N = self.num_patches

        out = self.dinov2(x)
        feats = out.last_hidden_state                   # (B,257,D)
        cls = feats[:, 0]                               # (B,D)
        patch = feats[:, 1:]                            # (B,N,D)

        # ── 步骤 1: cls (B,D) → 边界分布 p (B,N+1) + 可导软前缀掩码 ──
        logits = self.select_head(cls)                  # (B,N+1)
        mask_soft, p, k = self.prefix_mask(logits)      # (B,N) 软掩码 / (B,N+1) / (B,)
        rate_soft = mask_soft.mean()                    # E[k]/N
        rate_hard = k.float().mean() / N

        # ── 步骤 2: ReEncoder 出 z_s，按掩码选择后进 decoder ──
        specials = self.special_bank(B, x.device)       # (B,N,D)
        enc_in = torch.cat([cls.unsqueeze(1), specials, patch], dim=1)  # (B,2N+1,D)
        z = self.re_encoder(enc_in)                     # (B,2N+1,D)
        z_cls = z[:, 0:1]                               # (B,1,D)
        z_s = z[:, 1:1 + N]                             # (B,N,D) ← 前缀选择对象

        if hard_mode:
            # 真实剪枝: 每图只保留 k_i 个特殊 token（+cls），pad 到 batch 最大
            max_k = int(k.max().item())
            seq = max_k + 1                             # cls + k specials
            dec_in = self.pad_token.expand(B, seq, -1).clone()
            dec_in[:, 0] = z_cls[:, 0]
            pad_mask = torch.ones(B, seq, dtype=torch.bool, device=x.device)
            for i in range(B):
                ki = int(k[i].item())
                if ki > 0:
                    dec_in[i, 1:1 + ki] = z_s[i, :ki]
                    pad_mask[i, 1:1 + ki] = False
                pad_mask[i, 0] = False
            F_hat = self.decoder(dec_in, key_padding_mask=pad_mask)
        else:
            # 软模式: 期望掩码插值 z_s ↔ pad_token，全程可导
            dec_in = torch.cat([
                z_cls,
                mask_soft.unsqueeze(-1) * z_s
                + (1.0 - mask_soft).unsqueeze(-1) * self.pad_token,
            ], dim=1)                                   # (B,N+1,D)
            F_hat = self.decoder(dec_in)                # (B,N,D)

        # ── 步骤 3: 损失（全部按最小化实现）──
        recon = F.l1_loss(F_hat, patch)                 # 主线: 特征重建
        loss = recon + self.lambda_rate * rate_soft     # 预算: 率惩罚
        if self.lambda_ent > 0:                         # 可选: 边界锐化
            H = -(p * torch.log(p.clamp_min(1e-9))).sum(dim=1).mean()
            loss = loss + self.lambda_ent * H

        return {
            "loss": loss,
            "recon": recon,
            "rate_soft": rate_soft,
            "rate_hard": rate_hard,
            "k": k,
            "mask_soft": mask_soft,
            "p": p,
            "F_hat": F_hat,
        }


# ═══════════════════════════════════════════════════════════════
# 自检（python model_v2.py）
#   1. 形状正确性（软/硬模式）
#   2. 梯度流向 SelectHead（旧版断链问题的回归测试）
#   3. 预算响应: lambda_rate 越大 → rate 越小
#   4. init_reencoder_from_dino 可跑
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from types import SimpleNamespace

    torch.manual_seed(0)

    # ── 假的 DINOv2：结构对齐 HF Dinov2Model（供 init_reencoder_from_dino）──
    def _mk_dino_layer(dim: int, mlp_ratio: float = 4.0) -> nn.Module:
        att = nn.Module()
        att.attention = nn.Module()
        att.attention.query = nn.Linear(dim, dim)
        att.attention.key = nn.Linear(dim, dim)
        att.attention.value = nn.Linear(dim, dim)
        att.output = nn.Module()
        att.output.dense = nn.Linear(dim, dim)
        mlp = nn.Module()
        mlp.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        mlp.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        layer = nn.Module()
        layer.norm1 = nn.LayerNorm(dim)
        layer.norm2 = nn.LayerNorm(dim)
        layer.attention = att
        layer.mlp = mlp
        return layer

    class FakeDino(nn.Module):
        def __init__(self, dim: int = 64, n_layers: int = 4, num_patches: int = 16):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([_mk_dino_layer(dim) for _ in range(n_layers)])
            self._np, self._dim = num_patches, dim

        def forward(self, pixel_values: Tensor):
            B = pixel_values.shape[0]
            return SimpleNamespace(
                last_hidden_state=torch.randn(B, self._np + 1, self._dim))

    N, D = 16, 64
    dino = FakeDino(dim=D, num_patches=N)
    model = SRPhase1V2(dino, num_patches=N, dim=D, lambda_rate=0.0,
                       reencoder_depth=2, decoder_depth=2)
    model.init_reencoder_from_dino(2)
    x = torch.randn(2, 3, 224, 224)

    # ── 1. 形状 ──
    out = model(x)
    assert out["F_hat"].shape == (2, N, D), out["F_hat"].shape
    assert out["mask_soft"].shape == (2, N)
    assert out["p"].shape == (2, N + 1)
    assert out["k"].shape == (2,)
    assert 0.0 <= out["rate_soft"].item() <= 1.0
    assert (out["k"] >= 0).all() and (out["k"] <= N).all()
    assert out["mask_soft"].min() >= 0.0 and out["mask_soft"].max() <= 1.0
    print(f"[ok] shapes: F_hat{tuple(out['F_hat'].shape)} "
          f"mask{tuple(out['mask_soft'].shape)} p{tuple(out['p'].shape)} "
          f"k={out['k'].tolist()} rate_soft={out['rate_soft'].item():.3f}")

    # ── 1.5 块状注意力掩码结构（causal specials）──
    am = model.re_encoder.attn_mask                  # (2N+1, 2N+1) bool
    assert am.shape == (2 * N + 1, 2 * N + 1)
    assert am[0].all()                               # cls 全局
    assert am[N + 1:].all()                          # patches 全局（全双向）
    for i in range(1, N + 1):
        assert am[i, :i + 1].all()                   # special i 见 cls + specials≤i
        assert not am[i, i + 1:N + 1].any()          # 不见后面的 special
        assert am[i, N + 1:].all()                   # special i 见全部 patches
    # 关掉 causal 时掩码全 1（回退旧行为）
    m_free = SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                        causal_specials=False, reencoder_depth=2, decoder_depth=2)
    assert m_free.re_encoder.attn_mask.all()
    print(f"[ok] block mask: causal_specials=True 结构正确, "
          f"causal_specials=False 全 1 回退")

    # ── 2. 梯度流向 SelectHead（旧版断链的回归测试）──
    out["loss"].backward()
    g = model.select_head.mlp[-1].weight.grad
    assert g is not None and g.abs().sum() > 0, "SelectHead 收不到梯度！"
    print(f"[ok] recon 梯度直达 SelectHead: |grad|={g.abs().sum().item():.4f}")

    # ── 3. 预算响应: λ=0 vs λ=10 各训 30 步，rate 应显著下降 ──
    def train_rates(lmb: float, steps: int = 30) -> float:
        m = SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                       lambda_rate=lmb, reencoder_depth=2, decoder_depth=2)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        r = None
        for _ in range(steps):
            opt.zero_grad()
            o = m(x)
            o["loss"].backward()
            opt.step()
            r = o["rate_soft"].item()
        return r

    r0 = train_rates(0.0)
    r1 = train_rates(10.0)
    print(f"[ok] 预算响应: λ=0 → rate={r0:.3f}, λ=10 → rate={r1:.3f}")
    assert r1 < r0, f"预算惩罚失效: {r0} → {r1}"

    # ── 4. 硬模式（真实剪枝）──
    model.eval()
    with torch.no_grad():
        out_h = model(x, hard_mode=True)
    assert out_h["F_hat"].shape == (2, N, D)
    assert (out_h["k"] >= 0).all() and (out_h["k"] <= N).all()
    print(f"[ok] hard_mode: k={out_h['k'].tolist()} rate_hard="
          f"{out_h['rate_hard'].item():.3f}")

    print("\nALL CHECKS PASSED")
