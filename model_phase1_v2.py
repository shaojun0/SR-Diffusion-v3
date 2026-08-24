"""
SR-Diffusion Phase 1 v2 — 整体重构（按损失函数提取思想）
=================================================================

设计思想（修正版：softmax 能量谱 → sigmoid 伯努利门控）:
  softmax 方案的缺陷（上版实测/推导）:
    · softmax 对整体平移不变、Σp≡1 → rate = mask.mean() 是零梯度恒等式，
      率损失推不动 SelectHead；
    · 累计能量截断 ρ 引入了与"逐位置重要性"无关的全局耦合。
  修正: SelectHead(cls) 输出 scores (B,N)，逐位置过 sigmoid 得到独立的
  保留概率 p = σ(scores) ∈ (0,1)^N（伯努利门控，无归一化耦合）。此时:
    · rate = mask.mean() 有真实梯度（∂mean(σ)/∂scores = σ' > 0）；
    · 阈值 0.5 是 sigmoid 的自然二值化点（"比保留更倾向保留"）；
    · k=0 安全: dec_in 恒含 z_cls（保底牌），一个候选都不选 decoder
      仍能重建 → 无需峰值锚定/k≥1 防御；k=0 可自愈（重建变差 →
      recon 梯度经 STE 把 p 拉回）。注意: 此简化依赖 STE 保持梯度路径。
  即: 神经网络学习"这张图哪些空间位置值得保留"，budget 由阈值和
  保留概率的形状共同决定（内容自适应），而非固定维度瓶颈。

三步实现:
  1. cls (B,D) --SelectHead--> scores (B,N) --sigmoid--> p (B,N) 保留概率
  2. BernoulliGate: hard = (p > threshold) 二值化 + STE 可微 0/1 掩码
     （k=0 允许，cls 是保底牌）；作用于 z_s = z[:, 1:1+N]（ReEncoder
     输出的特殊 token 表示）→ decoder
  3. 损失（全部按最小化实现，未实现项标注 TODO）:
       L_recon = L1(F_hat, F_patch)       — 最小化特征重建误差（主线：驱动
                                             保留概率学会"少留也能重建"）
       L_rate  = λ_rate · mask.mean()     — 最小化保留 token 比例（sigmoid 下
                                             有真实梯度；默认 0.1，同 v1）
       L_ent   = -λ_ent · H(平均选择)     — 反塌缩：最小化负熵，防止选择退化
                                             为"所有图都选同一批位置"

未实现（TODO，后续按模块化嵌入）:
  [TODO] 阈值两阶段退火：stage-1 threshold=0（全保留，先教重建）→
         stage-2 退火到 0.5（对应 v1 的 fixed_tau 两阶段思想）
  [TODO] 温度退火 T（当前固定 1.0；T 控制 sigmoid 的软硬程度）
  [TODO] 可学习阈值 τ（v1 RateHead 思路：阈值随内容变化而非固定 0.5；
         若固定阈值压不动再启用）
  [TODO] per-sample z-score 校准（v1 里"排序与计数分离、防军备竞赛"技巧；
         若 recon 与率项在同一组 logits 上打架时再启用）
  [TODO] 预算信任域（v1 BudgetTrustRegion：KL 限速、k 守卫、recon 力放大）
  [TODO] k 硬守卫 [k_min, k_max]（当前无下界；若 k=0 瞬态重建不可接受再加）
  [TODO] sigmoid 饱和陷阱：scores 整体偏负时 σ'≈0，SelectHead 梯度被衰减
         （实测 σ'(-5)≈0.0067 → 梯度弱 ~125×），k=0 自愈变慢——若训练中
         k 长期为 0，启用 v1 的 raw clamp + 排斥正则或 z-score 校准
  [TODO] 物理剪枝（当前零填充 N+1 槽位、train/eval 一致；"只算 k 个 token"
          的算力收益待实现，对应 v1 的 _gather_selected / hard_mode）
  [TODO] 训练期硬/软输入混合（v1 hard_input_prob，消除 decoder 记忆分布差）
  [TODO] train_phase1_v2.py（两阶段调度 + 数据管线，复刻 train_phase1.py）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from model_phase1 import SpecialTokenBank, ReEncoder, FeatureDecoder


# ═══════════════════════════════════════════════════════════════
# SelectHead — cls (B,D) → scores (B,N)：全局描述子 → 位置重要性分数
# ═══════════════════════════════════════════════════════════════
# 步骤 1 的"cls (B,D) → (B,N)"。sigmoid（概率化）在 BernoulliGate 里做。
# 注意：cls 是全局描述子，只能学"区域级"重要性（如人脸区域重要），
# 感知不到具体 patch 内容——这是纯 cls 选择的已知局限（v1 的 ScoreHead
# 作用在 z_s 上能看到逐位置内容；如需恢复可把输入换成 [cls; z_s]）。

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
    """p = σ(scores/T) ∈ (0,1)^N：每个候选位置的独立保留概率。

    二值化: hard_i = (p_i > threshold)，阈值默认 0.5（sigmoid 自然中点）。

    k=0 是允许的: dec_in 恒含 z_cls（ReEncoder 的 cls 输出），即使一个
    候选 token 都没选，decoder 仍能靠 cls 重建（保底牌）——因此不需要
    峰值锚定/k≥1 的硬性防御；若 k=0 导致重建变差，recon 梯度会经 STE
    自动把 p 拉回，k=0 只是可自愈的过渡态。

    可微性（STE）: mask = p + (hard - p).detach()   → 前向是 one-hot 式
      硬选择，反向梯度经 p 流入 SelectHead。
      ⚠ 不能直接返回 hard: (p > threshold) 是比较运算、无 grad_fn，
      mask 会 requires_grad=False → recon/rate/ent 的梯度全部到不了
      SelectHead（"路径静默死掉"，v1 ThresholdGate 实测过的坑）。
    threshold=0 时全保留（等价 v1 stage-1 预热，见模块头 TODO）。
    """

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
    """自适应 token 预算 tokenizer（v2，纯 cls 全局选择 + sigmoid 伯努利门控）。

    Args:
        dinov2: frozen Dinov2Model (transformers)。输入 224×224, patch 14
                → 256 patch tokens + cls。
        num_patches: 候选 token 数（224²/patch14 → 256）
        dim: DINO hidden size（base=768, giant=1536）
        threshold: sigmoid 二值化阈值（默认 0.5；0 = 全保留预热，见 TODO）
        T: sigmoid 温度（默认 1.0；退火见 TODO）
        lambda_rate: 保留比例惩罚权重（默认 0.1，同 v1）
        lambda_ent: 反塌缩熵惩罚权重（默认 0.01，同 v1）
    """

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
        for p in self.dinov2.parameters():
            p.requires_grad_(False)
        self.dinov2.eval()

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

    # ── 从冻结 DINO 热启动 ReEncoder（同 v1，避免从零学）──
    def init_reencoder_from_dino(self, num_layers: int = 4):
        """Copy weights from pretrained DINO encoder layers into the ReEncoder.

        DINO layer layout (transformers 5.x):
            norm1 → attention(query/key/value) → output.dense → layer_scale1
                 → norm2 → mlp(fc1/fc2) → layer_scale2
        TransformerEncoderLayer layout:
            norm1 → self_attn(in_proj / out_proj) → linear1 → linear2 → norm2
        Shapes match 1:1, so we copy parameter by parameter.
        """
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
        """x (B,3,224,224) → sigmoid 伯努利选择 + 特征重建。

        Returns dict with loss + stats.
        """
        x = pixel_values                                # (B,3,224,224)
        B = x.shape[0]
        N = self.num_patches

        # pass 1: 冻结 DINO
        out = self.dinov2(x)
        feats = out.last_hidden_state                   # (B,257,D)
        cls = feats[:, 0]                               # (B,D)
        patch = feats[:, 1:]                            # (B,N,D)

        # ── 步骤 1: cls (B,D) → 保留概率 p (B,N) ──
        scores = self.select_head(cls)                  # (B,N)
        top_scores = nn.functional.softmax(scores,dim=1)
        mask = scores>torch.max(top_scores,dim=1,keepdim=True)
        # ── 步骤 2: ReEncoder 出 z_s，按 mask 选择后进 decoder ──
        specials = self.special_bank(B, x.device)       # (B,N,D)
        enc_in = torch.cat([cls.unsqueeze(1), specials, patch], dim=1)  # (B,2N+1,D)
        z = self.re_encoder(enc_in)                     # (B,2N+1,D)
        z_cls = z[:, 0:1]                               # (B,1,D)
        z_s = z[:, 1:1 + N]                             # (B,N,D) ← 掩码选择对象（top-k 候选）
        # [TODO] 物理剪枝：当前零填充 N+1 槽位（train/eval 一致），
        #        "只算 k 个 token"的算力收益待实现（见模块头 TODO）。
        dec_in = torch.cat([z_cls, z_s * mask.unsqueeze(-1)], dim=1)   # (B,N+1,D)
        F_hat = self.decoder(dec_in)                    # (B,N,D)

        # ── 步骤 3: 损失（全部按最小化实现）──
        # L_recon: 最小化特征重建误差（主线；驱动保留概率学会"少留也能重建"）
        recon = F.l1_loss(F_hat, patch)                 # scalar,0~1
        # L_rate: 最小化保留 token 比例。sigmoid 下 mask.mean() 有真实梯度
        #         （∂mean(σ)/∂scores = σ'/N > 0）——softmax 方案下这是
        #         零梯度恒等式，正是弃用 softmax 的原因（见模块头）。

        rate = mask.mean()                              # scalar 保留比例（前向 = k/N）
        # L_ent: 反塌缩——最小化负熵；avg_sel 为跨 batch 平均选择，
        #        退化"所有图选同一批位置"时 H→0，此项惩罚它
        avg_sel = mask.mean(dim=0)                      # (N,)
        ent = -(avg_sel * torch.log(avg_sel + 1e-8) +
                (1 - avg_sel) * torch.log(1 - avg_sel + 1e-8)).mean()  # scalar
        loss = recon + self.lambda_rate * rate - self.lambda_ent * ent
        # [TODO] 其余损失/机制（阈值退火、可学习 τ、z-score 校准、信任域、
        #        k 守卫、温度退火、硬/软输入混合、recon 力放大）见模块头 TODO。

        with torch.no_grad():
            hard = (mask > 0.5).float()                 # (B,N) 前向即硬选择
            k_used = hard.sum(dim=1)                    # (B,)
            def _t(v):
                return torch.as_tensor(v, device=x.device)
            stats = {
                "recon_l1": _t(recon),
                "rate": _t(rate),
                "entropy": _t(ent),
                "threshold": _t(self.threshold),
                "k_used_mean": _t(k_used.float().mean()),
                "k_used_min": _t(k_used.float().min()),
                "k_used_max": _t(k_used.float().max()),
            }

        return {"loss": loss, "F_hat": F_hat, "mask": mask, "p": p, "stats": stats}
