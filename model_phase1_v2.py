"""
SR-Diffusion Phase 1 v2 — 整体重构（按损失函数提取思想）
=================================================================

设计思想（非线性 SVD 类比）:
  SelectHead(cls) 把全局描述子映射为 N 个候选 token 位置上的"能量谱"
  p = softmax(scores)（类比 SVD 奇异值归一化谱；README 的 SVD 预处理就是
  按奇异值能量做 99% 截断）。按累计能量截断 ρ 自动决定保留多少个 token：
  谱越尖（信息集中）→ k 越小；谱越平（信息分散）→ k 越大。
  即：让神经网络近似拟合一个"非线性 SVD"——能量谱的形状由图像内容决定，
  而不是固定维度瓶颈。softmax 的峰值位置（argmax）永远在保留集合内。

三步实现:
  1. cls (B,D) --SelectHead--> scores (B,N) --softmax--> p (B,N) 能量谱
  2. EnergyGate: 累计能量 ≥ ρ 的前 k 个位置 → 0/1 掩码（STE 可微），
     作用于 z_s = z[:, 1:1+N]（ReEncoder 输出的特殊 token 表示）→ decoder
  3. 损失（全部按最小化实现，未实现项标注 TODO）:
       L_recon = L1(F_hat, F_patch)       — 最小化特征重建误差（主线，驱动
                                             能量谱学会"截断 ρ 仍能重建"）
       L_rate  = λ_rate · H(p)            — 最小化能量谱熵（谱越尖 → 越少
                                             token 越过 ρ 截断；默认 λ_rate=0，
                                             预算主旋钮是 ρ，此项作软正则备选）
       L_ent   = -λ_ent · H(平均选择)     — 反塌缩：最小化负熵，防止选择退化
                                             为"所有图都选同一批位置"

  注：v1 里 rate = mask.mean() 的朴素率损失在 softmax 下是恒等零梯度
      （softmax 对整体平移不变、Σp≡1），故 v2 改用 H(p) 作为率项。

未实现（TODO，后续按模块化嵌入）:
  [TODO] ρ 两阶段退火：stage-1 ρ=1.0（全保留，先教重建）→ stage-2 退火到目标
  [TODO] 温度退火 T（当前固定 1.0；退火可让能量谱从软变硬，配合 STE）
  [TODO] per-sample z-score 校准（v1 里"排序与计数分离、防军备竞赛"的技巧；
         若 recon 与率项在同一组 logits 上打架时再启用）
  [TODO] 预算信任域（v1 BudgetTrustRegion：KL 限速、k 守卫、recon 力放大）
  [TODO] k 硬守卫 [k_min, k_max]（能量截断天然给 k；需要硬界时在此加 clamp）
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
# 步骤 1 的"cls (B,D) → (B,N)"。softmax（能量谱化）在 EnergyGate 里做。
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
# EnergyGate — SVD 式能量截断 top-k（可微 one-hot 选择）
# ═══════════════════════════════════════════════════════════════

class EnergyGate(nn.Module):
    """p = softmax(scores/T) 视作候选 token 的归一化能量谱（Σ=1）。

    按 p 降序累计，保留"累计质量首次 ≥ ρ"的前 k 个位置：
        k_i = min{ k : Σ_{j≤k} p_sorted[j] ≥ ρ }，且 k_i ≥ 1
    k 完全由谱形状决定（内容自适应）：谱越尖 k 越小，谱越平 k 越大。
    softmax 的峰值位置（argmax，排序名次 0）恒在保留集合内。

    可微性（STE）：hard 为 0/1 掩码（前向），soft = p（梯度路径）：
        mask = p + (hard - p).detach()   → 前向是 one-hot 式硬选择，
                                           反向梯度经 p 流入 SelectHead。

    rho=1.0 时保留全部（等价 v1 stage-1 的全保留预热）。
    """

    def __init__(self, rho: float = 0.99, T: float = 1.0):
        super().__init__()
        self.rho = rho
        self.T = T

    def forward(self, scores: Tensor) -> Tuple[Tensor, Tensor]:
        B, N = scores.shape
        p = F.softmax(scores / self.T, dim=-1)             # (B,N) 能量谱, Σ=1
        order = torch.argsort(p, dim=1, descending=True)   # (B,N) 降序排列的位置
        sorted_p = p.gather(1, order)                      # (B,N) 降序能量
        cum = torch.cumsum(sorted_p, dim=1)                # (B,N) 累计质量
        k = (cum < self.rho).sum(dim=1) + 1                # (B,) 首次累计 ≥ ρ 的个数（≥1）
        rank = torch.argsort(order, dim=1)                 # (B,N) 每个位置的排序名次
        hard = (rank < k.unsqueeze(1)).float()             # (B,N) 0/1: 名次 < k_i 的位置保留
        mask = p + (hard - p).detach()                     # (B,N) STE
        return mask, p


# ═══════════════════════════════════════════════════════════════
# SRPhase1V2 — 主模型（复用 v1 的 SpecialTokenBank / ReEncoder / FeatureDecoder）
# ═══════════════════════════════════════════════════════════════

class SRPhase1V2(nn.Module):
    """自适应 token 预算 tokenizer（v2，纯 cls 全局选择 + 能量截断）。

    Args:
        dinov2: frozen Dinov2Model (transformers)。输入 224×224, patch 14
                → 256 patch tokens + cls。
        num_patches: 候选 token 数（224²/patch14 → 256）
        dim: DINO hidden size（base=768, giant=1536）
        rho: 能量截断阈值（0,1]，默认 0.99；1.0 = 全保留（stage-1 预热）
        T: softmax 温度（默认 1.0；退火见 TODO）
        lambda_rate: 能量谱熵惩罚权重（默认 0.0——预算主旋钮是 ρ）
        lambda_ent: 反塌缩熵惩罚权重（默认 0.01，同 v1）
    """

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        dim: int = 768,
        rho: float = 0.99,
        T: float = 1.0,
        lambda_rate: float = 0.0,
        lambda_ent: float = 0.01,
    ):
        super().__init__()
        self.dinov2 = dinov2
        for p in self.dinov2.parameters():
            p.requires_grad_(False)
        self.dinov2.eval()

        self.num_patches = num_patches
        self.dim = dim
        self.rho = rho
        self.T = T
        self.lambda_rate = lambda_rate
        self.lambda_ent = lambda_ent

        self.select_head = SelectHead(in_dim=dim, num_patches=num_patches)
        self.gate = EnergyGate(rho=rho, T=T)
        self.special_bank = SpecialTokenBank(num_patches=num_patches, dim=dim)
        self.re_encoder = ReEncoder(dim=dim, num_patches=num_patches)
        self.decoder = FeatureDecoder(dim=dim, num_patches=num_patches)

    # ── 预算旋钮：ρ（stage-1 设 1.0 全保留，stage-2 退火到目标值）──
    def set_rho(self, rho: float):
        """设置能量截断阈值。[TODO] 两阶段退火调度见模块头。"""
        self.rho = rho
        self.gate.rho = rho

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
        """x (B,3,224,224) → 能量谱选择 + 特征重建。

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

        # ── 步骤 1: cls (B,D) → 能量谱 p (B,N) ──
        scores = self.select_head(cls)                  # (B,N)
        mask, p = self.gate(scores)                     # mask (B,N) STE, p (B,N) 能量谱

        # ── 步骤 2: ReEncoder 出 z_s，按 mask 选择后进 decoder ──
        specials = self.special_bank(B, x.device)       # (B,N,D)
        enc_in = torch.cat([cls.unsqueeze(1), specials, patch], dim=1)  # (B,2N+1,D)
        z = self.re_encoder(enc_in)                     # (B,2N+1,D)
        z_cls = z[:, 0:1]                               # (B,1,D)
        z_s = z[:, 1:1 + N]                             # (B,N,D) ← top-k 唯一候选
        # [TODO] 物理剪枝：当前零填充 N+1 槽位（train/eval 一致），
        #        "只算 k 个 token"的算力收益待实现（见模块头 TODO）。
        dec_in = torch.cat([z_cls, z_s * mask.unsqueeze(-1)], dim=1)   # (B,N+1,D)
        F_hat = self.decoder(dec_in)                    # (B,N,D)

        # ── 步骤 3: 损失（全部按最小化实现）──
        # L_recon: 最小化特征重建误差（主线；驱动能量谱学会"截断 ρ 仍能重建"）
        recon = F.l1_loss(F_hat, patch)                 # scalar
        # L_rate: 最小化能量谱熵 → 谱越尖、越过 ρ 截断的 token 越少。
        #         （v1 的 rate=mask.mean() 在 softmax 下是零梯度恒等式，见模块头注）
        rate = -(p * torch.log(p + 1e-8)).sum(dim=1).mean()   # scalar H(p)
        # L_ent: 反塌缩——最小化负熵；avg_sel 为跨 batch 平均选择，
        #        退化"所有图选同一批位置"时 H→0，此项惩罚它
        avg_sel = mask.mean(dim=0)                      # (N,)
        ent = -(avg_sel * torch.log(avg_sel + 1e-8) +
                (1 - avg_sel) * torch.log(1 - avg_sel + 1e-8)).mean()  # scalar
        loss = recon + self.lambda_rate * rate - self.lambda_ent * ent
        # [TODO] 其余损失/机制（信任域、z-score 校准、k 守卫、温度退火、
        #        硬/软输入混合、recon 力放大）见模块头 TODO 列表。

        with torch.no_grad():
            hard = (mask > 0.5).float()                 # (B,N) 前向即硬选择
            k_used = hard.sum(dim=1)                    # (B,)
            def _t(v):
                return torch.as_tensor(v, device=x.device)
            stats = {
                "recon_l1": _t(recon),
                "rate_ent": _t(rate),
                "entropy": _t(ent),
                "rho": _t(self.rho),
                "k_used_mean": _t(k_used.float().mean()),
                "k_used_min": _t(k_used.float().min()),
                "k_used_max": _t(k_used.float().max()),
            }

        return {"loss": loss, "F_hat": F_hat, "mask": mask, "p": p, "stats": stats}
