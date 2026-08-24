"""
SR-Diffusion Phase 1 v2.5 — sigmoid 伯努利门控 + 拉格朗日预算约束
=================================================================

沿革（v2 → v2.5）:
  · v2（daf8bf1）: SelectHead(cls) → sigmoid 保留概率 → 固定阈值 0.5 二值化
    + STE。rate 靠手调 λ_rate 惩罚——想打到指定预算必须网格搜索（v1 同病）。
  · v2.5: 保留 v2 的干净架构（一个 SelectHead + sigmoid + STE），把预算控制
    换成 v3 的拉格朗日求解器（LagrangianBudget，从 model_phase1_v3 复用，
    单一来源）——预算从"手调 λ_rate 碰运气"变成"对偶变量 λ 自动收敛到
    KKT 乘子"。

问题形式化（拉格朗日约束优化）:
    min_θ  L_recon(θ)
    s.t.   R(θ) ≤ R_target          (R = 送入 decoder 的 token 比例 k/N)

    L(θ, λ) = L_recon + λ·(R_soft − R_target_eff) + (ρ/2)·ReLU(R_soft − R_target_eff)²
    原步（外部优化器）:  θ ← θ − η·∇L，λ 视为常数（detach）
    对偶步（每 train step）: λ ← clamp(λ + η_λ·(R_ema − R_target_eff), λ_min, λ_max)
    R_target_eff: 1.0 → R_target 线性退火（stage-2 平滑过渡）
    收敛时: R ≈ R_target，λ* = KKT 乘子 = "再省一个 token 的边际还原代价"

损失（全部按最小化）:
    L_recon = L1(F_hat, F_patch)        — 主线（驱动选择学会"少留也能重建"）
    L_lag   = λ·(R_soft − R_target_eff) + (ρ/2)·ReLU(R_soft − R_target_eff)²
                                        — 预算约束（λ 自适应，见 LagrangianBudget）
    L_ent   = −λ_ent·H(平均选择)         — 反塌缩（防所有图选同一批位置）
    stage-1 预热: 只开 L_recon + L_ent（threshold=0 全保留，rate 项关闭）

与 v3 的差异（v2.5 = v2 架构 + v3 预算器）:
    · v3:   ScoreHead(z_s) 内容感知 + RateHead 每图阈值 τ + z-score 校准
    · v2.5: 纯 cls 全局选择 + 固定阈值 0.5（无 RateHead/τ）——拉格朗日惩罚
      梯度经 sigmoid 直接推 scores（σ′>0 处处可导），一个头 + 一个对偶变量。
      分工: recon 决定"删哪些位置"（逐位置梯度），λ 决定"删多少"（全局压力）。
    · 已知局限（继承 v2）: 纯 cls 只能学"区域级"重要性（如人脸区域），
      感知不到具体 patch 内容——如需恢复可换 ScoreHead(z_s)（v3 的 select_on）。

弃用思路备忘（用户 ea8352f 草稿，已进 git 历史）:
  softmax + argmax 切点 + 前缀掩码 (positions < argmax) + BCE(softmax, onehot(argmax−1)):
    · 前缀掩码是比较运算 → 无梯度（"路径静默死掉"）；argmax 本身不可微；
    · 前缀 = 空间序列左段，不是按内容选 top-k；且峰值位置永不保留；
    · BCE 的 label = argmax−1 是移动靶（把峰往左推一格）——无 R_target、无 λ，
      不是预算约束而是"永远缩一格"的漂移压力；
    · 结论: "一个标量决定预算"的直觉正确，但正确实现是 v3 的可微 τ
      （或 v2.5 的 λ + 固定阈值），不是 argmax。

TODO（已知边界，不影响主线）:
    [TODO] 物理剪枝的算力收益（当前零填充 N+1 槽位；hard_mode 已能物理剪枝）
    [TODO] 每样本 λ（per-image multiplier；当前全局标量 = 标准平均率形式）
    [TODO] 训练期硬/软输入混合（v1/v3 的 hard_input_prob，decoder 记忆对齐）
    [TODO] 若固定阈值压不动预算，升级为 v3 的可学习 τ（RateHead + TTSA）
    [TODO] sigmoid 饱和陷阱（scores 偏负时 σ′≈0、梯度弱；可加 v1 的 raw clamp）
    [TODO] 对偶步与梯度累积步长对齐（当前每 forward 更新，同 v3）
    [TODO] train_phase1_v2.py（两阶段调度 + 数据管线，复刻 train_phase1.py）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from model_phase1 import SpecialTokenBank, ReEncoder, FeatureDecoder
from model_phase1_v3 import LagrangianBudget        # 复用 v3 的拉格朗日求解器


# ═══════════════════════════════════════════════════════════════
# SelectHead — cls (B,D) → scores (B,N)：全局描述子 → 位置重要性分数
# ═══════════════════════════════════════════════════════════════

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

    二值化: hard_i = (p_i > threshold)。threshold=0 时全保留（stage-1 预热）。
    k=0 是允许的: dec_in 恒含 z_cls（ReEncoder 的 cls 输出，保底牌），
    一个候选都不选 decoder 仍能重建——无需 k≥1 防御；k=0 可自愈
    （重建变差 → recon 梯度经 STE 把 p 拉回）。

    可微性（STE）: mask = p + (hard − p).detach() → 前向 one-hot 式硬选择，
    反向梯度经 p 流入 SelectHead。
    ⚠ 不能直接返回 hard: (p > threshold) 是比较运算、无 grad_fn，mask 会
    requires_grad=False → recon/lag/ent 的梯度全部到不了 SelectHead
    （"路径静默死掉"，v1 ThresholdGate 实测过的坑）。
    """

    def __init__(self, threshold: float = 0.5, T: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.T = T

    def forward(self, scores: Tensor) -> Tuple[Tensor, Tensor]:
        B, N = scores.shape
        p = torch.sigmoid(scores / self.T)               # (B,N) 保留概率
        hard = (p > self.threshold).float()              # (B,N) 0/1 阈值二值化
        mask = p + (hard - p).detach()                   # (B,N) STE
        return mask, p


# ═══════════════════════════════════════════════════════════════
# SRPhase1V2 — 主模型（v2 架构 + 拉格朗日预算控制器）
# ═══════════════════════════════════════════════════════════════

class SRPhase1V2(nn.Module):
    """自适应 token 预算 tokenizer（v2.5，纯 cls 全局选择 + 拉格朗日预算）。

    Args:
        dinov2: frozen Dinov2Model (transformers)。输入 224×224, patch 14
                → 256 patch tokens + cls。
        num_patches: 候选 token 数（224²/patch14 → 256）
        dim: DINO hidden size（base=768, giant=1536）
        threshold: stage-2 的 sigmoid 二值化阈值（默认 0.5；set_stage 控制）
        T: sigmoid 温度（默认 1.0；退火见 TODO）
        lambda_ent: 反塌缩熵惩罚权重（默认 0.01）
        lambda_rate: 无 lagrangian 时的固定 λ_rate 惩罚（ablation，默认 0）
        use_lagrangian: 启用拉格朗日预算器（默认 True）
        rate_target: 预算 R_target（默认 0.25）
        **lag_kwargs: 传给 LagrangianBudget 的超参（eta_lambda/rho/anneal 等）

    用法（两阶段，同 v1/v3）:
        model.set_stage(1)  → threshold=0 全保留、rate 项关闭（预热教重建）
        model.set_stage(2)  → threshold=threshold、拉格朗日开启（预算生效）
    """

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        dim: int = 768,
        threshold: float = 0.5,
        T: float = 1.0,
        lambda_ent: float = 0.01,
        lambda_rate: float = 0.0,
        use_lagrangian: bool = True,
        rate_target: float = 0.25,
        **lag_kwargs,
    ):
        super().__init__()
        self.dinov2 = dinov2
        for p in self.dinov2.parameters():
            p.requires_grad_(False)
        self.dinov2.eval()

        self.num_patches = num_patches
        self.dim = dim
        self.threshold = threshold          # stage-2 的目标阈值
        self.T = T
        self.lambda_ent = lambda_ent
        self.lambda_rate = lambda_rate
        self.rate_term_on = False           # stage-1 关闭，set_stage(2) 打开

        self.lagrangian = (LagrangianBudget(rate_target=rate_target, **lag_kwargs)
                           if use_lagrangian else None)

        self.select_head = SelectHead(in_dim=dim, num_patches=num_patches)
        # 初始 threshold=0 → 全保留（stage-1 预热语义；set_stage 切换）
        self.gate = BernoulliGate(threshold=0.0, T=T)
        self.special_bank = SpecialTokenBank(num_patches=num_patches, dim=dim)
        self.re_encoder = ReEncoder(dim=dim, num_patches=num_patches)
        self.decoder = FeatureDecoder(dim=dim, num_patches=num_patches)

    # ── 两阶段控制 ──
    def set_stage(self, stage: int):
        """stage=1: threshold=0 全保留预热、rate 项关闭（纯重建）。
        stage=2: threshold=self.threshold、拉格朗日开启（λ 从头学）。"""
        if stage == 1:
            self.gate.threshold = 0.0
            self.rate_term_on = False
        else:
            self.gate.threshold = self.threshold
            self.rate_term_on = True
            if self.lagrangian is not None:
                self.lagrangian.reset()

    def set_threshold(self, threshold: float):
        """设置 stage-2 的 sigmoid 二值化阈值（当前生效值由 set_stage 切换）。"""
        self.threshold = float(threshold)
        if self.rate_term_on:
            self.gate.threshold = self.threshold

    def set_rate_target(self, value: float):
        """设置预算 R_target（拉格朗日约束的目标保留率）。"""
        if self.lagrangian is not None:
            self.lagrangian.rate_target = float(value)

    @classmethod
    def build_model(cls, dinov2: nn.Module, num_patches: int = 256, dim: int = 768,
                    threshold: float = 0.5, T: float = 1.0,
                    lambda_ent: float = 0.01, rate_target: float = 0.25,
                    init_reencoder: bool = True, **lag_kwargs) -> "SRPhase1V2":
        """Build v2.5 model: LagrangianBudget + optional DINO warm-start."""
        model = cls(dinov2, num_patches=num_patches, dim=dim,
                    threshold=threshold, T=T, lambda_ent=lambda_ent,
                    rate_target=rate_target, **lag_kwargs)
        if init_reencoder:
            model.init_reencoder_from_dino()
        return model

    # ── 从冻结 DINO 热启动 ReEncoder（同 v1/v3，避免从零学）──
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
    def forward(self, pixel_values: Tensor, hard_mode: bool = False) -> dict:
        """x (B,3,224,224) → sigmoid 选择 + 拉格朗日预算约束 + 特征重建。

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

        # ── 步骤 1: cls (B,D) → 保留概率 p (B,N) → STE 掩码 ──
        scores = self.select_head(cls)                  # (B,N)
        mask, p = self.gate(scores)                     # mask (B,N) STE, p (B,N) 保留概率
        R_soft = mask.mean()                            # scalar 可微保留率（前向 = k/N）
        hard = (mask > 0.5).float()                     # (B,N) 0/1
        R_hard = hard.mean()                            # scalar 真实保留率（报告用）

        # ── 步骤 2: ReEncoder 出 z_s，按 mask 选择后进 decoder ──
        specials = self.special_bank(B, x.device)       # (B,N,D)
        enc_in = torch.cat([cls.unsqueeze(1), specials, patch], dim=1)  # (B,2N+1,D)
        z = self.re_encoder(enc_in)                     # (B,2N+1,D)
        z_cls = z[:, 0:1]                               # (B,1,D)
        z_s = z[:, 1:1 + N]                             # (B,N,D) ← 掩码选择对象
        # z[:, 1+N:] 是 patch 位置的输出，直接丢弃（同 v1/v2）
        if hard_mode:
            sel, _ = self._gather_selected(z_s, hard)   # 推理物理剪枝
            dec_in = torch.cat([z_cls, sel], dim=1)     # (B, k+1, D)
        else:
            # [TODO] 训练期物理剪枝（当前零填充 N+1 槽位，train/eval 一致）
            dec_in = torch.cat([z_cls, z_s * mask.unsqueeze(-1)], dim=1)  # (B,N+1,D)
        F_hat = self.decoder(dec_in)                    # (B,N,D)

        # ── 步骤 3: 损失（全部按最小化）──
        # L_recon: 主线——驱动保留概率学会"少留也能重建"
        recon = F.l1_loss(F_hat, patch)                 # scalar
        # L_ent: 反塌缩——退化"所有图选同一批位置"时 H→0，此项惩罚它
        avg_sel = mask.mean(dim=0)                      # (N,)
        ent = -(avg_sel * torch.log(avg_sel + 1e-8) +
                (1 - avg_sel) * torch.log(1 - avg_sel + 1e-8)).mean()  # scalar

        if self.rate_term_on and self.lagrangian is not None:
            # L_lag: 拉格朗日惩罚（λ 对原步是常数）+ 对偶步更新 λ
            if self.training:
                self.lagrangian.tick()
            target = self.lagrangian.current_target()
            lag_pen = self.lagrangian.penalty(R_soft, target)
            if self.training:
                # 对偶步用平滑的 R_soft（detach）+ EMA，见 LagrangianBudget.dual_step
                self.lagrangian.dual_step(R_soft, target)
            lam = self.lagrangian.lambda_
            loss = recon + lag_pen - self.lambda_ent * ent
        elif self.rate_term_on:
            # 固定 λ_rate 惩罚（ablation: lagrangian=None 时）: min D + λ_rate·R
            lam = R_soft.new_tensor(0.0)
            target = 1.0
            loss = recon + self.lambda_rate * R_soft - self.lambda_ent * ent
        else:
            # stage-1 预热: 纯重建（rate 项关闭）
            lam = recon.new_tensor(0.0)
            target = 1.0
            loss = recon - self.lambda_ent * ent

        with torch.no_grad():
            k_used = hard.sum(dim=1)                    # (B,)
            def _t(v):
                return torch.as_tensor(v, device=x.device)
            stats = {
                "recon_l1": _t(recon),
                "rate_soft": _t(R_soft),        # 可微保留率（loss 里那个）
                "rate": _t(R_hard),             # 真实保留率（对偶步/报告用）
                "lambda": _t(lam),
                "target": _t(target),
                "threshold": _t(self.gate.threshold),
                "entropy": _t(ent),
                "k_used_mean": _t(k_used.float().mean()),
                "k_used_min": _t(k_used.float().min()),
                "k_used_max": _t(k_used.float().max()),
            }

        return {"loss": loss, "F_hat": F_hat, "mask": mask,
                "p": p, "stats": stats}

    @staticmethod
    def _gather_selected(tokens: Tensor, hard_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """按原始空间顺序物理剪枝选中的 token（同 v1/v3）。返回 (selected, lengths)。"""
        lengths = hard_mask.sum(1).long()                # (B,) per-image k
        max_k = int(lengths.max().item())
        idx = torch.argsort(hard_mask, dim=1, descending=True,
                            stable=True)[:, :max_k]      # (B,max_k)
        sel = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        return sel, lengths
