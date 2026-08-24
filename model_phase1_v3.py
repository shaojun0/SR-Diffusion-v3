"""
SR-Diffusion Phase 1 v3 — Lagrangian 预算约束 tokenizer
=================================================================

问题（把目标写成约束优化）:
    在最小化"输入到解码器的 token 数量"的同时最小化还原损失。
    形式化为带不等式约束的最小化:

        min_θ  L_recon(θ)
        s.t.   R(θ) ≤ R_target

    · θ        = {ScoreHead/ClsSelectHead, RateHead, ReEncoder, FeatureDecoder}（原变量）
    · L_recon  = L1(F_hat, F_patch)          —— 特征空间还原损失
    · R(θ)     = E[ k/N ]                    —— 送入 decoder 的候选 token 比例
                                                (k = 硬门控选中的数量, N = 候选总数)
    · R_target ∈ (0,1]                      —— 预算（如 0.25 = 平均只留 25%）

拉格朗日函数（约束优化 → 无约束单目标）:
    L(θ, λ) = L_recon(θ) + λ·(R_soft(θ) − R_target) + (ρ/2)·ReLU(R_soft(θ) − R_target)²
    · λ ≥ 0  拉格朗日乘子（对偶变量）—— 由对偶上升自动调整，不是超参
    · ρ ≥ 0  增广项（method of multipliers；ρ=0 退化为纯拉格朗日对偶上升）

求解（Uzawa / 对偶上升，原变量与对偶变量交替更新）:
    原步（外部优化器做）:  θ ← θ − η_θ·∇_θ L(θ, λ)     λ 视为常数（detach）
    对偶步（本模块做）:    λ ← clamp(λ + η_λ·(R_ema − R_target_eff), λ_min, λ_max)
        · R_ema:    硬保留率 R_hard = k/N 的 EMA（慢外环，防对偶步抖动）
        · R_target_eff: 1.0 → R_target 线性退火（stage-2 平滑过渡，消除 k 突变冲击）
    收敛时: R(θ*) ≈ R_target，λ* = KKT 乘子 = "再省一个 token 的边际还原代价"

为什么是"拉格朗日方法"而不是 v1/v2 的固定 λ_rate 惩罚:
    · v1/v2:  loss = L_recon + λ_rate·R —— 惩罚法。λ_rate 是手调超参，
      想打到指定预算必须网格搜索；λ_rate 太小不压缩、太大会塌缩
      （v1 为此补了信任域/死区铰链/力放大/TTUR 一堆机制）。
    · v3:     约束被显式写入对偶变量 λ。λ 自行增长/衰减，直到 R ≈ R_target
      —— 预算成为"需求"而非"碰运气"。R < R_target 时惩罚项为负，
      原变量被允许"多花 token 换更低的还原损失"（惩罚法做不到）。
    · 固定 λ_rate 惩罚可看作 v3 的特例：sweep λ 就是描画 R-D 帕累托前沿
      （D(λ), R(λ)）；约束形式则是在前沿上选 R = R_target 的点。

实现（相对 v1/v2 的取舍）:
    · 门控: p = σ((scores − τ)/T)，τ 每图一个（RateHead(cls)），STE 硬掩码
      hard = (p > 0.5)。不用 v1 的 Gumbel 噪声（探索交给 SGD），不用 v2 的
      固定阈值 0.5（τ 是学出来的；0.5 只是硬二值化点）。
    · scores: 每样本 z-score + clamp ±8（τ∈[−1.8,1.8] 与 z-score 语义自洽，
      也是 v1 实测的 NaN 防御）。
    · select_on="zs": ScoreHead(z_s) 逐位置内容感知（默认；v2 纯 cls 的局限见
      model_phase1_v2.py 头）；select_on="cls": ClsSelectHead(cls) 全局描述子。
    · 对偶变量 λ 是 buffer：进 state_dict（断点续训保留乘子）、随 .to(device)
      移动、DDP 启动时广播；不进优化器 —— 主优化器只管原变量。
    · 保留: SpecialTokenBank/ReEncoder/FeatureDecoder（复用 v1）、cls 保底牌
      （k=0 安全）、反塌缩熵、hard_mode 推理剪枝、k 守卫（可选）、
      硬/软输入混合（train/eval decoder 记忆对齐）。
    · 双时间尺度（TTSA）: RateHead(τ) 用 learning_rate/10（train 脚本）——
      这是原-对偶收敛理论的标准配置，不是 v1 式的补偿机制；dk/dτ≈−100，
      τ 必须比主网络慢一个量级，否则 k 在预算点附近振荡。
    · 对偶步的信号用平滑的 R_soft（detach）而非阶梯状 R_hard（见
      LagrangianBudget.dual_step 注释）。
    · 移除: v1 的 BudgetTrustRegion/信任域/死区铰链/recon 力放大 ——
      这些是为固定 λ_rate 惩罚的不稳定做的补偿，乘子 + TTSA 替代。

TODO（已知边界，不影响主线）:
    [TODO] 物理剪枝的算力收益（当前软填充 N+1 槽位，同 v1/v2）
    [TODO] 每样本 λ（per-image multiplier；当前全局标量 = 标准平均率形式）
    [TODO] 对偶步与梯度累积步长对齐（当前每 micro-batch 更新，偏快；无碍）
    [TODO] 稀疏注意力的真正 token 级计算（替换零填充）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from model_phase1 import (
    RAW_MAX, RateHead, ScoreHead, SpecialTokenBank, ReEncoder, FeatureDecoder,
)


# ═══════════════════════════════════════════════════════════════
# SigmoidGate — sigmoid((scores − τ)/T) 伯努利门控 + STE
# ═══════════════════════════════════════════════════════════════
# τ 在 sigmoid 内部 ⇒ ∂recon/∂τ ≠ 0（还原损失能推 τ 往下、多留 token），
# 拉格朗日惩罚项也能推 τ 往上（少留 token）—— 力平衡由 λ 自动完成。
# hard = (p > 0.5) ⟺ scores > τ（σ 单调），阈值 0.5 只是二值化点。

class SigmoidGate(nn.Module):
    """p = σ((scores − τ)/T)：(B,N) 保留概率；hard = (p > threshold) 二值化 + STE。

    可微性（STE）: mask = p + (hard − p).detach() → 前向硬选择、反向经 p 流入
    ScoreHead/RateHead。⚠ 不能直接返回 hard——比较运算无 grad_fn，梯度路径
    静默死掉（v1 实测的坑）。
    """

    def __init__(self, T: float = 1.0, threshold: float = 0.5):
        super().__init__()
        self.T = T
        self.threshold = threshold

    def forward(self, scores: Tensor, tau: Tensor) -> Tuple[Tensor, Tensor]:
        p = torch.sigmoid((scores - tau) / self.T)      # (B,N) 保留概率
        hard = (p > self.threshold).float()             # (B,N) 0/1, detached
        mask = p + (hard - p).detach()                  # (B,N) STE
        return mask, p


# ═══════════════════════════════════════════════════════════════
# ClsSelectHead — cls (B,D) → scores (B,N)：全局描述子 → 逐位置重要性
# ═══════════════════════════════════════════════════════════════
# select_on="cls" 时使用。cls 只能学"区域级"重要性（如人脸区域重要），
# 感知不到具体 patch 内容——这是纯 cls 选择的已知局限（见 v2 模块头）。

class ClsSelectHead(nn.Module):
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
        return self.mlp(cls)                            # (B,N) raw scores


# ═══════════════════════════════════════════════════════════════
# LagrangianBudget — 对偶上升求解器: min_θ L_recon s.t. R ≤ R_target
# ═══════════════════════════════════════════════════════════════
# 见模块头"拉格朗日函数 / 求解"两节。核心: 惩罚项进 loss（λ 对原步是常数），
# 对偶步在 no_grad 下用 R_hard 更新 λ —— 这是"约束被满足"而不是"碰运气"。

class LagrangianBudget(nn.Module):
    """拉格朗日乘子 λ 的自适应求解器（Uzawa / 投影次梯度对偶上升）。

    penalty(rate_soft):  λ·(R_soft − R_target_eff) + (ρ/2)·ReLU(R_soft − R_target_eff)²
                         —— 加进 loss 的原步惩罚项（λ 已 detach）
    dual_step(rate_hard): λ ← clamp(λ + η_λ·(R_ema − R_target_eff), λ_min, λ_max)
                         —— 每 train step 更新一次（no_grad）

    λ / R_ema / step 都是 buffer：进 state_dict、随 .to(device) 移动；
    λ 不进优化器（requires_grad=False），由对偶步独占更新。
    """

    def __init__(
        self,
        rate_target: float = 0.25,      # 预算: 平均保留 token 比例 ≤ R_target
        lambda_init: float = 0.1,       # λ 起点（>0 起步就有预算压力；0 也对偶可恢复）
        eta_lambda: float = 0.05,       # 对偶步长（rate 单位；太大 λ 振荡，太小收敛慢）
        lambda_min: float = 0.0,
        lambda_max: float = 100.0,
        rho: float = 1.0,               # 增广项权重（method of multipliers；0 = 纯对偶上升）
        ema_alpha: float = 0.05,        # R_hard 的 EMA 平滑（慢外环）
        anneal_steps: int = 2000,       # R_target_eff: 1.0 → rate_target 的退火步数
    ):
        super().__init__()
        assert 0.0 < rate_target <= 1.0, "rate_target ∈ (0, 1]"
        assert lambda_min >= 0.0 and lambda_max > lambda_min
        self.rate_target = float(rate_target)
        self.lambda_init = float(lambda_init)
        self.eta_lambda = float(eta_lambda)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.rho = float(rho)
        self.ema_alpha = float(ema_alpha)
        self.anneal_steps = int(anneal_steps)

        self.register_buffer("lambda_", torch.tensor(lambda_init, dtype=torch.float32))
        self.register_buffer("rate_ema", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    # ── 状态 ──
    def reset(self):
        """stage-2 入口调用: 步数/EMA/乘子回到初始（λ 从 lambda_init 重新学）。"""
        self.step.zero_()
        self.rate_ema.fill_(1.0)
        self.lambda_.fill_(self.lambda_init)

    def tick(self):
        self.step.add_(1)

    # ── 退火目标: 1.0 → rate_target（stage-2 平滑过渡，消除 k 突变冲击）──
    def current_target(self) -> float:
        f = min(1.0, float(self.step) / max(1, self.anneal_steps))
        return self.rate_target + (1.0 - self.rate_target) * (1.0 - f)

    # ── 拉格朗日惩罚项（加进 loss；λ 对原步是常数）──
    def penalty(self, rate_soft: Tensor, target: Optional[float] = None) -> Tensor:
        if target is None:
            target = self.current_target()
        target = torch.as_tensor(target, dtype=torch.float32, device=rate_soft.device)
        viol = rate_soft - target
        quad = F.relu(viol).square()        # 增广项只惩罚超预算（不等式约束）
        # ⚠ 必须 clone：detach() 返回的是与 buffer 共享版本计数器的视图，对偶步的
        #   fill_ 会把它顶到 version 1 → backward 报 "modified by inplace operation"
        #   （实测）。clone() 得到独立存储，记录进图的 λ 是对偶步前的旧值（Uzawa
        #   语义: 原步用当前 λ，对偶步在 loss 之后更新）。
        lam = self.lambda_.detach().clone()
        return lam * viol + 0.5 * self.rho * quad

    # ── 对偶步（投影次梯度 / Uzawa；纯 fp32 防 bf16 精度坑，同 v1）──
    # ⚠ 对偶信号用 R_soft.detach()（平滑的约束代理）而非 R_hard（阶梯状）：
    #   R_hard 对 τ 是 ~100× 放大的阶梯函数（v1 实测），直接喂给对偶步会把
    #   抖动放大进 λ；R_soft 与 R_hard 轨迹一致但平滑，对偶动力学稳定得多。
    #   真实硬约束满足度由 stats 的 rate (R_hard) 与 eval 脚本验证。
    def dual_step(self, rate: Tensor, target: Optional[float] = None):
        with torch.no_grad():
            if target is None:
                target = self.current_target()
            ema = self.rate_ema.float()
            ema = (1 - self.ema_alpha) * ema + self.ema_alpha * rate.detach().float()
            self.rate_ema.fill_(ema)
            viol = torch.nan_to_num(
                ema - torch.as_tensor(target, dtype=torch.float32, device=ema.device),
                nan=0.0, posinf=0.0, neginf=0.0)
            new_lam = torch.clamp(self.lambda_.float() + self.eta_lambda * viol,
                                  self.lambda_min, self.lambda_max)
            if torch.isfinite(new_lam):
                self.lambda_.fill_(new_lam)


# ═══════════════════════════════════════════════════════════════
# SRPhase1V3 — 主模型（复用 v1 的 SpecialTokenBank / ReEncoder / FeatureDecoder）
# ═══════════════════════════════════════════════════════════════

class SRPhase1V3(nn.Module):
    """拉格朗日预算约束的 tokenizer（v3）：min L_recon s.t. R ≤ R_target。

    Args:
        dinov2: frozen Dinov2Model (transformers)。输入 224×224, patch 14
                → 256 patch tokens + cls。
        num_patches: 候选 token 数（224²/patch14 → 256）
        dim: DINO hidden size（base=768, giant=1536）
        T: 门控温度（stage-2 退火到 T_min，见 train 脚本）
        select_on: "zs"（默认，ScoreHead(z_s) 逐位置内容感知）| "cls"
        fixed_tau: stage-1 模式——固定 τ（-2.0 ≈ 全保留），rate 项关闭
        lagrangian: LagrangianBudget 实例；None → 退化为固定 λ_rate 惩罚
        lambda_rate: 无 lagrangian 时的固定惩罚权重（ablation / 描帕累托前沿）
        lambda_ent: 反塌缩熵权重（-H(avg_sel)，防所有图选同一批位置）
        k_min / k_max: 可选硬守卫（topk 分位数 clamp τ；默认关）
    """

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        dim: int = 768,
        T: float = 1.0,
        select_on: str = "zs",
        fixed_tau: Optional[float] = None,
        lagrangian: Optional[LagrangianBudget] = None,
        lambda_rate: float = 0.0,
        lambda_ent: float = 0.01,
        k_min: int = 0,
        k_max: Optional[int] = None,
    ):
        super().__init__()
        self.dinov2 = dinov2
        for p in self.dinov2.parameters():
            p.requires_grad_(False)
        self.dinov2.eval()

        self.num_patches = num_patches
        self.dim = dim
        self.T = T
        assert select_on in ("zs", "cls")
        self.select_on = select_on
        self.fixed_tau = fixed_tau
        self.lagrangian = lagrangian
        self.lambda_rate = lambda_rate
        self.lambda_ent = lambda_ent
        self.k_min = k_min
        self.k_max = k_max
        self.hard_input_prob = 0.0      # 训练期硬/软输入混合（train/eval 对齐）
        self._stage = 2                 # 默认 stage-2（rate 项开）；train 脚本先 set_stage(1)
        self.rate_term_on = True

        if select_on == "zs":
            self.score_head = ScoreHead(in_dim=dim)
        else:
            self.score_head = ClsSelectHead(in_dim=dim, num_patches=num_patches)
        self.rate_head = RateHead(in_dim=dim)
        self.gate = SigmoidGate(T=T)
        self.special_bank = SpecialTokenBank(num_patches=num_patches, dim=dim)
        self.re_encoder = ReEncoder(dim=dim, num_patches=num_patches)
        self.decoder = FeatureDecoder(dim=dim, num_patches=num_patches)

        # zero-init RateHead 输出：stage-2 起点 τ≈0（z-score 语义下 k≈N/2），
        # recon 与率惩罚的梯度都非零 → 无门控饱和死区（同 v1）
        nn.init.zeros_(self.rate_head.mlp[3].weight)
        nn.init.zeros_(self.rate_head.mlp[3].bias)

    # ── 两阶段控制 ──
    def set_stage(self, stage: int):
        """stage=1: 固定 τ=-2（≈全保留），rate 项关闭，纯教重建。
        stage=2: 启用 RateHead + 拉格朗日乘子（入口重置乘子状态）。"""
        if stage == self._stage:
            return
        self._stage = stage
        if stage == 1:
            self.fixed_tau = -2.0
            self.rate_term_on = False
        else:
            self.fixed_tau = None
            self.rate_term_on = True
            if self.lagrangian is not None:
                self.lagrangian.reset()

    def set_rate_target(self, value: float):
        if self.lagrangian is not None:
            self.lagrangian.rate_target = float(value)

    def set_lambda_rate(self, value: float):
        self.lambda_rate = float(value)

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

    @classmethod
    def build_model(cls, dinov2: nn.Module, num_patches: int = 256, dim: int = 768,
                    T: float = 1.0, select_on: str = "zs", lambda_ent: float = 0.01,
                    rate_target: float = 0.25, lambda_init: float = 0.1,
                    eta_lambda: float = 0.05, lambda_max: float = 100.0,
                    rho: float = 1.0, k_min: int = 0, k_max: Optional[int] = None,
                    init_reencoder: bool = True, use_lagrangian: bool = True,
                    **lag_kwargs) -> "SRPhase1V3":
        """Build v3 model: LagrangianBudget + optional DINO warm-start."""
        lag = (LagrangianBudget(rate_target=rate_target, lambda_init=lambda_init,
                                eta_lambda=eta_lambda, lambda_max=lambda_max,
                                rho=rho, **lag_kwargs)
               if use_lagrangian else None)
        model = cls(dinov2, num_patches=num_patches, dim=dim, T=T,
                    select_on=select_on, lagrangian=lag,
                    lambda_ent=lambda_ent, k_min=k_min, k_max=k_max)
        if init_reencoder:
            model.init_reencoder_from_dino()
        return model

    # ── forward ──
    def forward(self, pixel_values: Tensor, hard_mode: bool = False) -> dict:
        """x (B,3,224,224) → 拉格朗日预算约束的硬选择 + 特征重建。

        Returns dict with loss + stats.
        """
        x = pixel_values
        B = x.shape[0]
        N = self.num_patches

        # pass 1: 冻结 DINO
        out = self.dinov2(x)
        feats = out.last_hidden_state                   # (B,257,D)
        cls = feats[:, 0]                               # (B,D)
        patch = feats[:, 1:]                            # (B,N,D)

        # ── pass 2: [cls; specials; patches] 组合进 ReEncoder（全量自注意力）──
        specials = self.special_bank(B, x.device)       # (B,N,D)
        enc_in = torch.cat([cls.unsqueeze(1), specials, patch], dim=1)  # (B,2N+1,D)
        z = self.re_encoder(enc_in)                     # (B,2N+1,D)
        z_cls = z[:, 0:1]                               # (B,1,D)
        z_s = z[:, 1:1 + N]                             # (B,N,D) ← 掩码选择对象
        # z[:, 1+N:] 是 patch 位置的输出，直接丢弃（同 v1/v2）

        # ── 逐位置重要性 scores ──
        if self.select_on == "zs":
            scores = self.score_head(z_s)               # (B,N) 内容感知
        else:
            scores = self.score_head(cls)               # (B,N) 全局描述子
        # 每样本 z-score: τ∈[−1.8,1.8] 才有跨图一致的语义（k ≈ N·(1−Φ(τ))），
        # 也防止 ScoreHead 军备竞赛；clamp ±8 防 bf16 梯度尖峰 NaN（v1 实测）。
        scores = (scores - scores.mean(dim=1, keepdim=True)) / \
                 (scores.std(dim=1, keepdim=True) + 1e-5)
        scores = scores.clamp(-8.0, 8.0)                # (B,N)

        # ── 阈值 τ（原变量，拉格朗日惩罚项与 recon 的力平衡点）──
        if self.fixed_tau is not None:
            tau = torch.full((B, 1), self.fixed_tau, device=x.device)
        else:
            raw = torch.clamp(self.rate_head(cls), -RAW_MAX, RAW_MAX)  # (B,1)
            tau = 2.0 * torch.tanh(raw)                 # (B,1) ∈ [−1.8, 1.8]
            if self.k_min > 0 or (self.k_max is not None and self.k_max < N):
                # k 硬守卫（可选）: clamp τ 到 [第 k_max+1 大, 第 k_min 大] 之间
                kmax = min(self.k_max, N - 1) if self.k_max is not None else N - 1
                kmin = max(self.k_min, 1)
                top_vals, _ = torch.topk(scores, kmax + 1, dim=1)   # (B,kmax+1)
                tau = torch.clamp(tau,
                                  top_vals[:, kmax].detach().unsqueeze(1),
                                  top_vals[:, kmin - 1].detach().unsqueeze(1))

        # ── sigmoid 伯努利门控 + STE ──
        mask, p = self.gate(scores, tau)                # mask (B,N) STE, p (B,N)
        R_soft = mask.mean()                            # scalar 可微保留率
        hard = (mask > 0.5).float()                     # (B,N) 0/1（⟺ scores>τ）
        R_hard = hard.mean()                            # scalar 真实保留率（对偶步用）

        # ── decoder 输入（同 v1: 训练软填充 / 推理硬剪枝 / 可选混合）──
        if hard_mode:
            sel, _ = self._gather_selected(z_s, hard)
            dec_in = torch.cat([z_cls, sel], dim=1)     # (B, k+1, D)
        elif (self.training and self.fixed_tau is None
                and self.hard_input_prob > 0.0
                and torch.rand(1, device=x.device).item() < self.hard_input_prob):
            sel, _ = self._gather_selected(z_s, hard)
            dec_in = torch.cat([z_cls, sel], dim=1)
        else:
            dec_in = torch.cat([z_cls, z_s * mask.unsqueeze(-1)], dim=1)  # (B,N+1,D)

        F_hat = self.decoder(dec_in)                    # (B,N,D)

        # ── 损失: 还原 + 拉格朗日惩罚 + 反塌缩熵（全部按最小化）──
        recon = F.l1_loss(F_hat, patch)                 # scalar L_recon
        avg_sel = mask.mean(dim=0)                      # (N,) 跨 batch 平均选择
        ent = -(avg_sel * torch.log(avg_sel + 1e-8) +
                (1 - avg_sel) * torch.log(1 - avg_sel + 1e-8)).mean()  # scalar H

        if self.rate_term_on and self.lagrangian is not None:
            if self.training:
                self.lagrangian.tick()
            target = self.lagrangian.current_target()
            lag_pen = self.lagrangian.penalty(R_soft, target)   # λ·viol + ρ/2·ReLU²
            if self.training:
                # 对偶步用平滑的 R_soft（detach），见 LagrangianBudget.dual_step 注释
                self.lagrangian.dual_step(R_soft, target)       # λ ← clamp(λ + η·viol)
            lam = self.lagrangian.lambda_
            loss = recon + lag_pen - self.lambda_ent * ent
        elif self.rate_term_on:
            # 固定 λ_rate 惩罚（ablation / 描 R-D 帕累托前沿）: min D + λ_rate·R
            lam = R_soft.new_tensor(0.0)
            target = 1.0
            loss = recon + self.lambda_rate * R_soft - self.lambda_ent * ent
        else:
            # stage-1 预热: 纯重建（率项关闭）
            lam = recon.new_tensor(0.0)
            target = 1.0
            loss = recon - self.lambda_ent * ent

        with torch.no_grad():
            k_used = hard.sum(dim=1)                    # (B,)
            def _t(v):
                return torch.as_tensor(v, device=x.device)  # DataParallel-gatherable
            stats = {
                "recon_l1": _t(recon),
                "rate_soft": _t(R_soft),        # 可微保留率（loss 里那个）
                "rate": _t(R_hard),             # 真实保留率（对偶步/报告用）
                "lambda": _t(lam),
                "target": _t(target),
                "tau_mean": _t(tau.mean()),
                "entropy": _t(ent),
                "k_used_mean": _t(k_used.float().mean()),
                "k_used_min": _t(k_used.float().min()),
                "k_used_max": _t(k_used.float().max()),
            }

        return {"loss": loss, "F_hat": F_hat, "mask": mask,
                "p": p, "tau": tau, "stats": stats}

    @staticmethod
    def _gather_selected(tokens: Tensor, hard_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """按原始空间顺序物理剪枝选中的 token（同 v1）。返回 (selected, lengths)。"""
        lengths = hard_mask.sum(1).long()                # (B,) per-image k
        max_k = int(lengths.max().item())
        idx = torch.argsort(hard_mask, dim=1, descending=True,
                            stable=True)[:, :max_k]      # (B,max_k)
        sel = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        return sel, lengths
