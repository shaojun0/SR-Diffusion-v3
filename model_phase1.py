"""
SR-Diffusion Phase 1: Adaptive Token Budget + Feature-Space Reconstruction
=========================================================================

Core idea (visual BPE): learn the data distribution to decide HOW MANY tokens
each image needs.  A blank image should compress to ~1 token; a dense image
keeps many.  No fixed-dimension bottleneck, no per-pixel reconstruction.

Architecture (two-pass, v2 — special-token selection):

    x (B,3,224,224)
      │  pass 1: frozen DINOv2
      ▼
    F_patch (B,256,768) + cls (B,768)
      │
      ├─ RateHead(cls) ──────────────────► τ (B,1)          ← learned budget
      ├─ SpecialTokenBank ───────────────► specials (B,256,768)
      │     输入完全相同（共享向量），仅位置编码不同
      │
      ▼  [cls; specials; patches] 组合 → pass 2: ReEncoder（全量自注意力）
    z (B,513,768)                         [cls + N specials + N patches]
      │
      ├─ z_s = z[:, 1:257]                ← 特殊 token 的输出（top-k 唯一候选）
      ├─ ScoreHead(z_s) ─────────────────► logits (B,256)
      │
      ▼  differentiable top-k (Gumbel-Sigmoid + STE)
    mask (B,256)
      │
      ▼  decoder 输入 = [cls; 选中的 z_s]（训练 soft 门控 / 推理硬剪枝）
    F_hat (B,256,768)
      │
      ▼
    L = L1(F_hat, F_patch) + λ_rate·k_soft + λ_consist·consistency + λ_ent·H

选择只在编码器输出的"特殊 token 表示"里进行；图像的原始 patch 编码只作为
编码器输入参与信息聚合，永远不进入 top-k 候选、也不进入 decoder（避免与
图像 patch 编码混在一起）。代价：编码器始终全量计算 2N+1 token，自适应预算
的算力收益只体现在 decoder 的 cross-attention（k vs N memory）。

Training is two-stage:
  Stage 1: fix k=1.0 (all tokens), λ_rate=0  → teach re-encoder + decoder to
           reconstruct the full DINO feature map.  (never dead-locks)
  Stage 2: enable RateHead, anneal λ_rate from 0 up  → mask sparsifies,
           budget becomes content-adaptive.  (the "soft shift-one-forward
           penalty" from the original idea, made continuous and learnable)

Gradient path: SVD is NOT used here.  All gradients flow through ordinary
tensors; the only discrete op (top-k) is relaxed with Gumbel-Sigmoid + STE.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


# ═══════════════════════════════════════════════════════════════
# RateHead — predicts the token-budget threshold τ from [cls]
# ═══════════════════════════════════════════════════════════════

class RateHead(nn.Module):
    """cls (B,768) → τ (B,1) unbounded logit threshold.

    τ participates DIRECTLY in the mask:  mask_i = sigmoid((logits_i - τ)/T).
    So gradients from BOTH the reconstruction loss (pushes τ ↓, keep more
    tokens) and the budget penalty λ·mask.mean() (pushes τ ↑, drop more
    tokens) reach this head.  Their balance = content-adaptive compression.
    """

    def __init__(self, in_dim: int = 768, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, cls: Tensor) -> Tensor:
        return self.mlp(cls)  # (B,1) unbounded threshold


# ═══════════════════════════════════════════════════════════════
# ScoreHead — per-token importance
# ═══════════════════════════════════════════════════════════════

class ScoreHead(nn.Module):
    """F_patch (B,N,768) → logits (B,N).  One score per candidate token."""

    def __init__(self, in_dim: int = 768, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, patch_tokens: Tensor) -> Tensor:
        return self.mlp(patch_tokens).squeeze(-1)  # (B,N)


# ═══════════════════════════════════════════════════════════════
# Threshold Gate — differentiable top-k via learned threshold τ
# ═══════════════════════════════════════════════════════════════

class ThresholdGate(nn.Module):
    """
    mask_i = sigmoid((logits_i + g_i - τ) / T)          (soft, gumbel noise g)
    hard   = (logits_i > τ)                             (hard 0/1 selection)
    mask   = soft + (hard - soft).detach()              (STE: forward hard)

    τ is a per-image threshold predicted by RateHead.  It appears INSIDE the
    soft mask, so d(recon)/dτ ≠ 0 — the reconstruction loss can push τ down
    (keep more tokens) while the rate penalty λ·mask.mean() pushes τ up
    (drop tokens).  No explicit "predict k as a count" needed: the count
    emerges from where τ lands relative to the logits distribution.

    forward: logits (B,N), tau (B,1) → (mask, soft)
    """

    def __init__(self, T: float = 1.0):
        super().__init__()
        self.T = T

    def forward(self, logits: Tensor, tau: Tensor) -> Tuple[Tensor, Tensor]:
        B, N = logits.shape
        g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        soft = torch.sigmoid((logits + g - tau) / self.T)   # (B,N) in (0,1)
        hard = (logits > tau).float()                       # (B,N) 0/1, detached
        mask = soft + (hard - soft).detach()                # STE
        return mask, soft


# ═══════════════════════════════════════════════════════════════
# SpecialTokenBank — 特殊 token 池（输入相同，仅位置编码不同）
# ═══════════════════════════════════════════════════════════════

class SpecialTokenBank(nn.Module):
    """每个 patch 位置一个特殊 token：共享可学习向量 + 逐位置可学习位置编码。

    token (1,1,D) 对所有位置/所有图像完全相同；pos (1,N,D) 提供位置区分。
    它们与 patch 组合后输入 ReEncoder，编码器输出中特殊 token 的表示 z_s
    是 top-k 的唯一候选（见 SRPhase1.forward）——图像的原始 patch 编码
    不参与选择、也不进入 decoder。
    """

    def __init__(self, num_patches: int, dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)          # 共享向量
        self.pos = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)  # 位置编码

    def forward(self, B: int, device: torch.device) -> Tensor:
        return self.token.expand(B, self.num_patches, -1) + self.pos      # (B,N,D)


# ═══════════════════════════════════════════════════════════════
# Re-Encoder — pass 2, 组合编码器 [cls; specials; patches]
# ═══════════════════════════════════════════════════════════════

class ReEncoder(nn.Module):
    """
    对 [cls; 特殊 token; patch] 组合序列做全量自注意力（特殊 token 与 patch
    在此"组合"：每个位置的输出既带位置身份又聚合图像内容）。

    输出中特殊 token 位置的表示 z_s 是 top-k 的唯一候选；patch 位置的输出
    直接丢弃。因此图像 patch 编码只作为编码器输入参与聚合，永不进入 decoder。

    与旧版（门控 N token、推理硬剪枝 O(k²)）不同：输入是完整的 2N+1 序列，
    编码器在训练/推理都全量计算（O((2N)²)）——自适应预算的算力收益只体现
    在 decoder 的 cross-attention（k vs N memory）。剪枝移到 decoder 输入侧。

    Input:  x (B, 2N+1, D) = [cls; special_1..N; patch_1..N]
    Output: z (B, 2N+1, D)
    """

    def __init__(self, dim: int = 768, num_patches: int = 256, depth: int = 4,
                 heads: int = 8, mlp_ratio: float = 4.0):
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

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embed[:, :x.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════
# Feature Decoder — reconstruct the FULL feature map from k tokens
# ═══════════════════════════════════════════════════════════════

class FeatureDecoder(nn.Module):
    """
    Cross-attention decoder: 256 position queries attend to the re-encoded
    semantic tokens z → F_hat (B,256,768).  Position ids are required so the
    decoder knows WHERE each reconstructed patch goes.

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

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: (B, Nz, D) re-encoded tokens (Nz = k+1 in hard mode, N+1 in soft)
        Returns:
            F_hat: (B, num_patches, D)
        """
        B = z.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)      # (B,256,D)
        q = q + self.pos_query.unsqueeze(0)
        for layer in self.layers:
            q = layer(q, z)
        return self.norm(q)


# ═══════════════════════════════════════════════════════════════
# Full Phase-1 Model
# ═══════════════════════════════════════════════════════════════

class SRPhase1(nn.Module):
    """
    Adaptive-budget visual tokenizer with feature-space reconstruction.

    v2 (special-token selection): 新增 SpecialTokenBank —— N 个特殊 token
    （输入相同、仅位置编码不同）与 DINO patch 组合后一起输入 ReEncoder；
    top-k 只从编码器输出的特殊 token 表示 z_s 中选，图像的原始 patch 编码
    不参与选择、也不进入 decoder（避免与图像 patch 编码混在一起）。

    Args:
        dinov2: frozen Dinov2Model (transformers).  Input 224×224, patch 14
                → 256 patch tokens + cls.
        num_patches: candidate token count (256 for 224²/patch14)
        dim: DINO hidden size (768 for base, 1536 for giant)
        T: gate temperature (annealed in stage 2)
        lambda_rate: budget penalty weight (annealed in stage 2)
        lambda_ent: entropy penalty on the averaged selection (anti-collapse)
        fixed_tau: stage-1 mode — if not None, τ pinned to this value.
                   logits are per-sample z-scored (~N(0,1)), so -2.0 keeps
                   ≈97.7% of tokens; use a very negative value (e.g. -8) for
                   ≈100%.
    """

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        dim: int = 768,
        T: float = 1.0,
        lambda_rate: float = 0.1,
        lambda_ent: float = 0.01,
        fixed_tau: Optional[float] = None,
    ):
        super().__init__()
        self.dinov2 = dinov2
        for p in self.dinov2.parameters():
            p.requires_grad_(False)
        self.dinov2.eval()

        self.num_patches = num_patches
        self.dim = dim
        self.T = T
        self.lambda_rate = lambda_rate
        self.lambda_ent = lambda_ent
        self.fixed_tau = fixed_tau

        self.rate_head = RateHead(in_dim=dim)
        self.score_head = ScoreHead(in_dim=dim)
        self.gate = ThresholdGate(T=T)
        self.special_bank = SpecialTokenBank(num_patches=num_patches, dim=dim)
        self.re_encoder = ReEncoder(dim=dim, num_patches=num_patches)
        self.decoder = FeatureDecoder(dim=dim, num_patches=num_patches)

        # zero-init the RateHead output so stage-2 starts at τ≈0 (soft≈0.5):
        # both recon and rate gradients are then non-zero → no gate saturation
        nn.init.zeros_(self.rate_head.mlp[3].weight)
        nn.init.zeros_(self.rate_head.mlp[3].bias)

        # PPO-style budget stabilization (optional, enabled by train script)
        self.use_trust_region = False
        self.rate_head_target = None   # EMA reference policy, built on demand
        self.trust_region = None       # BudgetTrustRegion instance
        # train/inference alignment: prob (stage-2 training only) of feeding the
        # DECODER the hard-pruned selected z_s instead of zero-padded full length.
        # 0.0 = off (training always sees zero-slots); >0 mixes in pruned inputs.
        self.hard_input_prob = 0.0

    # ── enable PPO-style trust region for stage-2 budget learning ──
    def enable_trust_region(self, btr: "BudgetTrustRegion"):
        import copy
        self.use_trust_region = True
        self.trust_region = btr
        self.rate_head_target = copy.deepcopy(self.rate_head)
        for p in self.rate_head_target.parameters():
            p.requires_grad_(False)
        self.rate_head_target.eval()

    # ── two-stage control ──
    def set_stage(self, stage: int):
        """stage=1: fix τ (all tokens kept), no rate penalty.
        stage=2: learn τ (content-adaptive budget)."""
        self.fixed_tau = -2.0 if stage == 1 else None

    def set_lambda_rate(self, value: float):
        self.lambda_rate = value

    # ── init from pretrained DINO (mirrors SRQwenVLv10.build_model) ──
    @classmethod
    def build_model(cls, dinov2: nn.Module, num_patches: int = 256, dim: int = 768,
                    T: float = 1.0, lambda_rate: float = 0.1,
                    lambda_ent: float = 0.01, fixed_tau: Optional[float] = None,
                    init_reencoder: bool = True) -> "SRPhase1":
        """Build Phase-1 model and optionally warm-start ReEncoder from the
        pretrained DINO encoder layers (so the re-encoder doesn't learn from
        scratch — same spirit as SRQwenVLv10.build_model swapping in DINO)."""
        model = cls(dinov2, num_patches=num_patches, dim=dim, T=T,
                    lambda_rate=lambda_rate, lambda_ent=lambda_ent,
                    fixed_tau=fixed_tau)
        if init_reencoder:
            model.init_reencoder_from_dino()
        return model

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
                # q/k/v → fused in_proj (row order: query, key, value)
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
        """
        Args:
            pixel_values: (B,3,224,224) normalized images
            hard_mode: inference — physically prune selected z_s for the decoder
        Returns dict with loss components + stats.
        """
        x = pixel_values
        B = x.shape[0]
        N = self.num_patches

        # ── pass 1: frozen DINOv2 ──
        with torch.no_grad():
            out = self.dinov2(x)
            feats = out.last_hidden_state      # (B, 257, D)
        cls = feats[:, 0]                      # (B, D)
        patch = feats[:, 1:]                   # (B, N, D)

        # ── 特殊 token：输入相同（共享向量），仅位置编码不同 ──
        specials = self.special_bank(B, x.device)          # (B, N, D)

        # ── pass 2: [cls; specials; patches] 组合进编码器（全量自注意力）──
        enc_in = torch.cat([cls.unsqueeze(1), specials, patch], dim=1)  # (B,2N+1,D)
        z = self.re_encoder(enc_in)                        # (B,2N+1,D)
        z_cls = z[:, 0:1]                                  # (B,1,D) cls 输出
        z_s = z[:, 1:1 + N]                                # (B,N,D) 特殊 token 输出 ← top-k 唯一候选
        # z[:, 1+N:] 是 patch 位置的输出，直接丢弃——原始 patch 编码不进 decoder

        # ── per-token importance（排序只依赖编码器输出的特殊 token 表示）──
        logits = self.score_head(z_s)        # (B,N) raw scores
        # per-sample normalize: ScoreHead learns WHICH tokens matter (ranking),
        # RateHead learns HOW MANY (absolute τ).  This prevents an arms race
        # where ScoreHead inflates all logits past a bounded τ → gate always on.
        logits = (logits - logits.mean(dim=1, keepdim=True)) / \
                 (logits.std(dim=1, keepdim=True) + 1e-5)
        # NaN 防御（实测：真实 ImageNet 图 ~970 步 grad_norm=NaN，随机噪声 1000 步稳定；
        # 真实图 logits 长尾在 bf16 下产生梯度尖峰 → 单参数 inf → HF Trainer 的
        # clip_grad_norm_(inf) 做 inf×0 → 全参 NaN）。限幅不改变排序，k 守卫/gate 不受影响。
        logits = logits.clamp(-8.0, 8.0)

        # ── threshold gate (τ inside mask ⇒ gradients reach RateHead) ──
        if self.fixed_tau is not None:
            tau = torch.full((B, 1), self.fixed_tau, device=x.device)
            mask, soft = self.gate(logits, tau)    # (B,256) each
            budget_loss = torch.zeros((), device=x.device)
            kl_loss = torch.zeros((), device=x.device)
            ent_loss = torch.zeros((), device=x.device)
            tr_extras = {}
        elif self.use_trust_region and self.trust_region is not None:
            # PPO-style budget stabilization (stage-2)
            tr = self.trust_region.gate_forward(self, logits, cls)
            mask, soft = tr["mask"], tr["soft"]
            tau = tr["tau"]                       # 真实 τ（供 recon 力放大）
            budget_loss, kl_loss, ent_loss = (tr["budget_loss"], tr["kl_loss"],
                                              tr["ent_loss"])
            tau_reg = tr["tau_reg"]
            tr_extras = tr["extras"]
            # stash KL for the callback's adaptive-β update
            self.trust_region._last_kl = float(kl_loss.detach().cpu())
        else:
            # bounded threshold: 2·tanh keeps τ in [-1.8, 1.8] (raw 先 clamp 防
            # tanh 饱和死锁，dτ/draw ≥ 0.38 永不消失)。与 trust-region 路径一致。
            raw = torch.clamp(self.rate_head(cls), -1.472, 1.472)   # insurance
            tau = 2.0 * torch.tanh(raw)                             # ∈ [-1.8, 1.8]
            mask, soft = self.gate(logits, tau)    # (B,256) each
            budget_loss = torch.zeros((), device=x.device)
            kl_loss = torch.zeros((), device=x.device)
            ent_loss = torch.zeros((), device=x.device)
            tr_extras = {}

        # ── gate decoder input (train: soft; eval: hard) ──
        # 门控位置从"编码器输入"移到"编码器输出的特殊 token 表示"：
        # 选中的 z_s（携带内容）进入 decoder，原始 patch 编码从不进入 decoder。
        hard_mask = (mask > 0.5).float()
        if hard_mode:
            sel, _ = self._gather_selected(z_s, hard_mask)
            dec_in = torch.cat([z_cls, sel], dim=1)          # (B, k+1, D)
        elif (self.training and self.fixed_tau is None
                and self.hard_input_prob > 0.0
                and torch.rand(1, device=x.device).item() < self.hard_input_prob):
            # Optional train/inference alignment (default OFF — see
            # --hard_mix_prob): occasionally feed the decoder the hard-pruned
            # selected z_s.  In training the decoder otherwise only ever sees N
            # slots with zeros where tokens were dropped; at inference those
            # slots are physically gone → the "zero-slot" signal disappears →
            # decoder memory distribution shift.  Mixing hard inputs removes it.
            # Budget/τ gradients still flow through `soft`; recon just trains
            # on pruned memory.  Stage-1 (k≈256, no pruning) is skipped via the
            # fixed_tau check.
            sel, _ = self._gather_selected(z_s, hard_mask)
            dec_in = torch.cat([z_cls, sel], dim=1)          # (B, k+1, D)
        else:
            dec_in = torch.cat([z_cls, z_s * mask.unsqueeze(-1)], dim=1)
            # (B, N+1, D) 梯度经 STE mask 流动

        # ── feature-space reconstruction ──
        F_hat = self.decoder(dec_in)           # (B, 256, D)

        # ── losses ──
        recon = F.l1_loss(F_hat, patch, reduction="mean")

        if (self.use_trust_region and self.trust_region is not None
                and self.fixed_tau is None):
            # v3: dead-zone hinge + recon 的 τ 路径梯度放大 ×s + 排斥正则
            btr = self.trust_region
            beta = btr.beta
            # ── 力放大：把 ∂recon/∂τ 放大 s 倍，且只影响 RateHead ──
            #    L_boost = (s-1)·Σ_j (∂recon/∂τ_j)·τ_j
            #    ∂L_boost/∂τ_j = (s-1)·∂recon/∂τ_j（g_tau detach 后视为常数）
            #    L_boost 不含 logits/decoder 变量 → ScoreHead/ReEncoder/Decoder
            #    的梯度不变（实测 recon 力仅 ~0.001-0.003，不放大则 τ 动力学冻结）。
            if tau.requires_grad and recon.requires_grad:
                g_tau = torch.autograd.grad(recon, tau, retain_graph=True,
                                            allow_unused=True)[0]
                if g_tau is None:
                    g_tau = torch.zeros_like(tau)
                g_tau = torch.nan_to_num(g_tau, nan=0.0, posinf=0.0, neginf=0.0)
                recon_boost = (btr.recon_tau_scale - 1.0) * (g_tau.detach() * tau).sum()
            else:
                recon_boost = recon.new_zeros(())
            loss = (recon + recon_boost
                    + budget_loss + beta * kl_loss
                    + self.lambda_ent * ent_loss
                    + btr.lambda_tau * tau_reg)
            rate = soft.mean()
            avg_sel = soft.mean(dim=0)
            ent = -(avg_sel * torch.log(avg_sel + 1e-8) +
                    (1 - avg_sel) * torch.log(1 - avg_sel + 1e-8)).mean()
        else:
            # rate penalty = expected token fraction (differentiable via soft mask)
            rate = soft.mean()
            # entropy anti-collapse: averaged selection should not degenerate
            avg_sel = soft.mean(dim=0)             # (256,)
            ent = -(avg_sel * torch.log(avg_sel + 1e-8) +
                    (1 - avg_sel) * torch.log(1 - avg_sel + 1e-8)).mean()
            loss = recon + self.lambda_rate * rate + self.lambda_ent * ent

        with torch.no_grad():
            k_used = (mask > 0.5).sum(dim=1)   # (B,) hard token count
            def _t(v):
                return torch.as_tensor(v, device=x.device)  # DataParallel-gatherable
            stats = {
                "recon_l1": _t(recon),
                "tau_mean": _t(tau.mean() if tau.numel() else rate.new_tensor(0.0)),
                "rate": _t(rate),
                "usage": _t((mask > 0.5).float().mean()),
                "k_used_mean": _t(k_used.float().mean()),
                "k_used_min": _t(k_used.float().min()),
                "k_used_max": _t(k_used.float().max()),
                "entropy": _t(ent),
            }
            stats.update({k: _t(v) for k, v in tr_extras.items()})

        return {"loss": loss, "F_hat": F_hat, "mask": mask,
                "tau": tau, "stats": stats}

    @staticmethod
    def _gather_selected(tokens: Tensor, hard_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """按原始空间顺序物理剪枝选中的 token（decoder memory 无位置 PE，
        顺序无关紧要，但保持原始顺序与训练布局一致）。返回 (selected, lengths)。"""
        lengths = hard_mask.sum(1).long()                    # (B,) per-image k
        max_k = int(lengths.max().item())
        idx = torch.argsort(hard_mask, dim=1, descending=True,
                            stable=True)[:, :max_k]          # (B,max_k)
        sel = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        return sel, lengths


# ═══════════════════════════════════════════════════════════════
# BudgetTrustRegion — PPO-style stabilization of the budget policy
# ═══════════════════════════════════════════════════════════════
# Root cause (measured): τ moves are amplified ~100× into k (dk/dτ≈-φ·256),
# the recon/rate gradients cancel near equilibrium, and gumbel noise turns
# τ into a random walk that slams into the tanh bound → k collapses (λ=0.3)
# or never compresses (λ=0.04).  Fixes (mirror PPO's trust-region idea):
#   ① τ output cap:  |τ - τ_ref| ≤ δ_τ per step  (Δk ≤ ~5 tokens/step)
#   ② k hard guard:  τ clamped to keep k ∈ [k_min, k_max] always
#   ③ KL trust region: β·KL(Bern(p_ref)‖Bern(p)) with dual-gradient β
#      → one-step damped system, cannot diverge (PPO-KL variant)
#   ④ gumbel → 0 + T anneal: removes the noise driving the random walk
#   ⑤ EMA budget interval: penalty only outside [r_min, r_max], slow α_r
#      feedback (cascade control: fast recon inner loop, slow budget outer)
#   ⑥ separate small lr for RateHead (TTUR) + grad clip (in train script)

class BudgetTrustRegion:
    """Holds controller state + forward-time computations for stage-2.

    Not an nn.Module (stateless w.r.t. params) — the train script drives
    update_ref / update_beta / anneal each step.
    """

    def __init__(
        self,
        n: int = 256,
        delta_tau: float = 0.05,
        kl_target: float = 0.005,
        beta: float = 1.0,
        beta_min: float = 1e-3,
        beta_max: float = 50.0,
        beta_eta: float = 0.3,
        kl_ema_alpha: float = 0.05,
        k_min: int = 8,
        k_max: int = 250,
        T: float = 1.0,
        T_min: float = 0.3,
        anneal_steps: int = 1000,
        gumbel_steps: int = 800,
        rate_ema_alpha: float = 0.01,
        rate_min: float = 0.03,
        rate_max: float = 0.25,
        lambda_rate: float = 5.0,          # dead-zone hinge 刚度（原 0.3 线性惩罚语义废弃）
        lambda_ent: float = 0.01,
        ref_momentum: float = 0.99,
        recon_tau_scale: float = 50.0,     # NEW: recon 的 τ 路径梯度放大倍数（力平衡标定）
        raw_max: float = 1.472,            # NEW: atanh(1.8/2) — raw clamp，dτ/draw ≥ 0.38
        tau_soft: float = 1.5,             # NEW: raw 空间排斥正则起点（τ=1.5）
        lambda_tau: float = 1.0,           # NEW: 排斥强度
    ):
        assert k_max < n, "k_max must be < n (topk index needs k_max <= n-1)"
        self.n = n
        self.delta_tau = delta_tau
        self.kl_target = kl_target
        self.beta = beta
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_eta = beta_eta
        self.kl_ema_alpha = kl_ema_alpha
        self.k_min = k_min
        self.k_max = k_max
        self.T = T
        self.T_min = T_min
        self.anneal_steps = anneal_steps
        self.gumbel_steps = gumbel_steps
        self.rate_ema_alpha = rate_ema_alpha
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.lambda_rate = lambda_rate
        self.lambda_ent = lambda_ent
        self.ref_momentum = ref_momentum
        self.kl_ema = 0.0
        self.rate_ema = None      # init to 1.0 at stage-2 start (was full-keep)
        self._last_kl = 0.0
        self.step = 0
        self.recon_tau_scale = recon_tau_scale
        self.raw_max = raw_max
        self.raw_soft = math.atanh(min(0.999, tau_soft / 2.0))   # ≈ 0.973 (τ=1.5)
        self.lambda_tau = lambda_tau

    # ── schedule helpers ──
    def anneal(self):
        f = min(1.0, self.step / max(1, self.anneal_steps))
        return self.T + (self.T_min - self.T) * f

    def gumbel_coef(self):
        return max(0.0, 1.0 - self.step / max(1, self.gumbel_steps))

    def update_ref(self, model: "SRPhase1"):
        """EMA of RateHead weights = reference policy (old π)."""
        with torch.no_grad():
            for p, pt in zip(model.rate_head.parameters(),
                             model.rate_head_target.parameters()):
                pt.data.mul_(self.ref_momentum).add_(p.data, alpha=1 - self.ref_momentum)

    def update_beta(self, kl_step: float):
        """Dual-gradient update for β (PPO's adaptive KL penalty).

        NaN 防御：KL 为 NaN/Inf（真实图 logits 长尾 + bf16 梯度尖峰）时保持 β
        不动，否则 NaN 会经 exp/min/max 传染进 loss（grad_norm=NaN 的源头之一）。
        """
        if not math.isfinite(kl_step) or not math.isfinite(self.kl_ema):
            return
        self.kl_ema = (1 - self.kl_ema_alpha) * self.kl_ema + self.kl_ema_alpha * kl_step
        new_beta = self.beta * math.exp(self.beta_eta * (self.kl_ema - self.kl_target))
        if math.isfinite(new_beta):
            self.beta = min(max(new_beta, self.beta_min), self.beta_max)

    # ── forward-time gate + budget (called from SRPhase1.forward) ──
    def gate_forward(self, model: "SRPhase1", logits: Tensor, cls: Tensor) -> dict:
        """
        logits: (B,N) per-sample normalized, requires_grad (ScoreHead path)
        cls:    (B,D)
        Returns dict: mask(STE, for recon), soft(clean, τ-grad only),
                      budget_loss, kl_loss, ent_loss, extras(stats)
        """
        B, N = logits.shape
        self.step += 1

        # 平滑压缩：rate_max 从 1.0 退火到目标值（消除 stage-2 起点 k 突变的冲击；
        # 也是 NaN 防御——re-encoder/decoder 先适应全保留输入再逐步压缩）
        f = min(1.0, self.step / max(1, self.anneal_steps))
        rate_max = self.rate_max + (1.0 - self.rate_max) * (1.0 - f)

        # ① raw clamp（网络 + target 都要）: dτ/draw = 2(1-tanh²(raw)) ≥ 0.38 永不消失
        #    （在 forward 处 clamp 而非 RateHead 输出层：target 共享同一条代码路径；
        #     梯度经 clamp 在界内为恒等，模块本身保持纯函数）
        raw = torch.clamp(model.rate_head(cls), -self.raw_max, self.raw_max)
        tau_net = 2.0 * torch.tanh(raw)                              # ∈ [-1.8, 1.8]
        with torch.no_grad():
            raw_ref = torch.clamp(model.rate_head_target(cls),
                                  -self.raw_max, self.raw_max)       # target 也必须 clamp！
            tau_ref = 2.0 * torch.tanh(raw_ref)
        tau = torch.clamp(tau_net, tau_ref - self.delta_tau,
                          tau_ref + self.delta_tau)

        # ② k hard guard: keep k ∈ [k_min, k_max]
        #   topk(k_max+1) so index k_max = (k_max+1)-th largest logit exists.
        top_vals, _ = torch.topk(logits, self.k_max + 1, dim=1)     # descending
        tau = torch.clamp(tau,
                          top_vals[:, self.k_max].detach().unsqueeze(1),
                          top_vals[:, self.k_min - 1].detach().unsqueeze(1))

        # ③ gate
        T = self.anneal()
        g = self.gumbel_coef()
        noise = g * (-torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8))
        soft_noisy = torch.sigmoid((logits + noise - tau) / T)      # exploration (mask path)
        hard = (logits > tau).float()
        # STE: forward=hard，梯度走 soft_noisy。
        # ⚠ 必须 soft 在前：写成 hard + (soft-hard).detach() 时，hard 是比较运算
        #   输出（无 grad_fn），加一个 detach 项后整张 mask requires_grad=False，
        #   recon 梯度到不了 τ 和 ScoreHead —— 路径静默死掉（smoke test 实测确认）。
        mask = soft_noisy + (hard - soft_noisy).detach()

        # ④ clean soft: logits detached → budget terms only push τ
        soft = torch.sigmoid((logits.detach() - tau) / T)
        with torch.no_grad():
            soft_ref = torch.sigmoid((logits - tau_ref) / T)

        # ⑤ trust region: KL(Bern(p_ref) ‖ Bern(p))
        #    fp32 computation: under bf16 autocast, clamp(1e-5) can't be
        #    represented (bf16 precision ~1e-3) → pc.log() → -inf → NaN
        #    (observed at step ~980 under bf16; NaN only in rate_head).
        eps = 1e-3   # safe lower bound even in bf16
        pr = soft_ref.float().clamp(eps, 1 - eps)
        pc = soft.float().clamp(eps, 1 - eps)
        kl = (pr * (pr.log() - pc.log()) +
              (1 - pr) * ((1 - pr).log() - (1 - pc).log())).mean()
        # entropy (on clean soft)
        ent = -(pc * pc.log() + (1 - pc) * (1 - pc).log()).mean()

        # ⑥ dead-zone hinge（per-sample、瞬时值、平方）—— 死区内预算力≡0，
        #    消除实测 30× 力失衡；梯度 ≈ 2·λh·err·φ。
        #    精确形式（v3）:
        #      err_j   = ReLU(rate_j - r_max) + ReLU(r_min - rate_j)
        #      budget  = λh · mean_j(err_j²)
        #    vs 上一个临时版本（rate.detach() + 线性直通）:
        #      ① 它只有"上方"正确，rate < r_min 时线性直通方向错误（把 rate 继续往下推），
        #         塌缩恢复（k=4 → r_min）会失效；② float(...cpu()) 每步 GPU→CPU 同步。
        rate = soft.mean()
        rate_j = soft.mean(dim=1)                        # (B,) per-sample usage
        err = (F.relu(rate_j - rate_max)                 # 用退火后的 rate_max
               + F.relu(self.rate_min - rate_j))         # 死区 [r_min, r_max] 内力=0
        budget = self.lambda_rate * err.square().mean()
        # rate_ema 仅作日志统计（不再进入 loss）
        if self.rate_ema is None:
            self.rate_ema = torch.ones_like(rate)
        self.rate_ema = ((1 - self.rate_ema_alpha) * self.rate_ema.detach()
                         + self.rate_ema_alpha * rate)

        # ⑦ raw 空间排斥正则：饱和区唯一存活的梯度信号（clamp 防复发、排斥管恢复）
        #    L_tau_reg = λ_tau · mean_j(ReLU(|raw_j| - raw_soft)²)
        #    ∂L/∂raw_j = 2·λ_tau·ReLU(|raw_j|-raw_soft)·sign(raw_j) —— 无 sigmoid/tanh 衰减
        tau_reg = F.relu(raw.abs() - self.raw_soft).square().mean()

        extras = {
            "tau_mean": tau.mean().item(),
            "tau_std": tau.std().item(),              # 内容自适应指标（崩前 ≈0.03）
            "raw_mean": raw.mean().item(),            # raw 是否接近 clamp 边界
            "rate_j_std": rate_j.std().item(),        # per-sample 使用率分化
            "rate_max": rate_max,
            "tau_reg": tau_reg.item(),
            "logits_std": logits.std(dim=1).mean().item(),   # NaN 诊断
            "tau_ref": tau_ref.mean().item(),
            "k_used_mean": hard.sum(1).float().mean().item(),
            "rate": rate.item(),
            "kl": kl.item(),
            "beta": self.beta,
            "T": T,
            "gumbel_coef": g,
        }
        return {"mask": mask, "soft": soft, "budget_loss": budget,
                "kl_loss": kl, "ent_loss": ent, "tau_reg": tau_reg,
                "tau": tau, "extras": extras}
