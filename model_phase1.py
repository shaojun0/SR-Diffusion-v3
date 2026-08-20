"""
SR-Diffusion Phase 1: Adaptive Token Budget + Feature-Space Reconstruction
=========================================================================

Core idea (visual BPE): learn the data distribution to decide HOW MANY tokens
each image needs.  A blank image should compress to ~1 token; a dense image
keeps many.  No fixed-dimension bottleneck, no per-pixel reconstruction.

Architecture (two-pass):

    x (B,3,224,224)
      │  pass 1: frozen DINOv2
      ▼
    F_patch (B,256,768) + cls (B,768)
      │
      ├─ RateHead(cls) ──────────────► k_soft (B,1) ∈ [0,1]   ← learned budget
      ├─ ScoreHead(F_patch) ─────────► logits (B,256)          ← per-token importance
      │
      ▼  differentiable top-k (Gumbel-Sigmoid + STE)
    mask (B,256)  ≈ one-hot over the 256 candidate tokens
      │
      ▼  pass 2: re-encoder consumes gated semantic tokens
    z (B,257,D)  [cls + k selected semantic tokens]
      │
      ▼  feature-space decoder (cross-attention over 256 positions)
    F_hat (B,256,768)
      │
      ▼
    L = L1(F_hat, F_patch) + λ_rate·k_soft + λ_consist·consistency + λ_ent·H

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
# Re-Encoder — pass 2, consumes only the selected semantic tokens
# ═══════════════════════════════════════════════════════════════

class ReEncoder(nn.Module):
    """
    Lightweight transformer that re-encodes the gated tokens.

    During training: all N tokens enter (gated by mask), stable gradients.
    During inference: only the top-k tokens enter (hard pruning), O(k²) cost.

    Input:  gated tokens (B, N, D) + cls prepended → (B, N+1, D)
    Output: z (B, N+1, D)
    """

    def __init__(self, dim: int = 768, depth: int = 4, heads: int = 8,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 1025, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, gated_tokens: Tensor, hard_mode: bool = False,
                hard_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            gated_tokens: (B,N,D) already masked (selected→value, others→~0)
            hard_mode: if True, physically prune to top-k tokens (inference)
            hard_mask: (B,N) 0/1 hard selection used in hard_mode
        Returns:
            z: (B, N+1, D)  (or (B, max_k+1, D) in hard_mode)
        """
        B, N, D = gated_tokens.shape
        pad_mask = None
        if hard_mode and hard_mask is not None:
            lengths = hard_mask.sum(1).long()                    # (B,) per-image k
            max_k = int(lengths.max().item())
            idx = hard_mask.topk(max_k, dim=1).indices           # (B,max_k)
            gated_tokens = gated_tokens.gather(
                1, idx.unsqueeze(-1).expand(-1, -1, D))          # (B,max_k,D)
            N = max_k
            # True where padded (beyond this image's k)
            pad_mask = (torch.arange(max_k, device=hard_mask.device)
                        .unsqueeze(0) >= lengths.unsqueeze(1))   # (B,max_k)
            pad_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool,
                                              device=hard_mask.device),
                                  pad_mask], dim=1)              # (B,max_k+1) cls never pad
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, gated_tokens], dim=1)                # (B, N+1, D)
        x = x + self.pos_embed[:, :N + 1, :]
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)
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

    Args:
        dinov2: frozen Dinov2Model (transformers).  Input 224×224, patch 14
                → 256 patch tokens + cls.
        num_patches: candidate token count (256 for 224²/patch14)
        dim: DINO hidden size (768 for base, 1536 for giant)
        T: gate temperature (annealed in stage 2)
        lambda_rate: budget penalty weight (annealed in stage 2)
        lambda_ent: entropy penalty on the averaged selection (anti-collapse)
        fixed_tau: stage-1 mode — if not None, τ pinned to this value
                   (e.g. -8 ⇒ all tokens kept)
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
        self.re_encoder = ReEncoder(dim=dim)
        self.decoder = FeatureDecoder(dim=dim, num_patches=num_patches)

        # zero-init the RateHead output so stage-2 starts at τ≈0 (soft≈0.5):
        # both recon and rate gradients are then non-zero → no gate saturation
        nn.init.zeros_(self.rate_head.mlp[3].weight)
        nn.init.zeros_(self.rate_head.mlp[3].bias)

        # PPO-style budget stabilization (optional, enabled by train script)
        self.use_trust_region = False
        self.rate_head_target = None   # EMA reference policy, built on demand
        self.trust_region = None       # BudgetTrustRegion instance

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
            hard_mode: inference — physically prune to selected tokens
        Returns dict with loss components + stats.
        """
        x = pixel_values
        B = x.shape[0]

        # ── pass 1: frozen DINOv2 ──
        with torch.no_grad():
            out = self.dinov2(x)
            feats = out.last_hidden_state      # (B, 257, D)
        cls = feats[:, 0]                      # (B, D)
        patch = feats[:, 1:]                   # (B, 256, D)

        # ── per-token importance (relative ranking only) ──
        logits = self.score_head(patch)        # (B,256) raw scores
        # per-sample normalize: ScoreHead learns WHICH tokens matter (ranking),
        # RateHead learns HOW MANY (absolute τ).  This prevents an arms race
        # where ScoreHead inflates all logits past a bounded τ → gate always on.
        logits = (logits - logits.mean(dim=1, keepdim=True)) / \
                 (logits.std(dim=1, keepdim=True) + 1e-5)

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
            tau = mask.new_zeros(B, 1)  # not used directly; extras carry stats
            budget_loss, kl_loss, ent_loss = (tr["budget_loss"], tr["kl_loss"],
                                              tr["ent_loss"])
            tr_extras = tr["extras"]
            # stash KL for the callback's adaptive-β update
            self.trust_region._last_kl = float(kl_loss.detach().cpu())
        else:
            # bounded threshold: 2·tanh keeps τ in [-2, 2], always INSIDE the
            # normalized logits distribution (N(0,1)).  If τ left the support,
            # sigmoid would saturate → soft(1-soft)→0 → both recon and rate
            # gradients vanish → τ stuck (deadlock we hit with 4·tanh).
            tau = 2.0 * torch.tanh(self.rate_head(cls))   # (B,1) ∈ [-2, 2]
            mask, soft = self.gate(logits, tau)    # (B,256) each
            budget_loss = torch.zeros((), device=x.device)
            kl_loss = torch.zeros((), device=x.device)
            ent_loss = torch.zeros((), device=x.device)
            tr_extras = {}

        # ── gate features (train: soft; eval: hard) ──
        if hard_mode:
            hard_mask = (mask > 0.5).float()
            gated = patch * hard_mask.unsqueeze(-1)
        else:
            gated = patch * mask.unsqueeze(-1)  # gradients flow via STE mask

        # ── pass 2: re-encode selected semantic tokens ──
        z = self.re_encoder(gated, hard_mode=hard_mode,
                            hard_mask=(mask > 0.5).float() if hard_mode else None)
        # (B, N+1, D) or (B, k+1, D)

        # ── feature-space reconstruction ──
        F_hat = self.decoder(z)                # (B, 256, D)

        # ── losses ──
        recon = F.l1_loss(F_hat, patch, reduction="mean")

        if self.use_trust_region and self.trust_region is not None:
            # PPO-KL variant: budget interval (slow EMA) + trust region KL
            beta = self.trust_region.beta
            loss = recon + budget_loss + beta * kl_loss + self.lambda_ent * ent_loss
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
        k_min: int = 2,
        k_max: int = 250,
        T: float = 1.0,
        T_min: float = 0.3,
        anneal_steps: int = 1000,
        gumbel_steps: int = 800,
        rate_ema_alpha: float = 0.01,
        rate_min: float = 0.03,
        rate_max: float = 0.25,
        lambda_rate: float = 0.3,
        lambda_ent: float = 0.01,
        ref_momentum: float = 0.99,
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
        """Dual-gradient update for β (PPO's adaptive KL penalty)."""
        self.kl_ema = (1 - self.kl_ema_alpha) * self.kl_ema + self.kl_ema_alpha * kl_step
        self.beta = min(max(self.beta * math.exp(self.beta_eta * (self.kl_ema - self.kl_target)),
                            self.beta_min), self.beta_max)

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

        # ① τ with cap around reference policy
        tau_net = 2.0 * torch.tanh(model.rate_head(cls))            # (B,1) ∈ [-2,2]
        with torch.no_grad():
            tau_ref = 2.0 * torch.tanh(model.rate_head_target(cls))
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
        mask = hard + (soft_noisy - hard).detach()                  # STE path (unchanged)

        # ④ clean soft: logits detached → budget terms only push τ
        soft = torch.sigmoid((logits.detach() - tau) / T)
        with torch.no_grad():
            soft_ref = torch.sigmoid((logits - tau_ref) / T)

        # ⑤ trust region: KL(Bern(p_ref) ‖ Bern(p))
        eps = 1e-5
        pr = soft_ref.clamp(eps, 1 - eps)
        pc = soft.clamp(eps, 1 - eps)
        kl = (pr * (pr.log() - pc.log()) +
              (1 - pr) * ((1 - pr).log() - (1 - pc).log())).mean()
        # entropy (on clean soft)
        ent = -(pc * pc.log() + (1 - pc) * (1 - pc).log()).mean()

        # ⑥ budget interval penalty (hinge, rate-dependent)
        #    Out-of-range decided by detach(rate) (stable); gradient flows
        #    through rate → τ.  Inside [r_min, r_max] → zero penalty, so no
        #    equilibrium gradient cancellation (the λ cliff we measured).
        rate = soft.mean()
        rate_d = rate.detach()
        budget = self.lambda_rate * (F.relu(rate_d - self.rate_max) ** 2
                                     + F.relu(self.rate_min - rate_d) ** 2)
        # gradient path: the squared hinge is w.r.t. rate_d (const), so attach
        # a linear pass-through so τ still receives push when out of range
        if float(budget.detach().cpu()) > 0:
            budget = budget + self.lambda_rate * 0.1 * rate

        extras = {
            "tau_mean": tau.mean().item(),
            "tau_ref": tau_ref.mean().item(),
            "k_used_mean": hard.sum(1).float().mean().item(),
            "rate": rate.item(),
            "kl": kl.item(),
            "beta": self.beta,
            "T": T,
            "gumbel_coef": g,
        }
        return {"mask": mask, "soft": soft, "budget_loss": budget,
                "kl_loss": kl, "ent_loss": ent, "extras": extras}
