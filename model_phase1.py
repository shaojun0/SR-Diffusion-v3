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

    # ── two-stage control ──
    def set_stage(self, stage: int):
        """stage=1: fix τ (all tokens kept), no rate penalty.
        stage=2: learn τ (content-adaptive budget)."""
        self.fixed_tau = -4.0 if stage == 1 else None

    def set_lambda_rate(self, value: float):
        self.lambda_rate = value

    # ── forward ──
    def forward(self, x: Tensor, hard_mode: bool = False) -> dict:
        """
        Args:
            x: (B,3,224,224) normalized images
            hard_mode: inference — physically prune to selected tokens
        Returns dict with loss components + stats.
        """
        B = x.shape[0]

        # ── pass 1: frozen DINOv2 ──
        with torch.no_grad():
            out = self.dinov2(x)
            feats = out.last_hidden_state      # (B, 257, D)
        cls = feats[:, 0]                      # (B, D)
        patch = feats[:, 1:]                   # (B, 256, D)

        # ── per-token importance ──
        logits = self.score_head(patch) * 3.0   # (B,256) fixed scale: widen
                                                # distribution so τ stays in range

        # ── threshold gate (τ inside mask ⇒ gradients reach RateHead) ──
        if self.fixed_tau is not None:
            tau = torch.full((B, 1), self.fixed_tau, device=x.device)
        else:
            # bounded threshold: 4·tanh keeps τ in [-4, 4] so the gate never
            # saturates beyond the logits scale (prevents budget collapse)
            tau = 4.0 * torch.tanh(self.rate_head(cls))   # (B,1) ∈ [-4, 4]
        mask, soft = self.gate(logits, tau)    # (B,256) each

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

        # rate penalty = expected token fraction (differentiable via soft mask)
        rate = soft.mean()
        # entropy anti-collapse: averaged selection should not degenerate
        avg_sel = soft.mean(dim=0)             # (256,)
        ent = -(avg_sel * torch.log(avg_sel + 1e-8) +
                (1 - avg_sel) * torch.log(1 - avg_sel + 1e-8)).mean()

        loss = recon + self.lambda_rate * rate + self.lambda_ent * ent

        with torch.no_grad():
            k_used = (mask > 0.5).sum(dim=1)   # (B,) hard token count
            stats = {
                "recon_l1": recon.item(),
                "tau_mean": tau.mean().item(),
                "rate": rate.item(),
                "usage": (mask > 0.5).float().mean().item(),
                "k_used_mean": k_used.float().mean().item(),
                "k_used_min": k_used.float().min().item(),
                "k_used_max": k_used.float().max().item(),
                "entropy": ent.item(),
            }

        return {"loss": loss, "F_hat": F_hat, "mask": mask,
                "tau": tau, "stats": stats}
