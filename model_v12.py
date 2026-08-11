"""
SR-Diffusion v12: SVD-Compressed Encoder + Diffusion UNet Decoder (WITHOUT VAE)

Architecture:
  224×224 Image → DINOv2-giant (unfrozen) → F ∈ R^(B,257,1536)
    → SVD + top-k truncation (straight-through) → F_hat ∈ R^(B,257,1536)
    → [CLS removed, 256 patch features] → (B, 256, 1536)
    → ConditioningProjector: Linear(1536→64) → reshape(16×16) → upsample(224×224)
    → DiffusionDecoder UNet: noisy_img (3) concat cond (64) → predicted noise
    → DDIM sampling → reconstructed image

Design: ALL parameters unfrozen — DINOv2, diffusion decoder, projector, discriminator.

CRITICAL FIX (v12):
  - STE formula corrected: F_out = F + (F_hat - F).detach()  [was inverted]
  - VAE decoder replaced with diffusion UNet decoder for high-fidelity reconstruction
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List
from transformers import Dinov2Model


# ═══════════════════════════════════════════════════════════════
# SVD Truncation Module (batched SVD + energy truncation + STE)
# ═══════════════════════════════════════════════════════════════

class SVDTruncation(nn.Module):
    """
    Batched SVD with energy-based truncation and straight-through estimator.

    Input:  F ∈ R^(B, N, D)   — feature matrix from DINOv2
    Output: F_hat ∈ R^(B, N, D) — low-rank approx (forward: truncated, backward: STE)
    Also returns k (kept singular values per sample) for logging.

    Args:
        energy_threshold: cumulative energy ratio to keep (default 0.99)
        max_k: hard cap on number of singular values
        min_k: minimum number of singular values to keep
    """

    def __init__(self, energy_threshold: float = 0.99, max_k: int = 64, min_k: int = 1):
        super().__init__()
        self.energy_threshold = energy_threshold
        self.max_k = max_k
        self.min_k = min_k

    def forward(self, F: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            F: (B, N, D) feature matrix
        Returns:
            F_out: (B, N, D) truncated + straight-through output
            k: (B,) integer tensor of kept singular values per sample
        """
        B, N, D = F.shape

        # ── Batched SVD ──
        U, S, Vh = torch.linalg.svd(F.float(), full_matrices=False)
        U = U.to(F.dtype)
        S = S.to(F.dtype)
        Vh = Vh.to(F.dtype)

        # ── Energy-based truncation ──
        S_sq = S ** 2
        total_energy = S_sq.sum(dim=1, keepdim=True)
        cum_energy = torch.cumsum(S_sq, dim=1)
        cond = cum_energy / (total_energy + 1e-8) < self.energy_threshold
        k = cond.sum(dim=1) + 1
        k = k.clamp(min=self.min_k, max=self.max_k)

        # ── Low-rank reconstruction (per-sample) ──
        F_hat_list = []
        for i in range(B):
            ki = int(k[i].item())
            F_hat_i = (U[i, :, :ki] * S[i, None, :ki]) @ Vh[i, :ki, :]
            F_hat_list.append(F_hat_i)
        F_hat = torch.stack(F_hat_list, dim=0)  # (B, N, D)

        # ── Straight-through estimator ──
        # CORRECT: forward uses F_hat (truncated), backward gradient flows through F
        # Standard STE pattern: output = x + sg(quantize(x) - x)
        # Here: output = F + sg(F_hat - F) = F + (F_hat - F).detach()
        # Forward: F_out = F + (F_hat - F) = F_hat  ✓ (truncated)
        # Backward: ∂L/∂F = ∂L/∂F_out (detach blocks gradient through F_hat→F path)
        F_out = F + (F_hat - F).detach()

        return F_out, k


# ═══════════════════════════════════════════════════════════════
# Noise Scheduler (Cosine schedule)
# ═══════════════════════════════════════════════════════════════

class NoiseScheduler:
    """
    Cosine noise schedule for diffusion.

    Args:
        timesteps: total diffusion steps (default 1000)
        schedule: 'cosine' only (cosine schedule better for reconstruction)
        clip_min: minimum beta clamp
        clip_max: maximum beta clamp
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = 'cosine',
        clip_min: float = 1e-4,
        clip_max: float = 0.02,
    ):
        self.timesteps = timesteps

        if schedule == 'cosine':
            s = 0.008
            steps = timesteps + 1
            t = torch.arange(steps, dtype=torch.float32)
            ft = torch.cos(((t / timesteps + s) / (1 + s)) * (math.pi / 2)) ** 2
            alphas_cumprod = ft / ft[0]  # normalize so alpha_bar_0 = 1
            betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.clamp(betas, min=clip_min, max=clip_max)

            self.betas = betas
            self.alphas = 1.0 - betas
            self.alphas_cumprod = alphas_cumprod[1:]  # (T,)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Pre-compute useful values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """
        Forward diffusion process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x: clean images (B, C, H, W)
            noise: random noise (B, C, H, W)
            t: timesteps (B,) long tensor
        Returns:
            noisy images (B, C, H, W)
        """
        sqrt_alpha = self.sqrt_alphas_cumprod.to(x.device)[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(x.device)[t][:, None, None, None]
        return sqrt_alpha * x + sqrt_one_minus * noise

    def to(self, device):
        """Move pre-computed tensors to device (no-op for now, handled per call)."""
        return self


# ═══════════════════════════════════════════════════════════════
# Conditioning Projector
# ═══════════════════════════════════════════════════════════════

class ConditioningProjector(nn.Module):
    """
    Maps SVD-compressed DINOv2 patch features to spatial conditioning grid.

    (B, 256, 1536) → Linear(1536→cond_dim) → reshape (B, cond_dim, 16, 16)
    → Conv refinement at 16×16 → bilinear upsample → (B, cond_dim, 224, 224)
    → refinement conv layers at 224×224 for smooth conditioning

    Args:
        in_dim: DINOv2 hidden dim (1536 for giant)
        cond_dim: conditioning channels output (default 64)
    """

    def __init__(self, in_dim: int = 1536, cond_dim: int = 64):
        super().__init__()
        self.cond_dim = cond_dim

        # Project token dimension and expand
        self.token_proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, cond_dim * 2),
            nn.GELU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )

        # Conv refinement at 16×16 (DINOv2 patch grid)
        self.conv_16 = nn.Sequential(
            nn.Conv2d(cond_dim, cond_dim, 3, padding=1),
            nn.GroupNorm(min(8, cond_dim), cond_dim),
            nn.SiLU(),
            nn.Conv2d(cond_dim, cond_dim, 3, padding=1),
            nn.GroupNorm(min(8, cond_dim), cond_dim),
            nn.SiLU(),
        )

        # Conv refinement at 224×224 (after upsampling)
        self.conv_224 = nn.Sequential(
            nn.Conv2d(cond_dim, cond_dim, 3, padding=1),
            nn.GroupNorm(min(8, cond_dim), cond_dim),
            nn.SiLU(),
            nn.Conv2d(cond_dim, cond_dim, 3, padding=1),
        )

    def forward(self, patch_tokens: Tensor) -> Tensor:
        """
        Args:
            patch_tokens: (B, 256, in_dim) — DINOv2 patch features (CLS removed)
        Returns:
            cond: (B, cond_dim, 224, 224) spatial conditioning grid
        """
        B = patch_tokens.shape[0]

        # Project tokens
        x = self.token_proj(patch_tokens)  # (B, 256, cond_dim)

        # Reshape to spatial: 256 = 16×16
        x = x.transpose(1, 2).reshape(B, self.cond_dim, 16, 16)  # (B, cond_dim, 16, 16)

        # Process at 16×16
        x = self.conv_16(x)  # (B, cond_dim, 16, 16)

        # Upsample to 224×224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Refine at 224×224
        x = self.conv_224(x)  # (B, cond_dim, 224, 224)

        return x


# ═══════════════════════════════════════════════════════════════
# Diffusion UNet Decoder (lightweight)
# ═══════════════════════════════════════════════════════════════

def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal time embedding (transformer-style)."""
    device = t.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
    emb = t[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class TimestepEmbedding(nn.Module):
    """Sinusoidal → Linear → SiLU → Linear time embedding."""

    def __init__(self, dim: int, time_emb_dim: int = 256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.linear1 = nn.Linear(time_emb_dim, dim)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, t: Tensor) -> Tensor:
        emb = sinusoidal_embedding(t, self.time_emb_dim)
        emb = self.linear1(emb)
        emb = self.silu(emb)
        emb = self.linear2(emb)
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W)
            t_emb: (B, time_emb_dim)
        Returns:
            (B, out_ch, H, W)
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # Time conditioning (broadcast)
        h = h + self.time_proj(t_emb)[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return self.skip(x) + h


class Attention(nn.Module):
    """Multi-head self-attention on 2D feature maps."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)  # 3 × (B, inner_dim, H, W)
        q, k, v = [t.reshape(B, self.heads, self.dim_head, H * W).transpose(-2, -1)
                    for t in qkv]  # each (B, heads, H*W, dim_head)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, H*W, H*W)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(-2, -1).reshape(B, self.heads * self.dim_head, H, W)
        return self.to_out(out)


class DownBlock(nn.Module):
    """Downsampling block: stride-2 conv → ResBlocks → optional Attention."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int,
                 num_resblocks: int = 2, use_attention: bool = False):
        super().__init__()
        self.downsample = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.resblocks = nn.ModuleList([
            ResBlock(out_ch, out_ch, time_emb_dim) for _ in range(num_resblocks)
        ])
        self.attn = Attention(out_ch) if use_attention else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        x = self.downsample(x)
        for resblock in self.resblocks:
            x = resblock(x, t_emb)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block: 2× upsample → conv → skip concat → ResBlocks → optional Attention."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, time_emb_dim: int,
                 num_resblocks: int = 2, use_attention: bool = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # First ResBlock: out_ch + skip_ch → out_ch
        # Subsequent ResBlocks: out_ch → out_ch
        self.resblocks = nn.ModuleList()
        self.resblocks.append(ResBlock(out_ch + skip_ch, out_ch, time_emb_dim))
        for _ in range(num_resblocks - 1):
            self.resblocks.append(ResBlock(out_ch, out_ch, time_emb_dim))

        self.attn = Attention(out_ch) if use_attention else nn.Identity()

    def forward(self, x: Tensor, skip: Tensor, t_emb: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.upsample_conv(x)
        x = torch.cat([x, skip], dim=1)
        for resblock in self.resblocks:
            x = resblock(x, t_emb)
        x = self.attn(x)
        return x


class MidBlock(nn.Module):
    """Middle block: ResBlock → Attention → ResBlock."""

    def __init__(self, dim: int, time_emb_dim: int, use_attention: bool = True):
        super().__init__()
        self.resblock1 = ResBlock(dim, dim, time_emb_dim)
        self.attn = Attention(dim) if use_attention else nn.Identity()
        self.resblock2 = ResBlock(dim, dim, time_emb_dim)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        x = self.resblock1(x, t_emb)
        x = self.attn(x)
        x = self.resblock2(x, t_emb)
        return x


class DiffusionDecoder(nn.Module):
    """
    Lightweight UNet for diffusion-based image reconstruction.

    Input:  noisy image (B, 3, 224, 224) concatenated with conditioning (B, 64, 224, 224)
    Output: predicted noise (B, 3, 224, 224)

    Architecture:
      - 4 down levels: 112→56→28→14
      - 4 up levels: 14→28→56→112→224
      - base_dim=128, dim_mults=[1, 2, 4, 4]
      - Self-attention at 28×28, 14×14
      - Conditioning concatenated at input (B, 3+64=67, 224, 224)

    Args:
        in_channels: noisy image channels + conditioning channels (3 + 64 = 67)
        out_channels: output noise channels (3)
        base_dim: base filter dimension (default 128)
        dim_mults: channel multipliers per level
        attention_levels: which levels use self-attention (0-indexed from top)
        time_emb_dim: time embedding dimension
    """

    def __init__(
        self,
        in_channels: int = 67,   # 3 (noisy img) + 64 (conditioning)
        out_channels: int = 3,
        base_dim: int = 128,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 4),
        attention_levels: Tuple[int, ...] = (2, 3),  # 28×28 and 14×14
        time_emb_dim: int = 512,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Time embedding: sinusoidal → MLP
        self.time_embed = TimestepEmbedding(time_emb_dim)

        # Channel dimensions for each level
        dims = [base_dim] + [base_dim * m for m in dim_mults]  # [128, 128, 256, 512, 512]
        num_levels = len(dim_mults)

        # ── Input convolution ──
        self.input_conv = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        # ── Down blocks ──
        self.downs = nn.ModuleList()
        for i in range(num_levels):
            use_attn = i in attention_levels
            self.downs.append(DownBlock(
                in_ch=dims[i], out_ch=dims[i + 1],
                time_emb_dim=time_emb_dim,
                num_resblocks=2,
                use_attention=use_attn,
            ))

        # ── Mid block ──
        self.mid = MidBlock(
            dim=dims[-1],
            time_emb_dim=time_emb_dim,
            use_attention=True,
        )

        # ── Up blocks ──
        self.ups = nn.ModuleList()
        for i in reversed(range(num_levels)):
            use_attn = i in attention_levels
            self.ups.append(UpBlock(
                in_ch=dims[i + 1],      # channels from previous level
                out_ch=dims[i],          # output channels
                skip_ch=dims[i + 1],     # channels from skip (corresponding down)
                time_emb_dim=time_emb_dim,
                num_resblocks=2,
                use_attention=use_attn,
            ))

        # ── Output convolution ──
        self.output_norm = nn.GroupNorm(min(8, dims[0]), dims[0])
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv2d(dims[0], out_channels, 3, padding=1)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: noisy image (B, 3, 224, 224)
            t: diffusion timestep (B,)
            cond: conditioning grid (B, 64, 224, 224)
        Returns:
            predicted noise (B, 3, 224, 224)
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (B, time_emb_dim)

        # Concatenate conditioning
        x = torch.cat([x, cond], dim=1)  # (B, 67, 224, 224)

        # Input conv
        x = self.input_conv(x)  # (B, 128, 224, 224)

        # Down path
        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x, t_emb)

        # Mid
        x = self.mid(x, t_emb)

        # Up path
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, t_emb)

        # Output
        x = self.output_norm(x)
        x = self.output_act(x)
        x = self.output_conv(x)  # (B, 3, 224, 224)

        return x


# ═══════════════════════════════════════════════════════════════
# NLayerPatchGAN Discriminator (pix2pix / CycleGAN style)
# ═══════════════════════════════════════════════════════════════

class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator with configurable number of layers.

    Input:  (B, 3, H, W)
    Output: (B, 1, H', W')  patch-level real/fake logits

    Args:
        in_channels: input channels (3 for RGB)
        ndf: base filter count (default 64)
        n_layers: number of discriminator layers (default 5)
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 5):
        super().__init__()
        norm_layer = nn.InstanceNorm2d

        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers - 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(
                    ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=4, stride=2 if n < n_layers - 2 else 1, padding=1,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** (n_layers - 1), 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


# ═══════════════════════════════════════════════════════════════
# GAN Loss functions
# ═══════════════════════════════════════════════════════════════

class GANLoss(nn.Module):
    """
    GAN loss supporting vanilla (BCE), LSGAN (MSE), and hinge modes.

    Args:
        gan_mode: 'vanilla' | 'lsgan' | 'hinge'
        target_real: label for real images
        target_fake: label for fake images
    """

    def __init__(self, gan_mode: str = 'hinge', target_real: float = 1.0, target_fake: float = 0.0):
        super().__init__()
        self.gan_mode = gan_mode
        self.register_buffer('real_label', torch.tensor(target_real))
        self.register_buffer('fake_label', torch.tensor(target_fake))

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction: Tensor, target_is_real: bool) -> Tensor:
        target_val = self.real_label if target_is_real else self.fake_label
        return target_val.expand_as(prediction)

    def __call__(self, prediction: Tensor, target_is_real: bool) -> Tensor:
        if self.gan_mode in ('lsgan', 'vanilla'):
            target = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target)
        elif self.gan_mode == 'hinge':
            if target_is_real:
                return F.relu(1.0 - prediction).mean()
            else:
                return F.relu(1.0 + prediction).mean()
        else:
            raise ValueError(f"Unknown gan_mode: {self.gan_mode}")


# ═══════════════════════════════════════════════════════════════
# SR-Diffusion v12 — Encoder + SVD + Diffusion Decoder
# ═══════════════════════════════════════════════════════════════

class SRDiffusionV12(nn.Module):
    """
    SVD-Compressed DINOv2 Encoder → Conditioning Projector → Diffusion UNet Decoder.

    All parameters unfrozen: DINOv2, projector, diffusion decoder.

    Differences from pre-fix v12:
      - VAE decoder replaced with DiffusionDecoder (iterative diffusion)
      - VAEBridge replaced with ConditioningProjector (direct condition map)
      - STE formula FIXED: F_out = F + (F_hat - F).detach()

    Args:
        dinov2: Dinov2Model instance (unfrozen)
        conditioning_projector: ConditioningProjector
        diffusion_decoder: DiffusionDecoder (UNet)
        noise_scheduler: NoiseScheduler for diffusion
        svd_energy_threshold: SVD energy threshold
        svd_max_k: max singular values
        image_size: target image size (default 224)
    """

    def __init__(
        self,
        dinov2: Dinov2Model,
        conditioning_projector: ConditioningProjector,
        diffusion_decoder: DiffusionDecoder,
        noise_scheduler: NoiseScheduler,
        svd_energy_threshold: float = 0.99,
        svd_max_k: int = 64,
        image_size: int = 224,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.conditioning_projector = conditioning_projector
        self.diffusion_decoder = diffusion_decoder
        self.noise_scheduler = noise_scheduler
        self.image_size = image_size
        self.T = noise_scheduler.timesteps

        # SVD truncation module
        self.svd_trunc = SVDTruncation(
            energy_threshold=svd_energy_threshold,
            max_k=svd_max_k,
            min_k=1,
        )

    def encode(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode images to conditioning grid.

        Args:
            images: (B, 3, 224, 224) DINOv2-normalized
        Returns:
            cond: (B, 64, 224, 224) conditioning grid
            k: (B,) singular values kept
        """
        dino_out = self.dinov2(pixel_values=images)
        features = dino_out.last_hidden_state  # (B, 257, 1536)
        features_trunc, k = self.svd_trunc(features)
        patch_tokens = features_trunc[:, 1:, :]  # (B, 256, 1536)
        cond = self.conditioning_projector(patch_tokens)  # (B, 64, 224, 224)
        return cond, k

    def forward(
        self,
        images: Tensor,
        return_recon: bool = False,
        ddim_steps: int = 5,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        """
        Training forward: compute diffusion noise prediction loss.
        Optionally also return a quick DDIM reconstruction for GAN/discriminator.

        Args:
            images: (B, 3, 224, 224) DINOv2-normalized
            return_recon: if True, also return a quick reconstruction
            ddim_steps: number of DDIM steps for reconstruction (only if return_recon)
        Returns:
            noise_loss: scalar L1 noise prediction loss
            recon: (B, 3, 224, 224) reconstructed image (or None)
            k: (B,) singular values kept
            features: (B, 257, 1536) raw DINOv2 features (for SVD loss)
        """
        B = images.size(0)
        device = images.device

        # ── Encode ──
        dino_out = self.dinov2(pixel_values=images)
        features = dino_out.last_hidden_state  # (B, 257, 1536)

        # SVD truncation with STE
        features_trunc, k = self.svd_trunc(features)

        # Patch tokens → conditioning
        patch_tokens = features_trunc[:, 1:, :]  # (B, 256, 1536)
        cond = self.conditioning_projector(patch_tokens)  # (B, 64, 224, 224)

        # ── Diffusion noise prediction ──
        t = torch.randint(1, self.T, (B,), device=device)
        noise = torch.randn_like(images)
        noisy = self.noise_scheduler.add_noise(images, noise, t)
        noise_pred = self.diffusion_decoder(noisy, t, cond)

        # L1 loss for sharper reconstruction
        noise_loss = F.l1_loss(noise_pred, noise)

        # ── Optional: quick reconstruction for GAN ──
        recon = None
        if return_recon:
            with torch.no_grad():
                recon = self.ddim_sample_cond(cond, steps=ddim_steps)

        return noise_loss, recon, k, features

    @torch.no_grad()
    def ddim_sample_cond(self, cond: Tensor, steps: int = 50, eta: float = 0.0) -> Tensor:
        """
        DDIM sampling from conditioning grid.

        Args:
            cond: (B, 64, 224, 224) conditioning
            steps: number of DDIM steps (default 50)
            eta: noise level (0 = deterministic DDIM, 1 = DDPM)
        Returns:
            (B, 3, 224, 224) reconstructed image
        """
        B = cond.size(0)
        device = cond.device
        image_size = self.image_size

        x = torch.randn(B, 3, image_size, image_size, device=device)

        # Timesteps spaced evenly
        timesteps = torch.linspace(self.T - 1, 0, steps, device=device).long()

        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        alphas_cumprod_prev = self.noise_scheduler.alphas_cumprod_prev.to(device)

        for i in range(steps):
            t = timesteps[i]
            t_prev = timesteps[i + 1] if i < steps - 1 else torch.tensor(-1, device=device)

            t_batch = t.unsqueeze(0).expand(B)

            # Predict noise
            eps_pred = self.diffusion_decoder(x, t_batch, cond)

            # DDIM update
            alpha_t = alphas_cumprod[t]
            if t_prev >= 0:
                alpha_t_prev = alphas_cumprod[t_prev]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)

            sigma = eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t)
            ) * torch.sqrt(1 - alpha_t / alpha_t_prev)

            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)

            # Direction pointing to xt
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * eps_pred

            # Add noise if eta > 0
            noise = sigma * torch.randn_like(x) if eta > 0 else 0

            x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + noise

        return x

    @torch.no_grad()
    def sample(self, images: Tensor, steps: int = 50, eta: float = 0.0) -> Tuple[Tensor, Tensor]:
        """
        Full inference: encode image → sample reconstruction via DDIM.

        Args:
            images: (B, 3, 224, 224) DINOv2-normalized
            steps: DDIM steps (default 50)
            eta: noise level
        Returns:
            recon: (B, 3, 224, 224) reconstructed image
            k: (B,) singular values kept
        """
        cond, k = self.encode(images)
        recon = self.ddim_sample_cond(cond, steps=steps, eta=eta)
        # Clamp to valid range (DINOv2 normalized)
        recon = torch.clamp(recon, -3.0, 3.0)
        return recon, k

    def train(self, mode: bool = True):
        """Override to ensure all submodules are trainable."""
        super().train(mode)
        self.dinov2.train(mode)
        return self

    def get_trainable_params(self, lr_enc: float = 1e-4, lr_dec: float = 1e-4):
        """
        Returns parameter groups with separate learning rates.

        Returns:
            list of dicts for optimizer param_groups
        """
        return [
            {
                'params': list(self.dinov2.parameters()) + list(self.svd_trunc.parameters()),
                'lr': lr_enc,
                'name': 'encoder',
            },
            {
                'params': (list(self.conditioning_projector.parameters())
                          + list(self.diffusion_decoder.parameters())),
                'lr': lr_dec,
                'name': 'decoder',
            },
        ]


# ═══════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════

def build_model_v12(
    dino_path: Optional[str] = None,
    bridge_dim: int = 64,
    svd_energy_threshold: float = 0.99,
    svd_max_k: int = 64,
    diffusion_timesteps: int = 1000,
    unet_base_dim: int = 128,
    unet_dim_mults: Tuple[int, ...] = (1, 2, 4, 4),
    device: Optional[torch.device] = None,
    **kwargs,
) -> Tuple[SRDiffusionV12, NLayerDiscriminator]:
    """
    Build SR-Diffusion v12 model (encoder + diffusion decoder) and discriminator.

    Args:
        dino_path: path to DINOv2-giant (local or HF repo id).
                   Default: "facebook/dinov2-giant"
        bridge_dim: conditioning projector output dim (default 64)
        svd_energy_threshold: SVD energy threshold
        svd_max_k: max singular values
        diffusion_timesteps: total diffusion steps
        unet_base_dim: UNet base filter dimension
        unet_dim_mults: UNet channel multipliers per level
        device: torch device

    Returns:
        (generator, discriminator) tuple
    """
    if dino_path is None:
        dino_path = "facebook/dinov2-giant"

    print(f"[build_model_v12] Loading DINOv2 from: {dino_path}")
    dinov2 = Dinov2Model.from_pretrained(dino_path)

    print(f"[build_model_v12] Building NoiseScheduler (cosine, {diffusion_timesteps} steps)")
    noise_scheduler = NoiseScheduler(timesteps=diffusion_timesteps, schedule='cosine')

    print(f"[build_model_v12] Building ConditioningProjector (1536 → {bridge_dim} → 224×224)")
    conditioning_projector = ConditioningProjector(in_dim=1536, cond_dim=bridge_dim)

    in_ch = 3 + bridge_dim
    print(f"[build_model_v12] Building DiffusionDecoder (UNet, in_ch={in_ch}, base_dim={unet_base_dim}, "
          f"dim_mults={unet_dim_mults})")
    diffusion_decoder = DiffusionDecoder(
        in_channels=in_ch,
        out_channels=3,
        base_dim=unet_base_dim,
        dim_mults=unet_dim_mults,
        attention_levels=(2, 3),
        time_emb_dim=512,
    )

    print(f"[build_model_v12] Constructing SR-Diffusion v12...")
    generator = SRDiffusionV12(
        dinov2=dinov2,
        conditioning_projector=conditioning_projector,
        diffusion_decoder=diffusion_decoder,
        noise_scheduler=noise_scheduler,
        svd_energy_threshold=svd_energy_threshold,
        svd_max_k=svd_max_k,
        image_size=224,
    )

    discriminator = NLayerDiscriminator(in_channels=3, ndf=64, n_layers=5)

    if device is not None:
        generator = generator.to(device)
        discriminator = discriminator.to(device)

    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters()) / 1e6
    g_trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad) / 1e6
    d_params = sum(p.numel() for p in discriminator.parameters()) / 1e6

    # Per-component counts
    dino_params = sum(p.numel() for p in dinov2.parameters()) / 1e6
    proj_params = sum(p.numel() for p in conditioning_projector.parameters()) / 1e6
    unet_params = sum(p.numel() for p in diffusion_decoder.parameters()) / 1e6

    print(f"[build_model_v12] Generator: {g_params:.1f}M params ({g_trainable:.1f}M trainable)")
    print(f"[build_model_v12]   DINOv2: {dino_params:.1f}M  Projector: {proj_params:.1f}M  "
          f"Diffusion UNet: {unet_params:.1f}M")
    print(f"[build_model_v12] Discriminator: {d_params:.1f}M params")
    print(f"[build_model_v12] SVD config: energy_threshold={svd_energy_threshold}, max_k={svd_max_k}")
    print(f"[build_model_v12] Diffusion: T={diffusion_timesteps}, cosine schedule, "
          f"UNet base_dim={unet_base_dim}, dim_mults={unet_dim_mults}")

    return generator, discriminator
