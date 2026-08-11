"""
SR-Diffusion v12: SVD-Compressed Visual Encoder with Adversarial Reconstruction

Architecture:
  224×224 Image → DINOv2-giant (unfrozen) → F ∈ R^(B,257,1536)
    → SVD + top-k truncation (straight-through) → F_hat ∈ R^(B,257,1536)
    → [CLS removed, 256 patch features]
    → Projection → Reshape → Conv layers → (B,4,28,28) latent
    → SD VAE Decoder (unfrozen) → (B,3,224,224) reconstruction
    → Discriminator (NLayerPatchGAN)

Design: ALL parameters unfrozen — DINOv2, VAE decoder, bridge, discriminator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
from transformers import Dinov2Model
from diffusers import AutoencoderKL


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
        # full_matrices=False → U:(B,N,min(N,D)), S:(B,min(N,D)), Vh:(B,min(N,D),D)
        U, S, Vh = torch.linalg.svd(F.float(), full_matrices=False)
        U = U.to(F.dtype)
        S = S.to(F.dtype)
        Vh = Vh.to(F.dtype)

        # ── Energy-based truncation ──
        S_sq = S ** 2                          # (B, r)  r = min(N,D)
        total_energy = S_sq.sum(dim=1, keepdim=True)  # (B, 1)
        cum_energy = torch.cumsum(S_sq, dim=1)  # (B, r)
        # keep k where cum_energy / total_energy >= threshold
        cond = cum_energy / (total_energy + 1e-8) < self.energy_threshold
        k = cond.sum(dim=1) + 1                # (B,)
        k = k.clamp(min=self.min_k, max=self.max_k)

        # ── Low-rank reconstruction (per-sample for correct per-sample k) ──
        F_hat_list = []
        for i in range(B):
            ki = int(k[i].item())
            # F_hat_i = U_i[:,:ki] @ diag(S_i[:ki]) @ Vh_i[:ki,:]
            F_hat_i = (U[i, :, :ki] * S[i, None, :ki]) @ Vh[i, :ki, :]
            F_hat_list.append(F_hat_i)
        F_hat = torch.stack(F_hat_list, dim=0)  # (B, N, D)

        # ── Straight-through estimator ──
        # forward uses truncated F_hat; backward propagates through F_hat (and thus F)
        # detach() breaks the backward path from the subtraction term
        F_out = F_hat + (F - F_hat).detach()

        return F_out, k


# ═══════════════════════════════════════════════════════════════
# VAE Bridge — DINOv2 patch features → SD VAE latent
# ═══════════════════════════════════════════════════════════════

class VAEBridge(nn.Module):
    """
    Maps DINOv2 patch tokens (B, 256, 1536) → SD VAE latent (B, 4, 28, 28).

    Pipeline:
      256 tokens × 1536 → Linear(1536→proj_dim) → (B, 256, proj_dim)
      → reshape → (B, proj_dim, 16, 16)           [16×16 patch grid]
      → Conv + Upsample → (B, 4, 28, 28)          [28×28 SD VAE latent]

    Args:
        in_dim: DINOv2 hidden dim (1536 for giant)
        proj_dim: intermediate projection dim (default 256)
        latent_channels: SD VAE latent channels (4)
        latent_size: SD VAE latent spatial size (28 for 224px input)
    """

    def __init__(
        self,
        in_dim: int = 1536,
        proj_dim: int = 256,
        latent_channels: int = 4,
        latent_size: int = 28,
    ):
        super().__init__()
        self.latent_size = latent_size

        # Project token dimension
        self.token_proj = nn.Linear(in_dim, proj_dim)

        # Conv decoder: (proj_dim, 16, 16) → (latent_channels, latent_size, latent_size)
        self.conv_net = nn.Sequential(
            # (proj_dim, 16, 16) → (proj_dim * 2, 16, 16)
            nn.Conv2d(proj_dim, proj_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, proj_dim * 2),
            nn.SiLU(),
            # (proj_dim * 2, 16, 16) → (proj_dim, 16, 16)
            nn.Conv2d(proj_dim * 2, proj_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, proj_dim),
            nn.SiLU(),
            # (proj_dim, 16, 16) → (proj_dim // 2, 16, 16)
            nn.Conv2d(proj_dim, proj_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, proj_dim // 2),
            nn.SiLU(),
        )

        # Upsampling + final conv to reach (latent_channels, latent_size, latent_size)
        self.upsample = nn.Upsample(size=(latent_size, latent_size), mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(proj_dim // 2, proj_dim // 4, kernel_size=3, padding=1),
            nn.GroupNorm(4, proj_dim // 4),
            nn.SiLU(),
            nn.Conv2d(proj_dim // 4, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, patch_tokens: Tensor) -> Tensor:
        """
        Args:
            patch_tokens: (B, 256, in_dim) — DINOv2 patch features (CLS removed)
        Returns:
            latent: (B, latent_channels, latent_size, latent_size)
        """
        B = patch_tokens.shape[0]

        # Project token dimension
        x = self.token_proj(patch_tokens)  # (B, 256, proj_dim)

        # Reshape to spatial: 256 = 16×16 patch grid
        proj_dim = x.shape[-1]
        x = x.transpose(1, 2).reshape(B, proj_dim, 16, 16)  # (B, proj_dim, 16, 16)

        # Conv processing
        x = self.conv_net(x)  # (B, proj_dim//2, 16, 16)

        # Upsample 16×16 → 28×28
        x = self.upsample(x)  # (B, proj_dim//2, 28, 28)

        # Final projection to VAE latent channels
        latent = self.final_conv(x)  # (B, 4, 28, 28)

        return latent


# ═══════════════════════════════════════════════════════════════
# NLayerPatchGAN Discriminator (pix2pix / CycleGAN style)
# ═══════════════════════════════════════════════════════════════

class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator with configurable number of layers.

    Input:  (B, 3, H, W)
    Output: (B, 1, H', W')  patch-level real/fake logits

    Default n_layers=5 gives receptive field ~70×70.

    Args:
        in_channels: input channels (3 for RGB)
        ndf: base filter count (default 64)
        n_layers: number of discriminator layers (default 5)
        norm_layer: normalization (default InstanceNorm2d, no learnable params)
    """

    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 5,
    ):
        super().__init__()
        norm_layer = nn.InstanceNorm2d

        # Input: (in_channels, H, W) → (ndf, H/2, W/2)
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Middle layers with doubling filters
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers - 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # cap at 512
            layers += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=2 if n < n_layers - 2 else 1,
                    padding=1,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Final output layer: (ndf*nf_mult, H', W') → (1, H', W')
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
# SR-Diffusion v12 — Full Generator (Encoder + SVD + Bridge + VAE Decoder)
# ═══════════════════════════════════════════════════════════════

class SRDiffusionV12(nn.Module):
    """
    SVD-Compressed DINOv2 Encoder → VAE Bridge → SD VAE Decoder → Reconstruction.

    All parameters unfrozen: DINOv2, VAE decoder, bridge, discriminator.

    Args:
        dinov2: Dinov2Model instance (unfrozen)
        vae_decoder: AutoencoderKL instance (decoder only, unfrozen)
        bridge_dim: VAEBridge projection dim
        svd_energy_threshold: SVD energy threshold
        svd_max_k: maximum singular values to keep
    """

    def __init__(
        self,
        dinov2: Dinov2Model,
        vae_decoder: AutoencoderKL,
        bridge_dim: int = 256,
        svd_energy_threshold: float = 0.99,
        svd_max_k: int = 64,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.vae_decoder = vae_decoder

        # SVD truncation module
        self.svd_trunc = SVDTruncation(
            energy_threshold=svd_energy_threshold,
            max_k=svd_max_k,
            min_k=1,
        )

        # VAE Bridge: 256 patch tokens × 1536 → (4, 28, 28) latent
        self.bridge = VAEBridge(
            in_dim=dinov2.config.hidden_size,  # 1536 for giant
            proj_dim=bridge_dim,
            latent_channels=4,
            latent_size=28,
        )

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            images: (B, 3, 224, 224) preprocessed (DINOv2 normalization)
        Returns:
            reconstruction: (B, 3, 224, 224)
            features: (B, 257, 1536) raw DINOv2 features (before SVD)
            k: (B,) number of kept singular values per sample
        """
        # ── 1. DINOv2 encoding ──
        dino_out = self.dinov2(pixel_values=images)
        features = dino_out.last_hidden_state  # (B, 257, 1536)

        # ── 2. SVD truncation with straight-through ──
        features_trunc, k = self.svd_trunc(features)  # (B, 257, 1536)

        # ── 3. Extract patch tokens (remove CLS at index 0) ──
        patch_tokens = features_trunc[:, 1:, :]  # (B, 256, 1536)

        # ── 4. Bridge: patch tokens → VAE latent ──
        latent = self.bridge(patch_tokens)  # (B, 4, 28, 28)

        # ── 5. VAE decode ──
        # Scale to VAE native range (SD VAE expects latents scaled by 1/0.18215)
        latent_scaled = latent / self.vae_decoder.config.scaling_factor  # 0.18215
        decoded = self.vae_decoder.decode(latent_scaled).sample  # (B, 3, 224, 224)

        return decoded, features, k

    def train(self, mode: bool = True):
        """Override to ensure all submodules are trainable."""
        super().train(mode)
        self.dinov2.train(mode)
        self.vae_decoder.train(mode)
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
                'params': list(self.bridge.parameters()) + list(self.vae_decoder.parameters()),
                'lr': lr_dec,
                'name': 'decoder',
            },
        ]


# ═══════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════

def build_model_v12(
    dino_path: Optional[str] = None,
    vae_path: str = "stabilityai/sd-vae-ft-ema",
    bridge_dim: int = 256,
    svd_energy_threshold: float = 0.99,
    svd_max_k: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[SRDiffusionV12, NLayerDiscriminator]:
    """
    Build SR-Diffusion v12 model and discriminator.

    Args:
        dino_path: path to DINOv2-giant (local or HF repo id).
                   Default: "facebook/dinov2-giant"
        vae_path: path to SD VAE (default: "stabilityai/sd-vae-ft-ema")
        bridge_dim: VAEBridge projection dim
        svd_energy_threshold: SVD energy threshold
        svd_max_k: max singular values
        device: torch device

    Returns:
        (generator, discriminator) tuple
    """
    if dino_path is None:
        dino_path = "facebook/dinov2-giant"

    print(f"[build_model_v12] Loading DINOv2 from: {dino_path}")
    dinov2 = Dinov2Model.from_pretrained(dino_path)

    print(f"[build_model_v12] Loading VAE decoder from: {vae_path}")
    vae = AutoencoderKL.from_pretrained(vae_path)

    print(f"[build_model_v12] Constructing SR-Diffusion v12...")
    generator = SRDiffusionV12(
        dinov2=dinov2,
        vae_decoder=vae,
        bridge_dim=bridge_dim,
        svd_energy_threshold=svd_energy_threshold,
        svd_max_k=svd_max_k,
    )

    discriminator = NLayerDiscriminator(in_channels=3, ndf=64, n_layers=5)

    if device is not None:
        generator = generator.to(device)
        discriminator = discriminator.to(device)

    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters()) / 1e6
    g_trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad) / 1e6
    d_params = sum(p.numel() for p in discriminator.parameters()) / 1e6

    print(f"[build_model_v12] Generator: {g_params:.1f}M params ({g_trainable:.1f}M trainable)")
    print(f"[build_model_v12] Discriminator: {d_params:.1f}M params")
    print(f"[build_model_v12] SVD config: energy_threshold={svd_energy_threshold}, max_k={svd_max_k}")
    print(f"[build_model_v12] Bridge: in_dim=1536 → proj_dim={bridge_dim} → (4, 28, 28)")

    return generator, discriminator
