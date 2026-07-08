"""
SR-Diffusion v2: SVD + DINOv2-giant → SD 2.1 U-Net Diffusion 超分辨率.

Pipeline:
  1. HR image → SVD → top-k left singular vectors
  2. eig vectors + LR tokens → DINOv2 Encoder → cross-attn tokens (1024d)
  3. LR bicubic → VAE Encoder → condition latent
  4. HR → VAE Encoder → latent → +noise → U-Net(cross_attn) → x₀ pred
  5. Loss: MSE(x₀_pred, hr_z) — pure VAE latent reconstruction

全参数训练 (2.09B), fp32 + 8-bit AdamW + gradient checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class Config:
    """训练超参 & 模型结构参数"""

    # ── Paths ──
    dino_dir: str = "dinov2-giant"
    sd_model_id: str = "stabilityai/stable-diffusion-2-1"

    # ── Model dims ──
    dino_hidden: int = 1536
    sd_cross_attn: int = 1024
    image_size: int = 1024
    latent_size: int = 128          # image_size / 8
    in_channels: int = 3
    latent_channels: int = 4

    # ── SVD ──
    lr_size: int = 32
    max_eig: int = 64
    energy_threshold: float = 0.95
    min_eig: int = 16

    # ── Diffusion ──
    train_timesteps: int = 1000
    inference_steps: int = 25
    beta_start: float = 0.00085
    beta_end: float = 0.012

    # ── Training ──
    lr: float = 5e-5
    batch_size: int = 1
    grad_accum: int = 4
    max_grad_norm: float = 1.0


# ═══════════════════════════════════════════════════════════════
# SVD Preprocessing
# ═══════════════════════════════════════════════════════════════

def svd_eigenvectors(
    img: Tensor,
    energy_threshold: float = 0.95,
    max_n: int = 64,
    min_n: int = 16,
):
    """(B, 1, H, W) grayscale → padded top-k left singular vectors (B, max_n, H)"""
    if img.dim() == 4:
        img = img[:, 0].float()
    else:
        img = img.float()

    B, H, W = img.shape
    eig_padded, n_list = [], []

    for b in range(B):
        u, s, _ = torch.linalg.svd(img[b])
        s2 = s ** 2
        n = torch.searchsorted(s2.cumsum(0), energy_threshold * s2.sum()).item() + 1
        n = max(min_n, min(n, max_n))
        eig_padded.append(u[:, :n].T)
        n_list.append(n)

    max_nb = max(n_list)
    out = torch.zeros(B, max_nb, H, device=img.device, dtype=img.dtype)
    for b, (e, n) in enumerate(zip(eig_padded, n_list)):
        out[b, :n] = e
    return out, n_list


def build_tokens(
    hr: Tensor,
    lr_size: int = 32,
    energy_threshold: float = 0.95,
    max_n: int = 64,
    min_n: int = 16,
):
    """HR (B, 3, H, W) → lr_flat (B, lr_size²), eig_tokens (B, N, H), n_list"""
    B, H, W = hr.size(0), hr.shape[2], hr.shape[3]

    lr = F.interpolate(hr, size=(lr_size, lr_size), mode="bicubic", align_corners=False)
    lr_gray = 0.2989 * lr[:, 0] + 0.5870 * lr[:, 1] + 0.1140 * lr[:, 2]
    lr_flat = lr_gray.reshape(B, lr_size * lr_size)

    hr_gray = 0.2989 * hr[:, 0] + 0.5870 * hr[:, 1] + 0.1140 * hr[:, 2]
    eig_tokens, n_list = svd_eigenvectors(
        hr_gray.unsqueeze(1), energy_threshold, max_n, min_n
    )
    return lr_flat, eig_tokens, n_list


# ═══════════════════════════════════════════════════════════════
# Noise Schedule (DDPM, SD 2.1 params)
# ═══════════════════════════════════════════════════════════════

class NoiseSchedule:
    def __init__(self, steps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, steps) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(0)
        self.steps = steps
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas = alphas_cumprod.sqrt()
        self.sqrt_one_minus = (1.0 - alphas_cumprod).sqrt()

    def add_noise(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        device = x0.device
        sa = self.sqrt_alphas[t.cpu()].to(device)
        som = self.sqrt_one_minus[t.cpu()].to(device)
        return sa.view(-1, 1, 1, 1) * x0 + som.view(-1, 1, 1, 1) * noise


# ═══════════════════════════════════════════════════════════════
# DINOv2 Encoder — SVD tokens → cross-attention features
# ═══════════════════════════════════════════════════════════════

class DinoEncoder(nn.Module):
    """DINOv2-giant encoder wrapper. All params trainable."""

    def __init__(self, cfg: Config, device: str = "cuda"):
        super().__init__()
        from transformers import Dinov2Model

        print(f"  [DINO] Loading from {cfg.dino_dir} ...")
        self.vit = Dinov2Model.from_pretrained(
            cfg.dino_dir, local_files_only=True
        ).to(device)
        self.vit.train()
        self.vit.requires_grad_(True)
        self.vit.gradient_checkpointing_enable()

        self.tok_proj = nn.Linear(cfg.lr_size ** 2, cfg.dino_hidden).to(device)
        self.eig_proj = nn.Linear(cfg.image_size, cfg.dino_hidden).to(device)
        self.pos = nn.Parameter(
            torch.randn(1, cfg.max_eig + 3, cfg.dino_hidden, device=device) * 0.02
        )
        self.cls = nn.Parameter(
            torch.randn(1, 1, cfg.dino_hidden, device=device)
        )

    def forward(
        self, lr_flat: Tensor, eig_tokens: Tensor, n_list: list
    ) -> Tensor:
        B = lr_flat.size(0)

        lr_tok = self.tok_proj(lr_flat).unsqueeze(1)          # (B, 1, H)
        eig_tok = self.eig_proj(eig_tokens)                    # (B, N, H)
        tokens = torch.cat([lr_tok, eig_tok], dim=1)           # (B, 1+N, H)

        cls_tok = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)           # (B, 2+N, H)

        S = tokens.size(1)
        pos = self.pos[:, :S, :] if S <= self.pos.size(1) else F.pad(
            self.pos, (0, 0, 0, S - self.pos.size(1))
        )
        tokens = tokens + pos

        enc = self.vit.encoder(tokens, output_hidden_states=True)
        return self.vit.layernorm(enc.last_hidden_state)


# ═══════════════════════════════════════════════════════════════
# Token Projector — 1536d → 1024d for SD cross-attention
# ═══════════════════════════════════════════════════════════════

class TokenProjector(nn.Module):
    def __init__(self, dino_dim: int = 1536, sd_dim: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dino_dim, sd_dim * 2),
            nn.GELU(),
            nn.Linear(sd_dim * 2, sd_dim),
            nn.LayerNorm(sd_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


# ═══════════════════════════════════════════════════════════════
# Diffusion Decoder — SD 2.1 U-Net + VAE
# ═══════════════════════════════════════════════════════════════

class DiffusionDecoder(nn.Module):
    """SD 2.1 U-Net (noise prediction) + VAE (latent encode/decode). All trainable."""

    def __init__(self, cfg: Config, device: str = "cuda"):
        super().__init__()
        from diffusers import UNet2DConditionModel, AutoencoderKL

        print(f"  [SD] Loading from {cfg.sd_model_id} ...")
        self.unet = UNet2DConditionModel.from_pretrained(
            cfg.sd_model_id, subfolder="unet", low_cpu_mem_usage=False
        )
        self.vae = AutoencoderKL.from_pretrained(
            cfg.sd_model_id, subfolder="vae", low_cpu_mem_usage=False
        )

        # ── Expand U-Net input 4→8 (noisy + condition latent) ──
        old = self.unet.conv_in
        new = nn.Conv2d(8, old.out_channels, **{
            k: getattr(old, k) for k in ["kernel_size", "stride", "padding"]
        })
        new.weight.data[:, :4] = old.weight.data
        new.weight.data[:, 4:] = old.weight.data[:, :4].clone() * 0.02
        new.bias.data = old.bias.data.clone()
        self.unet.conv_in = new
        self.unet.enable_gradient_checkpointing()

        self.scale = self.vae.config.scaling_factor

        # ── Move to GPU ──
        self.to(device)

    def encode(self, x: Tensor) -> Tensor:
        """Image → latent (B, 4, H/8, W/8)"""
        return self.vae.encode(x).latent_dist.sample() * self.scale

    def decode(self, z: Tensor) -> Tensor:
        """Latent → image (B, 3, H, W)"""
        return self.vae.decode(z / self.scale).sample

    def forward(
        self,
        noisy: Tensor,
        cond: Tensor,
        t: Tensor,
        cross_tokens: Tensor,
    ) -> Tensor:
        """Predict noise. Input: (B,8,H,W), output: (B,4,H,W)"""
        x = torch.cat([noisy, cond], dim=1)
        return self.unet(x, t, encoder_hidden_states=cross_tokens, return_dict=False)[0]


# ═══════════════════════════════════════════════════════════════
# SRDiffusion — full DINOv2 + SD 2.1 pipeline
# ═══════════════════════════════════════════════════════════════

class SRDiffusion(nn.Module):
    """DINOv2 + SD 2.1 U-Net diffusion 超分。全参数训练。"""

    def __init__(self, cfg: Config, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.noise_schedule = NoiseSchedule(
            cfg.train_timesteps, cfg.beta_start, cfg.beta_end
        )

        self.dino = DinoEncoder(cfg, device)
        self.projector = TokenProjector(cfg.dino_hidden, cfg.sd_cross_attn).to(device)
        self.decoder = DiffusionDecoder(cfg, device)

        self._log_params()

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dino = sum(p.numel() for p in self.dino.parameters())
        unet = sum(p.numel() for p in self.decoder.unet.parameters())
        vae  = sum(p.numel() for p in self.decoder.vae.parameters())
        print(
            f"  [Model] Total: {total/1e9:.2f}B | Trainable: {train/1e9:.2f}B | "
            f"DINO: {dino/1e6:.0f}M | U-Net: {unet/1e6:.0f}M | VAE: {vae/1e6:.0f}M"
        )

    def forward(self, hr: Tensor):
        """Training forward: hr (B,3,H,W) → loss, noise_pred, noise"""
        B, device = hr.size(0), hr.device

        # ── 1. SVD tokens via DINOv2 ──
        lr_flat, eig_tokens, n_list = build_tokens(
            hr, self.cfg.lr_size, self.cfg.energy_threshold,
            self.cfg.max_eig, self.cfg.min_eig,
        )
        svd = self.dino(lr_flat, eig_tokens, n_list)
        cross = self.projector(svd)

        # ── 2. LR condition latent ──
        lr_img = F.interpolate(
            F.interpolate(
                hr, size=(self.cfg.lr_size, self.cfg.lr_size), mode="bicubic"
            ),
            size=(self.cfg.image_size, self.cfg.image_size), mode="bicubic",
        )
        cond = self.decoder.encode(lr_img)

        # ── 3. HR → latent → add noise ──
        hr_z = self.decoder.encode(hr)
        noise = torch.randn_like(hr_z)
        t = torch.randint(0, self.cfg.train_timesteps, (B,), device=device)
        noisy = self.noise_schedule.add_noise(hr_z, noise, t)

        # ── 4. U-Net predict noise → x₀ from noise ──
        pred = self.decoder(noisy, cond, t, cross)
        alpha_bar = self.noise_schedule.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
        x0_pred = (noisy - (1.0 - alpha_bar).sqrt() * pred) / alpha_bar.sqrt().clamp(min=1e-8)

        # ── 5. Loss: pure VAE latent reconstruction ──
        loss = F.mse_loss(x0_pred, hr_z)

        return loss, pred, noise

    @torch.no_grad()
    def sample(self, hr: Tensor, steps: int = 25) -> Tensor:
        """DDIM sampling for inference."""
        B, device = hr.size(0), hr.device

        lr_flat, eig_tokens, n_list = build_tokens(
            hr, self.cfg.lr_size, self.cfg.energy_threshold,
            self.cfg.max_eig, self.cfg.min_eig,
        )
        cross = self.projector(self.dino(lr_flat, eig_tokens, n_list))

        lr_img = F.interpolate(
            F.interpolate(
                hr, size=(self.cfg.lr_size, self.cfg.lr_size), mode="bicubic"
            ),
            size=(self.cfg.image_size, self.cfg.image_size), mode="bicubic",
        )
        cond = self.decoder.encode(lr_img)

        step_ratio = self.cfg.train_timesteps // steps
        timesteps = (
            torch.arange(steps - 1, -1, -1) * step_ratio
        ).long().to(device)

        z = torch.randn(
            B, self.cfg.latent_channels,
            self.cfg.latent_size, self.cfg.latent_size, device=device,
        )

        ac = self.noise_schedule.alphas_cumprod
        for t in timesteps:
            t_b = t.expand(B)
            pred = self.decoder(z, cond, t_b, cross)
            a_t = ac[t.cpu()].to(device).view(-1, 1, 1, 1)
            a_prev = (
                ac[t.cpu() - 1].to(device).view(-1, 1, 1, 1)
                if t > 0 else torch.ones_like(a_t)
            )
            z0 = (z - (1 - a_t).sqrt() * pred) / a_t.sqrt().clamp(min=1e-8)
            z = a_prev.sqrt() * z0 + (1 - a_prev).sqrt() * pred

        return self.decoder.decode(z)


def build_model(cfg: Config, device: str = "cuda") -> SRDiffusion:
    print("=" * 60)
    print("SR-Diffusion v2: DINOv2-giant + SD 2.1 U-Net")
    print("=" * 60)
    return SRDiffusion(cfg, device=device)
