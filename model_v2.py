"""
SR-Diffusion v2: SVD + DINOv2-giant → SD 2.1 U-Net Diffusion 超分辨率.

Pipeline:
  1. HR image → SVD → top-k left singular vectors
  2. eig vectors + LR tokens → DINOv2 Encoder → cross-attn tokens (1024d)
  3. LR bicubic → VAE Encoder → condition latent
  4. HR → VAE Encoder → latent → +noise → U-Net(cross_attn) → x₀ pred
  5. Loss: ε-MSE + 0.5·x₀-MSE — noise pred + latent KL constraint

全参数训练 (2.09B), fp32 + 8-bit AdamW + gradient checkpointing.
支持 AutoModel 加载: AutoModel.from_pretrained("path", trust_remote_code=True)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class SRDiffusionConfig(PretrainedConfig):
    """训练超参 & 模型结构参数"""
    model_type = "sr_diffusion_v2"

    def __init__(
        self,
        # ── Paths ──
        dino_dir: str = "dinov2-giant",
        sd_model_id: str = "stabilityai/stable-diffusion-2-1",
        # ── Model dims ──
        dino_hidden: int = 1536,
        sd_cross_attn: int = 1024,
        image_size: int = 1024,
        latent_size: int = 128,
        in_channels: int = 3,
        latent_channels: int = 4,
        # ── SVD ──
        lr_size: int = 32,
        max_eig: int = 64,
        energy_threshold: float = 0.95,
        min_eig: int = 16,
        # ── Diffusion ──
        train_timesteps: int = 1000,
        inference_steps: int = 25,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        # ── Training ──
        lr: float = 5e-5,
        batch_size: int = 1,
        grad_accum: int = 4,
        max_grad_norm: float = 1.0,
        cond_dropout: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dino_dir = dino_dir
        self.sd_model_id = sd_model_id
        self.dino_hidden = dino_hidden
        self.sd_cross_attn = sd_cross_attn
        self.image_size = image_size
        self.latent_size = latent_size
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.lr_size = lr_size
        self.max_eig = max_eig
        self.energy_threshold = energy_threshold
        self.min_eig = min_eig
        self.train_timesteps = train_timesteps
        self.inference_steps = inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.lr = lr
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.cond_dropout = cond_dropout
        self.auto_map = {
            "AutoConfig": "model_v2.SRDiffusionConfig",
            "AutoModel": "model_v2.SRDiffusion",
        }

    @classmethod
    def from_legacy(cls):
        """兼容旧的 Config() 调用方式"""
        return cls()


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

class NoiseSchedule(nn.Module):
    """注册 buffer 以便 save_pretrained / from_pretrained 正确保存恢复"""

    def __init__(self, steps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        super().__init__()
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, steps) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus", (1.0 - alphas_cumprod).sqrt())

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

    def __init__(self, cfg: SRDiffusionConfig, device: str = "cuda", skip_pretrained: bool = False):
        super().__init__()
        from transformers import Dinov2Model, Dinov2Config

        if skip_pretrained:
            print(f"  [DINO] Creating from config (no pretrained weights) ...")
            dino_cfg = Dinov2Config.from_pretrained(cfg.dino_dir)
            self.vit = Dinov2Model(dino_cfg).to(device)
        else:
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

    def __init__(self, cfg: SRDiffusionConfig, device: str = "cuda", skip_pretrained: bool = False):
        super().__init__()
        from diffusers import UNet2DConditionModel, AutoencoderKL

        if skip_pretrained:
            print(f"  [SD] Creating from config (no pretrained weights) ...")
            unet_cfg = UNet2DConditionModel.load_config(
                cfg.sd_model_id, subfolder="unet"
            )
            self.unet = UNet2DConditionModel.from_config(unet_cfg)
            vae_cfg = AutoencoderKL.load_config(
                cfg.sd_model_id, subfolder="vae"
            )
            self.vae = AutoencoderKL.from_config(vae_cfg)
        else:
            print(f"  [SD] Loading from {cfg.sd_model_id} ...")
            self.unet = UNet2DConditionModel.from_pretrained(
                cfg.sd_model_id, subfolder="unet", low_cpu_mem_usage=False
            )
            self.vae = AutoencoderKL.from_pretrained(
                cfg.sd_model_id, subfolder="vae", low_cpu_mem_usage=False
            )

        # ── Pre-fusion: concat(noisy, cond) → small CNN → 4ch → vanilla conv_in ──
        # 不改动 U-Net 预训练 conv_in，前置一个小型融合网络
        # 所有维度从 U-Net config 推导，不硬编码
        latent_ch = self.unet.config.in_channels          # SD: 4
        model_ch = self.unet.conv_in.out_channels          # SD: 320
        hidden_ch = model_ch // 5                          # SD: 64
        self.cond_fusion = nn.Sequential(
            nn.Conv2d(latent_ch * 2, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, latent_ch, kernel_size=3, stride=1, padding=1),
        )
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
        """Predict noise. noisy/cond (B,4,H,W) → cond_fusion → (B,4,H,W) → U-Net"""
        x = self.cond_fusion(torch.cat([noisy, cond], dim=1))
        return self.unet(x, t, encoder_hidden_states=cross_tokens, return_dict=False)[0]


# ═══════════════════════════════════════════════════════════════
# SRDiffusion — full DINOv2 + SD 2.1 pipeline
# ═══════════════════════════════════════════════════════════════

class SRDiffusion(PreTrainedModel):
    """DINOv2 + SD 2.1 U-Net diffusion 超分。全参数训练。
    
    用法:
        # 新建
        config = SRDiffusionConfig(...)
        model = SRDiffusion(config)
        
        # 保存
        model.save_pretrained("./checkpoint")
        
        # 加载 (AutoModel 兼容)
        from transformers import AutoModel
        model = AutoModel.from_pretrained("./checkpoint", trust_remote_code=True)
        
        # 或直接用类
        model = SRDiffusion.from_pretrained("./checkpoint")
    """
    config_class = SRDiffusionConfig

    def __init__(self, config: SRDiffusionConfig, device: str = "cuda", skip_pretrained: bool = False):
        super().__init__(config)
        self.config = config
        self.noise_schedule = NoiseSchedule(
            config.train_timesteps, config.beta_start, config.beta_end
        )

        self.dino = DinoEncoder(config, device, skip_pretrained=skip_pretrained)
        self.projector = TokenProjector(config.dino_hidden, config.sd_cross_attn).to(device)
        self.decoder = DiffusionDecoder(config, device, skip_pretrained=skip_pretrained)

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

    def forward(self, hr: Tensor, return_dict: bool = True, **kwargs):
        """Training forward: hr (B,3,H,W) → loss[dict], noise_pred, noise"""
        B, device = hr.size(0), hr.device

        # ── 1. SVD tokens via DINOv2 ──
        lr_flat, eig_tokens, n_list = build_tokens(
            hr, self.config.lr_size, self.config.energy_threshold,
            self.config.max_eig, self.config.min_eig,
        )
        svd = self.dino(lr_flat, eig_tokens, n_list)
        cross = self.projector(svd)

        # CFG-style conditioning dropout (15%) — force U-Net to use cross-attn
        if self.training and torch.rand(1).item() < self.config.cond_dropout:
            cross = torch.zeros_like(cross)

        # ── 2. LR condition latent ──
        lr_img = F.interpolate(
            F.interpolate(
                hr, size=(self.config.lr_size, self.config.lr_size), mode="bicubic"
            ),
            size=(self.config.image_size, self.config.image_size), mode="bicubic",
        )
        cond = self.decoder.encode(lr_img)

        # ── 3. HR → latent → add noise ──
        hr_z = self.decoder.encode(hr)
        noise = torch.randn_like(hr_z)
        t = torch.randint(0, self.config.train_timesteps, (B,), device=device)
        noisy = self.noise_schedule.add_noise(hr_z, noise, t)

        # ── 4. U-Net predict noise → ε-MSE + x₀ KL constraint ──
        pred = self.decoder(noisy, cond, t, cross)
        loss_noise = F.mse_loss(pred, noise)

        alpha_bar = self.noise_schedule.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
        x0_pred = (noisy - (1.0 - alpha_bar).sqrt() * pred) / alpha_bar.sqrt().clamp(min=1e-8)
        loss_recon = F.mse_loss(x0_pred, hr_z)

        # ── 5. Loss: ε-MSE (prevent collapse) + 0.5·x₀-MSE (latent constraint) ──
        loss = loss_noise + loss_recon * 0.5

        if return_dict:
            return {"loss": loss, "pred": pred, "noise": noise}
        return loss, pred, noise

    @torch.no_grad()
    def sample(self, hr: Tensor, steps: int = 25) -> Tensor:
        """DDIM sampling for inference."""
        B, device = hr.size(0), hr.device

        lr_flat, eig_tokens, n_list = build_tokens(
            hr, self.config.lr_size, self.config.energy_threshold,
            self.config.max_eig, self.config.min_eig,
        )
        cross = self.projector(self.dino(lr_flat, eig_tokens, n_list))

        lr_img = F.interpolate(
            F.interpolate(
                hr, size=(self.config.lr_size, self.config.lr_size), mode="bicubic"
            ),
            size=(self.config.image_size, self.config.image_size), mode="bicubic",
        )
        cond = self.decoder.encode(lr_img)

        step_ratio = self.config.train_timesteps // steps
        timesteps = (
            torch.arange(steps - 1, -1, -1) * step_ratio
        ).long().to(device)

        z = torch.randn(
            B, self.config.latent_channels,
            self.config.latent_size, self.config.latent_size, device=device,
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


def build_model(
    config: SRDiffusionConfig | None = None,
    device: str = "cuda",
    skip_pretrained: bool = False,
) -> SRDiffusion:
    """便捷构建函数。兼容旧接口 cfg=... 调用。
    
    新用法:
        model = build_model(config=SRDiffusionConfig(), device="cuda")
    
    加载:
        model = SRDiffusion.from_pretrained("./checkpoint")
    """
    if config is None:
        config = SRDiffusionConfig()

    print("=" * 60)
    mode = "from config" if skip_pretrained else "from pretrained weights"
    print(f"SR-Diffusion v2: DINOv2-giant + SD 2.1 U-Net [{mode}]")
    print("=" * 60)
    return SRDiffusion(config, device=device, skip_pretrained=skip_pretrained)
