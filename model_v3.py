"""
SR-Diffusion v2: SVD + DINOv2-giant → SD 2.1 U-Net Diffusion 超分辨率.
PreTrainedModel 兼容 — 支持 save_pretrained / AutoModel.from_pretrained.

Pipeline:
  1. HR → SVD → top-k left singular vectors
  2. eig vectors + LR tokens → DINOv2 Encoder → cross-attn tokens (1536d)
  3. LR → VAE Encoder → condition latent
  4. HR → VAE Encoder → +noise → U-Net(cross_attn) → ε pred
  5. Loss: ε-MSE + 0.5·x₀-MSE

用法:
    # 新建
    config = SRDiffusionConfig()
    model = SRDiffusion(config)
    model.build_model(dino_dir="dinov2-giant", sd_model_id="stabilityai/stable-diffusion-2-1")

    # 保存
    model.save_pretrained("./checkpoint")

    # 加载 (无需手动拼模块)
    from transformers import AutoModel
    model = AutoModel.from_pretrained("./checkpoint", trust_remote_code=True)

全参数训练 (2.09B), fp32 + 8-bit AdamW + gradient checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Dinov2Config, PretrainedConfig, PreTrainedModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class SRDiffusionConfig(PretrainedConfig):
    """训练超参 & 模型结构参数. 子模型配置嵌套在 dino / sd_unet / sd_vae 中."""
    model_type = "sr_diffusion_v2"

    def __init__(
        self,
        # ── 子模型配置（嵌套字典，自包含架构参数）──
        dino: dict | None = None,
        sd_unet: dict | None = None,
        sd_vae: dict | None = None,
        # ── 便捷维度字段（从子模型配置提取）──
        dino_hidden: int = 1536,
        sd_cross_attn: int = 1024,
        # ── Image / Latent ──
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

        # ── DINOv2-giant 默认配置（对齐实际 config.json）──
        if dino is None:
            dino = {
                "model_id": "dinov2-giant",
                "attention_probs_dropout_prob": 0.0,
                "drop_path_rate": 0.0,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.0,
                "hidden_size": 1536,
                "image_size": 518,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-06,
                "layerscale_value": 1.0,
                "mlp_ratio": 4,
                "num_attention_heads": 24,
                "num_channels": 3,
                "num_hidden_layers": 40,
                "patch_size": 14,
                "qkv_bias": True,
                "use_swiglu_ffn": True,
            }
        self.dino = dino

        # ── SD 2.1 U-Net 默认配置（对齐实际 config.json）──
        if sd_unet is None:
            sd_unet = {
                "model_id": "stabilityai/stable-diffusion-2-1",
                "act_fn": "silu",
                "attention_head_dim": [5, 10, 20, 20],
                "block_out_channels": [320, 640, 1280, 1280],
                "center_input_sample": False,
                "cross_attention_dim": 1024,
                "down_block_types": [
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ],
                "downsample_padding": 1,
                "dual_cross_attention": False,
                "flip_sin_to_cos": True,
                "freq_shift": 0,
                "in_channels": 4,
                "layers_per_block": 2,
                "mid_block_scale_factor": 1,
                "norm_eps": 1e-05,
                "norm_num_groups": 32,
                "num_class_embeds": None,
                "only_cross_attention": False,
                "out_channels": 4,
                "sample_size": 96,
                "up_block_types": [
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                ],
                "use_linear_projection": True,
                "upcast_attention": True,
            }
        self.sd_unet = sd_unet

        # ── SD 2.1 VAE 默认配置（对齐实际 config.json）──
        if sd_vae is None:
            sd_vae = {
                "model_id": "stabilityai/stable-diffusion-2-1",
                "act_fn": "silu",
                "block_out_channels": [128, 256, 512, 512],
                "down_block_types": [
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                ],
                "in_channels": 3,
                "latent_channels": 4,
                "layers_per_block": 2,
                "mid_block_add_attention": True,
                "norm_num_groups": 32,
                "out_channels": 3,
                "sample_size": 768,
                "scaling_factor": 0.18215,
                "up_block_types": [
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                ],
            }
        self.sd_vae = sd_vae

        # ── 便捷字段 ──
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

        # ── AutoModel 兼容: 告诉 transformers 去哪里找模型类 ──
        self.auto_map = {
            "AutoConfig": "model_v3.SRDiffusionConfig",
            "AutoModel": "model_v3.SRDiffusion",
        }

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
    B, _, H, W = hr.shape

    lr = F.interpolate(hr, size=(lr_size, lr_size), mode="bicubic", align_corners=False)
    lr_gray = 0.2989 * lr[:, 0] + 0.5870 * lr[:, 1] + 0.1140 * lr[:, 2]
    lr_flat = lr_gray.reshape(B, lr_size * lr_size)

    hr_gray = 0.2989 * hr[:, 0] + 0.5870 * hr[:, 1] + 0.1140 * hr[:, 2]
    eig_tokens, n_list = svd_eigenvectors(
        hr_gray.unsqueeze(1), energy_threshold, max_n, min_n
    )
    return lr_flat, eig_tokens, n_list

# ═══════════════════════════════════════════════════════════════
# Noise Schedule — DDPM, uses registered buffers for serialization
# ═══════════════════════════════════════════════════════════════

class NoiseSchedule:
    """DDPM noise schedule helper. Backed by registered buffers for save/load."""

    def __init__(self, alphas_cumprod: Tensor, sqrt_alphas: Tensor, sqrt_one_minus: Tensor):
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas = sqrt_alphas
        self.sqrt_one_minus = sqrt_one_minus

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

    def __init__(self, cfg: SRDiffusionConfig):
        super().__init__()
        dino_config = Dinov2Config.from_dict(cfg.dino)
        self.vit_encoder = Dinov2Encoder(dino_config)
        self.vit_layernorm = nn.LayerNorm(
            dino_config.hidden_size, eps=dino_config.layer_norm_eps
        )

        self.tok_proj = nn.Linear(cfg.lr_size ** 2, cfg.dino_hidden)
        self.eig_proj = nn.Linear(cfg.image_size, cfg.dino_hidden)
        self.pos = nn.Parameter(
            torch.randn(1, cfg.max_eig + 3, cfg.dino_hidden) * 0.02
        )
        self.cls = nn.Parameter(torch.randn(1, 1, cfg.dino_hidden))

    def forward(
        self, lr_flat: Tensor, eig_tokens: Tensor, n_list: list  # noqa: ARG002
    ) -> Tensor:
        B = lr_flat.size(0)

        # Handle variable image sizes — interpolate eig_tokens to match eig_proj
        H_actual = eig_tokens.size(2)
        H_expected = self.eig_proj.in_features
        if H_actual != H_expected:
            B_eig, N_eig = eig_tokens.shape[0], eig_tokens.shape[1]
            eig_tokens = eig_tokens.reshape(B_eig * N_eig, 1, H_actual)
            eig_tokens = F.interpolate(
                eig_tokens, size=H_expected,
                mode="linear", align_corners=False
            )
            eig_tokens = eig_tokens.reshape(B_eig, N_eig, H_expected)

        lr_tok = self.tok_proj(lr_flat).unsqueeze(1)     # (B, 1, H)
        eig_tok = self.eig_proj(eig_tokens)               # (B, N, H_expected)
        tokens = torch.cat([lr_tok, eig_tok], dim=1)     # (B, 1+N, H)

        cls_tok = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)     # (B, 2+N, H)

        S = tokens.size(1)
        pos = (
            self.pos[:, :S, :]
            if S <= self.pos.size(1)
            else F.pad(self.pos, (0, 0, 0, S - self.pos.size(1)))
        )
        tokens = tokens + pos

        enc = self.vit_encoder(tokens, output_hidden_states=True)
        return self.vit_layernorm(enc.last_hidden_state)

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
# Diffusion Decoder — SD 2.1 U-Net + VAE (cond_fusion 代替 conv_in hack)
# ═══════════════════════════════════════════════════════════════

class DiffusionDecoder(nn.Module):
    """SD 2.1 U-Net (noise prediction) + VAE (latent encode/decode). All trainable.
    
    使用 cond_fusion 模块将 noisy + condition latent (2×4ch) 融合为 4ch 再送入 U-Net，
    替代原先直接修改 conv_in 的做法，更干净且易于保存/加载。
    """

    def __init__(self, cfg: SRDiffusionConfig):
        super().__init__()
        from diffusers import UNet2DConditionModel, AutoencoderKL

        self.unet = UNet2DConditionModel.from_config(cfg.sd_unet)
        self.vae = AutoencoderKL.from_config(cfg.sd_vae)

        # cond_fusion: concat(noisy 4ch, cond 4ch) → 4ch for U-Net
        latent_ch = cfg.sd_unet["in_channels"]  # 4
        hidden_ch = self.unet.conv_in.out_channels  # 320
        self.cond_fusion = nn.Sequential(
            nn.Conv2d(latent_ch * 2, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, latent_ch, kernel_size=3, stride=1, padding=1),
        )

        self.scale = self.vae.config.scaling_factor

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
        """Predict noise. Input: noisy (B,4,H,W) + cond (B,4,H,W), output: (B,4,H,W)"""
        x = self.cond_fusion(torch.cat([noisy, cond], dim=1))
        return self.unet(x, t, encoder_hidden_states=cross_tokens, return_dict=False)[0]

# ═══════════════════════════════════════════════════════════════
# SRDiffusion — full pipeline, PreTrainedModel compatible
# ═══════════════════════════════════════════════════════════════

class SRDiffusion(PreTrainedModel):
    """DINOv2 + SD 2.1 U-Net diffusion 超分。全参数训练。

    用法:
        # 新建 + build
        config = SRDiffusionConfig()
        model = SRDiffusion(config)
        model.build_model(dino_dir="dinov2-giant",
                          sd_model_id="stabilityai/stable-diffusion-2-1")

        # 保存 (config.json 自包含所有子模型配置)
        model.save_pretrained("./checkpoint")

        # 加载 — AutoModel 直接加载，无需手动拼 DINO/SD
        from transformers import AutoModel
        model = AutoModel.from_pretrained("./checkpoint", trust_remote_code=True)
    """
    config_class = SRDiffusionConfig

    # ── PreTrainedModel compatibility helpers ──
    @property
    def _tied_weights_keys(self):
        return []

    def __init__(self, config: SRDiffusionConfig):
        super().__init__(config)
        self.config = config

        # Ensure all_tied_weights_keys exists (compat with transformers 5.x)
        if not hasattr(self, 'all_tied_weights_keys') or self.all_tied_weights_keys is None:
            self.all_tied_weights_keys = {}

        # ── 注册 noise schedule buffers — 保证 save/load 一致 ──
        betas = torch.linspace(
            config.beta_start ** 0.5, config.beta_end ** 0.5, config.train_timesteps
        ) ** 2
        alphas = 1.0 - betas
        ac = alphas.cumprod(0)
        self.register_buffer("_alphas_cumprod", ac, persistent=True)
        self.register_buffer("_sqrt_alphas", ac.sqrt(), persistent=True)
        self.register_buffer(
            "_sqrt_one_minus", (1.0 - ac).sqrt(), persistent=True
        )

        self.noise_schedule = NoiseSchedule(
            self._alphas_cumprod, self._sqrt_alphas, self._sqrt_one_minus
        )

        # ── 子模块 (arch only, weights 在 build_model 中加载) ──
        self.dino = DinoEncoder(config)
        self.projector = TokenProjector(config.dino_hidden, config.sd_cross_attn)
        self.decoder = DiffusionDecoder(config)

        # gradient checkpointing (DINO encoder + SD U-Net)
        self._enable_gradient_checkpointing()

    def build_model(
        self,
        dino_dir: str | None = None,
        sd_model_id: str | None = None,
        device: str = "cuda",
    ) -> "SRDiffusion":
        """加载预训练权重到子模块。调用后即可 save_pretrained / AutoModel 加载。

        Args:
            dino_dir: DINOv2-giant 本地目录 (含 model.safetensors 或 pytorch_model.bin)
            sd_model_id: SD 2.1 HuggingFace model id (如 "stabilityai/stable-diffusion-2-1")
            device: 目标设备

        Returns:
            self — 链式调用
        """
        print("=" * 60)
        print("SR-Diffusion v2: build_model — loading pretrained weights")
        print("=" * 60)

        # ── 1. DINOv2-giant ──
        if dino_dir is not None:
            print(f"  [DINO] Loading pretrained weights from {dino_dir} ...")
            from transformers import Dinov2Model

            dino_pretrained = Dinov2Model.from_pretrained(
                dino_dir, local_files_only=True
            )
            # 映射权重: Dinov2Model → 我们的 DinoEncoder
            pt_state = dino_pretrained.state_dict()
            our_state = {}

            # vit_encoder 权重 (Dinov2Encoder)
            for k, v in pt_state.items():
                if k.startswith("encoder."):
                    our_state["vit_encoder." + k[len("encoder."):]] = v
                elif k.startswith("layernorm."):
                    our_state["vit_layernorm." + k[len("layernorm."):]] = v
                # 跳过 patch_embed, mask_token 等 DINO 特定层

            # 合并: 只加载匹配的 key (保留我们新增的 tok_proj/eig_proj/pos/cls 随机初始化)
            missing, unexpected = self.dino.load_state_dict(our_state, strict=False)
            print(f"    Loaded {len(our_state)} keys, "
                  f"missing={len(missing)} (tok_proj/eig_proj/pos/cls — expected), "
                  f"unexpected={len(unexpected)}")
            del dino_pretrained

        # ── 2. SD 2.1 U-Net + VAE ──
        if sd_model_id is not None:
            print(f"  [SD] Loading pretrained weights from {sd_model_id} ...")
            from diffusers import UNet2DConditionModel, AutoencoderKL

            # U-Net (strict=False: 我们的 cond_fusion 不在 pretrained 里)
            unet_pt = UNet2DConditionModel.from_pretrained(
                sd_model_id, subfolder="unet", low_cpu_mem_usage=True
            )
            missing, unexpected = self.decoder.unet.load_state_dict(
                unet_pt.state_dict(), strict=False
            )
            print(f"    U-Net: missing={len(missing)} (cond_fusion — expected), "
                  f"unexpected={len(unexpected)}")
            del unet_pt

            # VAE
            vae_pt = AutoencoderKL.from_pretrained(
                sd_model_id, subfolder="vae", low_cpu_mem_usage=True
            )
            self.decoder.vae.load_state_dict(vae_pt.state_dict(), strict=True)
            print("    VAE: loaded (strict)")
            del vae_pt

        # ── 3. Move to device ──
        self.to(device)

        print("=" * 60)
        self._log_params()
        print("  Ready for save_pretrained / AutoModel.from_pretrained ✓")
        return self

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on DINO encoder and SD U-Net."""
        self.dino.vit_encoder.gradient_checkpointing = True
        if hasattr(self.decoder.unet, 'enable_gradient_checkpointing'):
            self.decoder.unet.enable_gradient_checkpointing()

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dino = sum(p.numel() for p in self.dino.parameters())
        unet = sum(p.numel() for p in self.decoder.unet.parameters())
        vae = sum(p.numel() for p in self.decoder.vae.parameters())
        print(
            f"  [Model] Total: {total/1e9:.2f}B | Trainable: {train/1e9:.2f}B | "
            f"DINO: {dino/1e6:.0f}M | U-Net: {unet/1e6:.0f}M | VAE: {vae/1e6:.0f}M"
        )

    def forward(self, hr: Tensor, return_dict: bool = True, **kwargs):
        """Training forward: hr (B,3,H,W) → {loss, pred, noise}"""
        B, device = hr.size(0), hr.device
        cfg = self.config

        # Resize input to expected resolution if needed
        if hr.shape[2] != cfg.image_size or hr.shape[3] != cfg.image_size:
            hr = F.interpolate(
                hr, size=(cfg.image_size, cfg.image_size),
                mode="bicubic", align_corners=False
            )

        # ── 1. SVD tokens via DINOv2 ──
        lr_flat, eig_tokens, n_list = build_tokens(
            hr, cfg.lr_size, cfg.energy_threshold, cfg.max_eig, cfg.min_eig,
        )
        svd = self.dino(lr_flat, eig_tokens, n_list)
        cross = self.projector(svd)

        # CFG-style conditioning dropout (15%)
        if self.training and torch.rand(1).item() < cfg.cond_dropout:
            cross = torch.zeros_like(cross)

        # ── 2. LR condition latent ──
        lr_img = F.interpolate(
            F.interpolate(hr, size=(cfg.lr_size, cfg.lr_size), mode="bicubic"),
            size=(cfg.image_size, cfg.image_size), mode="bicubic",
        )
        cond = self.decoder.encode(lr_img)

        # ── 3. HR → latent → add noise ──
        hr_z = self.decoder.encode(hr)
        noise = torch.randn_like(hr_z)
        t = torch.randint(0, cfg.train_timesteps, (B,), device=device)
        noisy = self.noise_schedule.add_noise(hr_z, noise, t)

        # ── 4. U-Net predict noise → ε-MSE (standard DDPM) ──
        pred = self.decoder(noisy, cond, t, cross)

        loss = F.mse_loss(pred, noise)

        if return_dict:
            return {"loss": loss, "pred": pred, "noise": noise}
        return loss, pred, noise

    @torch.no_grad()
    def sample(self, hr: Tensor, steps: int = 25) -> Tensor:
        """DDIM sampling for inference."""
        B, device = hr.size(0), hr.device
        cfg = self.config

        # Resize input to expected resolution if needed
        if hr.shape[2] != cfg.image_size or hr.shape[3] != cfg.image_size:
            hr = F.interpolate(
                hr, size=(cfg.image_size, cfg.image_size),
                mode="bicubic", align_corners=False
            )

        lr_flat, eig_tokens, n_list = build_tokens(
            hr, cfg.lr_size, cfg.energy_threshold, cfg.max_eig, cfg.min_eig,
        )
        cross = self.projector(self.dino(lr_flat, eig_tokens, n_list))

        lr_img = F.interpolate(
            F.interpolate(hr, size=(cfg.lr_size, cfg.lr_size), mode="bicubic"),
            size=(cfg.image_size, cfg.image_size), mode="bicubic",
        )
        cond = self.decoder.encode(lr_img)

        step_ratio = cfg.train_timesteps // steps
        timesteps = (
            torch.arange(steps - 1, -1, -1) * step_ratio
        ).long().to(device)

        z = torch.randn(
            B, cfg.latent_channels,
            cfg.latent_size, cfg.latent_size, device=device,
        )

        ac = self._alphas_cumprod
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
