"""
SR-Diffusion v4: DINOv2 整图 patch + 动态 special tokens 信息瓶颈.
PreTrainedModel 兼容.

Pipeline:
  1. Image(1024²) → resize 448² → DINOv2 patch_embed → 1024 patch tokens (1536d)
  2. SVD on grayscale → energy threshold → n (dynamic, 每张图不同)
  3. n learnable special tokens + 1024 patches → DINOv2 Encoder (self-attention)
  4. n special token outputs → Projector → U-Net cross-attn (1024d)
  5. Diffusion: ε-MSE + x₀-MSE (same as v3)

Changes from v3:
  - 去掉 SVD eig_tokens 作为输入特征 → 改为只用于定 n
  - DINOv2 输入改为 [CLS, special_n, patches_1024]
  - DINOv2 输出只取 special tokens → decoder cross-attn (信息瓶颈)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Dinov2Config, PretrainedConfig, PreTrainedModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder, Dinov2Embeddings


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class SRDiffusionV4Config(PretrainedConfig):
    model_type = "sr_diffusion_v4"

    def __init__(
        self,
        # ── 子模型配置 ──
        dino: dict | None = None,
        sd_unet: dict | None = None,
        sd_vae: dict | None = None,
        # ── DINO / SVD ──
        dino_hidden: int = 1536,
        sd_cross_attn: int = 1024,
        image_size: int = 1024,
        dino_input_size: int = 448,       # resize target: 32*14=448 for 1024 patches
        num_patches: int = 1024,           # 32×32
        max_special: int = 64,             # 最大特殊 token 数
        energy_threshold: float = 0.95,    # SVD 能量阈值 → 动态 n
        min_special: int = 16,             # 最小特殊 token 数
        # ── Latent ──
        latent_size: int = 128,
        in_channels: int = 3,
        latent_channels: int = 4,
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
        # ── Loss weights ──
        x0_loss_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

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

        if sd_unet is None:
            sd_unet = {
                "model_id": "stabilityai/stable-diffusion-2-1",
                "act_fn": "silu",
                "attention_head_dim": [5, 10, 20, 20],
                "block_out_channels": [320, 640, 1280, 1280],
                "center_input_sample": False,
                "cross_attention_dim": 1024,
                "down_block_types": [
                    "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D", "DownBlock2D",
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
                    "UpBlock2D", "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                ],
                "use_linear_projection": True,
                "upcast_attention": True,
            }
        self.sd_unet = sd_unet

        if sd_vae is None:
            sd_vae = {
                "model_id": "stabilityai/stable-diffusion-2-1",
                "act_fn": "silu",
                "block_out_channels": [128, 256, 512, 512],
                "down_block_types": [
                    "DownEncoderBlock2D", "DownEncoderBlock2D",
                    "DownEncoderBlock2D", "DownEncoderBlock2D",
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
                    "UpDecoderBlock2D", "UpDecoderBlock2D",
                    "UpDecoderBlock2D", "UpDecoderBlock2D",
                ],
            }
        self.sd_vae = sd_vae

        self.dino_hidden = dino_hidden
        self.sd_cross_attn = sd_cross_attn
        self.image_size = image_size
        self.dino_input_size = dino_input_size
        self.num_patches = num_patches
        self.max_special = max_special
        self.energy_threshold = energy_threshold
        self.min_special = min_special
        self.latent_size = latent_size
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.train_timesteps = train_timesteps
        self.inference_steps = inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.lr = lr
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.cond_dropout = cond_dropout
        self.x0_loss_weight = x0_loss_weight

        self.auto_map = {
            "AutoConfig": "model_v4.SRDiffusionV4Config",
            "AutoModel": "model_v4.SRDiffusionV4",
        }


# ═══════════════════════════════════════════════════════════════
# SVD — 只用于定 n (与 v3 完全一致)
# ═══════════════════════════════════════════════════════════════

def compute_n_from_svd(
    img: Tensor,
    energy_threshold: float = 0.95,
    max_n: int = 64,
    min_n: int = 16,
) -> list[int]:
    """(B, 3, H, W) → n_list: 每张图的特殊 token 数量

    算法与 v3 完全一致: grayscale → SVD → 累计能量阈值 → n
    """
    gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]  # (B, H, W)
    B = gray.size(0)
    n_list = []

    for b in range(B):
        _, s, _ = torch.linalg.svd(gray[b].float(), full_matrices=False)
        s2 = s ** 2
        total = s2.sum()
        if total > 0:
            n = torch.searchsorted(s2.cumsum(0), energy_threshold * total).item() + 1
        else:
            n = min_n
        n_list.append(max(min_n, min(n, max_n)))

    return n_list


# ═══════════════════════════════════════════════════════════════
# DINOv2 Encoder v4 — patch tokens + dynamic special tokens
# ═══════════════════════════════════════════════════════════════

class DinoEncoderV4(nn.Module):
    """DINOv2 encoder: image patches + learnable special tokens → encoder → special outputs.

    Input:
        img (B, 3, 1024, 1024)
    Process:
        1. resize 448² → DINOv2 patch_embed → 1024 tokens (1536d)
        2. add max_special learnable special tokens
        3. add position embeddings (interpolated from DINOv2)
        4. DINOv2 transformer encoder
    Output:
        special_token_features (B, max_special, 1536) — 每张图用前 n 个
    """

    def __init__(self, cfg: SRDiffusionV4Config):
        super().__init__()
        self.max_special = cfg.max_special
        self.num_patches = cfg.num_patches
        self.dino_input_size = cfg.dino_input_size
        self.hidden_size = cfg.dino_hidden

        # ── DINOv2 native patch embed (Conv2d 3→1536, k14 s14) ──
        dino_config = Dinov2Config.from_dict(cfg.dino)
        self.patch_embed = Dinov2Embeddings(dino_config)

        # ── Learnable special tokens ──
        self.special_tokens = nn.Parameter(
            torch.randn(1, cfg.max_special, cfg.dino_hidden) * 0.02
        )

        # ── CLS token (reuse DINOv2's or init new) ──
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, cfg.dino_hidden) * 0.02
        )

        # ── DINOv2 Transformer Encoder ──
        self.encoder = Dinov2Encoder(dino_config)
        self.layernorm = nn.LayerNorm(
            cfg.dino_hidden, eps=dino_config.layer_norm_eps
        )

        # ── Position embedding placeholder (从 DINOv2 weights 插值) ──
        # DINOv2 native: (1, 1+37×37, 1536)
        # We need:       (1, 1+max_special+32×32, 1536)
        # 在 build_model 中从 pretrained 加载并插值
        num_native_patches = (518 // 14) * (518 // 14)  # 37×37 = 1369
        total_positions = 1 + cfg.max_special + cfg.num_patches  # 1 + 64 + 1024 = 1089
        self.pos_embed = nn.Parameter(
            torch.randn(1, total_positions, cfg.dino_hidden) * 0.02
        )
        self._num_native_patches = num_native_patches

    def _interpolate_pos_embed(self, native_pos: Tensor, native_size: int = 37,
                                target_size: int = 32):
        """Interpolate DINOv2 native position embeddings to target grid size.

        native_pos: (num_patches, hidden) or (1, num_patches, hidden)
        Returns: (target_size², hidden)
        """
        if native_pos.dim() == 3:
            native_pos = native_pos[0]  # (N, H)
        return F.interpolate(
            native_pos.reshape(1, native_size, native_size, -1).permute(0, 3, 1, 2),
            size=(target_size, target_size),
            mode="bicubic", align_corners=False,
        ).permute(0, 2, 3, 1).reshape(target_size * target_size, -1)

    def load_interpolated_pos_embed(self, dino_state_dict: dict):
        """从 DINOv2 pretrained 加载并插值 position embeddings。"""
        native_key = "embeddings.position_embeddings"
        cls_key = "embeddings.cls_token"

        if native_key in dino_state_dict:
            # DINOv2 position: (1, 1370, 1536) = [CLS, patches_1369]
            native_pos = dino_state_dict[native_key]  # (1, 1370, 1536)

            # Extract CLS position + patch positions
            cls_pos = native_pos[:, :1, :]             # (1, 1, 1536)
            patch_pos = native_pos[:, 1:, :]            # (1, 1369, 1536)

            # Interpolate patches: 37×37 → 32×32
            interpolated = self._interpolate_pos_embed(patch_pos, 37, 32)  # (1024, 1536)

            # Build new position embeddings
            # [CLS, special_64, patches_1024]
            new_pos = torch.cat([
                cls_pos,                                     # (1, 1, 1536)
                self.pos_embed[:, 1:1 + self.max_special, :],  # (1, 64, 1536) — keep random init
                interpolated.unsqueeze(0),                   # (1, 1024, 1536)
            ], dim=1)

            self.pos_embed.data.copy_(new_pos)
            print(f"    Position embeddings: 37×37 → 32×32 (1369 → 1024 patches)")

        if cls_key in dino_state_dict:
            self.cls_token.data.copy_(dino_state_dict[cls_key])
            print(f"    CLS token: loaded from pretrained")

    def _resize_for_patches(self, img: Tensor) -> Tensor:
        """Resize image so DINOv2 native patch_embed yields exactly num_patches tokens."""
        # Patch embed: kernel=14, stride=14
        # Target: 32×32 patches → image size = 32×14 = 448
        return F.interpolate(
            img, size=(self.dino_input_size, self.dino_input_size),
            mode="bicubic", align_corners=False,
        )

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img: (B, 3, 1024, 1024)
        Returns:
            special_features: (B, max_special, hidden) — special token encoder outputs
        """
        B = img.size(0)
        device = img.device

        # ── 1. Resize + patch embed → 1024 tokens ──
        img_resized = self._resize_for_patches(img)
        patch_tokens = self.patch_embed(img_resized)  # (B, 1024, 1536)

        # ── 2. CLS + special tokens ──
        cls_tok = self.cls_token.expand(B, -1, -1)                           # (B, 1, 1536)
        special_tok = self.special_tokens.expand(B, -1, -1)                  # (B, max_special, 1536)

        # ── 3. Concatenate: [CLS, special, patches] ──
        tokens = torch.cat([cls_tok, special_tok, patch_tokens], dim=1)      # (B, 1+N+1024, 1536)
        tokens = tokens + self.pos_embed

        # ── 4. DINOv2 Encoder ──
        enc_out = self.encoder(tokens, output_hidden_states=True)
        hidden = self.layernorm(enc_out.last_hidden_state)                   # (B, 1+N+1024, 1536)

        # ── 5. Extract special token outputs ──
        special_features = hidden[:, 1:1 + self.max_special, :]              # (B, max_special, 1536)

        return special_features


# ═══════════════════════════════════════════════════════════════
# Token Projector — 1536d → 1024d
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
# Noise Schedule
# ═══════════════════════════════════════════════════════════════

class NoiseSchedule:
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
# Diffusion Decoder — SD 2.1 U-Net + VAE
# ═══════════════════════════════════════════════════════════════

class DiffusionDecoder(nn.Module):
    def __init__(self, cfg: SRDiffusionV4Config):
        super().__init__()
        from diffusers import UNet2DConditionModel, AutoencoderKL

        self.unet = UNet2DConditionModel.from_config(cfg.sd_unet)
        self.vae = AutoencoderKL.from_config(cfg.sd_vae)

        latent_ch = cfg.sd_unet["in_channels"]
        hidden_ch = self.unet.conv_in.out_channels
        self.cond_fusion = nn.Sequential(
            nn.Conv2d(latent_ch * 2, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, latent_ch, kernel_size=3, stride=1, padding=1),
        )
        self.scale = self.vae.config.scaling_factor

    def encode(self, x: Tensor) -> Tensor:
        return self.vae.encode(x).latent_dist.sample() * self.scale

    def decode(self, z: Tensor) -> Tensor:
        return self.vae.decode(z / self.scale).sample

    def forward(self, noisy: Tensor, cond: Tensor, t: Tensor, cross_tokens: Tensor) -> Tensor:
        x = self.cond_fusion(torch.cat([noisy, cond], dim=1))
        return self.unet(x, t, encoder_hidden_states=cross_tokens, return_dict=False)[0]


# ═══════════════════════════════════════════════════════════════
# SRDiffusionV4 — full pipeline
# ═══════════════════════════════════════════════════════════════

class SRDiffusionV4(PreTrainedModel):
    config_class = SRDiffusionV4Config

    @property
    def _tied_weights_keys(self):
        return []

    def __init__(self, config: SRDiffusionV4Config):
        super().__init__(config)
        self.config = config

        if not hasattr(self, 'all_tied_weights_keys') or self.all_tied_weights_keys is None:
            self.all_tied_weights_keys = {}

        # ── Noise schedule ──
        betas = torch.linspace(
            config.beta_start ** 0.5, config.beta_end ** 0.5, config.train_timesteps
        ) ** 2
        alphas = 1.0 - betas
        ac = alphas.cumprod(0)
        self.register_buffer("_alphas_cumprod", ac, persistent=True)
        self.register_buffer("_sqrt_alphas", ac.sqrt(), persistent=True)
        self.register_buffer("_sqrt_one_minus", (1.0 - ac).sqrt(), persistent=True)
        self.noise_schedule = NoiseSchedule(self._alphas_cumprod, self._sqrt_alphas,
                                            self._sqrt_one_minus)

        # ── Submodules ──
        self.dino = DinoEncoderV4(config)
        self.projector = TokenProjector(config.dino_hidden, config.sd_cross_attn)
        self.decoder = DiffusionDecoder(config)

        self._enable_gradient_checkpointing()

    def build_model(
        self,
        dino_dir: str | None = None,
        sd_model_id: str | None = None,
        device: str = "cuda",
    ) -> "SRDiffusionV4":
        print("=" * 60)
        print("SR-Diffusion v4: build_model")
        print("=" * 60)

        # ── 1. DINOv2-giant ──
        if dino_dir is not None:
            print(f"  [DINO] Loading pretrained weights from {dino_dir} ...")
            from transformers import Dinov2Model
            dino_pt = Dinov2Model.from_pretrained(dino_dir, local_files_only=True)
            pt_state = dino_pt.state_dict()

            # 加载 patch_embed
            pe_state = {}
            for k, v in pt_state.items():
                if k.startswith("embeddings.patch_embeddings."):
                    pe_state[k[len("embeddings.patch_embeddings."):]] = v
            missing, unexpected = self.dino.patch_embed.patch_embeddings.load_state_dict(
                pe_state, strict=False
            )
            print(f"    patch_embed: loaded, missing={len(missing)}, unexpected={len(unexpected)}")

            # 加载 encoder + layernorm
            enc_state = {}
            for k, v in pt_state.items():
                if k.startswith("encoder."):
                    enc_state[k[len("encoder."):]] = v
                elif k.startswith("layernorm."):
                    enc_state["layernorm." + k[len("layernorm."):]] = v
            missing, unexpected = self.dino.load_state_dict(enc_state, strict=False)
            print(f"    encoder+ln: missing={len(missing)} (special_tokens/cls/pos — expected), "
                  f"unexpected={len(unexpected)}")

            # 插值 position embeddings
            self.dino.load_interpolated_pos_embed(pt_state)

            del dino_pt

        # ── 2. SD 2.1 U-Net + VAE ──
        if sd_model_id is not None:
            from diffusers import UNet2DConditionModel, AutoencoderKL
            print(f"  [SD] Loading pretrained weights from {sd_model_id} ...")
            unet_pt = UNet2DConditionModel.from_pretrained(
                sd_model_id, subfolder="unet", low_cpu_mem_usage=True
            )
            missing, unexpected = self.decoder.unet.load_state_dict(unet_pt.state_dict(), strict=False)
            print(f"    U-Net: missing={len(missing)} (cond_fusion — expected), unexpected={len(unexpected)}")
            del unet_pt

            vae_pt = AutoencoderKL.from_pretrained(
                sd_model_id, subfolder="vae", low_cpu_mem_usage=True
            )
            self.decoder.vae.load_state_dict(vae_pt.state_dict(), strict=True)
            print("    VAE: loaded (strict)")
            del vae_pt

        self.to(device)
        print("=" * 60)
        self._log_params()
        print("  Ready ✓")
        return self

    def _enable_gradient_checkpointing(self):
        self.dino.encoder.gradient_checkpointing = True
        if hasattr(self.decoder.unet, 'enable_gradient_checkpointing'):
            self.decoder.unet.enable_gradient_checkpointing()

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dino = sum(p.numel() for p in self.dino.parameters())
        unet = sum(p.numel() for p in self.decoder.unet.parameters())
        vae = sum(p.numel() for p in self.decoder.vae.parameters())
        print(f"  [Model] Total: {total/1e9:.2f}B | Trainable: {train/1e9:.2f}B | "
              f"DINO: {dino/1e6:.0f}M | U-Net: {unet/1e6:.0f}M | VAE: {vae/1e6:.0f}M")

    def _get_cross_tokens(self, img: Tensor, n_list: list[int] | None = None) -> tuple[Tensor, Tensor]:
        """Compute cross-attention tokens from image.

        Returns:
            cross: (B, max_special, 1024) — cross-attention tokens (masked)
            mask: (B, max_special) — True for active tokens, False for padded
        """
        B = img.size(0)
        cfg = self.config

        # DINOv2 encode → special token features
        special_features = self.dino(img)  # (B, max_special, 1536)

        # Project to SD cross-attention dim
        cross = self.projector(special_features)  # (B, max_special, 1024)

        # Build mask based on n_list (dynamic per image)
        if n_list is not None:
            mask = torch.zeros(B, cfg.max_special, device=img.device, dtype=torch.bool)
            for i, n in enumerate(n_list):
                mask[i, :n] = True
            # Zero out padded tokens
            cross = cross * mask.unsqueeze(-1).float()
        else:
            mask = torch.ones(B, cfg.max_special, device=img.device, dtype=torch.bool)

        return cross, mask

    def forward(self, hr: Tensor, return_dict: bool = True, **kwargs):
        """Training forward."""
        B, device = hr.size(0), hr.device
        cfg = self.config

        # Resize if needed
        if hr.shape[2] != cfg.image_size or hr.shape[3] != cfg.image_size:
            hr = F.interpolate(hr, size=(cfg.image_size, cfg.image_size),
                               mode="bicubic", align_corners=False)

        # ── 1. SVD → n_list (dynamic) ──
        n_list = compute_n_from_svd(hr, cfg.energy_threshold, cfg.max_special, cfg.min_special)

        # ── 2. DINOv2 → special token features → project → cross-attn tokens ──
        cross, mask = self._get_cross_tokens(hr, n_list)

        # CFG conditioning dropout
        if self.training and torch.rand(1).item() < cfg.cond_dropout:
            cross = torch.zeros_like(cross)

        # ── 3. LR condition latent (down-up bicubic) ──
        lr_size = 32  # fixed; could be configurable
        lr_img = F.interpolate(
            F.interpolate(hr, size=(lr_size, lr_size), mode="bicubic", align_corners=False),
            size=(cfg.image_size, cfg.image_size), mode="bicubic", align_corners=False,
        )
        cond = self.decoder.encode(lr_img)

        # ── 4. HR → latent → add noise → predict ──
        hr_z = self.decoder.encode(hr)
        noise = torch.randn_like(hr_z)
        t = torch.randint(0, cfg.train_timesteps, (B,), device=device)
        noisy = self.noise_schedule.add_noise(hr_z, noise, t)

        # ── 5. U-Net predict noise ──
        pred = self.decoder(noisy, cond, t, cross)

        # ── 6. Loss: ε-MSE + x₀-MSE ──
        loss_eps = F.mse_loss(pred, noise)

        # x₀ prediction for latent constraint
        a_t = self._alphas_cumprod[t.cpu()].to(device).view(-1, 1, 1, 1)
        x0_pred = (noisy - (1 - a_t).sqrt() * pred) / a_t.sqrt().clamp(min=1e-8)
        loss_x0 = F.mse_loss(x0_pred, hr_z)

        loss = loss_eps + cfg.x0_loss_weight * loss_x0

        if return_dict:
            return {"loss": loss, "loss_eps": loss_eps, "loss_x0": loss_x0,
                    "pred": pred, "noise": noise, "n_list": n_list}
        return loss, pred, noise

    @torch.no_grad()
    def sample(self, img: Tensor, steps: int = 25) -> tuple[Tensor, list[int]]:
        """DDIM sampling. img is the input image (training: HR, inference: LR upscaled)."""
        B, device = img.size(0), img.device
        cfg = self.config

        if img.shape[2] != cfg.image_size or img.shape[3] != cfg.image_size:
            img = F.interpolate(img, size=(cfg.image_size, cfg.image_size),
                                mode="bicubic", align_corners=False)

        # SVD → n
        n_list = compute_n_from_svd(img, cfg.energy_threshold, cfg.max_special, cfg.min_special)

        # DINOv2 → cross-attn tokens
        cross, _ = self._get_cross_tokens(img, n_list)

        # LR condition latent
        lr_size = 32
        lr_img = F.interpolate(
            F.interpolate(img, size=(lr_size, lr_size), mode="bicubic", align_corners=False),
            size=(cfg.image_size, cfg.image_size), mode="bicubic", align_corners=False,
        )
        cond = self.decoder.encode(lr_img)

        # DDIM
        step_ratio = cfg.train_timesteps // steps
        timesteps = (torch.arange(steps - 1, -1, -1) * step_ratio).long().to(device)

        z = torch.randn(B, cfg.latent_channels, cfg.latent_size, cfg.latent_size, device=device)
        ac = self._alphas_cumprod

        for t in timesteps:
            t_b = t.expand(B)
            pred = self.decoder(z, cond, t_b, cross)
            a_t = ac[t.cpu()].to(device).view(-1, 1, 1, 1)
            a_prev = (ac[t.cpu() - 1].to(device).view(-1, 1, 1, 1)
                      if t > 0 else torch.ones_like(a_t))
            z0 = (z - (1 - a_t).sqrt() * pred) / a_t.sqrt().clamp(min=1e-8)
            z = a_prev.sqrt() * z0 + (1 - a_prev).sqrt() * pred

        return self.decoder.decode(z), n_list
