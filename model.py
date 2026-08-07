"""
SR-Qwen-VL v11: SVD(1024×1024) → DINOv2 Encoder → MLP → Qwen3.5-4B → Text

Architecture:
  1. [离线] 1024×1024 → 32×32 patches → SVD → (2n, 1024) matrix
  2. SVD Proj: Linear(1024→1536) + CLS + pos_embed
  3. DINOv2-giant Encoder (40 layers, frozen, auto-loaded from dino_dir)
  4. MLP Projector: 1536 → 5120 → 2560
  5. Qwen3.5-4B (auto-loaded from qwen_dir)

Design (v11 — entropy reduction):
  - Tokenizer: NOT attached to model; caller loads independently
  - Auto-load: DINO + Qwen loaded in __init__, no build_model()
  - Standard HF: no save_pretrained / from_pretrained overrides
  - Checkpoint stores only trainable weights (svd_proj, pos_embed, cls_token, projector)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    Dinov2Model,
    Qwen3_5ForConditionalGeneration,Qwen3_5Config
)
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder
from transformers import Dinov2Config as DinoCfg


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class SRQwenVLConfig(PretrainedConfig):
    model_type = "sr_qwen_vl_v10"

    def __init__(
        self,
        dino_config: dict = None,
        qwen_config: dict = None,
        projector_hidden: int = 1560,
        svd_max_eig: int = 128,
        svd_patch_dim: int = 1024,
        svd_image_size: int = 1024,
        svd_patch_size: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dino_config = dino_config
        self.qwen_config = qwen_config
        self.projector_hidden = projector_hidden
        self.svd_max_eig = svd_max_eig
        self.svd_patch_dim = svd_patch_dim
        self.svd_image_size = svd_image_size
        self.svd_patch_size = svd_patch_size
        self.auto_map = {
            "AutoConfig": "model.SRQwenVLConfig",
            "AutoModel": "model.SRQwenVLv10",
        }


# ═══════════════════════════════════════════════════════════════
# SVD Encoder — SVD tokens → DINOv2 transformer
# ═══════════════════════════════════════════════════════════════

class SVDEncoder(nn.Module):
    """
    SVD eigen-tokens → DINOv2-giant transformer encoder → features.
    Receives a pre-loaded Dinov2Encoder (frozen weights from dino_dir).

    Input:  (B, 2n, 1024)  SVD matrix (U^T stacked on V^T)
    Output: (B, 2n+1, 1536) encoded tokens with CLS
    """

    def __init__(self, cfg: SRQwenVLConfig):
        super().__init__()
        self.vit_encoder = Dinov2Encoder(DinoCfg(**cfg.dino_config))
        self.vit_layernorm = nn.LayerNorm(cfg.dino_config["hidden_size"], eps=cfg.dino_config["layer_norm_eps"])
        self.svd_proj = nn.Linear(cfg.svd_patch_dim, cfg.dino_config["hidden_size"])
        max_tokens = 2 * cfg.svd_max_eig + 1  # 257
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, cfg.dino_config["hidden_size"]) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.dino_config["hidden_size"]))

    def forward(self, svd_matrix: Tensor) -> Tensor:
        B, S, _ = svd_matrix.shape
        tokens = self.svd_proj(svd_matrix.to(self.svd_proj.weight.dtype))
        cls_tok = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :S + 1, :]
        enc = self.vit_encoder(tokens, output_hidden_states=True)
        return self.vit_layernorm(enc.last_hidden_state)


# ═══════════════════════════════════════════════════════════════
# MLP Projector
# ═══════════════════════════════════════════════════════════════

class MLPProjector(nn.Module):
    """2-layer GELU: in_dim → hidden → out_dim."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


# ═══════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════

class SRQwenVLv10(PreTrainedModel):
    """
    SR-Qwen-VL v11: SVD → DINO Encoder → MLP → Qwen → Text

    Entropy-reduced design:
    - DINO + Qwen auto-loaded in __init__ from config.dino_dir / config.qwen_dir
    - Tokenizer NOT attached (caller manages independently)
    - No build_model(), no custom save/load overrides
    - Only trainable params (svd_proj/pos_embed/cls_token/projector) saved to checkpoint
    """
    config_class = SRQwenVLConfig
    base_model_prefix = "sr_qwen_vl"

    def __init__(self, config: SRQwenVLConfig):
        super().__init__(config)
        self.encoder = SVDEncoder(config)

        # ── 2. MLP Projector ──
        # in = DINO hidden (1536), out = Qwen text hidden (2560)
        qwen_text_hidden = config.qwen_config["text_config"]["hidden_size"]
        self.projector = MLPProjector(
            in_dim=config.dino_config["hidden_size"],
            hidden_dim=config.projector_hidden,
            out_dim=qwen_text_hidden,
        )

        # ── 3. Qwen3.5（自动从 qwen_dir 加载）──
        self.lm_model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**config.qwen_config))

    @classmethod
    def build_model(cls, dino_path,qwen_path):
        dino_model = Dinov2Model.from_pretrained(dino_path, local_files_only=True)
        lm = Qwen3_5ForConditionalGeneration.from_pretrained(
            qwen_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        config = SRQwenVLConfig(dino_config=dino_model.config.to_dict(),qwen_config=lm.config.to_dict())
        model = cls(config)
        model.encoder.vit_encoder = dino_model.encoder
        model.lm_model = lm
        del dino_model,lm
        return model


    def _encode(self, svd_matrix: Tensor) -> Tensor:
        encoded = self.encoder(svd_matrix)      # (B, 2n+1, 1536)
        visual_embeds = self.projector(encoded)  # (B, 2n+1, 2560)
        return visual_embeds.to(torch.bfloat16)

    # ═══════════════════════════════════════════════════════════
    # forward
    # ═══════════════════════════════════════════════════════════

    def forward(
        self,
        svd_matrix: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        B, device = svd_matrix.size(0), svd_matrix.device
        visual_prefix = self._encode(svd_matrix)
        N_vis = visual_prefix.size(1)

        text_embeds = self.lm_model.get_input_embeddings()(input_ids)
        vis_mask = torch.ones(B, N_vis, device=device, dtype=attention_mask.dtype)
        inputs_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
        attn_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # Qwen3.5: 文本子模型 + lm_head
        text_out = self.lm_model.model.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attn_mask
        )
        logits = self.lm_model.lm_head(text_out.last_hidden_state)
        logits = logits[:, N_vis:, :]

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_dict:
            return {"loss": loss, "logits": logits}
        return (loss, logits)

    # ═══════════════════════════════════════════════════════════
    # generate
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate(
        self,
        svd_matrix: Tensor,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """tokenizer 由调用方传入（不再绑定在 model 上）"""
        device = svd_matrix.device
        visual_prefix = self._encode(svd_matrix)

        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_embeds = self.lm_model.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([visual_prefix, prompt_embeds], dim=1)

        vis_mask = torch.ones(prompt_ids.size(0), visual_prefix.size(1), device=device)
        text_mask = torch.ones_like(prompt_ids)
        attention_mask = torch.cat([vis_mask, text_mask], dim=1)

        outputs = self.lm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        return tokenizer.decode(outputs[0][prompt_ids.size(1):], skip_special_tokens=True)
