"""
SR-Qwen-VL v10: SVD(1024×1024) → DINOv2 Encoder → MLP → Qwen3.5-4B → Text

Architecture:
  1. [离线] 1024×1024 → 32×32 patches → SVD → (2n, 1024) matrix
     - 前 n 行: U[:,:n]^T (行空间)
     - 后 n 行: V[:,:n]^T (列空间)
     - n = min(energy_k, max_eig=128)
  2. SVD Proj: Linear(1024→1536) + CLS + pos_embed
  3. DINOv2-giant Encoder (40 transformer layers, frozen)
  4. MLP Projector: 1536 → 5120 → 2560
  5. Qwen3.5-4B: [visual_prefix | text] → CE loss

PreTrainedModel 兼容 — 支持 save_pretrained / from_pretrained.
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
    Dinov2Config,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder
from safetensors.torch import load_file as safetensors_load


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class SRQwenVLConfig(PretrainedConfig):
    model_type = "sr_qwen_vl_v10"

    def __init__(
        self,
        # ── 路径 ──
        dino_dir: str = "",
        qwen_dir: str = "",
        # ── 子模型配置 ──
        dino: Optional[dict] = None,
        qwen: Optional[dict] = None,
        # ── 维度 ──
        dino_dim: int = 1536,
        qwen_dim: int = 2560,
        qwen_max_length: int = 256,
        projector_hidden: int = 5120,
        # ── SVD ──
        svd_max_eig: int = 128,        # n, 2n=256 tokens
        svd_patch_dim: int = 1024,      # 32×32 patch
        svd_energy_threshold: float = 0.99,
        svd_image_size: int = 1024,
        svd_patch_size: int = 32,
        # ── DINO 是否训练 ──
        dino_trainable: bool = False,
        # ── Training ──
        lr: float = 1e-4,
        batch_size: int = 2,
        grad_accum: int = 4,
        max_grad_norm: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = qwen_dim  # for DeepSpeed ZeRO auto-config

        if dino is None:
            dino = {
                "hidden_size": 1536,
                "num_attention_heads": 24,
                "num_hidden_layers": 40,
                "mlp_ratio": 4,
                "hidden_act": "gelu",
                "layer_norm_eps": 1e-06,
                "attention_probs_dropout_prob": 0.0,
                "hidden_dropout_prob": 0.0,
                "drop_path_rate": 0.0,
                "qkv_bias": True,
                "use_swiglu_ffn": True,
                "initializer_range": 0.02,
                "layerscale_value": 1.0,
                "image_size": 518,
                "patch_size": 14,
                "num_channels": 3,
                "_attn_implementation": "sdpa",
            }
        self.dino = dino

        if qwen is None:
            qwen = {
                "hidden_size": 2560,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "vocab_size": 151936,
            }
        self.qwen = qwen

        self.dino_dir = dino_dir
        self.qwen_dir = qwen_dir
        self.dino_dim = dino_dim
        self.qwen_dim = qwen_dim
        self.qwen_max_length = qwen_max_length
        self.projector_hidden = projector_hidden
        self.dino_trainable = dino_trainable

        self.svd_max_eig = svd_max_eig
        self.svd_patch_dim = svd_patch_dim
        self.svd_energy_threshold = svd_energy_threshold
        self.svd_image_size = svd_image_size
        self.svd_patch_size = svd_patch_size

        self.lr = lr
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm

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

    Input:  (B, 2n, 1024)  SVD matrix (U^T stacked on V^T)
    Output: (B, 2n+1, 1536) encoded tokens with CLS
    """

    def __init__(self, cfg: SRQwenVLConfig):
        super().__init__()
        dino_config = Dinov2Config.from_dict(cfg.dino)
        self.vit_encoder = Dinov2Encoder(dino_config)
        self.vit_layernorm = nn.LayerNorm(
            dino_config.hidden_size, eps=dino_config.layer_norm_eps
        )

        # Project SVD token dim (1024) → DINO dim (1536)
        self.svd_proj = nn.Linear(cfg.svd_patch_dim, cfg.dino_dim)

        # Position embeddings: 2n+1 (CLS + 2n tokens)
        max_tokens = 2 * cfg.svd_max_eig + 1  # 257
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_tokens, cfg.dino_dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.dino_dim))

    def forward(self, svd_matrix: Tensor) -> Tensor:
        """
        Args:
            svd_matrix: (B, S, 1024) where S = 2n
        Returns:
            encoded: (B, S+1, 1536)
        """
        B, S, D = svd_matrix.shape

        # Project → DINO dim
        tokens = self.svd_proj(svd_matrix)  # (B, S, 1536)

        # Prepend CLS
        cls_tok = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)  # (B, S+1, 1536)

        # Add position embedding
        pos = self.pos_embed[:, :S + 1, :]
        tokens = tokens + pos

        # Through DINOv2 transformer encoder
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
    """SR-Qwen-VL v10: SVD → DINO Encoder → MLP → Qwen → Text"""
    config_class = SRQwenVLConfig
    base_model_prefix = "sr_qwen_vl"
    supports_gradient_checkpointing = True

    def __init__(self, config: SRQwenVLConfig):
        super().__init__(config)
        cfg = config

        # ── SVD Encoder (DINOv2 transformer) ──
        self.encoder = SVDEncoder(cfg)

        # ── MLP Projector ──
        self.projector = MLPProjector(
            in_dim=cfg.dino_dim,
            hidden_dim=cfg.projector_hidden,
            out_dim=cfg.qwen_dim,
        )

        # ── Qwen (延迟加载) ──
        self.lm_model = None
        self.tokenizer = None
        self._is_built = False

    # ═══════════════════════════════════════════════════════════
    # build_model
    # ═══════════════════════════════════════════════════════════

    def build_model(
        self,
        dino_dir: Optional[str] = None,
        qwen_dir: Optional[str] = None,
        device: str = "cuda",
    ) -> "SRQwenVLv10":
        """加载预训练权重。"""
        cfg = self.config
        print("=" * 60)
        print("SR-Qwen-VL v10: build_model — loading pretrained weights")
        print("=" * 60)

        # ── 1. DINOv2 encoder ──
        dino_path = dino_dir or cfg.dino_dir
        if dino_path and os.path.isdir(dino_path):
            print(f"  [DINO] Loading encoder weights from {dino_path} ...")
            from transformers import Dinov2Model

            dino_pt = Dinov2Model.from_pretrained(dino_path, local_files_only=True)
            pt_state = dino_pt.state_dict()
            our_state = {}

            # Map: Dinov2Model → our SVDEncoder
            for k, v in pt_state.items():
                if k.startswith("encoder."):
                    our_state["vit_encoder." + k[len("encoder."):]] = v
                elif k.startswith("layernorm."):
                    our_state["vit_layernorm." + k[len("layernorm."):]] = v
                # Skip patch_embed, mask_token → our svd_proj/pos_embed/cls_token

            missing, unexpected = self.encoder.load_state_dict(our_state, strict=False)
            print(f"    Loaded {len(our_state)} keys, "
                  f"missing={len(missing)} (svd_proj/pos/cls — expected), "
                  f"unexpected={len(unexpected)}")
            del dino_pt

            if not cfg.dino_trainable:
                self.encoder.eval()
                for p in self.encoder.parameters():
                    p.requires_grad = False
                print("    DINO encoder: frozen ✓")
        else:
            print("  [DINO] Skipped, random init")

        # ── 2. Qwen3.5-4B ──
        qwen_path = qwen_dir or cfg.qwen_dir
        if qwen_path and os.path.isdir(qwen_path):
            print(f"  [Qwen] Loading from {qwen_path} ...")
            self.lm_model = AutoModelForCausalLM.from_pretrained(
                qwen_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            )
            self.lm_model = self.lm_model.bfloat16()
            self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"    Qwen3.5-4B loaded ✓")
        else:
            print("  [Qwen] Skipped, no LM")

        self.to(device)
        self._is_built = True
        self._log_params()
        return self

    def _enable_gradient_checkpointing(self):
        if self.lm_model is not None:
            self.lm_model.gradient_checkpointing_enable()

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [Model] Total: {total/1e9:.2f}B | Trainable: {trainable/1e9:.2f}B")

    # ═══════════════════════════════════════════════════════════
    # encode — SVD matrix → visual tokens
    # ═══════════════════════════════════════════════════════════

    def _encode(self, svd_matrix: Tensor) -> Tensor:
        """
        Args:
            svd_matrix: (B, 2n, 1024) SVD eigen-matrix
        Returns:
            visual_embeds: (B, 2n+1, qwen_dim) CLS + 2n tokens
        """
        encoded = self.encoder(svd_matrix)  # (B, 2n+1, 1536)
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
        """
        Args:
            svd_matrix: (B, 2n, 1024) precomputed SVD tokens
            input_ids: (B, T)
            attention_mask: (B, T)
            labels: (B, T)
        Returns:
            {"loss": ..., "logits": ...}
        """
        B, device = svd_matrix.size(0), svd_matrix.device
        visual_prefix = self._encode(svd_matrix)
        N_vis = visual_prefix.size(1)

        text_embeds = self.lm_model.get_input_embeddings()(input_ids)
        vis_mask = torch.ones(B, N_vis, device=device)
        inputs_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
        attn_mask = torch.cat([vis_mask, attention_mask], dim=1)

        outputs = self.lm_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        logits = outputs.logits[:, N_vis:, :]

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
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        device = svd_matrix.device
        visual_prefix = self._encode(svd_matrix)

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_embeds = self.lm_model.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([visual_prefix, prompt_embeds], dim=1)

        B = prompt_ids.size(0)
        vis_mask = torch.ones(B, visual_prefix.size(1), device=device)
        text_mask = torch.ones(B, prompt_ids.size(1), device=device)
        attention_mask = torch.cat([vis_mask, text_mask], dim=1)

        outputs = self.lm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return self.tokenizer.decode(
            outputs[0][prompt_ids.size(1):], skip_special_tokens=True
        )

    def save_pretrained(self, save_directory, safe_serialization=True, **kwargs):
        super().save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)


# ═══════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing SRQwenVLv10 (SVD → DINO Encoder → MLP → Qwen)...")

    config = SRQwenVLConfig(
        dino_dir="/root/autodl-tmp/sr_dinov2/models/AI-ModelScope--dinov2-giant",
        qwen_dir="/root/autodl-tmp/qwen3.5-4B",
    )
    model = SRQwenVLv10(config)
    model.build_model()
    model._enable_gradient_checkpointing()
    model._log_params()

    B = 2
    svd_matrix = torch.randn(B, 2 * config.svd_max_eig, config.svd_patch_dim).cuda().bfloat16()
    input_ids = torch.randint(0, 1000, (B, 64)).cuda()
    attention_mask = torch.ones(B, 64).cuda()
    labels = input_ids.clone()

    outputs = model(svd_matrix=svd_matrix, input_ids=input_ids,
                    attention_mask=attention_mask, labels=labels)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Visual tokens: {2 * config.svd_max_eig + 1} (CLS + 2×{config.svd_max_eig})")
    print("✅ Forward OK")
