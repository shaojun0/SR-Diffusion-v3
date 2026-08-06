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
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class SRQwenVLConfig(PretrainedConfig):
    model_type = "sr_qwen_vl_v10"

    def __init__(
        self,
        dino_dir: str = "",
        qwen_dir: str = "",
        dino_dim: int = 1536,
        qwen_dim: int = 2560,
        qwen_max_length: int = 256,
        projector_hidden: int = 5120,
        svd_max_eig: int = 128,
        svd_patch_dim: int = 1024,
        svd_energy_threshold: float = 0.99,
        svd_image_size: int = 1024,
        svd_patch_size: int = 32,
        dino_trainable: bool = False,
        lr: float = 1e-4,
        batch_size: int = 2,
        grad_accum: int = 4,
        max_grad_norm: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = qwen_dim
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
    Receives a pre-loaded Dinov2Encoder (frozen weights from dino_dir).

    Input:  (B, 2n, 1024)  SVD matrix (U^T stacked on V^T)
    Output: (B, 2n+1, 1536) encoded tokens with CLS
    """

    def __init__(self, cfg: SRQwenVLConfig, vit_encoder: Dinov2Encoder, vit_layernorm: nn.LayerNorm):
        super().__init__()
        self.vit_encoder = vit_encoder
        self.vit_layernorm = vit_layernorm
        self.svd_proj = nn.Linear(cfg.svd_patch_dim, cfg.dino_dim)
        max_tokens = 2 * cfg.svd_max_eig + 1  # 257
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, cfg.dino_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.dino_dim))

    def forward(self, svd_matrix: Tensor) -> Tensor:
        B, S, _ = svd_matrix.shape
        tokens = self.svd_proj(svd_matrix)
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
    supports_gradient_checkpointing = True

    def __init__(self, config: SRQwenVLConfig):
        super().__init__(config)
        cfg = config

        # ── 1. DINOv2 encoder（自动从 dino_dir 加载）──
        dino_encoder, dino_layernorm = self._load_dino_encoder(cfg)
        self.encoder = SVDEncoder(cfg, dino_encoder, dino_layernorm)

        # ── 2. MLP Projector ──
        self.projector = MLPProjector(
            in_dim=cfg.dino_dim,
            hidden_dim=cfg.projector_hidden,
            out_dim=cfg.qwen_dim,
        )

        # ── 3. Qwen（自动从 qwen_dir 加载）──
        self.lm_model = self._load_qwen_lm(cfg)

    # ═══════════════════════════════════════════════════════════
    # Auto-load helpers
    # ═══════════════════════════════════════════════════════════

    def _load_dino_encoder(self, cfg: SRQwenVLConfig):
        """Load DINOv2-giant encoder from dino_dir. Returns (encoder, layernorm)."""
        if cfg.dino_dir and os.path.isdir(cfg.dino_dir):
            print(f"[DINO] Auto-loading from {cfg.dino_dir}")
            dino_model = Dinov2Model.from_pretrained(cfg.dino_dir, local_files_only=True)
            encoder = dino_model.encoder
            layernorm = dino_model.layernorm
            del dino_model  # free the full model, keep encoder + layernorm refs

            if not cfg.dino_trainable:
                encoder.eval()
                for p in encoder.parameters():
                    p.requires_grad_(False)
                for p in layernorm.parameters():
                    p.requires_grad_(False)
                print("[DINO] Frozen ✓")
            return encoder, layernorm
        else:
            print("[DINO] dino_dir not found, using random init")
            from transformers import Dinov2Config as DinoCfg
            dino_cfg = DinoCfg(hidden_size=cfg.dino_dim, num_hidden_layers=40,
                               num_attention_heads=24, image_size=518, patch_size=14)
            return Dinov2Encoder(dino_cfg), nn.LayerNorm(cfg.dino_dim)

    def _load_qwen_lm(self, cfg: SRQwenVLConfig):
        """Load Qwen3.5-4B from qwen_dir."""
        if cfg.qwen_dir and os.path.isdir(cfg.qwen_dir):
            print(f"[Qwen] Auto-loading from {cfg.qwen_dir}")
            lm = AutoModelForCausalLM.from_pretrained(
                cfg.qwen_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
            )
            print(f"[Qwen] Loaded ✓ — {sum(p.numel() for p in lm.parameters())/1e9:.2f}B params")
            return lm
        else:
            print("[Qwen] qwen_dir not found, creating empty model")
            from transformers import AutoConfig as AC
            qwen_cfg = AC.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
            return AutoModelForCausalLM.from_config(qwen_cfg, torch_dtype=torch.bfloat16)

    # ═══════════════════════════════════════════════════════════
    # Keys to ignore on save（仅保存可训练参数）
    # ═══════════════════════════════════════════════════════════

    @property
    def _keys_to_ignore_on_save(self):
        """DINO + Qwen loaded from directories — never saved in checkpoint."""
        keys = []
        sd = self.state_dict()
        for k in sd:
            if k.startswith("encoder.vit_encoder.") or k.startswith("encoder.vit_layernorm."):
                keys.append(k)
            if k.startswith("lm_model."):
                keys.append(k)
        return keys

    # ═══════════════════════════════════════════════════════════
    # Gradient checkpointing
    # ═══════════════════════════════════════════════════════════

    def _enable_gradient_checkpointing(self):
        if self.lm_model is not None:
            self.lm_model.gradient_checkpointing_enable()

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total: {total/1e9:.2f}B | Trainable: {trainable/1e9:.2f}B")

    # ═══════════════════════════════════════════════════════════
    # encode — SVD matrix → visual tokens
    # ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile

    dino_dir = "/root/autodl-tmp/sr_dinov2/models/AI-ModelScope--dinov2-giant"
    qwen_dir = "/root/autodl-tmp/qwen3_5_4B"

    print("=" * 60)
    print("SR-Qwen-VL v11: Self-test")
    print("=" * 60)

    # ── Test 1: Create model (auto-load) ──
    print("\n[1] Creating model (auto-load DINO + Qwen)...")
    config = SRQwenVLConfig(dino_dir=dino_dir, qwen_dir=qwen_dir)
    model = SRQwenVLv10(config)
    model._log_params()

    # ── Test 2: Forward pass ──
    print("\n[2] Forward pass...")
    model = model.cuda().bfloat16()
    B = 2
    svd_matrix = torch.randn(B, 2 * config.svd_max_eig, config.svd_patch_dim).cuda().bfloat16()
    input_ids = torch.randint(0, 1000, (B, 64)).cuda()
    attention_mask = torch.ones(B, 64).cuda()
    labels = input_ids.clone()

    outputs = model(svd_matrix=svd_matrix, input_ids=input_ids,
                    attention_mask=attention_mask, labels=labels)
    print(f"  Logits: {outputs['logits'].shape}, Loss: {outputs['loss'].item():.4f}")
    print("  ✅ Forward OK")

    # ── Test 3: Save → Load cycle ──
    print("\n[3] save_pretrained → from_pretrained cycle...")
    model = model.cpu()
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d, safe_serialization=True)

        # Inspect saved weights
        from safetensors.torch import load_file
        for f in sorted(os.listdir(d)):
            if f.endswith('.safetensors'):
                sd = load_file(os.path.join(d, f))
                for k in sorted(sd):
                    if 'pos_embed' in k.lower():
                        assert sd[k].numel() > 0, f"❌ pos_embed is EMPTY!"
                        print(f"  Saved: {k} shape={sd[k].shape} ✅")

        # Load back
        loaded = SRQwenVLv10.from_pretrained(d)
        pe = loaded.encoder.pos_embed
        assert pe.numel() > 0, "❌ pos_embed is EMPTY after load!"
        print(f"  Loaded: pos_embed shape={pe.shape} ✅")

    # ── Test 4: Has no tokenizer bound ──
    print(f"\n[4] Tokenizer bound: {hasattr(model, 'tokenizer')}")
    assert not hasattr(model, 'tokenizer'), "❌ tokenizer should NOT be bound to model"
    print("  ✅ Tokenizer decoupled")

    # ── Test 5: Has no build_model ──
    print(f"\n[5] build_model exists: {hasattr(model, 'build_model')}")
    assert not hasattr(model, 'build_model'), "❌ build_model should NOT exist"
    print("  ✅ No build_model")

    print("\n" + "=" * 60)
    print("All tests passed ✅")
    print("=" * 60)
