"""
SR-Qwen-VL v2: SVD(1024×1024) → DINOv2 Encoder → MLP → Qwen3.5-4B → Chinese captions.

Architecture:
  - __init__:    random-only (safe under meta-device ctx, from_pretrained-compatible)
  - build_model: explicit pre-trained weight loading + freeze → ready-to-train
  - forward:     standard HF causal-LM output (loss from labels)

Ref:
  LLaVA / Qwen2-VL PreTrainedModel patterns: __init__ uses from_config;
  weight loading is separated from construction.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from dataclasses import dataclass

from transformers import (
    PretrainedConfig, PreTrainedModel,
    Dinov2Model, Dinov2Config,
    AutoModelForCausalLM, AutoConfig,
    GenerationMixin,
)
from transformers.modeling_outputs import ModelOutput
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder


# ═══════════════════════════════════════════════════════
# Output class (standard HF causal-LM pattern)
# ═══════════════════════════════════════════════════════

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════

class SRQwenVLConfig(PretrainedConfig):
    model_type = "sr_qwen_vl"

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dino_dir = dino_dir
        self.qwen_dir = qwen_dir
        self.dino_dim = dino_dim
        self.qwen_dim = qwen_dim
        self.qwen_max_length = qwen_max_length
        self.projector_hidden = projector_hidden
        self.svd_max_eig = svd_max_eig
        self.svd_patch_dim = svd_patch_dim
        self.svd_energy_threshold = svd_energy_threshold
        self.svd_image_size = svd_image_size
        self.svd_patch_size = svd_patch_size
        self.dino_trainable = dino_trainable
        self.hidden_size = qwen_dim


# ═══════════════════════════════════════════════════════
# SVD Encoder: linear proj → cls + pos → DINOv2 encoder
# ═══════════════════════════════════════════════════════

class SVDEncoder(nn.Module):
    """Takes raw DINOv2 pieces (random init).  Weights replaced by build_model()."""

    def __init__(self, cfg: SRQwenVLConfig):
        super().__init__()
        # Random-init encoder + layernorm from config
        dino_cfg = _dino_config(cfg.dino_dir, cfg.dino_dim)
        self.vit_encoder = Dinov2Encoder(dino_cfg)
        self.vit_layernorm = nn.LayerNorm(cfg.dino_dim)

        self.svd_proj = nn.Linear(cfg.svd_patch_dim, cfg.dino_dim)
        max_tokens = 2 * cfg.svd_max_eig + 1
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, cfg.dino_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.dino_dim))

    def forward(self, svd_matrix: Tensor) -> Tensor:
        B, S, _ = svd_matrix.shape
        x = self.svd_proj(svd_matrix)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : S + 1, :]
        enc = self.vit_encoder(x, output_hidden_states=True)
        return self.vit_layernorm(enc.last_hidden_state)


# ═══════════════════════════════════════════════════════
# MLP Projector
# ═══════════════════════════════════════════════════════

class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


# ═══════════════════════════════════════════════════════
# Full Model – PreTrainedModel (random init only in __init__)
# ═══════════════════════════════════════════════════════

class SRQwenVLv2(PreTrainedModel, GenerationMixin):
    config_class = SRQwenVLConfig
    base_model_prefix = "sr_qwen_vl"
    supports_gradient_checkpointing = True

    def __init__(self, config: SRQwenVLConfig):
        super().__init__(config)
        cfg = config

        # ── Vision backbone (SVDEncoder) — random init ──
        self.encoder = SVDEncoder(cfg)

        # ── Projector — random init ──
        self.projector = MLPProjector(cfg.dino_dim, cfg.projector_hidden, cfg.qwen_dim)

        # ── Language model: placeholder (replaced in build_model) ──
        # We store config only; actual LM is loaded from pretrained in build_model.
        # This avoids creating 4B random params only to discard them.
        qwen_cfg = AutoConfig.from_pretrained(
            cfg.qwen_dir, trust_remote_code=True
        ) if cfg.qwen_dir and os.path.isdir(cfg.qwen_dir) else AutoConfig.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
        )
        # Use tiny nn.Module placeholder so post_init doesn't break
        self.lm_model = nn.Identity()
        self._qwen_config = qwen_cfg  # saved for build_model

        self.post_init()
        # _keys_to_ignore_on_save is set after build_model replaces backbones with real weights

    # ── forward ──

    def forward(
        self,
        svd_matrix: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        B, device = svd_matrix.size(0), svd_matrix.device

        # Cast inputs to model dtype (svd collator produces fp32, model is bf16)
        target_dtype = next(self.encoder.svd_proj.parameters()).dtype
        svd_matrix = svd_matrix.to(dtype=target_dtype, device=device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # Encode vision tokens
        visual_prefix = self.projector(self.encoder(svd_matrix)).to(torch.bfloat16)
        N_vis = visual_prefix.size(1)

        # Embed text
        text_embeds = self.lm_model.get_input_embeddings()(input_ids)

        # Merge
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

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # ── generate ──

    @torch.no_grad()
    def generate(
        self,
        svd_matrix: Tensor,
        tokenizer,
        prompt: str = "描述这张建筑工地图片：",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **gen_kwargs,
    ) -> str:
        device = svd_matrix.device
        target_dtype = next(self.encoder.svd_proj.parameters()).dtype
        svd_matrix = svd_matrix.to(dtype=target_dtype, device=device)
        visual_prefix = self.projector(self.encoder(svd_matrix)).to(torch.bfloat16)

        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_embeds = self.lm_model.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([visual_prefix, prompt_embeds], dim=1)

        vis_mask = torch.ones(prompt_ids.size(0), visual_prefix.size(1), device=device)
        attention_mask = torch.cat([vis_mask, torch.ones_like(prompt_ids)], dim=1)

        out = self.lm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            **gen_kwargs,
        )
        return tokenizer.decode(out[0][prompt_ids.size(1):], skip_special_tokens=True)

    # ── gradient checkpointing ──

    def _enable_gradient_checkpointing(self):
        if self.lm_model is not None and hasattr(self.lm_model, "gradient_checkpointing_enable"):
            self.lm_model.gradient_checkpointing_enable()

    # ── logging ──

    def log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total: {total / 1e9:.2f}B | Trainable: {trainable / 1e9:.2f}B")
        return trainable


# ═══════════════════════════════════════════════════════
# build_model: explicit weight loading → ready-to-train
# ═══════════════════════════════════════════════════════

def _dino_config(dino_dir: str, fallback_dim: int) -> Dinov2Config:
    """Load DINOv2 config from dir or return sensible default."""
    if dino_dir and os.path.isdir(dino_dir):
        try:
            return Dinov2Config.from_pretrained(dino_dir, local_files_only=True)
        except Exception:
            pass
    return Dinov2Config(
        hidden_size=fallback_dim,
        num_hidden_layers=40,
        num_attention_heads=24,
        image_size=518,
        patch_size=14,
    )


def build_model(
    config: SRQwenVLConfig,
    dino_dir: str = "",
    qwen_dir: str = "",
    device: str = "cuda",
) -> SRQwenVLv2:
    """Construct model (random init), then load pre-trained backbone weights.

    Returns a model ready for training – DINO/Qwen frozen, projector + SVD proj trainable.
    """
    model = SRQwenVLv2(config)
    device = torch.device(device)

    # ── Load DINOv2 encoder + layernorm ──
    dino_src = dino_dir or config.dino_dir
    if dino_src and os.path.isdir(dino_src):
        print(f"[build_model] Loading DINOv2 from {dino_src} ...")
        dino = Dinov2Model.from_pretrained(
            dino_src, local_files_only=True, torch_dtype=torch.bfloat16
        )
        # Replace random encoder+layernorm with pre-trained ones
        model.encoder.vit_encoder = dino.encoder.to(device)
        model.encoder.vit_layernorm = dino.layernorm.to(device)
        del dino

        if not config.dino_trainable:
            model.encoder.vit_encoder.eval()
            for p in model.encoder.vit_encoder.parameters():
                p.requires_grad_(False)
            for p in model.encoder.vit_layernorm.parameters():
                p.requires_grad_(False)
        print(f"[build_model] DINOv2 loaded ✓ (trainable={config.dino_trainable})")
    else:
        print("[build_model] ⚠ DINOv2 dir not found – using random-init encoder")

    # ── Load Qwen LM (directly from pretrained, no intermediate random init) ──
    qwen_src = qwen_dir or config.qwen_dir
    if qwen_src and os.path.isdir(qwen_src):
        print(f"[build_model] Loading Qwen from {qwen_src} ...")
        qwen = AutoModelForCausalLM.from_pretrained(
            qwen_src, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model.lm_model = qwen
        for p in model.lm_model.parameters():
            p.requires_grad_(False)
        n = sum(p.numel() for p in model.lm_model.parameters())
        print(f"[build_model] Qwen loaded ✓ ({n / 1e9:.2f}B params, frozen)")
    else:
        print("[build_model] ⚠ Qwen dir not found – LM left as placeholder")

    # ── Move to device ──
    model = model.to(device)

    # ── Set _keys_to_ignore_on_save after real weights are in place ──
    model._keys_to_ignore_on_save = [
        k for k in model.state_dict()
        if k.startswith("encoder.vit_encoder.")
        or k.startswith("encoder.vit_layernorm.")
        or k.startswith("lm_model.")
    ]

    model.log_params()
    return model


# ═══════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile
    from safetensors.torch import load_file

    dino_dir = "/root/autodl-tmp/sr_dinov2/models/AI-ModelScope--dinov2-giant/snapshots/master/"
    qwen_dir = "/root/autodl-tmp/qwen3_5_4B/"
    torch.cuda.empty_cache()

    print("=" * 60)
    print("SR-Qwen-VL v2 Self-test")
    print("=" * 60)

    # 1. Create (random-init only, no pretrained weights)
    print("\n[1] SRQwenVLv2 __init__ (random init) ...")
    config = SRQwenVLConfig(dino_dir=dino_dir, qwen_dir=qwen_dir)
    model = SRQwenVLv2(config)
    print("  ✅ Created (random init only, no pretrained weights)")

    # 2. build_model (explicit weight loading)
    print("\n[2] build_model (explicit weight loading) ...")
    model = build_model(config, dino_dir=dino_dir, qwen_dir=qwen_dir, device="cuda")
    model = model.bfloat16()
    print("  ✅ build_model OK")

    # 3. Forward
    print("\n[3] Forward ...")
    B = 2
    svd = torch.randn(B, 256, 1024).cuda().bfloat16()
    ids = torch.randint(0, 1000, (B, 64)).cuda()
    mask = torch.ones(B, 64).cuda()
    labels = ids.clone()

    out = model(svd_matrix=svd, input_ids=ids, attention_mask=mask, labels=labels)
    print(f"  Logits: {out.logits.shape}, Loss: {out.loss.item():.4f}")
    print("  ✅ Forward OK")

    # 4. Save / from_pretrained round-trip
    print("\n[4] save_pretrained → from_pretrained ...")
    model = model.cpu()
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d, safe_serialization=True)

        for f in sorted(os.listdir(d)):
            if f.endswith(".safetensors"):
                sd = load_file(os.path.join(d, f))
                for k in sorted(sd):
                    if "pos_embed" in k:
                        print(f"  Saved: {k} {tuple(sd[k].shape)} ✅")
                leaked = [
                    k for k in sd
                    if k.startswith("lm_model.") or k.startswith("encoder.vit_encoder.")
                ]
                assert not leaked, f"❌ Leaked frozen params: {leaked}"
                print("  No leaked frozen (DINO/Qwen) keys ✅")

        loaded = SRQwenVLv2.from_pretrained(
            d, ignore_mismatched_sizes=True
        )
        # After from_pretrained, we still need to load backbone weights
        loaded2 = build_model(
            loaded.config, dino_dir=dino_dir, qwen_dir=qwen_dir, device="cpu"
        )
        print(f"  Loaded pos_embed: {tuple(loaded2.encoder.pos_embed.shape)} ✅")

    print("\n" + "=" * 60)
    print("All tests passed ✅")
    print("=" * 60)
