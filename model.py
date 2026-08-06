"""
SR-Qwen-VL v12: SVD(1024x1024) -> DINOv2 Encoder -> MLP -> Qwen3.5-4B -> Text

Simplified: DINO from_dir (Dinov2Model.from_pretrained loads full model, we extract encoder+layernorm),
           Qwen lazy-loaded, only trainable params saved.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from transformers import (
    PretrainedConfig, PreTrainedModel,
    Dinov2Model, Dinov2Config,
    AutoModelForCausalLM, AutoConfig,
)
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder

# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════

class SRQwenVLConfig(PretrainedConfig):
    model_type = "sr_qwen_vl"

    def __init__(self, dino_dir="", qwen_dir="", dino_dim=1536, qwen_dim=2560,
                 qwen_max_length=256, projector_hidden=5120, svd_max_eig=128,
                 svd_patch_dim=1024, svd_energy_threshold=0.99,
                 svd_image_size=1024, svd_patch_size=32, dino_trainable=False, **kwargs):
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


# ═══════════════════════════════════════════════════
# SVD Encoder (SVD tokens -> DINOv2 encoder)
# ═══════════════════════════════════════════════════

class SVDEncoder(nn.Module):
    """Receives pre-built Dinov2Encoder + LayerNorm. Input (B, 2n, 1024) -> (B, 2n+1, 1536)."""

    def __init__(self, cfg: SRQwenVLConfig, vit_encoder: Dinov2Encoder, vit_layernorm: nn.LayerNorm):
        super().__init__()
        self.vit_encoder = vit_encoder
        self.vit_layernorm = vit_layernorm
        self.svd_proj = nn.Linear(cfg.svd_patch_dim, cfg.dino_dim)
        max_tokens = 2 * cfg.svd_max_eig + 1
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, cfg.dino_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.dino_dim))

    def forward(self, svd_matrix: Tensor) -> Tensor:
        B, S, _ = svd_matrix.shape
        x = self.svd_proj(svd_matrix)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :S + 1, :]
        enc = self.vit_encoder(x, output_hidden_states=True)
        return self.vit_layernorm(enc.last_hidden_state)


# ═══════════════════════════════════════════════════
# MLP Projector
# ═══════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════

class SRQwenVLv12(PreTrainedModel):
    config_class = SRQwenVLConfig
    base_model_prefix = "sr_qwen_vl"
    supports_gradient_checkpointing = True

    def __init__(self, config: SRQwenVLConfig):
        super().__init__(config)
        cfg = config

        # DINOv2 encoder (random init, pretrained weights loaded later)
        dino_cfg = self._load_dino_cfg(cfg)
        self.encoder = SVDEncoder(cfg, Dinov2Encoder(dino_cfg), nn.LayerNorm(cfg.dino_dim))

        # MLP Projector
        self.projector = MLPProjector(cfg.dino_dim, cfg.projector_hidden, cfg.qwen_dim)

        # Qwen (random init, pretrained weights loaded later)
        qwen_cfg = AutoConfig.from_pretrained(cfg.qwen_dir, trust_remote_code=True) if cfg.qwen_dir and os.path.isdir(cfg.qwen_dir) else AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
        self.lm_model = AutoModelForCausalLM.from_config(qwen_cfg, torch_dtype=torch.bfloat16)

        self._pretrained_loaded = False

    @staticmethod
    def _load_dino_cfg(cfg):
        if cfg.dino_dir and os.path.isdir(cfg.dino_dir):
            try:
                return Dinov2Config.from_pretrained(cfg.dino_dir, local_files_only=True)
            except Exception:
                pass
        return Dinov2Config(hidden_size=cfg.dino_dim, num_hidden_layers=40,
                            num_attention_heads=24, image_size=518, patch_size=14)

    # ── Lazy-load pretrained DINO + Qwen ──

    def _ensure_pretrained(self):
        if self._pretrained_loaded:
            return
        self._pretrained_loaded = True
        cfg = self.config

        # Get device from existing parameters
        device = next(self.encoder.svd_proj.parameters()).device

        # DINOv2: load full model, extract encoder + layernorm
        if cfg.dino_dir and os.path.isdir(cfg.dino_dir):
            dino = Dinov2Model.from_pretrained(cfg.dino_dir, local_files_only=True, torch_dtype=torch.bfloat16)
            self.encoder.vit_encoder = dino.encoder.to(device)
            self.encoder.vit_layernorm = dino.layernorm.to(device)
            del dino
            if not cfg.dino_trainable:
                self.encoder.vit_encoder.eval()
                for p in self.encoder.vit_encoder.parameters():
                    p.requires_grad_(False)
                for p in self.encoder.vit_layernorm.parameters():
                    p.requires_grad_(False)
            print(f"[DINO] Loaded ✓ (trainable={cfg.dino_trainable})")

        # Qwen (frozen — acts as fixed language model backend)
        if cfg.qwen_dir and os.path.isdir(cfg.qwen_dir):
            print(f"[Qwen] Loading from {cfg.qwen_dir} ...")
            qwen = AutoModelForCausalLM.from_pretrained(cfg.qwen_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
            self.lm_model.load_state_dict(qwen.state_dict())
            del qwen
            for p in self.lm_model.parameters():
                p.requires_grad_(False)
            n = sum(p.numel() for p in self.lm_model.parameters())
            print(f"[Qwen] Loaded ✓ ({n/1e9:.2f}B params, frozen)")

    # ── Ignore non-trainable params on save ──

    @property
    def all_tied_weights_keys(self):
        return {}

    @property
    def _keys_to_ignore_on_save(self):
        return [k for k in self.state_dict() if k.startswith("encoder.vit_encoder.") or k.startswith("encoder.vit_layernorm.") or k.startswith("lm_model.")]

    def _enable_gradient_checkpointing(self):
        if self.lm_model is not None:
            self.lm_model.gradient_checkpointing_enable()

    def log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total: {total/1e9:.2f}B | Trainable: {trainable/1e9:.2f}B")
        return trainable

    # ── encode ──

    def _encode(self, svd_matrix: Tensor) -> Tensor:
        self._ensure_pretrained()
        return self.projector(self.encoder(svd_matrix)).to(torch.bfloat16)

    # ── forward ──

    def forward(self, svd_matrix: Tensor, input_ids: Tensor, attention_mask: Tensor,
                labels: Optional[Tensor] = None, return_dict: bool = True, **kwargs):
        self._ensure_pretrained()
        # Ensure bf16 (DataParallel threads don't inherit autocast)
        if svd_matrix.dtype != torch.bfloat16:
            svd_matrix = svd_matrix.to(torch.bfloat16)
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
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=-100)
        if return_dict:
            return {"loss": loss, "logits": logits}
        return (loss, logits)

    # ── generate ──

    @torch.no_grad()
    def generate(self, svd_matrix: Tensor, tokenizer, prompt: str = "描述这张建筑工地图片：",
                 max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
        self._ensure_pretrained()
        device = svd_matrix.device
        visual_prefix = self._encode(svd_matrix)

        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_embeds = self.lm_model.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([visual_prefix, prompt_embeds], dim=1)

        vis_mask = torch.ones(prompt_ids.size(0), visual_prefix.size(1), device=device)
        attention_mask = torch.cat([vis_mask, torch.ones_like(prompt_ids)], dim=1)

        out = self.lm_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                     temperature=temperature, top_p=top_p,
                                     pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][prompt_ids.size(1):], skip_special_tokens=True)


# ═══════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile
    from safetensors.torch import load_file

    dino_dir = "/root/autodl-tmp/sr_dinov2/models/AI-ModelScope--dinov2-giant/snapshots/master/"
    qwen_dir = "/root/autodl-tmp/qwen3_5_4B/"
    torch.cuda.empty_cache()

    print("=" * 50)
    print("SR-Qwen-VL v12 Self-test")
    print("=" * 50)

    # 1. Create
    print("\n[1] Creating model...")
    config = SRQwenVLConfig(dino_dir=dino_dir, qwen_dir=qwen_dir)
    model = SRQwenVLv12(config)
    print("  ✅ Created (random init)")

    # 2. Forward
    print("\n[2] Forward + lazy-load...")
    model = model.cuda().bfloat16()
    B = 2
    svd = torch.randn(B, 256, 1024).cuda().bfloat16()
    ids = torch.randint(0, 1000, (B, 64)).cuda()
    mask = torch.ones(B, 64).cuda()
    labels = ids.clone()

    out = model(svd_matrix=svd, input_ids=ids, attention_mask=mask, labels=labels)
    print(f"  Logits: {out['logits'].shape}, Loss: {out['loss'].item():.4f}")
    model.log_params()
    print("  ✅ Forward OK")

    # 3. Save/Load
    print("\n[3] save_pretrained -> from_pretrained...")
    model = model.cpu()
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d, safe_serialization=True)

        for f in sorted(os.listdir(d)):
            if f.endswith('.safetensors'):
                sd = load_file(os.path.join(d, f))
                for k in sorted(sd):
                    if 'pos_embed' in k:
                        print(f"  Saved: {k} {tuple(sd[k].shape)} ✅")
                leaked = [k for k in sd if k.startswith('lm_model.') or k.startswith('encoder.vit_encoder.')]
                assert not leaked, f"❌ Leaked: {leaked}"
                print(f"  No leaked DINO/Qwen keys ✅")

        loaded = SRQwenVLv12.from_pretrained(d, ignore_mismatched_sizes=True)
        print(f"  Loaded pos_embed: {tuple(loaded.encoder.pos_embed.shape)} ✅")

    print("\n" + "=" * 50)
    print("All tests passed ✅")
    print("=" * 50)
