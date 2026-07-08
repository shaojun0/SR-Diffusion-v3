"""
SR-Diffusion v3 CPU smoke test
Test: import → config → model create → forward → save/load round-trip
"""
import os, sys, time, tempfile, shutil

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, "/root/autodl-tmp")

import torch
from model_v3_test import SRDiffusionConfig, SRDiffusion

# ── Test config: use small image_size for fast CPU testing ──
TEST_SIZE = 128
TEST_LATENT = TEST_SIZE // 8

# ── Test 1: Import ──
print("=" * 60)
print("TEST 1: Import model_v3")
print("=" * 60)
print("OK")

# ── Test 2: Config ──
print("\n" + "=" * 60)
print("TEST 2: Create config (CPU, from scratch)")
print("=" * 60)
config = SRDiffusionConfig(
    image_size=TEST_SIZE,
    latent_size=TEST_LATENT,
)
print(f"  model_type: {config.model_type}")
print(f"  dino_hidden: {config.dino_hidden}")
print(f"  image_size: {config.image_size}")
print(f"  latent_size: {config.latent_size}")
print(f"  lr_size: {config.lr_size}")
print(f"  max_eig: {config.max_eig}")

sf = config.sd_vae.get("scaling_factor", "MISSING")
mma = config.sd_vae.get("mid_block_add_attention", "MISSING")
print(f"  scaling_factor: {sf}")
print(f"  mid_block_add_attention: {mma}")
print("OK")

# ── Test 3: Model create ──
print("\n" + "=" * 60)
print("TEST 3: Create model (CPU)")
print("=" * 60)
t0 = time.time()
model = SRDiffusion(config)
t1 = time.time()
print(f"  Created in {t1-t0:.1f}s")

print(f"  dino: {type(model.dino).__name__}")
print(f"  projector: {type(model.projector).__name__}")
print(f"  decoder: {type(model.decoder).__name__}")
print(f"  unet: {type(model.decoder.unet).__name__}")
print(f"  vae: {type(model.decoder.vae).__name__}")
print(f"  noise buffer shape: {model._alphas_cumprod.shape}")
print(f"  DINO grad_ckpt: {model.dino.vit_encoder.gradient_checkpointing}")
print("OK")

# ── Test 4: Forward ──
print("\n" + "=" * 60)
print("TEST 4: Forward pass (CPU, {}x{})".format(TEST_SIZE, TEST_SIZE))
print("=" * 60)
model.eval()
hr = torch.randn(1, 3, TEST_SIZE, TEST_SIZE)
print(f"  Input: {hr.shape}")
t0 = time.time()
with torch.no_grad():
    out = model(hr)
t1 = time.time()
print(f"  Took {t1-t0:.1f}s")
print(f"  Loss: {out['loss'].item():.6f}")
print(f"  Pred shape: {out['pred'].shape}")
print(f"  Noise shape: {out['noise'].shape}")
print("OK")

# ── Test 5: save/load round-trip ──
print("\n" + "=" * 60)
print("TEST 5: save_pretrained / from_pretrained round-trip")
print("=" * 60)
tmpdir = tempfile.mkdtemp(prefix="sr_test_")
try:
    model.save_pretrained(tmpdir)
    files = os.listdir(tmpdir)
    print(f"  Saved: {files}")

    loaded = SRDiffusion.from_pretrained(tmpdir)
    print(f"  Loaded: {type(loaded).__name__}")

    assert loaded.config.model_type == config.model_type
    assert loaded.config.dino_hidden == config.dino_hidden
    print("  Config round-trip OK")

    assert torch.allclose(model._alphas_cumprod, loaded._alphas_cumprod)
    print("  Noise buffers round-trip OK")

    with torch.no_grad():
        out2 = loaded(hr)
    print(f"  Loss after reload: {out2['loss'].item():.6f}")
    print("  Forward after reload OK")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

# ── Test 6: build_model on CPU (no pretrained, just arch check) ──
print("\n" + "=" * 60)
print("TEST 6: build_model (no pretrained, arch-only verify)")
print("=" * 60)
model2 = SRDiffusion(SRDiffusionConfig())
result = model2.build_model(dino_dir=None, sd_model_id=None, device="cpu")
assert result is model2
print(f"  build_model returned self OK")
print(f"  Total params: {sum(p.numel() for p in model2.parameters())/1e9:.2f}B")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
