"""
Build SR-Diffusion v3 with pretrained weights and save.
CPU-only to avoid interfering with training on GPU.
Fixed paths — weights from HF cache snapshots.
"""
import os, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, "/root/autodl-tmp")
from model_v3_test import SRDiffusionConfig, SRDiffusion

DINO_DIR = "/root/autodl-tmp/sr_dinov2/models/AI-ModelScope--dinov2-giant/snapshots/master"
SD_MODEL = "/root/autodl-tmp/sd_models/models/AI-ModelScope--stable-diffusion-2-1/snapshots/master"
OUTPUT_DIR = "/root/autodl-tmp/sr_diffusion_v3_weights"

print("=" * 60)
print("SR-Diffusion v3: build_model with pretrained weights")
print("=" * 60)
print(f"  DINOv2: {DINO_DIR}")
print(f"  SD 2.1: {SD_MODEL}")
print(f"  Output: {OUTPUT_DIR}")

# Step 1: Create config
print("\n[1/4] Creating config...")
config = SRDiffusionConfig(image_size=1024, latent_size=128)
print("  OK")

# Step 2: Create model (arch only)
print("\n[2/4] Creating model architecture...")
t0 = time.time()
model = SRDiffusion(config)
t1 = time.time()
print(f"  OK ({t1-t0:.1f}s)")

# Step 3: build_model — load DINOv2-giant + SD 2.1 pretrained weights
print("\n[3/4] build_model — loading pretrained weights...")
t0 = time.time()
model.build_model(dino_dir=DINO_DIR, sd_model_id=SD_MODEL, device="cpu")
t1 = time.time()
print(f"  OK ({t1-t0:.1f}s)")

# Step 4: Save
print(f"\n[4/4] Saving to {OUTPUT_DIR} ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
t0 = time.time()
model.save_pretrained(OUTPUT_DIR)
t1 = time.time()
print(f"  OK ({t1-t0:.1f}s)")

# Verify
files = os.listdir(OUTPUT_DIR)
sizes = {f: os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1e9 for f in files}
print(f"\n  Saved files:")
for f, s in sorted(sizes.items()):
    print(f"    {f}: {s:.2f} GB")

print(f"\n  Total: {sum(sizes.values()):.2f} GB")
print("\n" + "=" * 60)
print("DONE — weights saved to:", OUTPUT_DIR)
print("=" * 60)
