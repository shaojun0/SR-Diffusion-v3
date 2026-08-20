"""Evaluate Phase 1: content-adaptive compression on blank vs dense images."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from PIL import Image
from transformers import Dinov2Model
import sys
sys.path.insert(0, "/root")
from model_phase1 import SRPhase1

ckpt = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/phase1_run6/final.pt"
data_dir = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/imagenet/val"

dino = Dinov2Model.from_pretrained("facebook/dinov2-base", cache_dir="/root/hf_cache")
dino = dino.cuda().eval()
for p in dino.parameters():
    p.requires_grad_(False)

model = SRPhase1(dino, num_patches=256, dim=768, T=1.0, lambda_rate=0.0).cuda()
ck = torch.load(ckpt, map_location="cuda")
model.load_state_dict(ck["model"], strict=False)
model.set_stage(2)
model.eval()
print(f"loaded {ckpt} (step {ck.get('step', '?')})")

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def to_tensor(img):
    arr = np.asarray(img.resize((224, 224), Image.BILINEAR), np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()

# ── blank / low-entropy images ──
blanks = {
    "pure_white": np.full((224, 224, 3), 255, np.uint8),
    "pure_black": np.full((224, 224, 3), 0, np.uint8),
    "flat_gray": np.full((224, 224, 3), 128, np.uint8),
    "noise_light": np.random.randint(120, 136, (224, 224, 3), np.uint8),
    "noise_heavy": np.random.randint(0, 255, (224, 224, 3), np.uint8),
}
print("\n=== blank / low-entropy ===")
for name, arr in blanks.items():
    x = to_tensor(Image.fromarray(arr))
    with torch.no_grad():
        out = model(x, hard_mode=True)
    s = out["stats"]
    print(f"{name:12s} k_used={s['k_used_mean']:6.1f}  tau={s['tau_mean']:.3f}")

# ── real images: min / median / max k in a sample ──
import glob
files = sorted(glob.glob(os.path.join(data_dir, "*.JPEG")))[:200]
ks = []
for f in files:
    img = Image.open(f).convert("RGB")
    x = to_tensor(img)
    with torch.no_grad():
        out = model(x, hard_mode=True)
    ks.append(out["stats"]["k_used_mean"])
ks = np.array(ks)
print("\n=== real ImageNet sample (n=%d) ===" % len(ks))
print(f"k distribution: min={ks.min():.0f}  p25={np.percentile(ks,25):.0f}  "
      f"median={np.median(ks):.0f}  p75={np.percentile(ks,75):.0f}  max={ks.max():.0f}")
print(f"compression ratio vs 256 tokens: {256/np.median(ks):.1f}x median")

# show a few extremes
order = np.argsort(ks)
for idx in [order[0], order[len(order)//2], order[-1]]:
    print(f"\n  k={ks[idx]:.0f}  img={os.path.basename(files[idx])}")
