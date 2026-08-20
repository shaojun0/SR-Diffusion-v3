"""Evaluate v3 checkpoint: k stability + content-adaptive distribution."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from PIL import Image
from transformers import Dinov2Model
import sys, glob
sys.path.insert(0, "/root")
from model_phase1 import SRPhase1, BudgetTrustRegion

ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/phase1_tr_v3/final"
data_dir = "/root/autodl-tmp/imagenet/val"

dino = Dinov2Model.from_pretrained("facebook/dinov2-base", cache_dir="/root/hf_cache")
dino = dino.cuda().eval()
for p in dino.parameters():
    p.requires_grad_(False)

model = SRPhase1.build_model(dino, num_patches=256, dim=768, T=1.0,
                             lambda_rate=0.0, init_reencoder=False).cuda()
p = os.path.join(ckpt_dir, "pytorch_model.bin")
if not os.path.exists(p):
    from safetensors.torch import load_file
    sd = load_file(os.path.join(ckpt_dir, "model.safetensors"))
else:
    sd = torch.load(p, map_location="cuda")
model.load_state_dict(sd, strict=False)
model.set_stage(2)
model.eval()

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def run(x):
    with torch.no_grad():
        out = model(x, hard_mode=True)
        return {k: (v.item() if hasattr(v, "item") else v)
                for k, v in out["stats"].items()}

print("=== blank / low-entropy ===")
for name, arr in {
    "pure_white": np.full((224, 224, 3), 255, np.uint8),
    "pure_black": np.full((224, 224, 3), 0, np.uint8),
    "flat_gray": np.full((224, 224, 3), 128, np.uint8),
    "noise_heavy": np.random.randint(0, 255, (224, 224, 3), np.uint8),
}.items():
    a = np.asarray(Image.fromarray(arr), np.float32) / 255.0
    a = (a - MEAN) / STD
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).cuda()
    s = run(x)
    print(f"{name:12s} k={s['k_used_mean']:6.1f} tau={s['tau_mean']:.3f} recon={s['recon_l1']:.4f}")

print("\n=== real ImageNet (n=300) ===")
all_files = sorted(glob.glob(os.path.join(data_dir, "*.JPEG")))
ks, taus, recs = [], [], []
for f in all_files[:300]:
    img = Image.open(f).convert("RGB").resize((224, 224), Image.BILINEAR)
    a = np.asarray(img, np.float32) / 255.0
    a = (a - MEAN) / STD
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).cuda()
    s = run(x)
    ks.append(s["k_used_mean"]); taus.append(s["tau_mean"]); recs.append(s["recon_l1"])
ks = np.array(ks); taus = np.array(taus); recs = np.array(recs)
print(f"k: min={ks.min():.0f} p25={np.percentile(ks,25):.0f} median={np.median(ks):.0f} "
      f"p75={np.percentile(ks,75):.0f} max={ks.max():.0f} std={ks.std():.0f}")
print(f"tau: mean={taus.mean():.3f} ± {taus.std():.3f} (content-adaptive if std>0.2)")
print(f"recon: mean={recs.mean():.4f} ± {recs.std():.4f}")
print(f"compression vs 256: {256/np.median(ks):.1f}x median")
