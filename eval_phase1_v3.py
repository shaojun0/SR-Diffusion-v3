"""Evaluate v3 checkpoint: k stability + content-adaptive distribution
+ 拉格朗日约束满足度（R 是否 ≈ R_target）。

Usage:
  python eval_phase1_v3.py <ckpt_dir> [--rate_target 0.25]
"""
import os, sys, glob, argparse
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch
from PIL import Image
from transformers import Dinov2Model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_phase1_v3 import SRPhase1V3

ap = argparse.ArgumentParser()
ap.add_argument("ckpt_dir", nargs="?", default="/root/autodl-tmp/phase1_v3/final")
ap.add_argument("--data_dir", default="/root/autodl-tmp/imagenet/val")
ap.add_argument("--dino", default="facebook/dinov2-base")
ap.add_argument("--cache_dir", default="/root/hf_cache")
ap.add_argument("--rate_target", type=float, default=0.25)
ap.add_argument("--n", type=int, default=300, help="ImageNet 采样数")
args = ap.parse_args()

dino = Dinov2Model.from_pretrained(args.dino, cache_dir=args.cache_dir)
dino = dino.cuda().eval()
for p in dino.parameters():
    p.requires_grad_(False)

model = SRPhase1V3.build_model(dino, num_patches=256, dim=dino.config.hidden_size,
                               T=1.0, rate_target=args.rate_target,
                               init_reencoder=False).cuda()
p = os.path.join(args.ckpt_dir, "pytorch_model.bin")
if os.path.exists(p):
    sd = torch.load(p, map_location="cuda")
else:
    from safetensors.torch import load_file
    sd = load_file(os.path.join(args.ckpt_dir, "model.safetensors"))
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
    print(f"{name:12s} k={s['k_used_mean']:6.1f} tau={s['tau_mean']:.3f} "
          f"rate={s['rate']:.3f} recon={s['recon_l1']:.4f}")

print(f"\n=== real ImageNet (n={args.n}) ===")
all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.JPEG")))
ks, taus, rates, recs = [], [], [], []
for f in all_files[:args.n]:
    img = Image.open(f).convert("RGB").resize((224, 224), Image.BILINEAR)
    a = np.asarray(img, np.float32) / 255.0
    a = (a - MEAN) / STD
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).cuda()
    s = run(x)
    ks.append(s["k_used_mean"]); taus.append(s["tau_mean"])
    rates.append(s["rate"]); recs.append(s["recon_l1"])
ks = np.array(ks); taus = np.array(taus)
rates = np.array(rates); recs = np.array(recs)
print(f"k: min={ks.min():.0f} p25={np.percentile(ks,25):.0f} median={np.median(ks):.0f} "
      f"p75={np.percentile(ks,75):.0f} max={ks.max():.0f} std={ks.std():.0f}")
print(f"tau: mean={taus.mean():.3f} ± {taus.std():.3f} (content-adaptive if std>0.2)")
print(f"recon: mean={recs.mean():.4f} ± {recs.std():.4f}")
print(f"rate: mean={rates.mean():.4f} (R_target={args.rate_target}) — "
      f"拉格朗日约束满足度 |R−R_target|={abs(rates.mean()-args.rate_target):.4f}")
print(f"compression vs 256: {256/np.median(ks):.1f}x median")
