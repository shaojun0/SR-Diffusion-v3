"""
SR-Diffusion Phase 1 training — adaptive token budget + feature reconstruction.

Two-stage schedule:
  Stage 1 (warmup):  k pinned to 1.0 (all tokens), λ_rate = 0.
                     Teaches re-encoder + decoder to reconstruct the full
                     DINO feature map.  Never dead-locks.
  Stage 2 (anneal):  RateHead enabled, λ_rate annealed 0 → target.
                     Mask sparsifies; budget becomes content-adaptive.

Usage:
  python train_phase1.py --data_dir /root/autodl-pub/ImageNet/val \
                         --dino facebook/dinov2-base \
                         --epochs 2 --batch_size 32 --stage1_steps 500

Data: any ImageFolder-style directory (class subdirs) or flat JPEG dir.
"""
import argparse, os, math, random, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import Dinov2Model, AutoImageProcessor

from model_phase1 import SRPhase1


# ═══════════════════════════════════════════════════════════════
# Dataset — flat JPEG dir or ImageFolder, on-the-fly 224² + DINO norm
# ═══════════════════════════════════════════════════════════════

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


class ImageDirDataset(Dataset):
    def __init__(self, root: str, size: int = 224, limit: int = 0):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        files = []
        for ext in exts:
            files += glob.glob(os.path.join(root, "**", ext), recursive=True)
        files = sorted(files)
        if limit > 0:
            files = files[:limit]
        self.files = files
        self.size = size
        print(f"[data] {len(files)} images from {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        # random-resized crop → 224 (light augmentation)
        w, h = img.size
        s = min(w, h)
        if s > self.size:
            x0 = random.randint(0, w - self.size)
            y0 = random.randint(0, h - self.size)
            img = img.crop((x0, y0, x0 + self.size, y0 + self.size))
        else:
            img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.array(DINO_MEAN, np.float32)) / np.array(DINO_STD, np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,224,224)


# ═══════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dino", default="facebook/dinov2-base",
                    help="HF id of frozen DINOv2 (base/small/large/giant)")
    ap.add_argument("--out", default="output/phase1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--stage1_steps", type=int, default=500,
                    help="steps with fixed k=1.0 before budget annealing")
    ap.add_argument("--anneal_steps", type=int, default=2000,
                    help="steps over which λ_rate ramps 0 → target")
    ap.add_argument("--lambda_rate_target", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=1.0, help="gumbel temp (annealed)")
    ap.add_argument("--tau_min", type=float, default=0.3)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="cap dataset size (0=all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── frozen DINOv2 ──
    print(f"[model] loading frozen {args.dino} ...")
    dino = Dinov2Model.from_pretrained(args.dino).to(device).eval()
    dim = dino.config.hidden_size
    num_patches = (224 // dino.config.patch_size) ** 2   # 256 for patch14
    print(f"[model] hidden={dim}, patches={num_patches}")

    model = SRPhase1(dino, num_patches=num_patches, dim=dim,
                     tau=args.tau, lambda_rate=0.0).to(device)
    model.set_stage(1)  # fixed k=1.0

    # trainable = everything except frozen dino
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[model] trainable params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # ── data ──
    ds = ImageDirDataset(args.data_dir, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, drop_last=True, persistent_workers=True)

    # ── loop ──
    step = 0
    log_keys = ["recon_l1", "rate_soft", "usage", "k_used_mean", "k_used_min",
                "k_used_max", "entropy"]
    for epoch in range(args.epochs):
        model.train()
        for x in loader:
            x = x.to(device, non_blocking=True)

            # ── stage control ──
            if step < args.stage1_steps:
                model.set_stage(1)
                model.set_lambda_rate(0.0)
            else:
                model.set_stage(2)
                # anneal λ_rate 0 → target over anneal_steps
                prog = min(1.0, (step - args.stage1_steps) / max(1, args.anneal_steps))
                lr_ = args.lambda_rate_target * prog
                model.set_lambda_rate(lr_)
                # anneal gumbel temp
                model.topk.tau = max(args.tau_min,
                                     args.tau * (1 - 0.7 * prog))
                model.rate_head.requires_grad_(True)

            out = model(x)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()

            if step % args.log_every == 0:
                s = out["stats"]
                log = " | ".join(f"{k}={s[k]:.4f}" for k in log_keys)
                print(f"[step {step:6d} ep{epoch}] λr={model.lambda_rate:.3f} "
                      f"τ={model.topk.tau:.2f} loss={loss.item():.4f} | {log}")
                with open(os.path.join(args.out, "log.txt"), "a") as f:
                    f.write(f"{step} {loss.item():.6f} " +
                            " ".join(f"{s[k]:.6f}" for k in log_keys) + "\n")

            if step % args.save_every == 0 and step > 0:
                ckpt = os.path.join(args.out, f"step{step}.pt")
                torch.save({
                    "step": step, "model": model.state_dict(),
                    "args": vars(args),
                }, ckpt)
                print(f"[save] {ckpt}")
            step += 1

    ckpt = os.path.join(args.out, "final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, ckpt)
    print(f"[done] {ckpt}")


if __name__ == "__main__":
    main()
