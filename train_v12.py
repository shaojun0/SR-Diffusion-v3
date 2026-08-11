"""
SR-Diffusion v12 Training Script (Diffusion Decoder version)

Training:
  - Diffusion noise prediction loss (L1, primary)
  - Optional adversarial loss (hinge GAN) on quick DDIM reconstruction
  - Optional SVD compression loss

Dataset: aswin00000/ConstructionSite (200 images for quick test)
  - 224×224 center crop + DINOv2 normalization

Usage:
  python train_v12.py [--dino_path /path/to/dinov2-giant] [--output_dir /path/to/output]

Environment:
  HF_ENDPOINT=https://hf-mirror.com  (set for China mirror)
"""
import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from datasets import load_dataset

# Add script dir to path for model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v12 import (
    build_model_v12, SRDiffusionV12, NLayerDiscriminator, GANLoss
)


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/sr_diffusion_v12_output"
DEFAULT_DINO_PATH = "facebook/dinov2-giant"

# Image
IMAGE_SIZE = 224

# SVD
SVD_ENERGY_THRESHOLD = 0.99
SVD_MAX_K = 64

# Diffusion
DIFFUSION_TIMESTEPS = 1000
DDIM_SAMPLE_STEPS = 50      # steps for visualization sampling
DDIM_QUICK_STEPS = 5        # steps for quick GAN reconstruction

# Training
BATCH_SIZE = 2
EPOCHS = 10
LR_ENCODER = 1e-4
LR_DECODER = 1e-4
LR_DISCRIMINATOR = 4e-4
WEIGHT_DECAY = 1e-4
GAN_WEIGHT = 0.05            # weight for adversarial loss (0 = disable)
SVD_COMPRESS_WEIGHT = 0.01   # weight for SVD compression loss (0 = disable)

# GAN
GAN_MODE = "hinge"  # 'hinge' | 'lsgan' | 'vanilla'

# Logging
LOG_EVERY = 10
SAVE_EVERY = 100
SAMPLE_EVERY = 50
NUM_SAMPLES = 200  # quick test


# ═══════════════════════════════════════════════════════════════
# Image Transform (224×224 center crop + DINOv2 normalization)
# ═══════════════════════════════════════════════════════════════

# DINOv2 normalization (ImageNet stats)
DINOV2_MEAN = [0.485, 0.456, 0.406]
DINOV2_STD = [0.229, 0.224, 0.225]

# Denormalization for saving/visualization
DENORM_MEAN = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
DENORM_STD = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]


def get_train_transform():
    """224×224 center crop + DINOv2 normalize."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert DINOv2-normalized tensor back to [0,1] for visualization."""
    denorm = transforms.Normalize(mean=DENORM_MEAN, std=DENORM_STD)
    return denorm(tensor).clamp(0, 1)


# ═══════════════════════════════════════════════════════════════
# Dataset — HF ConstructionSite → (image,)
# ═══════════════════════════════════════════════════════════════
import io as io_module


class ConstructionSiteDataset(torch.utils.data.Dataset):
    """
    Wraps HF dataset, applies image transforms.

    Expected HF dataset structure: dict with 'image' key (PIL Image or path).
    """

    def __init__(self, hf_dataset, transform=None, max_samples: int = None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples is not None:
            return min(len(self.hf_dataset), self.max_samples)
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        img = sample['image']

        # Handle various image formats from HF datasets
        if isinstance(img, dict):
            if 'bytes' in img:
                img = Image.open(io_module.BytesIO(img['bytes']))
            elif 'path' in img:
                img = Image.open(img['path'])
            else:
                raise ValueError(f"Unknown image dict format: {img.keys()}")
        elif isinstance(img, bytes):
            img = Image.open(io_module.BytesIO(img))
        elif isinstance(img, str):
            img = Image.open(img)
        elif not isinstance(img, Image.Image):
            raise TypeError(f"Unexpected image type: {type(img)}")

        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img  # (3, 224, 224) normalized tensor


# ═══════════════════════════════════════════════════════════════
# SVD Compression Loss (optional)
# ═══════════════════════════════════════════════════════════════

def svd_compression_loss(S: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Encourage energy concentration in top-k singular values.

    Args:
        S: (B, r) singular values
        k: (B,) number of kept singular values
    Returns:
        scalar loss
    """
    log_S = torch.log(S + 1e-8)
    kept_log = torch.zeros(S.shape[0], device=S.device)
    total_log = log_S.sum(dim=1)

    for i in range(S.shape[0]):
        ki = int(k[i].item())
        kept_log[i] = log_S[i, :ki].sum()

    ratio = kept_log / (total_log + 1e-8)
    loss = (1.0 - ratio).mean()
    return loss


# ═══════════════════════════════════════════════════════════════
# Sample saving (uses DDIM sampling)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def save_samples(generator, images, step, output_dir, ddim_steps=50):
    """Save comparison: original vs DDIM reconstruction."""
    import torchvision.utils as vutils

    generator.eval()
    recon, k = generator.sample(images, steps=ddim_steps)
    generator.train()

    # Denormalize for visualization
    orig_vis = denormalize(images[:4])
    recon_vis = denormalize(recon[:4])

    # Concatenate: [orig1, recon1, orig2, recon2, ...]
    comparison = []
    for i in range(min(4, images.size(0))):
        comparison.append(orig_vis[i])
        comparison.append(recon_vis[i])

    grid = vutils.make_grid(comparison, nrow=2, padding=4, normalize=False)
    vutils.save_image(grid, os.path.join(output_dir, f"samples_step_{step:06d}.png"))

    # Log k values
    k_avg = k.float().mean().item()
    k_min = k.min().item()
    k_max = k.max().item()
    return k_avg, k_min, k_max


# ═══════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")
    print(f"[Train] Output dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)

    # ── Build model ──
    generator, discriminator = build_model_v12(
        dino_path=args.dino_path,
        bridge_dim=args.bridge_dim,
        svd_energy_threshold=args.svd_energy_threshold,
        svd_max_k=args.svd_max_k,
        diffusion_timesteps=args.diffusion_timesteps,
        unet_base_dim=args.unet_base_dim,
        unet_dim_mults=tuple(args.unet_dim_mults),
        device=device,
    )

    # ── Load dataset ──
    print(f"[Train] Loading dataset: aswin00000/ConstructionSite (max {args.num_samples} samples)")
    hf_ds = load_dataset("aswin00000/ConstructionSite", split="train")
    transform = get_train_transform()
    dataset = ConstructionSiteDataset(hf_ds, transform=transform, max_samples=args.num_samples)
    print(f"[Train] Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizers ──
    param_groups = generator.get_trainable_params(lr_enc=args.lr_encoder, lr_dec=args.lr_decoder)
    optimizer_G = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    optimizer_D = torch.optim.AdamW(
        discriminator.parameters(),
        lr=args.lr_discriminator,
        weight_decay=args.weight_decay,
        betas=(0.5, 0.999),
    )

    # ── Losses ──
    gan_loss_fn = GANLoss(gan_mode=args.gan_mode)

    # ── Training state ──
    global_step = 0
    metrics_history = defaultdict(list)

    use_gan = args.gan_weight > 0

    print(f"\n{'='*70}")
    print(f"SR-Diffusion v12 Training (Diffusion Decoder)")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}  Samples: {len(dataset)}")
    print(f"  LR: enc={args.lr_encoder} dec={args.lr_decoder} disc={args.lr_discriminator}")
    print(f"  GAN: {'enabled' if use_gan else 'disabled'} ({args.gan_mode}, weight={args.gan_weight})")
    print(f"  SVD: energy_threshold={args.svd_energy_threshold} max_k={args.svd_max_k}")
    print(f"  Diffusion: T={args.diffusion_timesteps}, cosine schedule, DDIM={args.ddim_steps} steps")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        epoch_metrics = defaultdict(float)
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)
            global_step += 1
            n_batches += 1

            # ═══════════════════════════════════════════════
            # Generator step
            # ═══════════════════════════════════════════════
            optimizer_G.zero_grad()

            # Forward: diffusion noise prediction loss
            # return_recon=True for GAN training
            noise_loss, recon, k, features = generator(
                images,
                return_recon=use_gan,
                ddim_steps=args.ddim_quick_steps,
            )

            loss_G = noise_loss

            # Optional: adversarial loss
            if use_gan and recon is not None:
                pred_fake = discriminator(recon)
                loss_G_gan = gan_loss_fn(pred_fake, target_is_real=True) * args.gan_weight
                loss_G = loss_G + loss_G_gan
            else:
                loss_G_gan = torch.tensor(0.0, device=device)

            # Optional: SVD compression loss
            if args.svd_compress_weight > 0:
                with torch.no_grad():
                    _, S_vals, _ = torch.linalg.svd(features.float(), full_matrices=False)
                loss_svd = svd_compression_loss(S_vals, k) * args.svd_compress_weight
                loss_G = loss_G + loss_svd
            else:
                loss_svd = torch.tensor(0.0, device=device)

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()

            # ═══════════════════════════════════════════════
            # Discriminator step (only if GAN is enabled)
            # ═══════════════════════════════════════════════
            if use_gan and recon is not None:
                optimizer_D.zero_grad()

                pred_real = discriminator(images)
                loss_D_real = gan_loss_fn(pred_real, target_is_real=True)

                pred_fake_d = discriminator(recon.detach())
                loss_D_fake = gan_loss_fn(pred_fake_d, target_is_real=False)

                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()
            else:
                loss_D = torch.tensor(0.0, device=device)

            # ── Metrics ──
            k_avg = k.float().mean().item()
            epoch_metrics['loss_G'] += loss_G.item()
            epoch_metrics['diff_loss'] += noise_loss.item()
            epoch_metrics['loss_G_gan'] += loss_G_gan.item() if use_gan else 0
            epoch_metrics['loss_D'] += loss_D.item() if use_gan else 0
            epoch_metrics['loss_svd'] += loss_svd.item()
            epoch_metrics['k_avg'] += k_avg
            epoch_metrics['k_min'] += k.min().item()
            epoch_metrics['k_max'] += k.max().item()

            # Update progress bar
            postfix = {
                'diff': f"{noise_loss.item():.3f}",
                'k': f"{k_avg:.1f}",
            }
            if use_gan:
                postfix['D'] = f"{loss_D.item():.3f}"
                postfix['G_gan'] = f"{loss_G_gan.item():.3f}"
            pbar.set_postfix(postfix)

            # ── Logging ──
            if global_step % args.log_every == 0:
                log_str = (
                    f"[Step {global_step:06d}] "
                    f"G={loss_G.item():.4f} diff={noise_loss.item():.4f} "
                    f"G_gan={loss_G_gan.item():.4f} "
                    f"D={loss_D.item():.4f} SVD={loss_svd.item():.4f} "
                    f"k_avg={k_avg:.1f} k_min={k.min().item()} k_max={k.max().item()}"
                )
                with open(os.path.join(args.output_dir, "train.log"), "a") as f:
                    f.write(f"{datetime.now().isoformat()} {log_str}\n")

            # ── Save samples ──
            if global_step % args.sample_every == 0:
                k_avg_s, k_min_s, k_max_s = save_samples(
                    generator, images, global_step,
                    os.path.join(args.output_dir, "samples"),
                    ddim_steps=args.ddim_steps,
                )

            # ── Save checkpoint ──
            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, "checkpoints", f"step_{global_step:06d}.pt")
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'args': vars(args),
                }, ckpt_path)
                print(f"\n[Checkpoint] Saved to {ckpt_path}")

        # ── End of epoch summary ──
        epoch_time = time.time() - epoch_start
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            metrics_history[key].append(epoch_metrics[key])

        summary = (
            f"\n[Epoch {epoch}/{args.epochs}] "
            f"G={epoch_metrics['loss_G']:.4f} diff={epoch_metrics['diff_loss']:.4f} "
            f"G_gan={epoch_metrics['loss_G_gan']:.4f} "
            f"D={epoch_metrics['loss_D']:.4f} k_avg={epoch_metrics['k_avg']:.1f} "
            f"Time={epoch_time:.1f}s"
        )
        print(summary)
        with open(os.path.join(args.output_dir, "train.log"), "a") as f:
            f.write(f"{datetime.now().isoformat()} {summary}\n")

        # Epoch checkpoint
        ckpt_path = os.path.join(args.output_dir, "checkpoints", f"epoch_{epoch:03d}.pt")
        torch.save({
            'step': global_step,
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'metrics_history': dict(metrics_history),
            'args': vars(args),
        }, ckpt_path)
        print(f"[Checkpoint] Epoch {epoch} saved to {ckpt_path}")

        # Save samples at end of epoch
        try:
            sample_batch = next(iter(dataloader)).to(device)
            save_samples(generator, sample_batch, f"epoch_{epoch:03d}",
                        os.path.join(args.output_dir, "samples"),
                        ddim_steps=args.ddim_steps)
        except Exception as e:
            print(f"[Warn] Could not save epoch sample: {e}")

    # ── Final save ──
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }, final_path)
    print(f"\n[Final] Model saved to {final_path}")
    print("[Done] Training complete.")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="SR-Diffusion v12 Training (Diffusion Decoder)")
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--dino_path', type=str, default=DEFAULT_DINO_PATH)
    parser.add_argument('--bridge_dim', type=int, default=64,
                        help='Conditioning projector output channels')
    parser.add_argument('--svd_energy_threshold', type=float, default=SVD_ENERGY_THRESHOLD)
    parser.add_argument('--svd_max_k', type=int, default=SVD_MAX_K)
    parser.add_argument('--diffusion_timesteps', type=int, default=DIFFUSION_TIMESTEPS)
    parser.add_argument('--ddim_steps', type=int, default=DDIM_SAMPLE_STEPS,
                        help='DDIM steps for visualization sampling')
    parser.add_argument('--ddim_quick_steps', type=int, default=DDIM_QUICK_STEPS,
                        help='DDIM steps for quick GAN reconstruction')
    parser.add_argument('--unet_base_dim', type=int, default=128,
                        help='UNet base filter dimension')
    parser.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4, 4],
                        help='UNet channel multipliers per level')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr_encoder', type=float, default=LR_ENCODER)
    parser.add_argument('--lr_decoder', type=float, default=LR_DECODER)
    parser.add_argument('--lr_discriminator', type=float, default=LR_DISCRIMINATOR)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--gan_weight', type=float, default=GAN_WEIGHT,
                        help='GAN adversarial loss weight (0 = disable)')
    parser.add_argument('--svd_compress_weight', type=float, default=SVD_COMPRESS_WEIGHT,
                        help='SVD compression loss weight (0 = disable)')
    parser.add_argument('--gan_mode', type=str, default=GAN_MODE, choices=['hinge', 'lsgan', 'vanilla'])
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=LOG_EVERY)
    parser.add_argument('--save_every', type=int, default=SAVE_EVERY)
    parser.add_argument('--sample_every', type=int, default=SAMPLE_EVERY)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
