#!/usr/bin/env python3
"""SR v3 — No DataLoader, pure PyTorch, guaranteed to work."""

import os, glob, time, math, random
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import numpy as np

torch.backends.cudnn.benchmark = True

OUT = "/root/autodl-tmp/sr_v3_output"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
BS, EP, LR = 8, 40, 1e-4
LSZ, PSZ, HSZ = 32, 256, 1024

class ResBlock(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.n = nn.Sequential(nn.Conv2d(c,c,3,1,1), nn.ReLU(True), nn.Conv2d(c,c,3,1,1))
    def forward(self, x): return x + self.n(x)

class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        c=64; self.h = nn.Conv2d(3,c,3,1,1)
        self.b = nn.Sequential(*[ResBlock(c) for _ in range(8)], nn.Conv2d(c,c,3,1,1))
        self.t = nn.Conv2d(c,3,3,1,1)
    def forward(self, x):
        f=self.h(x); return self.t(self.b(f)+f)+x

print("="*50, flush=True)
print("SR v3 — Manual Training Loop", flush=True)
print("="*50, flush=True)

# Load images
files = sorted(glob.glob(os.path.join("/root/autodl-tmp/DIV2K_train_HR", "*.png")))
print(f"  Loading {len(files)} images...", end=" ", flush=True)
imgs = []
for f in files:
    img = Image.open(f).convert("RGB").resize((HSZ,HSZ), Image.BICUBIC)
    imgs.append(torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1))
data = torch.stack(imgs)
print(f"Done! {data.shape}", flush=True)

N = data.size(0)
PPI = (HSZ//PSZ)**2  # 16
total = N * PPI
n_batch = total // BS

print(f"  Images: {N}, patches: {total}, batches/epoch: {n_batch}", flush=True)
model = SRNet().to(DEV)
print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP)

# Pre-compute patch indices for efficiency
indices = [(i, r, c) for i in range(N) for r in range(HSZ//PSZ) for c in range(HSZ//PSZ)]

best_loss = float("inf")
t0 = time.time()

for ep in range(1, EP+1):
    model.train()
    random.shuffle(indices)
    ep_loss, nb = 0.0, 0
    
    for bi in range(0, len(indices), BS):
        batch_idx = indices[bi:bi+BS]
        if len(batch_idx) < BS: continue
        
        lr_b, hr_b = [], []
        for i, r, c in batch_idx:
            hp = data[i, :, r*PSZ:(r+1)*PSZ, c*PSZ:(c+1)*PSZ]
            ls = F.interpolate(hp.unsqueeze(0), size=(LSZ,LSZ), mode='bicubic')
            lp = F.interpolate(ls, size=(PSZ,PSZ), mode='bicubic').squeeze(0)
            # aug
            if random.random()>0.5: lp, hp = torch.flip(lp,[-1]), torch.flip(hp,[-1])
            if random.random()>0.5: lp, hp = torch.flip(lp,[-2]), torch.flip(hp,[-2])
            lr_b.append(lp); hr_b.append(hp)
        
        lr_t = torch.stack(lr_b).to(DEV)
        hr_t = torch.stack(hr_b).to(DEV)
        pred = model(lr_t)
        loss = F.l1_loss(pred, hr_t)
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss += loss.item(); nb += 1
    
    sch.step()
    avg = ep_loss / nb
    
    # Validation
    if ep % 5 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            ht = data[0:1].to(DEV)
            ls = F.interpolate(ht, size=(LSZ,LSZ), mode='bicubic')
            li = F.interpolate(ls, size=(HSZ,HSZ), mode='bicubic')
            so = model(li)
        vm = F.mse_loss(so, ht).item(); bm = F.mse_loss(li, ht).item()
        vp = 20*math.log10(1.0)-10*math.log10(vm) if vm>0 else 100
        bp = 20*math.log10(1.0)-10*math.log10(bm) if bm>0 else 100
        print(f"  [Ep {ep:3d}] loss={avg:.4f} | PSNR={vp:.2f}dB(bic={bp:.2f}) | lr={sch.get_last_lr()[0]:.2e}", flush=True)
    else:
        print(f"  [Ep {ep:3d}] loss={avg:.4f} | lr={sch.get_last_lr()[0]:.2e}", flush=True)
    
    if avg < best_loss:
        best_loss = avg
        torch.save({"model":model.state_dict(),"epoch":ep,"loss":best_loss}, os.path.join(OUT,"best.pt"))
    if ep % 10 == 0:
        torch.save({"model":model.state_dict(),"epoch":ep,"loss":avg}, os.path.join(OUT,f"ckpt_ep{ep}.pt"))

h = (time.time()-t0)/3600
print(f"\nDone. {h:.1f}h, best={best_loss:.4f}", flush=True)
torch.save({"model":model.state_dict(),"epoch":EP}, os.path.join(OUT,"final.pt"))
