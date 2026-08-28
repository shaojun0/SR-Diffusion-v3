import glob, torch, numpy as np
from torch.utils.data import DataLoader
from transformers import Dinov2Model
from data_v2 import ParquetImageDataset, V2Collator, DINO_MEAN, DINO_STD
from model_v2 import SRPhase1V2

torch.manual_seed(0)
W, H = 448, 252
N = (W//14)*(H//14)
PATCH_PX = 14*14*3
dino = Dinov2Model.from_pretrained('/root/autodl-tmp/models/dinov2-large')
if getattr(dino.config,'use_mask_token',False):
    dino.config.use_mask_token=False; del dino.embeddings.mask_token
model = SRPhase1V2(dinov2=dino, num_patches=N, dim=dino.config.hidden_size, reencoder_depth=4)
sd = torch.load('/root/autodl-tmp/sr-diffusion-v3-test/output/phase1_v2_fp32_uniform/final_model.pt', map_location='cpu')
model.load_state_dict(sd, strict=True)
model.eval().cuda()

files = sorted(glob.glob('/root/autodl-tmp/construction_site/test-*.parquet'))
ds = ParquetImageDataset(files, limit=32)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=V2Collator(model_size=(W,H)))

P_all, zs_all, F_all, pix_all = [], [], [], []
with torch.no_grad():
    for batch in loader:
        x = batch['pixel_values'].cuda()
        # register_specials 兼容: encode() 按模式分派（ReEncoder / DINO register）
        z_cls, z_s = model.encode(x)
        F = model.decoder(z_cls, z_s)
        P = model.dinov2(x).last_hidden_state[:, 1:]     # 原始 DINO patch 特征(基线)
        x_np = x.cpu().numpy().transpose(0,2,3,1) * DINO_STD + DINO_MEAN
        x_np = np.clip(x_np, 0, 255)
        B = x_np.shape[0]
        pix = np.zeros((B, N, PATCH_PX), np.float32)
        for bi in range(B):
            for py in range(H//14):
                for px in range(W//14):
                    blk = x_np[bi, py*14:(py+1)*14, px*14:(px+1)*14]
                    pix[bi, py*(W//14)+px] = blk.reshape(-1)
        P_all.append(P.float().cpu().numpy())
        zs_all.append(z_s.float().cpu().numpy())
        F_all.append(F.float().cpu().numpy())
        pix_all.append(pix)
P = np.concatenate(P_all); Z = np.concatenate(zs_all); F = np.concatenate(F_all)
pix = np.concatenate(pix_all)                       # (M,N,588)
M = P.shape[0]

def within_std(T):
    return float(T.std(axis=1).mean(axis=(0,1)))

def patch_seq_to_img(seq):                          # (M,N,14,14,3) -> (M,H,W,3)
    img = np.zeros((M, H, W, 3), np.float32)
    for py in range(H//14):
        for px in range(W//14):
            k = py*(W//14)+px
            img[:, py*14:(py+1)*14, px*14:(px+1)*14] = seq[:, k]
    return img

gt_img = patch_seq_to_img(pix.reshape(M, N, 14, 14, 3))

def dec_l1(T, name):
    X = T.reshape(-1, T.shape[-1])
    Wm, *_ = np.linalg.lstsq(X, pix.reshape(-1, PATCH_PX), rcond=1e-6)
    recon = (X @ Wm).reshape(M, N, 14, 14, 3)
    rimg = patch_seq_to_img(recon)
    l1 = float(np.abs(rimg - gt_img).mean())
    print(f"{name} → 像素 L1 = {l1:.2f}")
    return l1

print("=== 空间信息(每图内跨位置变异性) 逐级追踪 ===")
p_std, z_std, f_std = within_std(P), within_std(Z), within_std(F)
print(f"DINO patch P   : within-std = {p_std:.3e}")
print(f"ReEncoder  z_s : within-std = {z_std:.3e}  ({z_std/p_std*100:.0f}% 相对P)")
print(f"Decoder  F_hat : within-std = {f_std:.3e}  ({f_std/p_std*100:.1f}% 相对P)")
print()
print("=== 像素可还原性 (线性解码 L1, 0-255 尺度, 越小越好) ===")
dec_l1(P, "真实 DINO patch P")
dec_l1(Z, "ReEncoder 输出 z_s")
dec_l1(F, "Decoder 输出 F_hat")
img_mean = np.broadcast_to(gt_img.mean(axis=(1,2), keepdims=True), gt_img.shape)
print(f"全图平均色参照          → 像素 L1 = {float(np.abs(img_mean-gt_img).mean()):.2f}")
