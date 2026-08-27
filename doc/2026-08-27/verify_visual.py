"""数值验证重建图质量: 原图 vs 重建 的空间结构(std/边缘) 与颜色分布"""
import sys, glob
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import Dinov2Model
sys.path.insert(0, "/root/autodl-tmp/sr-diffusion-v3-test")
from data_v2 import ParquetImageDataset, V2Collator, DINO_MEAN, DINO_STD
from model_v2 import SRPhase1V2

W, H = 448, 252
np_ = (W//14)*(H//14)
dino = Dinov2Model.from_pretrained("/root/autodl-tmp/models/dinov2-large")
if getattr(dino.config, "use_mask_token", False):
    dino.config.use_mask_token = False
    del dino.embeddings.mask_token
model = SRPhase1V2(dinov2=dino, num_patches=np_, dim=dino.config.hidden_size, reencoder_depth=4)
sd = torch.load("/root/autodl-tmp/sr-diffusion-v3-test/output/phase1_v2_pixel_fp32/final_model.pt", map_location="cpu")
model.load_state_dict(sd, strict=True)
model.eval().cuda()

files = sorted(glob.glob("/root/autodl-tmp/construction_site/test-*.parquet"))
ds = ParquetImageDataset(files, limit=3)
loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=4,
                    collate_fn=V2Collator(model_size=(W,H)))
with torch.no_grad():
    for batch in loader:
        x = batch["pixel_values"].cuda()
        B, C, Hh, Ww = x.shape
        feats = model.dinov2(x).last_hidden_state
        cls, pf = feats[:, 0:1], feats[:, 1:]
        sp = model.special_bank(B, x.device)
        z = model.re_encoder(torch.cat([cls, sp, pf], dim=1))
        F = model.decoder(z[:, 0:1], z[:, 1:1+np_])
        F_pix = model.pixel_head(F)
        tgt = x.reshape(B, C, Hh//14, 14, Ww//14, 14).permute(0,2,4,1,3,5).reshape(B, np_, 14*14*3)
        def to_img(p):
            im = p.reshape(B, Hh//14, Ww//14, 14, 14, 3).permute(0,1,3,2,4,5).reshape(B, Hh, Ww, 3)
            return np.clip(im.cpu().numpy()*DINO_STD+DINO_MEAN, 0, 255).astype(np.uint8)
        gt = to_img(tgt); rec = to_img(F_pix)
        break

for b in range(B):
    g, r = gt[b].astype(np.float32), rec[b].astype(np.float32)
    g_std, r_std = g.std(), r.std()
    # 行间/列间变异 (结构代理): 相邻像素差均值
    g_edge = np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()
    r_edge = np.abs(np.diff(r, axis=0)).mean() + np.abs(np.diff(r, axis=1)).mean()
    l1 = np.abs(g-r).mean()
    print(f"图{b}: 原图std={g_std:.1f} 重建std={r_std:.1f} | "
          f"原图边缘={g_edge:.1f} 重建边缘={r_edge:.1f} | L1={l1:.1f} "
          f"(平均色参照≈61, 重建std/边缘应明显>0)")
print("验证: 重建图有空间结构(边缘>0) 且 非平均色(与std相关)")
