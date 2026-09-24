"""常识性检查：模型是否真的在重建（而不是停在「预测均值」的平凡解）。

1) 平凡解基线：用训练集逐像素均值图去预测 val，算 L1 / PSNR。
   若模型 L1 >= 均值基线，说明它什么都没学到（参考项目 README 记载过这个坑）。
2) 导出 best.pt 的重建样例，目视确认。
"""
import json, os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from model import build_model

DATA = "/root/autodl-tmp/rrnet/data"
CKPT = "/root/autodl-tmp/rrnet/ckpt"
OUT = "/root/autodl-tmp/rrnet/results"
os.makedirs(OUT, exist_ok=True)

rows = []
for res in (224, 448):
    tr = np.load(os.path.join(DATA, f"div2k_train_{res}.npy"), mmap_mode="r")
    va = np.load(os.path.join(DATA, f"div2k_val_{res}.npy"), mmap_mode="r")
    # 逐像素均值图（[0,1]）
    mean_img = np.asarray(tr, dtype=np.float32).mean(axis=0) / 255.0   # (S,S,3)
    m = torch.from_numpy(mean_img).permute(2, 0, 1)[None]
    v = torch.from_numpy(np.asarray(va[:100], dtype=np.float32) / 255.0).permute(0, 3, 1, 2)
    l1 = F.l1_loss(m.expand_as(v), v).item()
    mse = F.mse_loss(m.expand_as(v), v).item()
    psnr = 10 * np.log10(1.0 / mse)
    rows.append(dict(res=res, mean_baseline_l1=round(l1, 5), mean_baseline_psnr=round(float(psnr), 3)))
    print(f"[{res}] 均值图基线: L1={l1:.5f}  PSNR={psnr:.3f}", flush=True)

    # 重建样例（用每个 backbone 的 best.pt）
    tiles = []
    for bb in ["resnet-10", "resnet-34", "resnet-50", "resnet-152"]:
        p = os.path.join(CKPT, f"{bb}_{res}", "best.pt")
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location="cuda", weights_only=False)
        net = build_model(bb, res).cuda().eval()
        net.load_state_dict(ck["model"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rec = net(v[:4].cuda()).float().clamp(0, 1)
        tiles.append((bb, v[:4], rec.cpu()))
        d = json.load(open(os.path.join(CKPT, f"{bb}_{res}", "metrics.json")))
        print(f"   {bb:11s} best_l1={d['best_val_l1']:.5f} vs mean {l1:.5f} "
              f"-> {'OK 优于基线' if d['best_val_l1'] < l1 else '!! 不优于基线'}", flush=True)
        del net, ck
        torch.cuda.empty_cache()

    # 拼图：行 = 原图 / 各模型重建
    if tiles:
        n = 4
        grid = [torch.cat([t[1][i] for t in tiles], dim=2) for i in range(n)]
        top = torch.cat(grid, dim=1)                                  # 原图
        bots = [torch.cat([t[2][i] for t in tiles], dim=2) for i in range(n)]
        bot = torch.cat(bots, dim=1)
        canvas = torch.cat([top, bot], dim=1)                         # 上原图 下重建
        arr = (canvas.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(OUT, f"samples_{res}.png"))
        print(f"   saved {OUT}/samples_{res}.png (上=原图, 下=重建, 列={[t[0] for t in tiles]})", flush=True)

json.dump(rows, open(os.path.join(OUT, "mean_baseline.json"), "w"), indent=2)
print("DONE")
