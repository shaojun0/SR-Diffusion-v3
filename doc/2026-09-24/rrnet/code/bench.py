"""对 12 个配置逐一测速 + 显存，用于排期（不改动训练口径）。"""
import os, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model import BACKBONES, build_model

SIZE = {"224": 224, "448": 448}

class Ds(Dataset):
    def __init__(self, p): self.a = np.load(p, mmap_mode="r")
    def __len__(self): return min(256, self.a.shape[0])
    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(self.a[i], np.float32) / 255.0).permute(2, 0, 1).contiguous()


def cycle(dl):
    while True:
        for b in dl:
            yield b

def main():
    data_dir = "/root/autodl-tmp/rrnet/data"
    out = []
    for res in (224, 448):
        ds = Ds(os.path.join(data_dir, f"div2k_train_{res}.npy"))
        bs = 16 if res == 224 else 8
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2)
        for bb in BACKBONES:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m = build_model(bb, res).cuda().train()
            amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            it = cycle(dl)
            for _ in range(2):  # warmup
                x = next(it).cuda()
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    loss = F.l1_loss(m(x), x)
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); t0 = time.time(); n = 6
            for _ in range(n):
                x = next(it).cuda()
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    loss = F.l1_loss(m(x), x)
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            s_it = (time.time() - t0) / n
            peak = torch.cuda.max_memory_allocated() / 1e9
            spe = 800 // bs
            rec = dict(backbone=bb, res=res, bs=bs, s_it=round(s_it, 3),
                       peak_GB=round(peak, 2), steps_per_epoch=spe,
                       epoch_min=round(s_it * spe / 60, 2), run40_min=round(s_it * spe * 40 / 60, 1))
            print(json.dumps(rec), flush=True)
            out.append(rec)
            del m, opt
    json.dump(out, open("/root/autodl-tmp/rrnet/logs/bench.json", "w"), indent=2)
    tot = sum(r["run40_min"] for r in out)
    print(f"TOTAL 12 runs (40 ep) = {tot/60:.2f} h")

if __name__ == "__main__":
    main()
