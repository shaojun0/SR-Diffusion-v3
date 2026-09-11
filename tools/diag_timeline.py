"""diag_timeline.py — 逐 step 训练动力学时间线探针（定位"谁先死"）。

自包含的最小训练循环, 复制 train_v2.py 的配方:
  AdamW(lr, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01)
  + get_scheduler("cosine", num_warmup_steps, num_training_steps)
  + clip_grad_norm_(1.0) + 先 optimizer.step() 后 scheduler.step()
  + DINOv2-large 不冻结, loss = 平权逐步累加像素 L1

每 `--probe_every` 步在**固定探针 batch**上记录:
  z_s_within_std      —— z_s 在 K 个 register 之间的 std（DINO 侧寄存器多样性）
  z_s_across_patch_std—— 每个 register 在 N 个 patch 之间的 std（对图像内容是否敏感）
  z_norm / z_cls_norm
  Y_*_std             —— 解码器逐步输出跨 patch std（解码器侧多样性）
  F_hat_std / Y_pix_std
  probe_loss / per-step loss
并同时记录该训练 step 的**逐模块 grad norm**（clip 前）, 找"哪个模块先死"。

用法:
  python diag_timeline.py --arm B --stack_dim 2048 --depth 4 --heads 16 \
      --dropout 0.05 --lr 1.5e-4 --warmup_ratio 0.4367 --max_steps 400 \
      --probe_every 10 --out /root/train_logs/stack2x_timeline_B.json
"""
import argparse, copy, glob, json, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Dinov2Model, get_scheduler, set_seed

from data_v2 import ParquetImageDataset, V2Collator
from model_v2 import SRPhase1V2

W, H, N = 448, 252, 576
STEPS = [1, 4, 9, 16, 25]
K = 35
DIM = 1024


def pstd(t):
    return float(t.std(dim=-2).mean().item())


def grad_norm(p):
    return 0.0 if p.grad is None else float(p.grad.detach().norm().item())


def make_groups(model):
    g = {"dinov2.embeddings": [p for n, p in model.named_parameters()
                               if n.startswith("embeddings")]}
    for i in (0, 11, 23):
        g[f"dinov2.L{i}"] = [p for n, p in model.named_parameters()
                             if f"encoder.layer.{i}." in n]
    g["dinov2.layernorm"] = [p for n, p in model.named_parameters()
                             if "layernorm" in n]
    for nm in ("special_bank",):
        g[f"decoder.{nm}"] = [p for n, p in model.named_parameters() if nm in n]
    g["decoder.query_base"] = [p for n, p in model.named_parameters()
                               if "query_base" in n]
    g["decoder.pos_embed"] = [p for n, p in model.named_parameters()
                              if "pos_embed" in n]
    g["decoder.pixel_head"] = [p for n, p in model.named_parameters()
                               if "pixel_head" in n]
    for nm in ("stack_in", "stack_out"):
        g[f"decoder.{nm}"] = [p for n, p in model.named_parameters()
                              if f".{nm}." in n]
    nl = len(model.decoder.stack.layers)
    for i in range(nl):
        g[f"decoder.stack.L{i}"] = [p for n, p in model.named_parameters()
                                    if f"stack.layers.{i}." in n]
    return g


@torch.no_grad()
def probe(model, pbatch):
    model.eval()
    x = pbatch
    z_cls, z_s = model.encode(x)
    out = model(x)
    Y = model.decoder(z_cls, z_s)              # (B,T,N,D)
    Ypix = model.pixel_head(Y)
    d = {
        "z_s_within_std": float(z_s.std(dim=1).mean()),          # 跨 K 个 register
        "z_s_feat_std": float(z_s.std(dim=2).mean()),            # 跨特征维
        "z_s_across_image_std": float(z_s.std(dim=0).mean()),    # 跨 batch(内容敏感度)
        "z_norm": float(z_s.norm(dim=-1).mean()),
        "z_cls_norm": float(z_cls.norm(dim=-1).mean()),
        "Y_across_patch_std": pstd(Y),
        "Y_abs_mean": float(Y.abs().mean()),
        "Y_step_std": [float(Y[:, i].std(dim=1).mean()) for i in range(len(STEPS))],
        "Ypix_across_patch_std": pstd(Ypix),
        "F_hat_across_patch_std": pstd(out["F_hat"]),
        "probe_loss": float(out["loss"]),
        "probe_recon": float(out["recon"]),
        "probe_per_step_loss": [float(v) for v in
                                F.l1_loss(Ypix, out["target_pix"].unsqueeze(1)
                                          .expand_as(Ypix), reduction="none")
                                .mean(dim=(0, 2, 3))],
    }
    model.train()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/construction_site")
    ap.add_argument("--dino_dir", default="/root/autodl-tmp/models/dinov2-large")
    ap.add_argument("--stack_dim", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.4367)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--probe_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    train_files = sorted(glob.glob(os.path.join(args.data_dir, "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    coll = V2Collator(model_size=(W, H), canvas=(1600, 900), angle_step=0.5)
    train_ds = ParquetImageDataset(train_files)
    prob_ds = ParquetImageDataset(test_files, limit=args.bs)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=8, collate_fn=coll, drop_last=True,
                              generator=g)
    prob_loader = DataLoader(prob_ds, batch_size=args.bs, shuffle=False,
                             num_workers=2, collate_fn=coll)
    pbatch = next(iter(prob_loader))["pixel_values"].cuda()

    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V2(dinov2=dino, num_patches=N, dim=DIM, heads=args.heads,
                       mlp_ratio=4.0, decoder_steps=STEPS,
                       decoder_depth=args.depth, skip_steps=0, max_steps=5,
                       num_specials=K, query_mask_mode="blockdiag",
                       stack_dim=args.stack_dim, decoder_dropout=args.dropout)
    model.cuda().train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    warmup = int(args.max_steps * args.warmup_ratio)
    print(f"[cfg] arm={args.arm} stack_dim={args.stack_dim} depth={args.depth} "
          f"heads={args.heads} dropout={args.dropout} lr={args.lr} "
          f"warmup={warmup}/{args.max_steps} params={n_train/1e6:.1f}M",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=args.weight_decay)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=warmup,
                          num_training_steps=args.max_steps)
    groups = make_groups(model)

    timeline = []
    step = 0
    t0 = time.time()
    done = False
    while not done:
        for batch in train_loader:
            x = batch["pixel_values"].cuda()
            out = model(x)
            loss = out["loss"]
            loss.backward()
            # 逐模块 grad norm 必须在 clip/zero_grad 之前取
            gn = {k: sum(grad_norm(p) ** 2 for p in v) ** 0.5
                  for k, v in groups.items()}
            gnorm_raw = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip))
            cur_lr = float(sched.get_last_lr()[0])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.probe_every == 0 or step == 1:
                rec = {"step": step, "lr": cur_lr, "loss": float(loss.detach()),
                       "grad_norm_raw": gnorm_raw}
                rec["grad_norms"] = gn
                rec.update(probe(model, pbatch))
                timeline.append(rec)
                print(json.dumps({"step": step, "lr": round(cur_lr, 9),
                                  "loss": round(rec["loss"], 5),
                                  "gn_raw": round(gnorm_raw, 5),
                                  "z_win": round(rec["z_s_within_std"], 5),
                                  "z_img": round(rec["z_s_across_image_std"], 5),
                                  "Y_ap": round(rec["Y_across_patch_std"], 5),
                                  "F_ap": round(rec["F_hat_across_patch_std"], 5),
                                  "probe_loss": round(rec["probe_loss"], 5),
                                  "gn_pixelhead": round(gn["decoder.pixel_head"], 4),
                                  "gn_stackout": round(gn["decoder.stack_out"], 4),
                                  "gn_dinoL0": round(gn["dinov2.L0"], 4),
                                  }), flush=True)
            if step >= args.max_steps:
                done = True
                break
    print(f"[done] {step} steps in {time.time()-t0:.0f}s", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "timeline": timeline}, f, indent=2)
        print("[out]", args.out)


if __name__ == "__main__":
    main()
