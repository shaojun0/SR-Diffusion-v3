"""
SR-Diffusion Phase 1 v2 — register 式推理测试（像素目标版, 2026-08-27 起）
=================================================
架构（model_v2.py, register 式唯一路径）: DINOv2-large + register
    specials(K) 直接拼进输入序列 → OutputQueryDecoder（输出查询注意力,
    分块掩码: 每采样步只 attend 自己的 z_s 块）→ PixelHead → 像素 patch
    预测。K = num_specials 与 N = num_patches 解耦（K 由训练侧最终采样
    步集推导/显式指定, 见 train_v2.py 与 model_v2.py derive_num_specials）。

测试项（像素目标 = 最终判据）:
    1) 全量重建像素 L1（归一化空间 + 反归一化 0-255 空间双口径）——
       对照: 全图平均色 baseline / 每 patch 平均色 baseline。
       像素 L1 必须显著优于"平均色"才有还原意义（特征空间 L1 是假象,
       已证实特征目标退化）。
    2) 渐进重建曲线（2026-08-31 用户需求: 结果沿采样步累加）——
       第 n 步预测的结果 = 第 n-1 步的结果 + 第 n 步的预测, 即
       Y_pix[:, n] = Σ_{t≤n} Y_t。对每步 n 度量像素 L1(Y_pix[:, n],
       target_pix), 得到"累积步数越多重建越精"的渐进曲线。

2026-08-31（分块掩码改造 + 边界实验对齐）:
    · decoder 掩码从 KV 因果前缀改为**分块掩码**: 步 t 只 attend 自己所在的
      z_s 块（块号 ⌊√t⌋）; 默认采样计划 = square_block_starts
      （块起点 = 平方数, 每块一步, 步数 = ⌊√N⌋）;
    · --slice_start/--slice_end 可选挑选分块子区间（默认 None = 全部分块）;
    · --decoder_steps 越界校验: 0 <= s <= N（K 的最终校验交给模型）。

2026-09-02（num_specials 对齐; 修复与 model_v2.py 接口脱节）:
    · 删除 --reencoder_depth / --register_specials（register 式唯一路径,
      model_v2.py 无这些接口）;
    · K 复现训练值: **优先读 model_info.json 的 num_specials**; 没有则
      --num_specials CLI（默认 0=auto, 按与训练一致的 slice/decoder_steps
      自动推导, 公式同 derive_num_specials）; 都没有 = 全量默认 K=N。
      K 必须与 final_model.pt 权重形状一致, 否则 strict load 直接崩。

用法:
    python infer_v2_test.py \
        --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --final_model output/phase1_v2_pixelfp32/final_model.pt \
        --output output/phase1_v2_pixelfp32/infer_test.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from data_v2 import ParquetImageDataset, V2Collator, DINO_MEAN, DINO_STD
from model_v2 import SRPhase1V2
import model_v2   # 读侧掩码开关对齐（SRV2_MEMORY_OPEN, 见下方 memory_open 强制对齐）


def parse_args():
    p = argparse.ArgumentParser(description="infer test (pixel target) for OutputQueryDecoder v2")
    p.add_argument("--data_dir", required=True, help="parquet 目录(test-*.parquet)")
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True, help="训练好的权重(final_model.pt)")
    p.add_argument("--output", default="output/phase1_v2_pixelfp32/infer_test.json")
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--canvas", default="1600x900")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条 test(0=全量)")
    p.add_argument("--num_specials", type=int, default=0,
                   help="register/specials 数 K: 0=自动（按训练一致的 "
                        "slice/decoder_steps 推导, 公式同训练）; >0=显式 K。"
                        "model_info.json 有 num_specials 字段时以它为准")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--decoder_depth", type=int, default=2,
                   help="OutputQueryDecoder 的 TransformerDecoder 层数(与训练一致)")
    p.add_argument("--slice_start", type=int, default=None,
                   help="可选挑选分块起点索引(与训练 --slice_start 一致); 默认 None = 全部分块")
    p.add_argument("--slice_end", type=int, default=None,
                   help="可选挑选分块终点索引(与训练 --slice_end 一致); 默认 None = 全部分块")
    p.add_argument("--decoder_steps", default=None,
                   help="必须与训练一致(逗号分隔); 默认 square_block_starts(N) (分块起点=平方数)")
    return p.parse_args()


def _patch_to_img(pix_patches, H, W):
    """(B,N,14,14,3) 归一化像素 patch → (B,H,W,3) 反归一化 0-255 图像。

    布局与 DINO 一致: row-major (先 y 后 x), N = (H/14)*(W/14)。
    """
    B = pix_patches.shape[0]
    img = pix_patches.reshape(B, H // 14, W // 14, 14, 14, 3) \
                     .permute(0, 1, 3, 2, 4, 5) \
                     .reshape(B, H, W, 3)
    img = img.cpu().numpy() * DINO_STD + DINO_MEAN
    return np.clip(img, 0, 255)


def main():
    args = parse_args()
    W, H = (int(v) for v in args.model_input.lower().split("x"))
    num_patches = (W // 14) * (H // 14)
    PATCH_PX = 14 * 14 * 3

    steps = None
    if args.decoder_steps:
        steps = [int(s) for s in args.decoder_steps.split(",") if s.strip()]
        # 采样时刻 t ∈ [0, N]; t ≤ K ≤ N 的最终校验由模型构造时完成
        # （与 train_v2.py 相同的解析/校验逻辑）
        assert steps and all(0 <= s <= num_patches for s in steps), \
            f"decoder_steps 越界: {steps} (N={num_patches})"

    # ── model_info.json 提前读取（K/采样对齐提示, 不强制）──
    # 训练侧把 num_specials / decoder_depth / slice_start / slice_end /
    # decoder_steps 写在 output_dir/model_info.json。K 错则 strict load 形状
    # 不匹配直接崩, 所以**构造模型前**先按训练侧配置对齐 num_specials。
    info_path = os.path.join(os.path.dirname(args.final_model), "model_info.json")
    train_info = None
    if os.path.exists(info_path):
        with open(info_path) as f:
            train_info = json.load(f)
        for k in ("slice_start", "slice_end", "decoder_depth"):
            if k in train_info and train_info[k] != args.__dict__[k]:
                print(f"[warn] model_info.json 记录 {k}={train_info[k]}, "
                      f"但 --{k}={args.__dict__[k]}: 与训练配置不一致, "
                      f"请按训练配置传参")
    # num_specials(K) 解析: ① model_info.json 优先; ② --num_specials CLI;
    # ③ 都没有 → None = 全量默认 K=N
    num_specials = None
    if train_info is not None and "num_specials" in train_info:
        num_specials = int(train_info["num_specials"])
        if args.num_specials and args.num_specials != num_specials:
            print(f"[warn] model_info.json 记录 num_specials={num_specials}, 与 "
                  f"--num_specials={args.num_specials} 不一致: 以 model_info 为准")
    elif args.num_specials:
        num_specials = args.num_specials
    if train_info is not None and "num_specials" not in train_info:
        # 旧 register 训练产物（K=N=num_patches, 无 num_specials 字段）:
        # 若 slice 非默认, 自动推导的 K 会与旧权重不符 → strict load 崩
        print(f"[info] model_info.json 无 num_specials 字段（旧 register 产物, "
              f"K=N={num_patches}）: 若 strict load 形状不符, 请显式 "
              f"--num_specials {num_patches}")

    # ── 模型: 训练好的重建权重 ──
    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size,
                       heads=args.heads, mlp_ratio=args.mlp_ratio,
                       decoder_steps=steps,
                       decoder_depth=args.decoder_depth,
                       skip_steps=args.slice_start,
                       max_steps=args.slice_end,
                       num_specials=num_specials)
    sd = torch.load(args.final_model, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().cuda()
    T_steps = model.decoder.steps
    print(f"[model] loaded {args.final_model}: N={num_patches}, "
          f"K(num_specials)={model.num_specials}, "
          f"decoder 采样 {len(T_steps)} 步 {T_steps[:6]}...{T_steps[-3:]}")

    # ── model_info.json 对齐提示（加载后完整对比, 不强制）──
    if train_info is not None:
        mism = []
        if ("num_specials" in train_info
                and int(train_info["num_specials"]) != model.num_specials):
            mism.append(f"num_specials: 训练 {train_info['num_specials']} "
                        f"!= 推理 {model.num_specials}")
        if ("decoder_steps" in train_info
                and list(train_info["decoder_steps"]) != T_steps):
            mism.append(f"decoder_steps: 训练 {train_info['decoder_steps']} "
                        f"!= 推理 {T_steps}")
        if mism:
            print(f"[warn] 推理参数与训练侧 model_info.json 不一致 ({info_path}):")
            for m in mism:
                print(f"    - {m}")
        else:
            print(f"[ok] 推理参数与训练侧 model_info.json 对齐 ({info_path})")

    # ── memory_open（读侧全开, 2026-09-04）: 掩码影响前向输出, 推理必须与
    # 训练同开关——以 model_info.json 记录为准强制对齐, 并打印证据 ──
    mem_open = bool(train_info.get("memory_open", False)) if train_info else False
    if mem_open != model_v2.SRV2_MEMORY_OPEN:
        print(f"[warn] model_info.json 记录 memory_open={mem_open} 与当前环境 "
              f"不一致: 以训练侧为准强制对齐 (model_v2.SRV2_MEMORY_OPEN := "
              f"{mem_open})")
        model_v2.SRV2_MEMORY_OPEN = mem_open
    else:
        print(f"[ok] memory_open={mem_open}（读侧掩码 "
              f"{'全开(全键可及)' if mem_open else '分块/首步前缀'}）与训练侧一致")

    # ── 数据: test 分片, 与训练同预处理（1600:900 画布 → 448x252）──
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    assert test_files, f"无 test-*.parquet in {args.data_dir}"
    coll = V2Collator(model_size=(W, H), canvas=(1600, 900))
    ds = ParquetImageDataset(test_files, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=coll)
    n_total = len(ds)
    print(f"[data] {n_total} 条 test")

    # ── 前向: 全量 test, 像素 L1（归一化空间 + 0-255 空间）──
    norm_sum, norm_sq, pix_sum, pix_sq, n = 0.0, 0.0, 0.0, 0.0, 0
    step_pix_sum = np.zeros(len(T_steps), np.float64)   # 每采样步像素 L1 (0-255)
    step_pix_sq = np.zeros(len(T_steps), np.float64)
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["pixel_values"].cuda()            # (B,3,H,W) 归一化
            B, C, Hh, Ww = x.shape
            out = model(x)                              # 同一 forward（两种模式通用）
            F_pix = out["F_hat"]                        # (B,N,588) 采样步平均
            Y_pix = out["Y_pix"]                        # (B,|T|,N,588) 每采样步
            target = out["target_pix"]                  # (B,N,588)
            # 归一化空间 L1
            l1_norm = (F_pix - target).abs().mean(dim=(1, 2))          # (B,)
            norm_sum += l1_norm.sum().item()
            norm_sq += (l1_norm ** 2).sum().item()
            # 0-255 空间: 反归一化 (patch 级, 与 pixel_recon_check 同口径)
            gt_255 = _patch_to_img(target, Hh, Ww)      # (B,H,W,3)
            recon_255 = _patch_to_img(F_pix, Hh, Ww)
            l1_pix = np.abs(recon_255 - gt_255).mean(axis=(1, 2, 3))   # (B,)
            pix_sum += l1_pix.sum().item()
            pix_sq += (l1_pix ** 2).sum().item()
            # 每采样步像素 L1 (0-255)
            for i in range(len(T_steps)):
                step_img = _patch_to_img(Y_pix[:, i], Hh, Ww)
                sl1 = np.abs(step_img - gt_255).mean(axis=(1, 2, 3))   # (B,)
                step_pix_sum[i] += sl1.sum().item()
                step_pix_sq[i] += (sl1 ** 2).sum().item()
            n += B
            if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                print(f"  ... {n}/{n_total} ({time.time() - t0:.0f}s)", flush=True)

    norm_mean = norm_sum / n
    pix_mean = pix_sum / n
    pix_std = np.sqrt(max(pix_sq / n - pix_mean ** 2, 0.0))
    step_pix_mean = step_pix_sum / n
    step_pix_std = np.sqrt(np.maximum(step_pix_sq / n - step_pix_mean ** 2, 0.0))

    print(f"\n[full] 全量重建像素 L1 (归一化空间) = {norm_mean:.6f}")
    print(f"[full] 全量重建像素 L1 (0-255 空间) = {pix_mean:.2f} ± {pix_std:.2f}")
    print(f"       参照(旧实验): 全图平均色≈61, 每patch平均色≈?, 质心基线见 pixel_recon_check")
    print(f"[steps] 渐进曲线 ({len(T_steps)} 步, 0-255 像素 L1, 累积结果):")
    for i, t in enumerate(T_steps):
        print(f"    步 {i + 1:2d} (块起点 {t:4d}, 前 {i + 1:3d} 步累积) L1 = {step_pix_mean[i]:.2f} ± {step_pix_std[i]:.2f}")
    head = step_pix_mean[:min(4, len(step_pix_mean))]
    tail = step_pix_mean[max(0, len(step_pix_mean) - 4):]
    print(f"[steps] 前段(少步累积) {head.mean():.2f} | 后段(多步累积) {tail.mean():.2f} | "
          f"最少/最多步累积 = {step_pix_mean[0]:.2f}/{step_pix_mean[-1]:.2f}")
    print(f"[time] {(time.time() - t0):.0f}s | {n} 图")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "n": n, "num_patches": num_patches,
            "num_specials": model.num_specials, "input": [W, H],
            "decoder_steps": T_steps,
            "full_norm_l1": float(norm_mean),
            "full_pixel_l1_255": float(pix_mean), "full_pixel_std_255": float(pix_std),
            "step_pixel_l1_255": [float(v) for v in step_pix_mean],
            "step_pixel_std_255": [float(v) for v in step_pix_std],
            "head_mean": float(head.mean()), "tail_mean": float(tail.mean()),
            "shortest_l1": float(step_pix_mean[0]), "longest_l1": float(step_pix_mean[-1]),
            "time_s": round(time.time() - t0, 1),
        }, f, indent=2, ensure_ascii=False)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
