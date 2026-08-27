"""
泄露诊断: 为什么 t=0 (仅 1 个键) 就能全量还原? 四个方向排查:
  1. 目标方差: DINO patch 特征本身是否几乎恒定(重建任务无意义低)?
  2. decoder 掩码生效性: 扰动被屏蔽的 z_s[j](j>t), Y_t 应不变;
     扰动可见的 z_s[j](j<=t), Y_t 应变。
  3. z_cls 信息量: 扰动 patch 特征, 看 z_cls / z_s 是否随之变化
     (ReEncoder 的 cls/specials 掩码是否让它们看到全部 patches)。
  4. query_base 角色: t=0 时 576 行查询共享 1 个键 —— 输出若几乎
     不依赖键, 说明是"查询基模板 + 常数偏置"在还原, 而非键的信息。
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from data_v2 import ParquetImageDataset, V2Collator
from model_v2 import SRPhase1V2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True)
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=64, help="诊断用小样本即可")
    return p.parse_args()


def main():
    args = parse_args()
    W, H = (int(v) for v in args.model_input.lower().split("x"))
    num_patches = (W // 14) * (H // 14)

    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size, reencoder_depth=4)
    sd = torch.load(args.final_model, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().cuda()

    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    coll = V2Collator(model_size=(W, H))
    ds = ParquetImageDataset(test_files, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=coll)

    T_steps = model.decoder.steps
    NT = len(T_steps)

    # ── 收集一批特征 ──
    xs, patches, z_cls_list, z_s_list, Y_list, F_list = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].cuda()
            feats = model.dinov2(x).last_hidden_state
            cls, patch = feats[:, 0:1], feats[:, 1:]
            specials = model.special_bank(x.shape[0], x.device)
            z = model.re_encoder(torch.cat([cls, specials, patch], dim=1))
            z_cls, z_s = z[:, 0:1], z[:, 1:1 + num_patches]
            F_hat = model.decoder(z_cls, z_s)
            Y = model.decoder.last_Y
            xs.append(x); patches.append(patch); z_cls_list.append(z_cls)
            z_s_list.append(z_s); Y_list.append(Y); F_list.append(F_hat)
    x = torch.cat(xs); patch = torch.cat(patches)
    z_cls = torch.cat(z_cls_list); z_s = torch.cat(z_s_list)
    Y = torch.cat(Y_list); F_hat = torch.cat(F_list)
    B = x.shape[0]
    print(f"[data] {B} 图, N={num_patches}, T={NT} 步")

    report = {"n": B, "num_patches": num_patches, "steps": T_steps}

    # ══════════════════════════════════════════════════════════
    # 1. 目标方差: DINO patch 特征各行(各 patch 位置)差异多大?
    #    · 行间方差 (不同 patch 位置的特征差异): 小 ⇒ 目标近乎常数
    #    · 行内方差 (同 patch 特征各维度): 参考
    # ══════════════════════════════════════════════════════════
    p_flat = patch.reshape(-1, patch.shape[-1])            # (B*N, D)
    between_patch_std = patch.std(dim=1).mean().item()     # 每图内跨 patch 的 std
    global_std = patch.std().item()
    per_dim_std = patch.reshape(-1, patch.shape[-1]).std(dim=0).mean().item()
    # 每张图: patch 行间均值向量(质心) → 各行到质心的距离
    centroid = patch.mean(dim=1, keepdim=True)
    dist_to_centroid = (patch - centroid).abs().mean().item()
    print(f"\n[1] patch 目标方差: 全局 std={global_std:.6f}, "
          f"每图跨位置 std={between_patch_std:.6f}, "
          f"行到质心平均 L1={dist_to_centroid:.6f}")
    # 关键对照: 与重建 L1 同量级吗?
    full_l1 = (F_hat - patch).abs().mean().item()
    print(f"    [关键] 重建 L1={full_l1:.6f} vs 行到质心距离={dist_to_centroid:.6f} "
          f"—— 若重建误差 ≈ 行间方差, 则模型在输出质心/常数, 未真还原")
    report["patch_global_std"] = global_std
    report["patch_between_pos_std"] = between_patch_std
    report["patch_dist_to_centroid"] = dist_to_centroid
    report["full_l1"] = full_l1

    # ══════════════════════════════════════════════════════════
    # 2. decoder 掩码生效性: 扰动 z_s 的某列, 看各采样步输出变化
    #    Y_t 应只依赖 z_s[0..t] (掩码正确) — 扰动 j>t 的列输出不变
    # ══════════════════════════════════════════════════════════
    with torch.no_grad():
        Y0 = Y.clone()   # (B,NT,N,D) 已从 last_Y 收集
        # 扰动一个"后面"的 z_s 列: j = 前 8 步之外的位置 (比如 64)
        j_far = 64 if num_patches > 64 else num_patches // 2
        z_s_pert = z_s.clone()
        z_s_pert[:, j_far] += torch.randn_like(z_s_pert[:, j_far]) * 0.5
        _ = model.decoder(z_cls, z_s_pert)
        Yp = model.decoder.last_Y                          # (B,NT,N,D)
        # 每步输出相对扰动列的变化 (应只在 t>=j_far 的步出现)
        delta = (Yp - Y0).abs().mean(dim=(0, 2, 3))        # (NT,)
        print(f"\n[2] decoder 掩码: 扰动 z_s[:, {j_far}] (幅度0.5)")
        for i, t in enumerate(T_steps):
            flag = "⚠️泄露!" if t < j_far and delta[i] > 1e-4 else "ok"
            print(f"    t={t:4d} (可见前缀≤{t:3d}) ΔY={delta[i].item():.2e} {flag}")
        report["mask_j_far"] = j_far
        report["mask_delta_per_step"] = [float(v) for v in delta.cpu().numpy()]

    # ══════════════════════════════════════════════════════════
    # 3. z_cls / z_s 信息量: 扰动 patch 特征, 看 z_cls 是否变化
    #    (ReEncoder cls 全局注意力 ⇒ z_cls 含全图信息; specials 见全部
    #     patches ⇒ z_s 也含全图信息 — 这是"前缀不渐进"的架构根源)
    # ══════════════════════════════════════════════════════════
    with torch.no_grad():
        feats0 = model.dinov2(x).last_hidden_state
        cls0, patch0 = feats0[:, 0:1], feats0[:, 1:]
        sp0 = model.special_bank(x.shape[0], x.device)
        z0 = model.re_encoder(torch.cat([cls0, sp0, patch0], dim=1))
        # 扰动单个 patch 列
        pj = 0
        patch_pert = patch0.clone()
        patch_pert[:, pj] += torch.randn_like(patch_pert[:, pj]) * 0.5
        z1 = model.re_encoder(torch.cat([cls0, sp0, patch_pert], dim=1))
        dz_cls = (z1[:, 0] - z0[:, 0]).abs().mean().item()
        dz_s_all = (z1[:, 1:1 + num_patches] - z0[:, 1:1 + num_patches]).abs().mean().item()
        dz_s_pj = (z1[:, 1 + pj] - z0[:, 1 + pj]).abs().mean().item()
        print(f"\n[3] ReEncoder 信息流: 扰动 patch[:, {pj}] (幅度0.5)")
        print(f"    Δz_cls={dz_cls:.2e} | Δz_s(全部)={dz_s_all:.2e} | "
              f"Δz_s[{pj}]={dz_s_pj:.2e}")
        print(f"    → z_cls 对任意 patch 扰动敏感 = cls 全局注意力聚合全图")
        print(f"    → z_s[{pj}] 敏感 = specials 行也看到全部 patches (设计如此)")
        report["dz_cls_per_patch_perturb"] = dz_cls
        report["dz_s_all"] = dz_s_all
        report["dz_s_at_perturbed"] = dz_s_pj

    # ══════════════════════════════════════════════════════════
    # 4. query_base 角色: t=0 (仅 1 键) 时 N 行输出是否依赖键?
    #    把 z_cls 换成随机向量再跑 t=0, 看输出变化幅度。
    #    · 几乎不变 ⇒ 输出由 query_base 模板主导, 键信息量小
    #    · 大变 ⇒ 键承载信息, t=0 的高精度来自 z_cls 聚合了全图
    # ══════════════════════════════════════════════════════════
    with torch.no_grad():
        z_cls_rand = torch.randn_like(z_cls) * 0.1
        z_s_rand = torch.randn_like(z_s) * 0.1
        # (a) 换随机 z_cls, 保持 z_s
        _ = model.decoder(z_cls_rand, z_s)
        Y_rand_cls = model.decoder.last_Y
        # (b) 全随机
        _ = model.decoder(z_cls_rand, z_s_rand)
        Y_rand_all = model.decoder.last_Y
        delta_cls = (Y_rand_cls - Y0).abs().mean(dim=(0, 2, 3))
        delta_all = (Y_rand_all - Y0).abs().mean(dim=(0, 2, 3))
        print(f"\n[4] query_base/键信息: 替换输入为随机 (幅度0.1)")
        for i, t in enumerate(T_steps[:6]):
            print(f"    t={t:4d}: 随机z_cls ΔY={delta_cls[i].item():.4f} | "
                  f"全随机 ΔY={delta_all[i].item():.4f}")
        print(f"    → 随机键后输出若几乎不变 ⇒ 输出≈query_base 模板函数, "
              f"键贡献小; 若大变 ⇒ 键(尤其 z_cls)承载还原信息")
        # t=0 步单独: 输出到底长什么样 — 与"质心"比对
        Y0_step0 = Y0[:, 0]                                   # (B,N,D)
        print(f"    t=0 输出: 行间 std={Y0_step0.std(dim=1).mean().item():.6f} "
              f"(目标 patch 行间 std={between_patch_std:.6f})")
        report["rand_cls_delta_step0"] = float(delta_cls[0])
        report["rand_all_delta_step0"] = float(delta_all[0])
        report["Y_step0_between_std"] = float(Y0_step0.std(dim=1).mean().item())

    with open(args.final_model.replace("final_model.pt", "leak_diag.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n[save] leak_diag.json")


if __name__ == "__main__":
    main()
