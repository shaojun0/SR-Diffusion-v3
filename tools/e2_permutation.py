"""E2 — 方案A 置换不变性实测（真实 DINOv2-large patch 特征）
================================================================

设计文档 §4: 置换不变性是**结构性保证**（GNN 节点级等变 + Readout 不变）,
§10 E2: "随机打乱 patch 顺序, 验证 z 不变而序列式基线漂移"。

本脚本给出**数字**:
    1. 用真实数据（construction_site test 图）→ DINOv2-large patch token
       特征 (B,576,1024) → 方案A 的 kNN 图;
    2. 置换设计:
       · split-half（主口径, 最苛刻）: 前半 288 个 patch 保持原位, 后半 288
         个随机打乱 —— 与"无置换"基准跑同一模块;
       · S 次独立全排列（汇总 mean/max）。
    3. 对照（顺序敏感基线, 同容量/同 Readout/同聚合方式, 只差"是否绑序号"）:
       · OrderSensitiveMLP(+PE)  —— **主对照**（唯一变量 = 可学习位置编码）
       · OrderSensitiveMLP(−PE)  —— 纯 DeepSets 版本（应不变; 用来证明
         "漂移来自 PE 而不是别的东西"）
       · SeqBaseline（Transformer + PE + sum 池化）
       · DINOv2 register 式路径的 z_s（把 K 个 register token 拼进 DINO 序列,
         取输出 register; 这是**现有管线**的顺序敏感来源实例）
       · DINOv2 patch 顺序 → z_cls（cls token 池化; 极稳健, 作"结构性对照"）

口径: 全部 fp32, 无 dropout（eval）, torch.no_grad; 报告
    max|Δ| / mean|Δ| / 相对差 = ‖Δ‖_F / ‖ref‖_F / cos / 单位球 L2 距离
    （Projector 输出 L2 归一化 ⇒ 下游是余弦空间, 单位球距离才是任务口径）。
判据: 不变性成立 ⇒ ≈ 0（浮点非结合性 1e-7 量级）; 顺序敏感基线 ⇒ 显著 > 0。

用法:
    python tools/e2_permutation.py \
        --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --n_images 8 --n_perm 8 --k 8 --num_proto 35 \
        --out /root/train_logs/e2_permutation_k8.json --device cuda:0
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_v2 import ParquetImageDataset, V2Collator          # noqa: E402
from model_gnn_schemeA import (OrderSensitiveMLP, SchemeAEncoder,  # noqa: E402
                               SeqBaseline, permute_adj)
from model_v2 import SpecialTokenBank                         # noqa: E402


def load_images(args):
    files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    assert files, f"无 test-*.parquet in {args.data_dir}"
    ds = ParquetImageDataset(files, limit=max(args.n_images * 3, 16))
    coll = V2Collator(model_size=(448, 252), canvas=(1600, 900))
    rows = [ds[i] for i in range(len(ds))]
    x = coll(rows)
    if isinstance(x, dict):
        x = x["pixel_values"]
    return x[:args.n_images]


def dino_forward(dino, seq):
    for layer in dino.encoder.layer:
        out = layer(seq)
        seq = out[0] if isinstance(out, (tuple, list)) else out
    return dino.layernorm(seq)


def dino_patch_feats(dino, x):
    """DINOv2-large 末层 patch token 特征 (B,N,D)。"""
    return dino_forward(dino, dino.embeddings(x))[:, 1:]


def split_half_perm(n, half, generator):
    """前半 half 个保持原位, 后半随机打乱; 返回 perm（新序第 i 位放原 perm[i]）。"""
    head = torch.arange(half)
    tail = torch.randperm(n - half, generator=generator) + half
    return torch.cat([head, tail])


def pair_stats(a, b):
    """对 (B,D) 输出: 绝对/相对差 + **余弦空间**差（下游只用余弦度量）。"""
    d = (a - b).abs()
    an, bn = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    cos = (an * bn).sum(-1)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rel_fro": float(d.norm() / b.norm().clamp_min(1e-12)),
            "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "l2_unit_max": float((an - bn).norm(dim=-1).max()),
            "ref_norm_mean": float(b.norm(dim=-1).mean())}


def multiset_stats(a, b):
    """两个集合（顺序无关）: 双向最近邻 max/mean（Hausdorff 风格）。"""
    D = torch.cdist(a, b)
    nn_ab, nn_ba = D.min(dim=1).values, D.min(dim=0).values
    return {"hausdorff_max": float(max(nn_ab.max(), nn_ba.max())),
            "nn_mean": float(torch.cat([nn_ab, nn_ba]).mean()),
            "rel_fro": float((a.norm() - b.norm()).abs() / b.norm().clamp_min(1e-12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dino_dir", required=True)
    ap.add_argument("--n_images", type=int, default=8)
    ap.add_argument("--n_perm", type=int, default=8, help="独立全排列次数 S")
    ap.add_argument("--half", type=int, default=288, help="split-half 保持原位的前缀长度")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--conv", default="gin")
    ap.add_argument("--num_proto", type=int, default=35)
    ap.add_argument("--arch_K", type=int, default=35, help="register 基线用的 K（= 管线 K）")
    ap.add_argument("--grid", default=None, help="'gh,gw' 网格邻接（并入图）")
    ap.add_argument("--grid_weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="e2_permutation_result.json")
    args = ap.parse_args()

    from transformers import Dinov2Model
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    dino = Dinov2Model.from_pretrained(args.dino_dir).to(dev).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    D = dino.config.hidden_size

    x = load_images(args).to(dev)
    with torch.no_grad():
        feats = dino_patch_feats(dino, x)           # (B,N,D)
    B, N, _ = feats.shape
    print(f"[e2] 图像 {B} 张, patch N={N}, D={D}; 特征 = DINOv2-large 末层 patch "
          f"token（fp32, 真实 {args.data_dir} 图像）")

    grid = tuple(int(v) for v in args.grid.split(",")) if args.grid else None

    def mk_gnn(readout):
        torch.manual_seed(args.seed)
        return SchemeAEncoder(in_dim=D, hid_dim=args.hid, out_dim=D,
                              num_layers=args.layers, readout=readout,
                              num_proto=args.num_proto, conv=args.conv,
                              k=args.k, grid=grid,
                              grid_weight=args.grid_weight).to(dev).eval()

    gnn_sum, gnn_proto = mk_gnn("sum"), mk_gnn("proto")
    torch.manual_seed(args.seed)
    os_pe = OrderSensitiveMLP(in_dim=D, hid_dim=args.hid, out_dim=D,
                              num_proto=args.num_proto, pos_mode="learned").to(dev).eval()
    torch.manual_seed(args.seed)
    os_nope = OrderSensitiveMLP(in_dim=D, hid_dim=args.hid, out_dim=D,
                                num_proto=args.num_proto, pos_mode="none").to(dev).eval()
    torch.manual_seed(args.seed)
    seqm = SeqBaseline(in_dim=D, hid_dim=args.hid, out_dim=D, num_layers=args.layers,
                       nhead=4, num_proto=args.num_proto).to(dev).eval()
    torch.manual_seed(args.seed)
    bank = SpecialTokenBank(num_tokens=args.arch_K, dim=D).to(dev).eval()
    # register 输出 → 定长向量的"读取器"（随机固定 MLP, 与方案A 同容量）:
    torch.manual_seed(args.seed)
    reg_proj = torch.nn.Sequential(torch.nn.Linear(D, args.hid), torch.nn.GELU(),
                                   torch.nn.Linear(args.hid, D)).to(dev).eval()

    def dino_register_zs(pixel_values, patch_perm=None):
        """现有 register 式路径: [cls; specials(K); patches] → z_s（K 个 register）。

        patch_perm 非 None 时按该顺序重排 patch 输入（模拟"patch 顺序被打乱"）,
        但 DINO 的位置编码仍绑序号 ⇒ 正是序列式路径的顺序依赖来源。
        """
        emb = dino.embeddings(pixel_values)                     # (B,1+N,D)
        pat = emb[:, 1:] if patch_perm is None else emb[:, 1:][:, patch_perm]
        sp = bank(pixel_values.shape[0], pixel_values.device)    # (B,K,D)
        seq = torch.cat([emb[:, :1], sp, pat], dim=1)
        seq = dino_forward(dino, seq)
        zs = seq[:, 1:1 + args.arch_K]                           # (B,K,D)
        return F.normalize(reg_proj(zs.mean(dim=1)), dim=-1)     # 读出成定长向量

    def inter_image_stats(z):
        """不同图之间的 z 余弦（正对照）: 若置换不变性来自"输出坍塌成常数",
        跨图余弦会 ≈1; 有真实区分度则 <1。"""
        zn = F.normalize(z.reshape(z.shape[0], -1), dim=-1)
        C = zn @ zn.t()
        off = C[~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)]
        return {"cross_image_cos_mean": float(off.mean()),
                "cross_image_cos_min": float(off.min())}

    res = {"meta": {"n_images": B, "N": N, "D": D, "n_perm": args.n_perm,
                    "half": args.half, "k": args.k, "conv": args.conv,
                    "layers": args.layers, "hid": args.hid,
                    "num_proto": args.num_proto, "arch_K": args.arch_K,
                    "grid": args.grid, "grid_weight": args.grid_weight,
                    "seed": args.seed, "data_dir": args.data_dir,
                    "dino_dir": args.dino_dir,
                    "note": "Projector 输出 L2 归一化; cos/l2_unit 是任务口径"}}

    with torch.no_grad():
        A = gnn_sum.build_graph(feats)              # (B,N,N)
        ref = {"schemeA_sum_z": gnn_sum(feats, A=A)["z"],
               "schemeA_h": gnn_sum(feats, A=A)["h"],
               "schemeA_proto": gnn_proto(feats, A=A)["proto"][0],
               "order_mlp_pe": os_pe(feats)["z"],
               "order_mlp_nope": os_nope(feats)["z"],
               "seq_transformer_pe": seqm(feats)["z"],
               "dino_registers_zs": dino_register_zs(x)}
        res["discriminability"] = {k: inter_image_stats(v) for k, v in ref.items()}

    def run_variant(perm):
        fp = feats if perm is None else feats[:, perm]
        Ap = A if perm is None else permute_adj(A, perm)
        with torch.no_grad():
            o1 = gnn_sum(fp, A=Ap)
            o2 = gnn_proto(fp, A=Ap)
            o3 = gnn_sum(fp)                        # 从置换后特征**重建** kNN 图
            o4 = gnn_proto(fp)
            o5 = os_pe(fp)
            o6 = os_nope(fp)
            o7 = seqm(fp)
            o8 = dino_register_zs(x, patch_perm=perm)
        return {
            "schemeA_sum_z_reuse_edges": pair_stats(o1["z"], ref["schemeA_sum_z"]),
            "schemeA_sum_z_rebuild_knn": pair_stats(o3["z"], ref["schemeA_sum_z"]),
            "schemeA_proto_multiset_reuse_edges": multiset_stats(
                o2["proto"][0], ref["schemeA_proto"]),
            "schemeA_proto_multiset_rebuild_knn": multiset_stats(
                o4["proto"][0], ref["schemeA_proto"]),
            "schemeA_h_multiset_rebuild_knn": multiset_stats(
                o3["h"][0], ref["schemeA_h"][0]),
            "order_mlp_pe_z": pair_stats(o5["z"], ref["order_mlp_pe"]),
            "order_mlp_nope_z": pair_stats(o6["z"], ref["order_mlp_nope"]),
            "seq_transformer_pe_z": pair_stats(o7["z"], ref["seq_transformer_pe"]),
            "dino_registers_zs": pair_stats(o8, ref["dino_registers_zs"]),
        }

    # ── (1) split-half: 前半不变, 后半打乱（长序列基线最敢漂的设计）──
    perm = split_half_perm(N, args.half, gen).to(dev)
    res["split_half"] = run_variant(perm)

    # ── (2) S 次独立全排列: 汇总 mean/max（越接近 0 越"结构不变"）──
    acc = {}
    for s in range(args.n_perm):
        pm = torch.randperm(N, generator=gen).to(dev)
        for key, st in run_variant(pm).items():
            acc.setdefault(key, []).append(st)
    summary = {}
    for key, rows in acc.items():
        summary[key] = {m: {"mean": float(np.mean([r[m] for r in rows])),
                            "max": float(np.max([r[m] for r in rows]))}
                        for m in rows[0]}
    res["full_permutation_S"] = {"S": args.n_perm, "stats": summary}

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n[e2] 结果已写 {args.out}")
    print("\nE2 DONE")


if __name__ == "__main__":
    main()
