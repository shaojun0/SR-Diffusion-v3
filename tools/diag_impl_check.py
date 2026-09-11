"""diag_impl_check.py — stack 加宽 (OutputQueryDecoder.stack_dim=2D) 的实现正确性
静态 + 数值双重核对。

检查项:
  [1] stack_dim=0 与 stack_dim=dim 的等价性（Identity 分支不引入差异）
  [2] forward 语义核对: stack_in(Y)/stack_in(A) 各恰好 1 次, stack_out 1 次;
      手写参考实现与 OutputQueryDecoder.forward 逐位一致; 掩码形状与基线一致
  [3] 初始化输出统计: 同一 seed/同一 batch, 基线 vs 加宽 的
      z_s_within_std / Y 跨 patch std / Y_pix 跨 patch std / F_hat 跨 patch std / loss
  [4] 逐模块梯度范数: 同一 batch 前向+反向, 分组报告 grad norm（找"谁先死"）
  [5] checkpoint 审计: stack_in/stack_out 相对初始化的变化幅度、谱、以及
      各模块激活跨 patch std（init vs ckpt）定位多样性在哪一层丢失

用法:
  python diag_impl_check.py --dino_dir /root/autodl-tmp/models/dinov2-large \
     --data_dir /root/autodl-tmp/construction_site \
     --ckpt /root/autodl-tmp/.../checkpoint-2000/model.safetensors \
     --out /root/train_logs/stack2x_impl_check.json
"""
import argparse, copy, json, os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as sf_load
from transformers import Dinov2Model

from data_v2 import ParquetImageDataset, V2Collator
import model_v2 as M
from model_v2 import SRPhase1V2

W, H, N, PX = 448, 252, 576, 588
STEPS = [1, 4, 9, 16, 25]
K = 35
DIM = 1024


def build(dino, stack_dim, depth, heads, dropout, seed=42):
    torch.manual_seed(seed)
    return SRPhase1V2(dinov2=dino, num_patches=N, dim=DIM, heads=heads,
                      mlp_ratio=4.0, decoder_steps=STEPS, decoder_depth=depth,
                      skip_steps=0, max_steps=5, num_specials=K,
                      query_mask_mode="blockdiag", stack_dim=stack_dim,
                      decoder_dropout=dropout)


def pstd(t):
    """跨 patch(N) 维度的 std, 再对 batch/其它维取均值。t: (...,N,...)"""
    return float(t.std(dim=-2).mean().item()) if t.dim() >= 2 else float(t.std().item())


def grad_norm(p):
    return 0.0 if p.grad is None else float(p.grad.detach().norm().item())


def groups(model, name_prefix):
    g = {}
    d = model.dinov2
    g["dinov2.embeddings"] = [p for n, p in d.named_parameters()
                              if n.startswith("embeddings")]
    for i in (0, 11, 23):
        g[f"dinov2.layer{i}"] = [p for n, p in d.named_parameters()
                                 if n.startswith(f"encoder.layer.{i}.")]
    g["dinov2.layernorm"] = [p for n, p in d.named_parameters()
                             if n.startswith("layernorm")]
    for nm in ("special_bank", "pixel_head", "query_base", "pos_embed"):
        g[f"decoder.{nm}"] = [p for n, p in model.named_parameters()
                              if nm in n]
    for nm in ("stack_in", "stack_out"):
        g[f"decoder.{nm}"] = [p for n, p in model.named_parameters()
                              if f".{nm}." in n]
    nl = len(model.decoder.stack.layers)
    for i in range(nl):
        g[f"decoder.stack.L{i}"] = [p for n, p in model.named_parameters()
                                    if f"stack.layers.{i}." in n]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/root/autodl-tmp/construction_site")
    ap.add_argument("--dino_dir", default="/root/autodl-tmp/models/dinov2-large")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {}

    # ---- 数据: 固定 batch（train 前 16 条, shuffle=False） ----
    files = sorted(__import__("glob").glob(os.path.join(args.data_dir, "train-*.parquet")))
    ds = ParquetImageDataset(files, limit=16)
    coll = V2Collator(model_size=(W, H), canvas=(1600, 900), angle_step=0.5)
    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(ds, batch_size=16, shuffle=False,
                                 num_workers=2, collate_fn=coll)))
    x = batch["pixel_values"].cuda()
    print(f"[data] batch {tuple(x.shape)}", flush=True)

    # ---- [1] Identity 等价性 ----
    dino0 = Dinov2Model.from_pretrained(args.dino_dir)
    m0 = build(copy.deepcopy(dino0), 0, 2, 8, 0.0, seed=42).cuda().eval()
    m0b = build(copy.deepcopy(dino0), DIM, 2, 8, 0.0, seed=42).cuda().eval()
    sd0, sd0b = m0.state_dict(), m0b.state_dict()
    same_keys = set(sd0) == set(sd0b)
    maxdiff = max((sd0[k] - sd0b[k]).abs().max().item() for k in sd0) if same_keys else None
    with torch.no_grad():
        o0, o0b = m0(x), m0b(x)
    eq = {k: float((o0[k] - o0b[k]).abs().max().item()) for k in ("loss", "recon", "F_hat")}
    res["identity_equiv"] = {"same_keys": same_keys,
                             "stack_dim0_vs_dim1024_max_wdiff": maxdiff,
                             "output_maxdiff": eq,
                             "note": "stack_dim=0 与 =dim 都应走 Identity, 逐位一致"}
    print(f"[1] identity equiv: keys={same_keys} wdiff={maxdiff} outdiff={eq}", flush=True)
    del m0b

    # ---- [2] forward 语义: 计数 + 手写参考 ----
    m2 = build(copy.deepcopy(dino0), 2048, 4, 16, 0.05, seed=42).cuda().eval()
    calls = {"stack_in": 0, "stack_out": 0}
    hs = [m2.decoder.stack_in.register_forward_hook(
        lambda *a, _n="stack_in": calls.__setitem__(_n, calls[_n] + 1))]
    hs.append(m2.decoder.stack_out.register_forward_hook(
        lambda *a, _n="stack_out": calls.__setitem__(_n, calls[_n] + 1)))
    with torch.no_grad():
        z_cls, z_s = m2.encode(x)
        Y_model = m2.decoder(z_cls, z_s)
        # 手写参考（严格按 docstring 语义: 同一 stack_in 投影 Y 与 A, stack 一次, stack_out 一次）
        A = torch.cat([z_cls, z_s], dim=1) + m2.decoder.pos_embed
        A_t = A[:, m2.decoder.steps]
        Yq = (A_t.unsqueeze(2) + m2.decoder.query_base).reshape(
            x.shape[0], len(m2.decoder.steps) * N, DIM)
        mask = M.build_block_mask(m2.decoder.num_specials, m2.decoder.steps,
                                  num_queries=N, device=A.device)
        tgt = M.build_causal_query_mask(len(m2.decoder.steps), N, device=A.device,
                                        mode=m2.decoder.query_mask_mode)
        Y_ref = m2.decoder.stack_out(m2.decoder.stack(
            m2.decoder.stack_in(Yq), m2.decoder.stack_in(A),
            memory_mask=mask, tgt_mask=tgt))
        Y_ref = Y_ref.reshape(x.shape[0], len(m2.decoder.steps), N, DIM)
    for h in hs:
        h.remove()
    res["forward_semantics"] = {
        "stack_in_calls_per_forward": calls["stack_in"],
        "stack_out_calls_per_forward": calls["stack_out"],
        "expected": {"stack_in": 2, "stack_out": 1},
        "ref_vs_model_maxdiff": float((Y_model - Y_ref).abs().max().item()),
        "memory_mask_shape": list(mask.shape),
        "tgt_mask_shape": list(tgt.shape),
        "tgt_mask_is_blockdiag": bool(tgt.abs().gt(1e8).any().item()),
        "Y_shape": list(Y_model.shape),
    }
    print(f"[2] forward calls={calls} refdiff="
          f"{res['forward_semantics']['ref_vs_model_maxdiff']:.3e} "
          f"mask={list(mask.shape)} tgt={list(tgt.shape)}", flush=True)

    # ---- [3] 初始化输出统计 ----
    with torch.no_grad():
        z_cls0, z_s0 = m0.encode(x)
        out0 = m0(x)
        Y0 = m0.decoder(z_cls0, z_s0)
        yp0 = m0.pixel_head(Y0)
        z_cls2, z_s2 = m2.encode(x)
        out2 = m2(x)
        Y2 = m2.decoder(z_cls2, z_s2)
        yp2 = m2.pixel_head(Y2)
    res["init_stats"] = {
        "baseline": dict(
            z_s_within_std=float(z_s0.std(dim=1).mean()),
            z_s_norm=float(z_s0.norm(dim=-1).mean()),
            Y_cross_patch_std=pstd(Y0),
            Y_pix_cross_patch_std=pstd(yp0),
            F_hat_cross_patch_std=pstd(out0["F_hat"]),
            Y_abs_mean=float(Y0.abs().mean()), Y_std=float(Y0.std()),
            loss=float(out0["loss"]), recon=float(out0["recon"])),
        "stack2x": dict(
            z_s_within_std=float(z_s2.std(dim=1).mean()),
            z_s_norm=float(z_s2.norm(dim=-1).mean()),
            Y_cross_patch_std=pstd(Y2),
            Y_pix_cross_patch_std=pstd(yp2),
            F_hat_cross_patch_std=pstd(out2["F_hat"]),
            Y_abs_mean=float(Y2.abs().mean()), Y_std=float(Y2.std()),
            loss=float(out2["loss"]), recon=float(out2["recon"])),
    }
    # z_cls/z_s 应逐位相同（DINO 权重相同、输入相同）
    res["init_stats"]["encode_identical"] = {
        "z_cls_maxdiff": float((z_cls0 - z_cls2).abs().max()),
        "z_s_maxdiff": float((z_s0 - z_s2).abs().max()),
    }
    print("[3] init stats:", json.dumps(res["init_stats"], indent=2,
                                        ensure_ascii=False), flush=True)

    # ---- [4] 逐模块 grad norm ----
    grad_res = {}
    for tag, mm in (("baseline", m0), ("stack2x", m2)):
        mm.zero_grad(set_to_none=True)
        mm.train()
        o = mm(x)
        o["loss"].backward()
        gr = groups(mm, tag)
        grad_res[tag] = {k: sum(grad_norm(p) ** 2 for p in v) ** 0.5
                         for k, v in gr.items()}
        grad_res[tag + "_loss"] = float(o["loss"])
        mm.zero_grad(set_to_none=True)
        mm.eval()
    res["grad_norms"] = grad_res
    print("[4] per-module grad norms:", json.dumps(grad_res, indent=2), flush=True)

    # ---- [5] checkpoint 审计 ----
    if args.ckpt:
        sd = (sf_load(args.ckpt) if args.ckpt.endswith(".safetensors")
              else torch.load(args.ckpt, map_location="cpu"))
        init_sd = build(copy.deepcopy(dino0), 2048, 4, 16, 0.05, seed=42).state_dict()
        ck = {}
        for k in ("decoder.stack_in.weight", "decoder.stack_in.bias",
                  "decoder.stack_out.weight", "decoder.stack_out.bias",
                  "decoder.query_base", "decoder.pos_embed",
                  "decoder.pixel_head.net.0.weight", "decoder.pixel_head.net.2.weight",
                  "special_bank.token", "special_bank.pos"):
            if k not in sd:
                continue
            a, b = sd[k].float(), init_sd[k].float()
            if b.dim() == 2:
                sv = torch.linalg.svdvals(b)
                sv_top5 = [float(v) for v in sv[:5]]
                sv_max, sv_min = float(sv.max()), float(sv.min())
            else:
                f = b.flatten()
                sv_top5 = [float(f.abs().max())]
                sv_max, sv_min = float(f.abs().max()), float(f.abs().min())
            ck[k] = {
                "init_norm": float(b.norm()), "ckpt_norm": float(a.norm()),
                "rel_change": float((a - b).norm() / b.norm().clamp_min(1e-12)),
                "cos_init_ckpt": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
                "ckpt_sv_top5": sv_top5,
                "ckpt_sv_max": sv_max, "ckpt_sv_min": sv_min,
            }
        # DINO 相对预训练的变化
        dinop = {"dinov2." + n: p for n, p in
                 Dinov2Model.from_pretrained(args.dino_dir).state_dict().items()}
        for nm, keys in (("dinov2.layer0", "dinov2.encoder.layer.0."),
                         ("dinov2.layer23", "dinov2.encoder.layer.23."),
                         ("dinov2.layernorm", "dinov2.layernorm.")):
            tot_n = tot_d = 0.0
            for k, v in dinop.items():
                if not k.startswith(keys):
                    continue
                if k in sd:
                    a, b = sd[k].float(), v.float()
                    tot_n += float(b.norm()) ** 2
                    tot_d += float((a - b).norm()) ** 2
            ck[nm + "_rel_change"] = (tot_d ** 0.5) / max(tot_n ** 0.5, 1e-12)
        res["ckpt_audit"] = ck
        print("[5] ckpt audit:", json.dumps(ck, indent=2), flush=True)

        # 激活跨 patch std: init vs ckpt, 逐层定位多样性丢失点
        try:
            m2.load_state_dict(sd, strict=True)
        except Exception as e:
            print("[5] load_state_dict failed:", e)
        m2.eval()

        def act_scan(model, xt):
            acts = {}

            def mk(nm):
                def h(mod, inp, out):
                    t = out if torch.is_tensor(out) else out[0]
                    acts[nm] = pstd(t) if t.dim() >= 2 else float(t.std())
                return h
            hooks = []
            hooks.append(model.special_bank.register_forward_hook(mk("special_bank")))
            hooks.append(model.decoder.stack_in.register_forward_hook(mk("stack_in")))
            for i, l in enumerate(model.decoder.stack.layers):
                hooks.append(l.register_forward_hook(mk(f"stack.L{i}")))
            hooks.append(model.decoder.stack_out.register_forward_hook(mk("stack_out")))
            hooks.append(model.pixel_head.register_forward_hook(mk("pixel_head")))
            with torch.no_grad():
                model(xt)
            for h in hooks:
                h.remove()
            return acts
        m2_init = build(copy.deepcopy(dino0), 2048, 4, 16, 0.05, seed=42).cuda()
        m2_init.eval()
        res["act_scan"] = {"init": act_scan(m2_init, x), "ckpt2000": act_scan(m2, x)}
        print("[5] act scan:", json.dumps(res["act_scan"], indent=2), flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("[out]", args.out)


if __name__ == "__main__":
    main()
