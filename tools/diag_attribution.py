"""diag_attribution.py — checkpoint-2000 塌缩来源 2x2 归因。

把 "DINO 权重" 与 "special_bank(pos/token)" 两个来源各自换成初始化值, 交叉组合,
看 z_s 的跨 register 多样性 (z_s_within_std) 与解码器输出多样性分别由谁决定:

  (DINO_ckpt, sp_ckpt)  = 崩溃模型            -> 应复现 z_s_within_std≈0.02
  (DINO_init, sp_init)  = 纯初始化模型        -> 应 ≈0.095 (DINO 预训练本身多样性)
  (DINO_init, sp_ckpt)  = 只坏 special_bank   -> ?
  (DINO_ckpt, sp_init)  = 只坏 DINO           -> ?

另: 敏感性检查 —— 把 z_s 置零/打乱后解码器输出变化量 (解码器是否还在用 memory)。
"""
import argparse, copy, glob, json, os
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file as sf_load
from transformers import Dinov2Model
from data_v2 import ParquetImageDataset, V2Collator
from model_v2 import SRPhase1V2

W, H, N, DIM, K = 448, 252, 576, 1024, 35
STEPS = [1, 4, 9, 16, 25]


def build(dino, s, d, h, p, seed=42):
    torch.manual_seed(seed)
    return SRPhase1V2(dinov2=dino, num_patches=N, dim=DIM, heads=h, mlp_ratio=4.0,
                      decoder_steps=STEPS, decoder_depth=d, skip_steps=0,
                      max_steps=5, num_specials=K, query_mask_mode="blockdiag",
                      stack_dim=s, decoder_dropout=p)


def pstd(t):
    return float(t.std(dim=-2).mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino_dir", default="/root/autodl-tmp/models/dinov2-large")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/construction_site")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tf = sorted(glob.glob(os.path.join(a.data_dir, "train-*.parquet")))
    coll = V2Collator(model_size=(W, H), canvas=(1600, 900), angle_step=0.5)
    ds = ParquetImageDataset(tf, limit=16)
    x = next(iter(DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                             collate_fn=coll)))["pixel_values"].cuda()

    dino_pretrained = Dinov2Model.from_pretrained(a.dino_dir)
    if getattr(dino_pretrained.config, "use_mask_token", False):
        dino_pretrained.config.use_mask_token = False
        del dino_pretrained.embeddings.mask_token
    dino_pre_sd = {"dinov2." + k: v.clone()
                   for k, v in dino_pretrained.state_dict().items()}

    m_init = build(copy.deepcopy(dino_pretrained), 2048, 4, 16, 0.05, seed=42)
    m_ck = build(copy.deepcopy(dino_pretrained), 2048, 4, 16, 0.05, seed=42)
    sd = (sf_load(a.ckpt) if a.ckpt.endswith(".safetensors")
          else torch.load(a.ckpt, map_location="cpu"))
    m_ck.load_state_dict(sd, strict=True)

    init_sd = m_init.state_dict()
    sp_keys = [k for k in sd if k.startswith("special_bank.")]
    dino_keys = [k for k in sd if k.startswith("dinov2.")]
    print(f"[cfg] special_bank keys={sp_keys} n_dino={len(dino_keys)}", flush=True)

    def make(dino_src, sp_src):
        m = build(copy.deepcopy(dino_pretrained), 2048, 4, 16, 0.05, seed=42)
        tgt = m.state_dict()
        for k in sd:
            if k.startswith("dinov2."):
                tgt[k] = (sd[k] if dino_src == "ck" else dino_pre_sd[k]).clone()
            elif k.startswith("special_bank."):
                tgt[k] = (sd[k] if sp_src == "ck" else init_sd[k]).clone()
            else:
                tgt[k] = sd[k].clone()
        m.load_state_dict(tgt, strict=True)
        return m.cuda().eval()

    res = {}
    with torch.no_grad():
        for tag, ds_, ss_ in (("DINO_ck+sp_ck", "ck", "ck"),
                              ("DINO_init+sp_init", "init", "init"),
                              ("DINO_init+sp_ck", "init", "ck"),
                              ("DINO_ck+sp_init", "ck", "init")):
            m = make(ds_, ss_)
            zc, zs = m.encode(x)
            Y = m.decoder(zc, zs)
            out = m(x)
            # 解码器 memory 敏感性
            zs0 = torch.zeros_like(zs)
            Y0 = m.decoder(zc, zs0)
            zsr = zs[:, torch.randperm(zs.shape[1])]
            Yr = m.decoder(zc, zsr)
            res[tag] = {
                "z_s_within_std": float(zs.std(dim=1).mean()),
                "z_s_feat_std": float(zs.std(dim=2).mean()),
                "z_s_norm": float(zs.norm(dim=-1).mean()),
                "Y_across_patch_std": pstd(Y),
                "Y_step_std_mean": float(Y.std(dim=1).mean()),
                "F_hat_across_patch_std": pstd(out["F_hat"]),
                "loss": float(out["loss"]),
                "dY_when_zs_zeroed": float((Y - Y0).abs().mean()),
                "dY_when_zs_shuffled": float((Y - Yr).abs().mean()),
                "Y_abs_mean": float(Y.abs().mean()),
            }
            print(f"[{tag}] " + json.dumps(res[tag], indent=1), flush=True)
            del m
            torch.cuda.empty_cache()

        # special_bank.pos 跨 register 多样性 init vs ckpt
        res["special_bank_pos"] = {
            "init_within_std": float(init_sd["special_bank.pos"].std(dim=1).mean()),
            "ckpt_within_std": float(sd["special_bank.pos"].std(dim=1).mean()),
            "init_norm": float(init_sd["special_bank.pos"].norm()),
            "ckpt_norm": float(sd["special_bank.pos"].norm()),
            "token_init_norm": float(init_sd["special_bank.token"].norm()),
            "token_ckpt_norm": float(sd["special_bank.token"].norm()),
        }
        print("[special_bank.pos]", json.dumps(res["special_bank_pos"]), flush=True)

        # DINO 权重变化幅度
        tot_n = tot_d = 0.0
        for k in dino_keys:
            b = dino_pre_sd[k].float()
            tot_n += float(b.norm()) ** 2
            tot_d += float((sd[k].float() - b).norm()) ** 2
        res["dino_rel_change"] = (tot_d ** 0.5) / max(tot_n ** 0.5, 1e-12)
        print("[dino_rel_change]", res["dino_rel_change"], flush=True)

    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print("[out]", a.out)


if __name__ == "__main__":
    main()
