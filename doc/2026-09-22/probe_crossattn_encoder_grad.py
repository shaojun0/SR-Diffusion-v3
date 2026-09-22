"""探针：解码器 → 编码器 的学习信号到底有多强？（BPTT vs detach）

问题（用户 2026-09-22 提问）：
    编码器的输出 z_s 只能经**交叉注意力**进解码器
    （`OutputQueryDecoder.forward`: `A_in[:, lo:hi+1]` 就是 memory）。
    BPTT 打开后，编码器收到的信号会不会太弱？

本探针**直接用仓库自己的类**（model_v2.SRPhase1V2 / OutputQueryDecoder /
PixelHead + 真实分块读窗口 + 真实 carry + 真实"直接预测"损失），只把
DINOv2 换成一个结构对齐的假编码器（CPU 无 GPU / 无 dinov2-large 权重）。
所有口径与 model_v2.py 逐位一致。

测什么（每个臂 = detach / BPTT）：
  ① 逐 step 的 ‖∂L_t/∂z_s‖、‖∂L_t/∂z_cls‖、‖∂L_t/∂query_base‖
     → 编码器 vs 解码器侧各自的直接学习信号
  ② 逐 step 的编码器参数梯度范数 / 解码器参数梯度范数（per-param RMS）
     → 编码器拿到的更新份额
  ③ 累加相干性 ‖Σ_t ∂L_t/∂z_s‖ / Σ_t‖∂L_t/∂z_s‖ 与两两 cos
     → 各 step 经交叉注意力送回来的信号是**叠加**还是**互相抵消**
     （这才是"太弱"最可能的机制：路径变多 ≠ 净信号变强）
  ④ 1/|T| 稀释：单步损失 vs mean_t 损失下 ‖∂L/∂z_s‖
  ⑤ 训练轨迹（同一 warm start，只差 carry_detach 一个开关）

用法:
    python doc/2026-09-22/probe_crossattn_encoder_grad.py            # N=64, 8 步
    python doc/2026-09-22/probe_crossattn_encoder_grad.py --quick    # 冒烟
    python doc/2026-09-22/probe_crossattn_encoder_grad.py --steps 12 # 更接近真实步数比
结果写 doc/2026-09-22/data/probe_crossattn_encoder_grad.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from model_v2 import SRPhase1V2, square_block_starts  # noqa: E402  仓库真实类

SEED = 42


# ───────────────────────── 假 DINO（结构对齐 HF Dinov2Model）─────────────────────────
class _FakeLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        att = nn.Module(); att.attention = nn.Module()
        att.attention.query = nn.Linear(dim, dim)
        att.attention.key = nn.Linear(dim, dim)
        att.attention.value = nn.Linear(dim, dim)
        att.output = nn.Module(); att.output.dense = nn.Linear(dim, dim)
        self.attention = att
        mlp = nn.Module()
        mlp.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        mlp.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.mlp = mlp

    def forward(self, h, head_mask=None, output_attentions=False):
        n = self.norm1(h)
        q, k, v = (self.attention.attention.query(n),
                   self.attention.attention.key(n),
                   self.attention.attention.value(n))
        a = F.softmax(q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        h = h + self.attention.output.dense(a @ v)
        h = h + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(h))))
        return (h, None)


class _FakeEmbed(nn.Module):
    def __init__(self, dim, num_patches, patch=14):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.patch_embeddings = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.dropout = nn.Identity()

    def forward(self, pixel_values, bool_masked_pos=None, interpolate_pos_encoding=None):
        B = pixel_values.shape[0]
        p = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
        emb = torch.cat([self.cls_token.expand(B, -1, -1), p], dim=1)
        pe = self.position_embeddings
        if emb.shape[1] != pe.shape[1]:
            pe = F.interpolate(pe.transpose(1, 2).unsqueeze(0), size=(emb.shape[1],),
                               mode="linear", align_corners=False).squeeze(0).transpose(1, 2)
        return self.dropout(emb + pe)


class FakeDino(nn.Module):
    def __init__(self, dim, n_layers, num_patches, patch=14):
        super().__init__()
        self.embeddings = _FakeEmbed(dim, num_patches, patch)
        self.encoder = nn.Module(); self.encoder.layer = nn.ModuleList(
            [_FakeLayer(dim) for _ in range(n_layers)])
        self.layernorm = nn.LayerNorm(dim)


# ───────────────────────── 数据：低维流形合成图（可学，非平凡解）─────────────────────────
def make_data(n, grid, patch, seed):
    g = torch.Generator().manual_seed(seed)
    img = grid * patch
    ax = torch.linspace(0, 1, img)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    out = []
    for _ in range(n):
        amp = torch.rand(3, generator=g)
        ph = torch.rand(6, generator=g) * 2 * math.pi
        acc = torch.zeros(3, img, img)
        for ch in range(3):
            for f in range(3):
                k = f + 1
                acc[ch] += amp[f] * torch.sin(2 * math.pi * k * xx + ph[f]) \
                    * torch.cos(2 * math.pi * k * yy + ph[f + 3])
        out.append(acc)
    x = torch.stack(out)
    return (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-6)


# ───────────────────────── 测量 ─────────────────────────
def param_groups(model):
    return {
        "enc": list(model.dinov2.parameters()) + list(model.special_bank.parameters()),
        "dec": list(model.decoder.parameters()),
        "head": list(model.pixel_head.parameters()),
    }


def grad_stats(groups):
    """每组参数的 ① per-param 梯度 RMS（抗参数量差异，跨组可比）② 总范数。"""
    out = {}
    for name, ps in groups.items():
        sq = num = 0.0
        for p in ps:
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum())
                num += p.grad.numel()
        out[name] = math.sqrt(sq / num) if num else 0.0
        out[name + "_norm"] = math.sqrt(sq) if num else 0.0
    return out


def fwd(model, x):
    z_cls, z_s = model.encode(x)
    z_cls.retain_grad(); z_s.retain_grad()
    out = model.decode(z_cls, z_s, x)
    Ypix, tgt = out["Y_pix"], out["target_pix"]          # (B,|T|,N,588)
    per_step = F.l1_loss(Ypix, tgt.unsqueeze(1).expand_as(Ypix),
                         reduction="none").mean(dim=(0, 2, 3))   # (|T|,)
    return z_cls, z_s, per_step


def measure(model, x, tag):
    """逐 step 反传 + 总损失反传，记录编码器/解码器两侧的信号。"""
    groups = param_groups(model)
    T = len(model.decoder.steps)
    per = []
    g_zs = []
    for i in range(T):
        model.zero_grad(set_to_none=True)
        z_cls, z_s, per_step = fwd(model, x)
        per_step[i].backward(retain_graph=False)
        r = grad_stats(groups)
        r.update({"step_idx": i, "t": model.decoder.steps[i],
                  "L_i": float(per_step[i].detach()),
                  "grad_zs": float(z_s.grad.norm()),
                  "grad_zcls": float(z_cls.grad.norm()),
                  "grad_qbase": float(model.decoder.query_base.grad.norm()),
                  "enc_over_dec": r["enc"] / (r["dec"] + 1e-30)})
        per.append(r)
        g_zs.append(z_s.grad.detach().reshape(-1).clone())

    # 总损失（= 训练口径 mean_t）
    model.zero_grad(set_to_none=True)
    z_cls, z_s, per_step = fwd(model, x)
    loss = per_step.mean()
    loss.backward()
    tot = grad_stats(groups)
    tot.update({"loss": float(loss.detach()),
                "grad_zs": float(z_s.grad.norm()),
                "grad_zcls": float(z_cls.grad.norm()),
                "grad_qbase": float(model.decoder.query_base.grad.norm()),
                "enc_over_dec": tot["enc"] / (tot["dec"] + 1e-30)})

    # ① 稀释：单步损失（首步 / 末步）vs mean_t 总损失
    singles = {}
    for label, i in (("step0_only", 0), ("last_step_only", T - 1)):
        model.zero_grad(set_to_none=True)
        z_cls, z_s, per_step = fwd(model, x)
        per_step[i].backward()
        st = grad_stats(groups)
        st.update({"grad_zs": float(z_s.grad.norm()),
                   "grad_zcls": float(z_cls.grad.norm()),
                   "grad_qbase": float(model.decoder.query_base.grad.norm())})
        singles[label] = st

    # ③ 相干性
    G = torch.stack(g_zs)                                  # (T, ·)
    nrm_sum = float(G.norm(dim=1).sum())
    coh = float(G.sum(dim=0).norm()) / (nrm_sum + 1e-30)
    Gn = G / (G.norm(dim=1, keepdim=True) + 1e-30)
    C = (Gn @ Gn.t())
    off = C[~torch.eye(T, dtype=torch.bool)]
    return {"tag": tag, "T": T, "steps": list(model.decoder.steps),
            "per_step": per, "total": tot, "single_step": singles,
            "coherence": {"norm_of_sum_over_sum_of_norms": coh,
                          "pairwise_cos_mean": float(off.mean()),
                          "pairwise_cos_min": float(off.min()),
                          "pairwise_cos_max": float(off.max())}}


def train(model, tr, ev, n, lr, seed, batch=16):
    g = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    t0 = time.time()
    for _ in range(n):
        idx = torch.randint(0, tr.shape[0], (batch,), generator=g)
        loss = model(tr[idx])["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        o = model(ev)
        curve = o["Y_pix"]
        tgt = o["target_pix"]
        per = F.l1_loss(curve, tgt.unsqueeze(1).expand_as(curve),
                        reduction="none").mean(dim=(0, 2, 3)).tolist()
    return per, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=8, help="网格边长 → N=grid²")
    ap.add_argument("--patch", type=int, default=14, help="patch 像素边长")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--enc-layers", type=int, default=6)
    ap.add_argument("--dec-depth", type=int, default=2)
    ap.add_argument("--warm", type=int, default=500, help="A 相小样本过拟合步数")
    ap.add_argument("--fit-n", type=int, default=16, help="过拟合的小样本张数")
    ap.add_argument("--fit-batch", type=int, default=8)
    ap.add_argument("--phaseb", type=int, default=300, help="B 相步数")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(HERE / "data" / "probe_crossattn_encoder_grad.json"))
    args = ap.parse_args()

    if args.quick:
        args.warm, args.phaseb = 40, 40

    torch.set_num_threads(min(8, torch.get_num_threads() or 8))
    N = args.grid ** 2
    steps = square_block_starts(N)
    D, EP = args.dim, args.patch
    print(f"[cfg] N=K={N}, steps={steps} (|T|={len(steps)}), D={D}, heads={args.heads}"
          f"(head_dim={D // args.heads}), enc/dec={args.enc_layers}/{args.dec_depth} 层, "
          f"warm={args.warm}, phaseB={args.phaseb}")

    tr = make_data(512, args.grid, EP, 1)
    te = make_data(128, args.grid, EP, 2)
    # A 相 = **小样本过拟合**（关键）: 目的不是泛化, 而是把解码器推到一个
    # **真正依赖 memory** 的非退化操作点。query_base 是全局共享参数, 不能
    # 逐图记忆 ⇒ 要压低这 16 张的损失, 图像信息**必须**经 z_s/cls 的交叉
    # 注意力进来。这一步保证下面测到的 ∂L/∂z_s 不是"解码器忽略 memory"的
    # 退化零点（首版用大训练集 warm start 300 步停在平凡解 0.81, 已弃用）。
    fit = tr[:args.fit_n]

    torch.manual_seed(SEED)
    dino = FakeDino(D, args.enc_layers, N, EP)
    model = SRPhase1V2(dino, num_patches=N, dim=D, heads=args.heads,
                       decoder_steps=steps, decoder_depth=args.dec_depth,
                       mlp_ratio=2.0, patch_px=3 * EP * EP)
    model.decoder.carry_detach = True                     # A 相 = 单步口径
    model.decoder.steps = [steps[-1]]                     # 单步 + 全读
    optA = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    t0 = time.time()
    for it in range(args.warm):
        idx = torch.randint(0, fit.shape[0], (args.fit_batch,),
                            generator=torch.Generator().manual_seed(it))
        loss = model(fit[idx])["loss"]
        optA.zero_grad(set_to_none=True)
        loss.backward()
        optA.step()
    with torch.no_grad():
        fit_l1 = model(fit)["loss"].item()
        te_l1 = model(te)["loss"].item()
    trivial = te.abs().mean().item()
    print(f"[A] 小样本过拟合 {args.warm} 步 ({time.time() - t0:.0f}s): "
          f"fit({args.fit_n} 张) L1={fit_l1:.4f} | test L1={te_l1:.4f}"
          f"（平凡解≈{trivial:.4f}）", flush=True)
    warm_state = {k: v.clone() for k, v in model.state_dict().items()}

    x = fit[:8]                                           # 测量点 = 过拟合过的图
    out = {"config": {"N": N, "K": N, "steps": steps, "dim": D, "heads": args.heads,
                      "patch": EP, "head_dim": D // args.heads,
                      "enc_layers": args.enc_layers, "dec_depth": args.dec_depth,
                      "warm": args.warm, "fit_n": args.fit_n, "fit_batch": args.fit_batch,
                      "fit_l1": fit_l1, "test_l1_at_warm": te_l1, "trivial_l1": trivial,
                      "phaseb": args.phaseb, "batch_for_measure": int(x.shape[0])},
           "arms": []}

    for detach in (True, False):
        torch.manual_seed(SEED)
        dino = FakeDino(D, args.enc_layers, N, EP)
        m = SRPhase1V2(dino, num_patches=N, dim=D, heads=args.heads,
                       decoder_steps=steps, decoder_depth=args.dec_depth, mlp_ratio=2.0,
                       patch_px=3 * EP * EP)
        m.load_state_dict(warm_state)
        m.decoder.carry_detach = detach
        tag = "detach" if detach else "BPTT"

        t1 = time.time()
        rec = measure(m, x, f"{tag}@fit")
        print(f"[{tag}] 测量完成 ({time.time() - t1:.0f}s)", flush=True)

        torch.manual_seed(SEED)
        dino = FakeDino(D, args.enc_layers, N, EP)
        m2 = SRPhase1V2(dino, num_patches=N, dim=D, heads=args.heads,
                        decoder_steps=steps, decoder_depth=args.dec_depth, mlp_ratio=2.0,
                        patch_px=3 * EP * EP)
        m2.load_state_dict(warm_state)
        m2.decoder.carry_detach = detach
        curve, secs = train(m2, fit, fit, args.phaseb, 5e-4, SEED,
                            batch=args.fit_batch)
        rec["train_curve"] = curve
        rec["train_secs"] = round(secs, 1)
        rec["traj_gain_pct"] = 100 * (curve[0] - min(curve)) / curve[0]
        rec["last"] = curve[-1]
        print(f"[{tag}] B 相 {args.phaseb} 步: " +
              " ".join(f"{c:.4f}" for c in curve) +
              f"  traj_gain={rec['traj_gain_pct']:.1f}%  ({secs:.0f}s)", flush=True)
        out["arms"].append(rec)

    base = {a["tag"].split("@")[0]: a["last"] for a in out["arms"]}
    for a in out["arms"]:
        k = a["tag"].split("@")[0]
        other = "detach" if k == "BPTT" else "BPTT"
        a["last_vs_other_pct"] = 100 * (base[other] - a["last"]) / base[other]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"-> 写入 {args.out}")


if __name__ == "__main__":
    main()
