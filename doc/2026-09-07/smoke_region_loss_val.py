"""region_loss 数值冒烟（Qwen3.8-max, 2026-09-07, 配套 DESIGN_v2_region_loss.md §8.2）

目的: 在服务器 GPU + 真实 Dinov2Model 上小规模验证分区掩码损失实现
（不启动任何训练/长任务; 随机固定输入, 无外部数据依赖）:
  A. region_loss=True: loss == 手工按区域掩码计算的均值（逐位）; backward 通;
     decoder.last_Y 在**非本步区域行**的梯度精确为 0（PixelHead 逐行 + carry
     detach ⇒ 结构性质）, 本步区域内非零;
  B. region_loss=False: loss == 旧"每步整图"公式的全图均值（逐位复现旧行为）;
     backward 通;
  C. 两口径同权重同输入下 F_hat / Y_pix / recon 逐位不变（开关只动 loss）。

运行（服务器, 临时目录内, 拷入改后的 model_v2.py）:
    cd /root/qwen_region_loss_val
    CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python smoke_region_loss_val.py
DINO 权重: /root/autodl-tmp/models/dinov2-large（与训练同源）。
配置 = slice27_v2 协议: 输入 448x252 → N=576, slice[2:7] → steps=[9,16,25,
36,49], K 自动推导=63, T=5, regions=[(0,115),(115,230),(230,345),(345,460),
(460,576)]。B=2, fp32。
"""
import torch
import torch.nn.functional as F
from transformers import Dinov2Model

from model_v2 import SRPhase1V2, region_slices

torch.manual_seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
DINO_DIR = "/root/autodl-tmp/models/dinov2-large"

dino = Dinov2Model.from_pretrained(DINO_DIR)
if getattr(dino.config, "use_mask_token", False):
    dino.config.use_mask_token = False
    del dino.embeddings.mask_token

N = 576                       # 448x252 → (252/14)*(448/14) = 18*32
model = SRPhase1V2(dinov2=dino, num_patches=N, dim=dino.config.hidden_size,
                   skip_steps=2, max_steps=7).to(dev)   # slice [2:7]
assert model.decoder.steps == [9, 16, 25, 36, 49], model.decoder.steps
assert model.num_specials == 63, model.num_specials     # K 自动推导
assert model.region_loss is True                        # 新默认
T = len(model.decoder.steps)
regs = region_slices(N, T)
assert regs == [(0, 115), (115, 230), (230, 345), (345, 460), (460, 576)], regs
print(f"[cfg] device={dev} N={N} T={T} K={model.num_specials} "
      f"steps={model.decoder.steps} regions={regs}", flush=True)

x = torch.randn(2, 3, 252, 448, device=dev)             # 归一化空间随机输入

# ── A. region_loss=True（新默认, 分区掩码）──
out = model(x)
Y_pix, tgt = out["Y_pix"], out["target_pix"]
manual = torch.stack([
    F.l1_loss(Y_pix[:, t, lo:hi], tgt[:, lo:hi], reduction="none").mean()
    for t, (lo, hi) in enumerate(regs)])                # 手工按区域掩码
print(f"[A] loss(region)={out['loss'].item():.8f}", flush=True)
print(f"[A] 手工 per_step={[round(v, 8) for v in manual.tolist()]}", flush=True)
print(f"[A] 手工 mean   ={manual.mean().item():.8f}", flush=True)
assert torch.equal(out["loss"], manual.mean()), "region 损失应逐位==手工掩码均值"
print("[A][ok] loss == 手工按区域掩码的均值（逐位, torch.equal）", flush=True)

# 梯度局部性: last_Y (B,T,N,D), 第 t 步损失只碰 region t 的行
gY = torch.autograd.grad(out["loss"], model.decoder.last_Y,
                         retain_graph=True)[0]
for t, (lo, hi) in enumerate(regs):
    outside = torch.cat([gY[:, t, :lo], gY[:, t, hi:]], dim=1)
    inside = gY[:, t, lo:hi]
    mo, mi = outside.abs().max().item(), inside.abs().max().item()
    assert mo == 0.0, f"step {t} 非区域行梯度应精确为 0, got {mo}"
    assert mi > 0, f"step {t} 区域行梯度应非零"
    print(f"[A] step {t}: region[{lo},{hi}) |grad|max={mi:.3e} | "
          f"非区域行 |grad|max={mo:.1e}", flush=True)
print("[A][ok] decoder.last_Y 非本步区域行梯度精确为 0（结构性质）", flush=True)

out["loss"].backward()
for name, p in [("decoder.query_base", model.decoder.query_base),
                ("special_bank.pos", model.special_bank.pos),
                ("pixel_head.net[0].weight", model.pixel_head.net[0].weight),
                # transformers 5.x: patch_embeddings 是模块(内含 Conv2d), 取其首参数
                ("dinov2.embeddings(首参数)",
                 next(model.dinov2.embeddings.parameters()))]:
    g = p.grad
    assert g is not None and g.abs().sum().item() > 0, f"{name} 无梯度"
print("[A][ok] backward 通: query_base/special_bank.pos/pixel_head/DINO 嵌入"
      " 梯度非零", flush=True)

# ── B. region_loss=False（对照, 旧"每步整图"）──
model.zero_grad(set_to_none=True)
model.region_loss = False
with torch.no_grad():
    out_b = model(x)
per_full = F.l1_loss(out_b["Y_pix"],
                     out_b["target_pix"].unsqueeze(1).expand_as(out_b["Y_pix"]),
                     reduction="none").mean(dim=(0, 2, 3))     # 旧公式
print(f"[B] loss(full)={out_b['loss'].item():.8f}", flush=True)
print(f"[B] 旧公式 per_step={[round(v, 8) for v in per_full.tolist()]}", flush=True)
assert torch.equal(out_b["loss"], per_full.mean()), \
    "region_loss=False 应逐位==旧全图平权均值"
print("[B][ok] loss == 旧'每步整图'全图均值（逐位, torch.equal）", flush=True)
out_b2 = model(x)
out_b2["loss"].backward()
assert model.decoder.query_base.grad.abs().sum().item() > 0
print("[B][ok] region_loss=False 下 backward 通", flush=True)

# ── C. 开关不改前向数值 ──
model.region_loss = True
with torch.no_grad():
    out_c = model(x)
assert torch.equal(out_c["F_hat"], out_b["F_hat"]), "F_hat 应逐位不变"
assert torch.equal(out_c["Y_pix"], out_b["Y_pix"]), "Y_pix 应逐位不变"
assert torch.equal(out_c["recon"], out_b["recon"]), "recon(全图口径)应逐位不变"
print(f"[C][ok] 两口径 F_hat/Y_pix/recon 逐位不变; recon(全图)="
      f"{out_c['recon'].item():.8f}", flush=True)

if dev == "cuda":
    print(f"[mem] GPU 峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
          flush=True)
print("\nSMOKE ALL PASSED (region_loss=True/False 双口径)", flush=True)
