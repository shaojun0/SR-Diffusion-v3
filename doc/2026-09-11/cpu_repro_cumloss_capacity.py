"""
SR-Diffusion-v3 · CPU 可证伪实验：累加平权损失 vs 解码器容量
================================================================
日期: 2026-09-11   设备: CPU (torch 2.8.0+cpu)   训练: 无 GPU、无真实数据、无 checkpoint

要回答的问题
------------
"后几步不干活" 是因为
  (H1) 解码器容量没到临界值 / 参数被 step-1 独占   ← 用户的假设
  (H2) 累加平权损失把 "Y_1 独扛、后步≈0" 设成了精确最优解   ← 前一轮分析的结论

怎么把两者分开
--------------
真实数据的问题是无法知道"信息够不够"。合成数据可以:
  构造 z_s 的**递增信息结构**（块 k 只含 patch 块 k 的隐因子），并算出
  **oracle 下界** = 用已解锁块线性最优预测 target 的 L1。oracle 单调下降
  且每个块都有大幅边际收益 ⇒ "信息给足、线性可解" 是**构造保证**的。
  于是若训练出来的模型后步仍≈0，H1 就被排除。
  （注: 合成编码器故意是**递增**的，这对"后步要真干活"是**有利**先验——
   真实 DINO register 反而是塌缩的；所以这是对 H1 最宽容的测试。）

三个臂（除标注外，掩码/查询基/共享参数结构完全同构）
--------------------------------------------------
  A  cumloss          : 原版结构（cumsum 平权 L1，固定 query_base，depth=2）
  B  cumloss+capacity : depth=6 + 逐 patch 内容查询 query_base + Linear(z_s[k])
                        （给足容量与信息通路，损失一字不改）
  C  regionloss       : 损失换成分区域私有目标（每步只监督自己那 1/5 patch 区）
                        容量同 A

判据
----
  step_scale_ratio = mean|pixel(Y_t)| / mean|target|   （每步自身输出量级）
  oracle L1 per step  vs  模型 progressive L1 per step
  H2 成立 ⇒ A/B 的 ratio 单调塌到 <0.1，C 的 ratio 在 ~0.2-1.0；
            且 B 与 A 同样塌（容量不是杠杆）。
  H1 成立 ⇒ B 明显不塌，或 C 与 A 无差别。

用法: python cpu_repro_cumloss_capacity.py [--steps N]
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ── 合成数据的固定配置 ──────────────────────────────────────────────
N_PATCH = 32          # patch 数（= 输出行数 N）；须被 len(BLOCKS) 整除
P_FAC = 24            # 每组隐因子维数
P_PATCH = 32          # 每 patch 的"像素"维（远小于 D，制造近零空间）
D = 96                # 特征维
K = 25                 # num_specials（register 数）; 位置 0 = z_cls, 1..24 = z_s
BLOCKS = [2 * k + 1 for k in range(1, math.isqrt(K - 1) + 1)]   # 块大小 [3,5,7,9]
W_OWN = 0.9            # target 的"本组因子"分量权重
W_GLOB = 0.35          # target 的"全部因子均值"分量权重
N_SAMPLE = 131072
N_TEST = 16384
STEPS = [k * k for k in range(1, len(BLOCKS) + 1)]              # [1,4,9,16] = square_block_starts
assert sum(BLOCKS) == K - 1, f"块大小之和 {sum(BLOCKS)} 应 = K-1 = {K-1}"
assert [(2 * k + 1) for k in range(1, len(BLOCKS) + 1)] == BLOCKS, BLOCKS
_TGT_SCALE = [1.0]     # make_data 写入的 target 归一化尺度
VIS_MODE = ["block"]   # "block"（块内排他, 现状）| "prefix"（第一个步前缀规则推广到所有步）


def block_bounds(k):
    """块 k 的 register 区间 [lo, hi]（1-based 位置, 对应 z_s 下标）。"""
    lo = k * k
    hi = min((k + 1) * (k + 1) - 1, K)
    return lo, hi


def make_data(seed, n=None):
    """构造 z_s（递增信息结构）与 target（正交分量 + 全局分量）。

    与 build_block_mask 的可见性**对齐**的信息结构:
      · 4 个采样步 [1,4,9,16]，第 i 步只读到第 i 个块（块内排他）
      · patch 分 4 组，组 i 的 target 含一个**只由块 i 的因子决定**的分量
        (own) —— 这是该步独有的信息
      · 另有全局分量 glob，依赖**全部**块因子（最后一步才补得全）——
        使后步能**修正**前步对早期 patch 的预测
      · 块 i 的每个 register = proj(f_i) + 小噪声（n_i 次独立读数 ⇒ 后验
        方差 1/(1+n_i·snr)，信息随 i 递增）
    ⇒ 闭式 MMSE 严格单调递减，且每步的边际都是"新解锁那一组"的真实信息。
    这是对"后步要真干活"最宽容的构造（信息给足、线性可解、结构递增），
    比真实 DINO register 的塌缩状态宽松得多 —— 若此设定下后步仍塌，容量假设被排除。
    """
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g)

    m = len(BLOCKS)
    per = N_PATCH // m                                     # 每组 patch 数
    assert per * m == N_PATCH, f"N_PATCH={N_PATCH} 须被 m={m} 整除"
    assert (W_OWN ** 2) + (W_GLOB ** 2) / m > 0

    FAC = rnd(n, m, P_FAC)                                 # 逐样本隐因子 (n,m,P_FAC)
    Wp = rnd(P_FAC, D) * (1.0 / math.sqrt(P_FAC))          # 因子→特征投影
    proj = FAC @ Wp                                        # (n, m, D)

    zs = torch.zeros(n, K, D)
    for k, (lo, hi) in enumerate([block_bounds(k) for k in range(m)]):
        npos = hi + 1 - lo                                 # 块内 register 数
        zs[:, lo:hi + 1] = proj[:, k:k + 1].expand(n, npos, D) \
                           + rnd(n, npos, D) * 0.05         # 同一因子的 npos 次读数
    zs[:, 0] = proj[:, 0] + rnd(n, D) * 0.05

    # target: 本组因子分量（该步独有信息）+ 逐 patch 均值分量（随步单调减少）
    #   prefix 模式下全局分量改用"该步可见块的均值" ⇒ 每步新增的块确实带来新信息
    W_own = rnd(P_FAC, P_PATCH) * W_OWN
    W_glob = rnd(m * P_FAC, P_PATCH) * (W_GLOB / math.sqrt(m * P_FAC))
    own = torch.stack([FAC[:, k // per] for k in range(N_PATCH)], 1) @ W_own  # (n,N,P)
    if VIS_MODE[0] == "prefix":
        Wg = rnd(m, P_FAC, P_PATCH) * (W_GLOB / math.sqrt(P_FAC))
        glob = torch.zeros(n, P_PATCH)
        for i in range(m):
            glob = glob + FAC[:, i] @ Wg[i]
    else:
        glob = FAC.reshape(n, -1) @ W_glob
    tgt = own + glob.unsqueeze(1) + rnd(n, N_PATCH, P_PATCH) * 0.05
    tgt = tgt + 0.1 * F.relu(-tgt)                         # 轻非线性

    # zs: (N, K, D) 逐样本; tgt: (1, N_PATCH, P) 固定目标（与 batch 广播）
    # 说明: target 不随样本变化是本实验的刻意简化——每个样本的差异只来自 z_s，
    # 等价于"给每个样本一组不同的 register，要求解码出同一结构的目标"。
    _TGT_SCALE[0] = float(tgt.std())
    return zs, tgt / _TGT_SCALE[0]


def visible_cols(i, t=None):
    """第 i 个采样步可见的 z_s 列（不含位置 0），由 VIS_MODE 决定。

    · "block"  : 与 model_v2.build_block_mask 一致——第一个步读 0..hi，
                 其余步只读自己那块 [k², hi]（读窗口互不重叠）
    · "prefix" : 全程用第一个步的前缀规则——第 i 步读 0..hi(k_i)
                 （**复现真实 v2 的关键情形**：step-1 的读窗口已覆盖很大范围，
                  后续步只是往外扩一点）
    """
    t = STEPS[i] if t is None else t
    k = math.isqrt(int(t))
    hi = min((k + 1) * (k + 1) - 1, K - 1)
    if VIS_MODE[0] == "prefix":
        lo = 1
    else:
        lo = 1 if i == 0 else max(k * k, 1)
    return list(range(lo, hi + 1))


def _features(zs, steps):
    """每步可读特征（可见性规则见 visible_cols）。

    只取 z_s 列（位置 ≥1），**不加** step one-hot——one-hot 会给每步一个额外的
    自由截距（而各步输出本就带 bias），在本设定下它是纯噪声源（本实验踩过）。
    """
    return [zs[:, visible_cols(i), :].reshape(zs.shape[0], -1)
            for i in range(len(steps))]


@torch.no_grad()
def oracle_per_step(zs_tr, tgt_tr, zs_te, tgt_te, steps, lam=1e-4):
    """经验 oracle：训练集拟合、测试集评估的**线性最优** L1 下界。

    特征 = 每步可读的 z_s 列（可见性规则与 build_block_mask 一致），逐列标准化。
    用**最小范数最小二乘**（pinv）而非岭回归：块内各 register 是同一因子的
    近共线读数（标准化后 SNR≈400），岭回归的收缩偏差会把预测压回均值、R² 变负
    （本实验踩过这个坑），pinv 则给出无偏的最小范数解。
    `cond` 一并返回，作为"该步特征是否病态"的体检指标。
    """
    ftr = _features(zs_tr, steps)
    fte = _features(zs_te, steps)
    ytr = tgt_tr.reshape(tgt_tr.shape[0], -1)
    yte = tgt_te.reshape(tgt_te.shape[0], -1)
    out, conds = [], []
    for Xtr, Xte in zip(ftr, fte):
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        Xtr = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1)], 1)
        Xte = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], 1)
        XtX = Xtr.T @ Xtr
        conds.append(torch.linalg.cond(XtX).item())
        W = torch.linalg.pinv(XtX.double()).float() @ (Xtr.T @ ytr)
        out.append((Xte @ W - yte).abs().mean().item())
    return out, conds


class FakeDino(nn.Module):
    """z_s 已由 make_data 构造在 D 维 ⇒ 编码器侧为恒等（信息不给也不夺）。

    真实模型里这一跳是 DINOv2 24 层（P→register 路由），本实验把该跳固定为
    "无损" —— 即对 H1(容量/信息不足) 最宽容的设定: 信息一定在。
    """

    def forward(self, x):
        return x


class Decoder(nn.Module):
    """与 model_v2.OutputQueryDecoder 同构（memory_mask + tgt_mask + 累加语义）。

    差别仅在: cap=True 时加深度 + 逐 patch 内容查询（给足容量/信息通路）。
    """

    def __init__(self, dim, n_patch, K, steps, depth=2, cap=False, p_patch=32):
        super().__init__()
        self.n_patch, self.K, self.steps = n_patch, K, steps
        self.cap = cap
        self.query_base = nn.Parameter(torch.randn(n_patch, dim) * 0.02)
        if cap:
            # 逐 patch 内容查询句柄: z_s 逐位置（K 个）→ N 个查询行（广播到所有步）
            self.query_handle = nn.Module()
            self.query_handle.net = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            self.patch_emb = nn.Parameter(torch.randn(n_patch, dim) * 0.02)
        else:
            self.query_handle = None
            self.patch_emb = None
        self.pos_embed = nn.Parameter(torch.randn(1, K, dim) * 0.02)   # 位置 0..K-1
        self.stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, 4, dim * 4, dropout=0.0,
                                       activation="gelu", batch_first=True,
                                       norm_first=True),
            num_layers=depth)
        # 掩码：与 build_block_mask 完全一致的语义
        rows = []
        for i, t in enumerate(steps):
            row = torch.full((K,), float("-inf"))
            cols = visible_cols(i, t)
            if i == 0 or VIS_MODE[0] == "prefix":
                row[0] = 0.0                                # 第一个步前缀规则含 z_cls
            if cols:
                row[cols[0]:cols[-1] + 1] = 0.0
            rows.append(row)
        self.register_buffer("mem_mask", torch.stack(rows).repeat_interleave(n_patch, 0))
        L = len(steps) * n_patch
        tm = torch.full((L, L), float("-inf"))
        for i in range(len(steps)):
            tm[i * n_patch:(i + 1) * n_patch, i * n_patch:(i + 1) * n_patch] = 0.0
        self.register_buffer("tgt_mask", tm)            # blockdiag（当前默认）

    def forward(self, z_cls, z_s):
        B = z_s.shape[0]
        A = torch.cat([z_cls, z_s], 1) + self.pos_embed
        A_t = A[:, self.steps]                          # (B,|T|,D)
        if self.cap:
            # 逐 patch 内容句柄: 把 K 个 register 的全局摘要广播成 N 行（每行 = 
            # 自己的 patch 模板 + 全部 register 的池化摘要）——这是"信息通路给足"
            # 的对照：patch k 的查询不再只依赖跨图固定模板。
            pooled = self.query_handle.net(z_s.mean(1, keepdim=True))   # (B,1,D)
            handle = pooled + self.patch_emb.unsqueeze(0)               # (B,N,D)
            base = self.query_base.unsqueeze(0) + handle                # (B,N,D)
            Y = (A_t.unsqueeze(2) + base.unsqueeze(1)) \
                .reshape(B, len(self.steps) * self.n_patch, -1)
        else:
            Y = (A_t.unsqueeze(2) + self.query_base).reshape(B, len(self.steps) * self.n_patch, -1)
        Y = self.stack(Y, A, memory_mask=self.mem_mask, tgt_mask=self.tgt_mask)
        return Y.reshape(B, len(self.steps), self.n_patch, -1)


class Head(nn.Module):
    def __init__(self, dim, p_patch, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, p_patch))

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    """arm: 'cum' | 'region'"""

    def __init__(self, arm, cap=False, depth=2):
        super().__init__()
        self.arm = arm
        self.enc = FakeDino()
        self.dec = Decoder(D, N_PATCH, K, STEPS, depth=depth, cap=cap)
        self.head = Head(D, P_PATCH)
        # 分区域（region 臂用）：每个采样步负责一个连续 patch 区
        per = N_PATCH // len(STEPS)
        regions = []
        for i in range(len(STEPS)):
            lo = i * per
            hi = (i + 1) * per if i < len(STEPS) - 1 else N_PATCH
            regions.append(list(range(lo, hi)))
        self.regions = regions

    def forward(self, zs, tgt):
        z = self.enc(zs)                                # (B,S,D)
        Y = self.dec(z[:, :1], z[:, 1:])                # (B,|T|,N,D)
        Y_cum = torch.cat([torch.zeros_like(Y[:, :1]),
                           Y.cumsum(1)[:, :-1]], 1).detach() + Y
        Y_pix = self.head(Y_cum)                        # (B,|T|,N,P)
        if self.arm == "cum":
            per_step = F.l1_loss(Y_pix, tgt.unsqueeze(1),
                                 reduction="none").mean(dim=(0, 2, 3))
            loss = per_step.mean()
        else:                                            # region: 每步只监督自己的区
            losses = []
            for i, cols in enumerate(self.regions):
                losses.append(F.l1_loss(Y_pix[:, i, cols], tgt[:, cols]))
            loss = torch.stack(losses).mean()
        F_pix = Y_pix[:, -1]
        return {"loss": loss, "Y_pix": Y_pix, "Y": Y,
                "recon": F.l1_loss(F_pix, tgt), "target": tgt}


def run_arm(name, arm, cap, depth, zs_tr, tgt_tr, zs_te, tgt_te, steps_n):
    torch.manual_seed(42)
    m = Model(arm, cap=cap, depth=depth)
    n_par = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=steps_n, pct_start=0.1)
    B = 128
    t0 = time.time()
    m.train()
    for it in range(steps_n):
        idx = torch.randint(0, zs_tr.shape[0], (B,))
        out = m(zs_tr[idx], tgt_tr[idx])
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if (it + 1) % max(1, steps_n // 8) == 0 or it == 0:
            print(f"    [{name}] it {it+1:>5}/{steps_n} loss {out['loss'].item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    m.eval()
    with torch.no_grad():
        out = m(zs_te, tgt_te)
        Y_pix = out["Y_pix"]                            # (B,|T|,N,P)
        tgt = out["target"]
        scale = (Y_pix.abs().mean((0, 2, 3)) / tgt.abs().mean()).tolist()
        # 与 oracle 同口径: 逐样本的"第 i 步累积预测 vs 目标" L1
        prog = [(Y_pix[:, i] - tgt).abs().mean().item() for i in range(len(STEPS))]
        # 参照: "后步什么都不做" = 最后一步仍是 step-1 的预测。若实际 progressive
        # 比它更差, 说明后步不是在修正, 而是在**帮倒忙**（发出负贡献把结果推偏）
        rep1 = (Y_pix[:, 0] - tgt).abs().mean().item()
        # 近零空间探针: 每步特征范数 vs 像素范数
        feat_norm = out["Y"].norm(dim=-1).mean((0, 2)).tolist()
        pix_norm = [Y_pix[:, i].std().item() for i in range(len(STEPS))]
    return {"name": name, "params": n_par, "step_scale_ratio": scale,
            "prog_l1": prog, "final_l1": prog[-1], "step1_only_l1": rep1,
            "recon_l1": out["recon"].item(),
            "Y_feat_norm": feat_norm, "Ypix_std": pix_norm,
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--visible", default="block", choices=["block", "prefix"],
                    help="读窗口规则: block=块内排他（现状）; prefix=全程前缀规则")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    VIS_MODE[0] = args.visible

    print("=" * 78)
    print("SR-Diffusion-v3 · CPU 可证伪实验：累加平权损失 vs 解码器容量")
    print(f"设备: CPU | K={K} N={N_PATCH} D={D} steps={STEPS} | 训练步数 {args.steps} "
          f"| 读窗口={args.visible}")
    print("=" * 78)

    # 关键: 训练/测试必须是**同一个数据集的两半**。曾经踩过的坑——用
    # make_data(0) / make_data(1) 各建一份，两者连 Wp/W_own/W_glob 都不同
    # （即"两个不同的世界"），模型学不到任何可迁移的东西，评测端 L1 连平凡
    # 常数解都不如，所有对照结论都会被这个 bug 淹没。
    N_TR, N_TE = 16384, 4096
    zs_all, tgt_all = make_data(0, N_TR + N_TE)
    zs_tr, tgt_tr = zs_all[:N_TR], tgt_all[:N_TR]
    zs_te, tgt_te = zs_all[N_TR:], tgt_all[N_TR:]
    assert zs_tr.dim() == 3 and tgt_tr.shape == (N_TR, N_PATCH, P_PATCH), \
        f"数据形状异常: zs{tuple(zs_tr.shape)} tgt{tuple(tgt_tr.shape)}"
    print(f"[data] zs train {tuple(zs_tr.shape)}  tgt {tuple(tgt_tr.shape)}  "
          f"(同一数据集切分)")
    print(f"[data] target 量级 mean|t|={tgt_tr.abs().mean():.4f} "
          f"std={tgt_tr.std():.4f}")

    # ── oracle: 信息给足时每步的线性最优 L1 ──
    orc, conds = oracle_per_step(zs_tr, tgt_tr, zs_te, tgt_te, STEPS)
    print("\n[oracle] 每步可及信息下的线性最优 L1（信息充要性证据）:")
    for t, v, c in zip(STEPS, orc, conds):
        print(f"    step {t:>2}  oracle L1 = {v:.4f}   (cond {c:.2e})")
    marg = [orc[i] - orc[i + 1] for i in range(len(orc) - 1)]
    print(f"    边际收益 = {[round(x,4) for x in marg]}")
    print(f"    ⇒ 后 3 步合计边际 {sum(marg[1:]):.4f} "
          f"({sum(marg[1:])/orc[0]*100:.1f}% of step-1 误差)")

    arms = [
        ("A cumloss (depth2, fixed q)", "cum", False, 2),
        ("B cumloss + capacity (depth6, per-patch q)", "cum", True, 6),
        ("C regionloss (depth2, fixed q)", "region", False, 2),
    ]
    results = {"oracle_l1": orc, "oracle_marginal": marg, "arms": {}}
    for name, arm, cap, depth in arms:
        print(f"\n[train] {name}")
        r = run_arm(name, arm, cap, depth, zs_tr, tgt_tr, zs_te, tgt_te, args.steps)
        results["arms"][name] = r
        print(f"    params {r['params']:,}  ({r['seconds']}s)")
        print(f"    step_scale_ratio = {[round(x,4) for x in r['step_scale_ratio']]}")
        print(f"    progressive L1   = {[round(x,4) for x in r['prog_l1']]}")
        print(f"    |Y_t| 特征范数   = {[round(x,3) for x in r['Y_feat_norm']]}")

    print("\n" + "=" * 78)
    print("裁定")
    print("=" * 78)
    a = results["arms"]["A cumloss (depth2, fixed q)"]
    b = results["arms"]["B cumloss + capacity (depth6, per-patch q)"]
    c = results["arms"]["C regionloss (depth2, fixed q)"]
    print(f"oracle 下界（线性可及）        : {[round(x,4) for x in orc]}")
    print(f"A cumloss   每步输出量级      : {[round(x,4) for x in a['step_scale_ratio']]}")
    print(f"B +capacity 每步输出量级      : {[round(x,4) for x in b['step_scale_ratio']]}")
    print(f"C regionloss 每步输出量级     : {[round(x,4) for x in c['step_scale_ratio']]}")
    print(f"A progressive L1              : {[round(x,4) for x in a['prog_l1']]}")
    print(f"B progressive L1              : {[round(x,4) for x in b['prog_l1']]}")
    print(f"C 区域 L1（各步自身区）        : {[round(x,4) for x in c['prog_l1']]}")
    tail_a = sum(a["step_scale_ratio"][1:]) / 3
    tail_b = sum(b["step_scale_ratio"][1:]) / 3
    tail_c = sum(c["step_scale_ratio"][1:]) / 3
    print(f"\n后 3 步/step-1 平均量级: A={tail_a:.3f}  B={tail_b:.3f}  C={tail_c:.3f}")
    if tail_b < 0.15 and tail_a < 0.15:
        print("⇒ H1(容量临界值) 被否: 容量×3 且给出逐 patch 内容句柄后, "
              "后步仍然塌到近零。")
    if tail_c > 2 * max(tail_a, 1e-6):
        print("⇒ H2(损失结构) 被支持: 只把'整图平权'换成'每步私有区域目标', "
              "后步立刻开始出力。")
    print("=" * 78)

    outdir = Path(__file__).parent / "data"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "cpu_repro_cumloss_capacity.json"
    p.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {p}")


if __name__ == "__main__":
    main()
