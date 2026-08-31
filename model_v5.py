"""
SR-Diffusion Phase 1 v5 — 扩散式渐进细化（2026-08-31 用户需求）
=================================================================

背景（v2 教训 → v5 设计）:
    v2 的"渐进重建"实际是**前缀截断**（每步从零独立预测整图, 步间零耦合,
    token 无增量性）→ 曲线"悬崖+平台"（t=1 已 ≈ t=576, 信息全堆进第一个
    token）。用户想法（正确）: t=0 预测大体结构, t=1 补细节, 循环补细节——
    即**渐进细化（coarse-to-fine refinement）**范式。本模型用**扩散**把这个
    思想真正实现:

    1) 步间耦合:  DDPM 反向过程 x_{t-1} 的输入就是 x_t（每步在上一步基础上
                    细化, 不是从零预测）。
    2) 每步目标不同: 不同 t 对应不同噪声水平, 模型学"去该噪声", 不再是
                    每步监督同一张整图。
    3) token 增量性: **NVAE 式渐进解锁**（参考 NVAE 的分层潜变量/渐进训练
                    思想）——噪声大(t 大, 粗结构)只给 z_cls + 少量 token;
                    噪声小(t 小, 细细节)逐步解锁全部 K 个 token。token 只在
                    "细节该出现时"才可用 ⇒ 梯度被迫让后面的 token 装新信息
                    ⇒ 平台被打破, 渐进曲线应为**阶梯**。

架构:
    pixel_values (B,3,448,252)
      → dinov2.embeddings(x)                     (B,1+N,D) [cls; patches]+PE
      → specials (B,K=128,D) 拼入: [cls; specials; patches] (B,1+K+N,D)
      → DINO 24 层（全双向, 默认不冻结）→ layernorm
      → z_cls (B,1,D), z_s (B,K,D)               ← K=128 固定压缩表示
      → 扩散解码器（DiT-lite, patch 级）:
            x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·ε          (x0 = 归一化像素 patch B,N,588)
            条件 = [z_cls; z_s[:m(t)]]            (m(t) = 渐进解锁数, NVAE 式)
            x̂0 = decoder(x_t, t_emb, ctx)         (x0 预测)
            L = MSE(x̂0, x0) + l1_weight·L1(x̂0, x0)
      推理: DDIM 确定性反向（从噪声出发, 逐步解锁 token → 渐进阶梯曲线）。

    · K=128 固定压缩（576 patch → 128 token）: 冗余解容量不够 ⇒ token 被迫分工。
    · 条件信息语义: 结构/布局/边界（活信息）必须来自 token; 纹理（死信息,
      项目文档不追）由扩散过程编造 ⇒ 探针语义 = "token 能否驱动还原活信息"。
    · CFG 可选: 训练时按 cfg_drop 概率丢弃条件, 推理 cfg_scale>1 增强 token
      作用（默认 1.0 不做引导, 探针保持确定性）。

预训练结构: DINOv2（本项目已有）= 预训练编码器骨干; 扩散解码器为 Phase 1
    训练脚手架（不需要额外预训练, 与 v2/v3/v4 的 decoder 同性质）。

用法
----
    model = SRPhase1V5(dinov2=dinov2_model, num_patches=576, dim=1024,
                       num_specials=128, freeze_dino=False)
    out = model(pixel_values)            # 训练步: {"loss","recon","x0_hat",
                                         #          "target","z_s","t","m"}
    res = model.sample(pixel_values, steps=100)   # DDIM 反向: {"pixels",
                                         #   "target","curve":[(t,m,l1),...]}
    x̂0, x0, eps = model.predict_at(pixel_values, t=500, m=64)  # m 扫描探针

    自检: python model_v5.py（形状 / 梯度含 DINO / 冻结开关 / DDIM 采样 /
          解锁单调性 / eval 同路径）
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════
# 噪声调度（cosine, Nichol & Dhariwal 2021）— ᾱ_t 表
# ═══════════════════════════════════════════════════════════════

def cosine_alpha_bar(T: int, s: float = 0.008) -> Tensor:
    """ᾱ_t = cos²((t/T+s)/(1+s)·π/2), t=0..T; ᾱ_0=1, ᾱ_T≈0。返回 (T+1,) 张量。"""
    steps = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos((steps / T + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
    return torch.clamp(f / f[0], min=1e-5, max=1.0)


# ═══════════════════════════════════════════════════════════════
# SpecialTokens — K 个 register special token（与 v4 同构）
# ═══════════════════════════════════════════════════════════════

class SpecialTokens(nn.Module):
    """K 个 register token: 共享可学习向量 + 逐位置可学习 pos。"""

    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)

    def forward(self, B: int, device: torch.device) -> Tensor:
        return self.token.expand(B, -1, -1) + self.pos       # (B,K,D)


# ═══════════════════════════════════════════════════════════════
# 扩散解码器（DiT-lite）— patch 级去噪器, 条件 = 解锁的 token
# ═══════════════════════════════════════════════════════════════

def time_embedding(t: Tensor, dim: int) -> Tensor:
    """t: (B,) float → (B, dim) 正弦时间嵌入。"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device)
                      / half)
    ang = t.unsqueeze(1) * freqs.unsqueeze(0)                # (B, half)
    emb = torch.cat([ang.sin(), ang.cos()], dim=-1)          # (B, dim)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class _DiTBlock(nn.Module):
    """pre-norm: 自注意力(patch×patch) + 交叉注意力(patch→条件 token) + FFN。
    ctx=None 时跳过交叉注意力（CFG 无条件路径）。"""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)),
                                 nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, h: Tensor, ctx: Tensor) -> Tensor:
        n = self.norm1(h)
        h = h + self.self_attn(n, n, n, need_weights=False)[0]
        if ctx is not None:
            n = self.norm2(h)
            h = h + self.cross_attn(n, ctx, ctx, need_weights=False)[0]
        h = h + self.ffn(self.norm3(h))
        return h


class PatchDiffusionDecoder(nn.Module):
    """patch 级去噪器: (B,N,P) 噪声 patch + t 嵌入 + 条件 token → (B,N,P) x̂0。

    输入 patch 是 588 维像素 patch（归一化空间）; 位置编码提供空间布局
    （row-major, 与 DINO patch 顺序一致）。"""

    def __init__(self, dim: int, num_patches: int, patch_px: int,
                 depth: int = 4, heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.patch_embed = nn.Linear(patch_px, dim)
        self.patch_pos = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                   nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            [_DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_px)

    def forward(self, x_t: Tensor, t_emb: Tensor, ctx: Tensor) -> Tensor:
        """x_t: (B,N,P) 噪声 patch; t_emb: (B,D); ctx: (B,C,D) 或 None。"""
        h = self.patch_embed(x_t) + self.patch_pos              # (B,N,D)
        h = h + self.t_mlp(t_emb).unsqueeze(1)                  # t 条件广播
        for blk in self.blocks:
            h = blk(h, ctx)
        return self.head(self.norm(h))                          # (B,N,P)


# ═══════════════════════════════════════════════════════════════
# SRPhase1V5 — register K 压缩 + 扩散式渐进细化主模型
# ═══════════════════════════════════════════════════════════════

class SRPhase1V5(nn.Module):
    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 576,
        dim: int = 1024,
        num_specials: int = 128,
        diffusion_steps: int = 1000,
        decoder_depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 2.0,
        freeze_dino: bool = False,
        patch_px: int = 14 * 14 * 3,
        l1_weight: float = 0.5,
        cfg_drop: float = 0.1,
        unlock: str = "linear",
        fixed_eval_t: int = None,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.dim = dim
        self.num_specials = num_specials
        self.T = diffusion_steps
        self.patch_px = patch_px
        self.l1_weight = l1_weight
        self.cfg_drop = cfg_drop
        self.unlock = unlock
        self.fixed_eval_t = fixed_eval_t
        self.freeze_dino = freeze_dino
        if freeze_dino:
            for p in dinov2.parameters():
                p.requires_grad_(False)

        self.specials = SpecialTokens(num_tokens=num_specials, dim=dim)
        self.decoder = PatchDiffusionDecoder(
            dim=dim, num_patches=num_patches, patch_px=patch_px,
            depth=decoder_depth, heads=heads, mlp_ratio=mlp_ratio)
        self.register_buffer("alphas_cumprod",
                             cosine_alpha_bar(diffusion_steps))

    # ── 编码: [cls; specials(K); patches] → DINO 24 层 → z_cls, z_s ──
    def encode(self, x: Tensor):
        """x: (B,3,H,W) 归一化像素 → (z_cls (B,1,D), z_s (B,K,D))。"""
        B = x.shape[0]
        if self.freeze_dino:
            with torch.no_grad():
                emb = self.dinov2.embeddings(x)                 # (B,1+N,D)
        else:
            emb = self.dinov2.embeddings(x)
        sp = self.specials(B, x.device)                         # (B,K,D)
        seq = torch.cat([emb[:, :1], sp, emb[:, 1:]], dim=1)    # (B,1+K+N,D)
        for layer in self.dinov2.encoder.layer:                 # DINO 24 层, 全双向
            out = layer(seq)
            seq = out[0] if isinstance(out, (tuple, list)) else out
        seq = self.dinov2.layernorm(seq)
        return seq[:, :1], seq[:, 1:1 + self.num_specials]

    # ── patch 提取: (B,3,H,W) → (B,N,588), row-major（与 DINO 一致）──
    def _patches(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        return x.reshape(B, C, H // 14, 14, W // 14, 14) \
                .permute(0, 2, 4, 1, 3, 5) \
                .reshape(B, self.num_patches, C * 14 * 14)

    # ── NVAE 式渐进解锁: t 大(噪声大,粗) → token 少; t 小(细) → 全解锁 ──
    def unlock_count(self, t: int) -> int:
        """给定扩散步 t (1..T) 返回应解锁的 special token 数 m (0..K)。"""
        if self.unlock == "none":
            return self.num_specials
        f = 1.0 - t / self.T
        if self.unlock == "sqrt":
            f = math.sqrt(max(f, 0.0))
        return int(round(self.num_specials * min(max(f, 0.0), 1.0)))

    def _time_embed_full(self, B: int, t: int, device) -> Tensor:
        tt = torch.full((B,), float(t), device=device)
        return time_embedding(tt, self.dim)                     # (B,D)

    def _ctx(self, z_cls: Tensor, z_s: Tensor, m: int) -> Tensor:
        """条件 token = [z_cls; z_s[:m]]; m=0 时只剩 z_cls（粗结构锚点）。"""
        if m <= 0:
            return z_cls
        return torch.cat([z_cls, z_s[:, :m]], dim=1)

    # ── 训练步: 采样一个 t, 加噪, 去噪, L2+L1 ──
    def forward(self, x: Tensor) -> dict:
        """x: (B,3,H,W) → {"loss","recon","x0_hat","target","z_s","t","m"}。

        t 采样: 训练随机 1..T; eval 用 fixed_eval_t（默认 None→随机）——
        eval 时建议传 fixed_eval_t=T//2 得到确定性的 eval_loss。
        """
        z_cls, z_s = self.encode(x)                             # (B,1,D),(B,K,D)
        x0 = self._patches(x)                                   # (B,N,588)
        B = x0.shape[0]
        if self.training or self.fixed_eval_t is None:
            t = int(torch.randint(1, self.T + 1, ()).item())
        else:
            t = int(self.fixed_eval_t)
        m = self.unlock_count(t)
        ctx = self._ctx(z_cls, z_s, m)
        if self.training and self.cfg_drop > 0 and \
                torch.rand(()).item() < self.cfg_drop:
            ctx = None                                          # CFG 无条件路径
        a = float(self.alphas_cumprod[t])
        eps = torch.randn_like(x0)
        x_t = math.sqrt(a) * x0 + math.sqrt(max(1.0 - a, 0.0)) * eps
        x_hat = self.decoder(x_t, self._time_embed_full(B, t, x.device), ctx)
        l2 = F.mse_loss(x_hat, x0)
        l1 = F.l1_loss(x_hat, x0)
        loss = l2 + self.l1_weight * l1
        return {"loss": loss, "recon": l1, "x0_hat": x_hat,
                "target": x0, "z_s": z_s, "t": t, "m": m}

    # ── 探针: 固定噪声水平 t、指定解锁数 m 的单步 x0 预测（m 扫描用）──
    def predict_at(self, x: Tensor, t: int, m: int = None,
                   eps: Tensor = None):
        """x0 加噪到 x_t（eps 可传入以复用同一噪声）→ 用前 m 个 token 预测 x̂0。

        返回 (x̂0, x0, eps)。t∈[1,T], m∈[0,K]; m=None → 用 unlock_count(t)。
        """
        z_cls, z_s = self.encode(x)
        x0 = self._patches(x)
        if eps is None:
            eps = torch.randn_like(x0)
        a = float(self.alphas_cumprod[t])
        x_t = math.sqrt(a) * x0 + math.sqrt(max(1.0 - a, 0.0)) * eps
        if m is None:
            m = self.unlock_count(t)
        ctx = self._ctx(z_cls, z_s, m)
        x_hat = self.decoder(x_t, self._time_embed_full(x0.shape[0], t,
                                                        x.device), ctx)
        return x_hat, x0, eps

    # ── DDIM 确定性反向: 从噪声出发, 每步解锁更多 token → 渐进阶梯 ──
    def sample(self, x: Tensor, steps: int = 100, cfg_scale: float = 1.0,
               cond: bool = True, record_curve: bool = True, seed: int = None):
        """DDIM(σ=0) 确定性采样, 条件 = 随 t 递减渐进解锁的 token。

        steps: 反向步数（从 T 均匀抽到 1）。record_curve=True 返回每步
        (t, m, L1(x̂0, target)) —— 渐进阶梯曲线（m 越大 token 越多）。
        cond=False: 全程无条件（token 消融, 看 token 的价值）。
        """
        if seed is not None:
            torch.manual_seed(seed)
        z_cls, z_s = self.encode(x)
        x0 = self._patches(x)
        B = x0.shape[0]
        # 递减时刻表: linspace 单调递减, unique_consecutive 去重且保序
        ts = torch.linspace(self.T, 1, steps).round().long()
        ts = ts.unique_consecutive()
        if int(ts[0]) != self.T:
            ts = torch.cat([torch.tensor([self.T]), ts])
        ts = ts.tolist()                                        # 严格递减
        x = torch.randn_like(x0)                                # x_T ≈ 纯噪声
        curve = []
        x_hat = None
        for i, t in enumerate(ts):
            m = self.unlock_count(t) if cond else 0
            ctx = self._ctx(z_cls, z_s, m) if cond else None
            t_emb = self._time_embed_full(B, t, x.device)
            x_hat = self.decoder(x, t_emb, ctx)
            if cfg_scale > 1.0 and cond:
                x_hat_u = self.decoder(x, t_emb, None)
                x_hat = x_hat_u + cfg_scale * (x_hat - x_hat_u)
            if record_curve:
                curve.append((t, m,
                              float(F.l1_loss(x_hat, x0).mean().detach())))
            if t > 1:
                t_prev = ts[i + 1] if i + 1 < len(ts) else t - 1
                a_t = float(self.alphas_cumprod[t])
                a_p = float(self.alphas_cumprod[max(t_prev, 0)])
                eps_hat = (x - math.sqrt(a_t) * x_hat) \
                    / math.sqrt(max(1.0 - a_t, 1e-5))
                x = math.sqrt(a_p) * x_hat \
                    + math.sqrt(max(1.0 - a_p, 0.0)) * eps_hat
        return {"pixels": x_hat, "target": x0, "curve": curve,
                "z_s": z_s, "steps": ts}


# ═══════════════════════════════════════════════════════════════
# 自检（python model_v5.py）
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)

    class _FakeEmbeddings(nn.Module):
        def __init__(self, dim: int, num_patches: int):
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.patch_embeddings = nn.Conv2d(3, dim, kernel_size=14, stride=14)
            self.position_embeddings = nn.Parameter(
                torch.randn(1, num_patches + 1, dim) * 0.02)
            self.dropout = nn.Identity()

        def forward(self, pixel_values, bool_masked_pos=None,
                    interpolate_pos_encoding=None):
            B = pixel_values.shape[0]
            patch = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
            cls = self.cls_token.expand(B, -1, -1)
            return self.dropout(torch.cat([cls, patch], dim=1)
                                + self.position_embeddings)

    class _FakeDinoLayer(nn.Module):
        def __init__(self, dim: int, mlp_ratio: float = 2.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            att = nn.Module()
            att.attention = nn.Module()
            att.attention.query = nn.Linear(dim, dim)
            att.attention.key = nn.Linear(dim, dim)
            att.attention.value = nn.Linear(dim, dim)
            att.output = nn.Module()
            att.output.dense = nn.Linear(dim, dim)
            self.attention = att
            mlp = nn.Module()
            mlp.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
            mlp.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
            self.mlp = mlp

        def forward(self, hidden_states, head_mask=None,
                    output_attentions=False):
            n = self.norm1(hidden_states)
            q = self.attention.attention.query(n)
            k = self.attention.attention.key(n)
            v = self.attention.attention.value(n)
            attn = F.softmax((q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5),
                             dim=-1)
            h = hidden_states + self.attention.output.dense(attn @ v)
            h = h + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(h))))
            return (h, None)

    class FakeDino(nn.Module):
        def __init__(self, dim: int = 64, n_layers: int = 2,
                     num_patches: int = 16):
            super().__init__()
            self.embeddings = _FakeEmbeddings(dim, num_patches)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList(
                [_FakeDinoLayer(dim) for _ in range(n_layers)])
            self.layernorm = nn.LayerNorm(dim)

    N, D, K, T = 16, 64, 4, 50
    PATCH_PX = 14 * 14 * 3
    x = torch.randn(2, 3, 56, 56)
    B, C, H, W = x.shape

    # ── 1. 形状 + 训练步 ──
    dino = FakeDino(dim=D, num_patches=N)
    model = SRPhase1V5(dino, num_patches=N, dim=D, num_specials=K,
                       diffusion_steps=T, decoder_depth=1, heads=4,
                       mlp_ratio=2.0, freeze_dino=False,
                       fixed_eval_t=T // 2)
    out = model(x)
    assert out["loss"].shape == () and out["recon"].shape == ()
    assert out["x0_hat"].shape == (B, N, PATCH_PX)
    assert out["z_s"].shape == (B, K, D)
    assert 1 <= out["t"] <= T and 0 <= out["m"] <= K
    target = x.reshape(B, C, H // 14, 14, W // 14, 14) \
               .permute(0, 2, 4, 1, 3, 5).reshape(B, N, PATCH_PX)
    assert torch.isclose(out["target"], target).all()
    print(f"[ok] shapes: z_s{tuple(out['z_s'].shape)} (K={K}) "
          f"x0_hat{tuple(out['x0_hat'].shape)} loss={out['loss'].item():.4f} "
          f"t={out['t']} m={out['m']}")

    # ── 2. 渐进解锁单调性: t 大 → m 小; t=1 → m=K; t=T → m=0 ──
    # ms 按 t=1..T 递增排列, 故 m 应非增
    ms = [model.unlock_count(t) for t in range(1, T + 1)]
    assert ms[0] == K, f"t=1 应全解锁, got {ms[0]}"
    assert ms[-1] == 0, f"t=T 应只留 z_cls, got {ms[-1]}"
    assert all(ms[i] >= ms[i + 1] for i in range(T - 1)), \
        "解锁数应随 t 增大单调不增"
    print(f"[ok] 渐进解锁: t=1→m={ms[0]}, t=T→m={ms[-1]}, 单调")

    # ── 3. 梯度（DINO 不冻结）: 全链路可训 ──
    # 用 eval + fixed_eval_t 固定 t（m>0, z_s 参与），避免随机 t=T 时 m=0
    model.eval()
    out = model(x)
    out["loss"].backward()
    for name, p in [("dino.embeddings.patch_embeddings.weight",
                     dino.embeddings.patch_embeddings.weight),
                    ("dino.encoder.layer.0.mlp.fc1.weight",
                     dino.encoder.layer[0].mlp.fc1.weight),
                    ("dino.layernorm.weight", dino.layernorm.weight),
                    ("specials.token", model.specials.token),
                    ("decoder.patch_embed.weight", model.decoder.patch_embed.weight),
                    ("decoder.blocks.0.cross_attn.in_proj_weight",
                     model.decoder.blocks[0].cross_attn.in_proj_weight),
                    ("decoder.head.weight", model.decoder.head.weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    assert out["m"] > 0, "梯度测试要求 m>0（z_s 参与）"
    print(f"[ok] 梯度(不冻结): DINO 嵌入/层/layernorm + specials + decoder 全部可训 "
          f"(t={out['t']}, m={out['m']})")

    # ── 4. 冻结开关 ──
    dino_f = FakeDino(dim=D, num_patches=N)
    m_f = SRPhase1V5(dino_f, num_patches=N, dim=D, num_specials=K,
                     diffusion_steps=T, decoder_depth=1, heads=4,
                     freeze_dino=True, fixed_eval_t=1)   # t=1 → m=K, z_s 全用
    m_f.eval()
    out_f = m_f(x)
    out_f["loss"].backward()
    assert dino_f.embeddings.patch_embeddings.weight.grad is None
    assert m_f.specials.token.grad.abs().sum() > 0
    print(f"[ok] 冻结开关: freeze_dino=True 时 DINO 无梯度")

    # ── 5. DDIM 采样 + 渐进阶梯曲线 ──
    res = model.sample(x, steps=8, cfg_scale=1.0, record_curve=True, seed=0)
    assert res["pixels"].shape == (B, N, PATCH_PX)
    assert len(res["curve"]) == len(res["steps"]) == 8
    ts_c, ms_c, l1s = zip(*res["curve"])
    assert list(ts_c) == sorted(ts_c, reverse=True), "t 应递减"
    assert list(ms_c) == sorted(ms_c), "解锁数应递增"
    # 最后一步应全解锁
    assert ms_c[-1] == K
    print(f"[ok] DDIM {len(res['steps'])} 步: 曲线 t={ts_c[0]}..{ts_c[-1]}, "
          f"m={ms_c[0]}..{ms_c[-1]}, 最终 L1={l1s[-1]:.4f}")

    # ── 6. 无条件消融（cond=False → 全程 m=0）──
    res_u = model.sample(x, steps=8, cond=False, record_curve=True, seed=0)
    assert res_u["pixels"].shape == (B, N, PATCH_PX)
    assert all(m == 0 for _, m, _ in res_u["curve"])
    print(f"[ok] 无条件消融: 全程 m=0")

    # ── 7. predict_at（m 扫描探针）──
    eps = torch.randn_like(target)
    l1_by_m = []
    for m in (0, 1, 2, 4):
        xh, x0p, e = model.predict_at(x, t=T // 2, m=m, eps=eps)
        assert xh.shape == (B, N, PATCH_PX) and torch.equal(e, eps)
        l1_by_m.append(float(F.l1_loss(xh, x0p).mean().detach()))
    print(f"[ok] predict_at m 扫描 (t={T // 2}): m=0..4 → L1={['%.4f' % v for v in l1_by_m]}")

    # ── 8. eval 同路径（fixed_eval_t 确定性）──
    model.eval()
    with torch.no_grad():
        o1 = model(x)
        o2 = model(x)
    assert o1["t"] == o2["t"] == T // 2, "eval 应固定 t"
    print(f"[ok] eval 同路径: fixed t={o1['t']}, loss={o1['loss'].item():.4f}")

    print("\nALL CHECKS PASSED")
