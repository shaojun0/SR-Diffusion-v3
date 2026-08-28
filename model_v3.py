"""
SR-Diffusion Phase 1 v3 — BLIP-2 式最小版（2026-08-28）
=================================================================

背景（register_specials 测试失败后, 用户拍板"完全按 BLIP-2, 先简单"）:
    · Q-Former 式编码: K 个可学习 query（默认 64）自注意力 + 交叉注意力
      读 DINOv2 特征 → z_s (B,K,D) 压缩表示。**无时序、无前缀、无渐进**。
    · 解码: N 个可学习查询行（行 k ↔ patch k）交叉注意力读 K 个 z_s 键
      → (B,N,D) → 像素头（2 层 MLP）→ 像素 patch (B,N,588)。
      单次前向直接出全部 patch, 没有采样步 / kv_causal / 渐进曲线。
    · 无 ReEncoder / 无 SpecialTokenBank（它们都属于 v2 的"编码器侧路由"方案）。
    · DINOv2 **默认冻结**（BLIP-2 惯例; 也回应"7009 张会不会过拟合"的担忧:
      可训练参数从 367M 降到 ~60M）; `freeze_dino=False` 可解冻（回应
      "解码器欠拟合"的担忧需单独调解码器容量: qformer_depth/decoder_depth/
      head_hidden 都是开关）。

目标（权威版 GOAL_compression_for_nlp.md v2）:
    Phase 1 中间验收 = **K 压缩 × 重建质量**。本 v3 默认 K=64 就是压缩实验
    （旧 k-sweep 基于特征目标模型, 不可外推）; 重建质量看"活信息"（布局/
    物体/边界）保真, 纹理级清晰度非目标。DINO 冻结时, 9.8 线性解码探针
    表明原始特征里信息足够——v3 验证的是"K=64 查询能否把信息读出+还原"。

参考: BLIP-2（Li et al. 2023, [2]）——冻结视觉编码器 + 可学习 query 桥接;
Perceiver 家族（[1]）——输出查询注意力读共享键。

用法
----
    model = SRPhase1V3(dinov2=dinov2_model, num_patches=576, dim=1024,
                       num_queries=64, freeze_dino=True)
    out = model(pixel_values)     # {"loss","recon","pixels","target","z_s"}
    loss = out["loss"]; loss.backward()
    model.eval()                  # 推理同路径

    自检: python model_v3.py（形状 / 梯度 / DINO 冻结与解冻 / eval 同路径）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════
# QFormer — BLIP-2 式可学习查询编码器（K 个 query 交叉注意力读 DINO 特征）
# ═══════════════════════════════════════════════════════════════

class _QFormerLayer(nn.Module):
    """自注意力(query×query) + 交叉注意力(query→DINO 特征) + FFN, 残差结构。"""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)),
                                 nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        n = self.norm1(q)
        q = q + self.self_attn(n, n, n, need_weights=False)[0]
        n = self.norm2(q)
        q = q + self.cross_attn(n, kv, kv, need_weights=False)[0]
        n = self.norm3(q)
        q = q + self.ffn(n)
        return q


class QFormer(nn.Module):
    """K 个可学习 query → 交叉注意力读 DINO 特征 → z_s (B,K,D)。

    queries (1,K,D) 与 BLIP-2 的 Q-Former query 同源: 可学习、跨图共享;
    位置/身份信息由训练决定（无逐 patch 绑定——这正是"K 压缩"的语义:
    K 个 token 编码整图, 与 patch 数无关）。
    """

    def __init__(self, dim: int = 1024, num_queries: int = 64, depth: int = 2,
                 heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.layers = nn.ModuleList(
            [_QFormerLayer(dim, heads, mlp_ratio) for _ in range(depth)])

    def forward(self, feats: Tensor) -> Tensor:
        """feats: (B,L,D) DINO 特征([cls; patches]) → z_s (B,K,D)"""
        B = feats.shape[0]
        q = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, feats)
        return q


# ═══════════════════════════════════════════════════════════════
# OutputQueryDecoder — 无时序输出查询解码（N 行查询 attend K 个 z_s 键）
# ═══════════════════════════════════════════════════════════════

class _DecoderLayer(nn.Module):
    """交叉注意力(N 行查询 → K 个 z_s 键) + FFN, 残差结构。"""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)),
                                 nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        q = q + self.cross_attn(self.norm1(q), kv, kv, need_weights=False)[0]
        q = q + self.ffn(self.norm2(q))
        return q


class OutputQueryDecoder(nn.Module):
    """N 个可学习查询行（行 k ↔ patch k）读 K 个 z_s 键 → (B,N,D)。

    单次前向（无采样步、无 kv_causal、无渐进曲线）——"不弄多余的时序"。
    每行查询与全部 K 个键做注意力, 输出 = 键的加权组合 + 共享 FFN;
    键有内容（QFormer 输出）时, 行 k 学到"挑出" patch k 所需信息。
    """

    def __init__(self, dim: int = 1024, num_patches: int = 576, depth: int = 2,
                 heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_patches = num_patches
        self.query_base = nn.Parameter(
            torch.randn(1, num_patches, dim) * 0.02)      # 行 k ↔ patch k
        self.layers = nn.ModuleList(
            [_DecoderLayer(dim, heads, mlp_ratio) for _ in range(depth)])

    def forward(self, z_s: Tensor) -> Tensor:
        """z_s: (B,K,D) → (B,N,D)"""
        B = z_s.shape[0]
        q = self.query_base.expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, z_s)
        return q


# ═══════════════════════════════════════════════════════════════
# PixelHead — 特征 → 像素 patch 解码头（2 层 MLP, 容量 > v2 单层线性）
# ═══════════════════════════════════════════════════════════════

class PixelHead(nn.Module):
    """每 patch 特征 (B,N,D) → 像素 patch (B,N,14*14*3)。

    2 层 MLP（dim→hidden→588）: 回应"解码器参数量不够欠拟合"的担忧
    （v2 是单层 Linear, 0.6M; 这里是 ~2.1M@dim=1024,hidden=2048, 且带
    非线性）。输出不加激活: 像素按 DINO_MEAN/STD 归一化(范围≈[-2,2]),
    L1 直接监督归一化空间, 评估时再反归一化。
    """

    def __init__(self, dim: int, patch_px: int = 14 * 14 * 3, hidden: int = 2048):
        super().__init__()
        self.patch_px = patch_px
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, patch_px))

    def forward(self, feat: Tensor) -> Tensor:
        """feat: (..., N, D) → (..., N, patch_px)"""
        return self.net(feat)


# ═══════════════════════════════════════════════════════════════
# SRPhase1V3 — 主模型（冻结 DINO + QFormer + 无时序解码 + 像素头）
# ═══════════════════════════════════════════════════════════════

class SRPhase1V3(nn.Module):
    """BLIP-2 式最小重建模型。

    forward: pixel_values (B,3,H,W) 归一化像素 →
        dinov2(冻结, no_grad) → feats (B,257,D)
        → QFormer → z_s (B,K,D)            ← K 压缩表示（默认 K=64）
        → OutputQueryDecoder → (B,N,D)     ← 单次前向, 无时序
        → PixelHead → pixels (B,N,588)
        → L = L1(pixels, target_pix)       ← 平权, 无加权体系
    """

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 576,
        dim: int = 1024,
        num_queries: int = 64,
        qformer_depth: int = 2,
        decoder_depth: int = 2,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        freeze_dino: bool = True,
        patch_px: int = 14 * 14 * 3,
        head_hidden: int = 2048,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.dim = dim
        self.patch_px = patch_px
        self.num_queries = num_queries
        self.freeze_dino = freeze_dino
        if freeze_dino:
            for p in dinov2.parameters():
                p.requires_grad_(False)

        self.qformer = QFormer(dim=dim, num_queries=num_queries,
                               depth=qformer_depth, heads=heads,
                               mlp_ratio=mlp_ratio)
        self.decoder = OutputQueryDecoder(dim=dim, num_patches=num_patches,
                                          depth=decoder_depth, heads=heads,
                                          mlp_ratio=mlp_ratio)
        self.pixel_head = PixelHead(dim=dim, patch_px=patch_px,
                                    hidden=head_hidden)

    def forward(self, pixel_values: Tensor) -> dict:
        """pixel_values: (B,3,H,W) 归一化像素 → dict{loss, recon, pixels,
        target, z_s}

        H,W 须为 14 的倍数, (W//14)*(H//14) == num_patches。
        """
        x = pixel_values
        B, C, H, W = x.shape
        N = self.num_patches
        assert W % 14 == 0 and H % 14 == 0, "输入须为 14 的倍数"
        assert (W // 14) * (H // 14) == N, \
            f"输入 {W}x{H} 产生 {(W//14)*(H//14)} patches, 但模型 num_patches={N}"

        if self.freeze_dino:
            with torch.no_grad():
                feats = self.dinov2(x).last_hidden_state      # (B,257,D)
        else:
            feats = self.dinov2(x).last_hidden_state          # (B,257,D)

        z_s = self.qformer(feats)                             # (B,K,D) 压缩表示
        h = self.decoder(z_s)                                 # (B,N,D) 单次前向
        pixels = self.pixel_head(h)                           # (B,N,588) 归一化像素

        # 像素目标: (B,3,H,W) → (B,N,588); 布局与 DINO row-major 一致
        target = x.reshape(B, C, H // 14, 14, W // 14, 14) \
                   .permute(0, 2, 4, 1, 3, 5) \
                   .reshape(B, N, C * 14 * 14)                # (B,N,588)

        loss = F.l1_loss(pixels, target)
        return {"loss": loss, "recon": loss, "pixels": pixels,
                "target": target, "z_s": z_s}


# ═══════════════════════════════════════════════════════════════
# 自检（python model_v3.py）
#   1. 形状（z_s / F / pixels / loss）
#   2. 梯度（QFormer queries / decoder query_base / pixel_head;
#      DINO 冻结时无梯度, 解冻时有）
#   3. eval 同路径
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from types import SimpleNamespace

    torch.manual_seed(0)

    class FakeDino(nn.Module):
        """v3 只需 dinov2(x).last_hidden_state; 带一个可训参数供解冻测试。"""
        def __init__(self, dim: int = 64, num_patches: int = 16):
            super().__init__()
            self._feat = torch.randn(4, num_patches + 1, dim)
            self._p = nn.Parameter(torch.zeros(1))

        def forward(self, pixel_values: Tensor):
            B = pixel_values.shape[0]
            return SimpleNamespace(
                last_hidden_state=self._feat[:B] + self._p.view(1, 1, 1) * 0.01)

    N, D, K = 16, 64, 4
    PATCH_PX = 14 * 14 * 3
    x = torch.randn(2, 3, 56, 56)
    B, C, H, W = x.shape

    # ── 1. 形状（默认冻结 DINO）──
    dino = FakeDino(dim=D, num_patches=N)
    model = SRPhase1V3(dino, num_patches=N, dim=D, num_queries=K,
                       qformer_depth=1, decoder_depth=1, head_hidden=128)
    out = model(x)
    assert out["z_s"].shape == (2, K, D), out["z_s"].shape
    assert out["pixels"].shape == (2, N, PATCH_PX), out["pixels"].shape
    assert out["loss"].shape == () and out["recon"].shape == ()
    target = x.reshape(B, C, H // 14, 14, W // 14, 14) \
               .permute(0, 2, 4, 1, 3, 5).reshape(B, N, PATCH_PX)
    assert torch.isclose(out["loss"], F.l1_loss(out["pixels"], target))
    assert not dino._p.requires_grad, "冻结模式 DINO 参数不应可训"
    print(f"[ok] shapes: z_s{tuple(out['z_s'].shape)} (K={K}) "
          f"pixels{tuple(out['pixels'].shape)} loss={out['loss'].item():.4f}")

    # ── 2a. 梯度（冻结 DINO）: QFormer/Decoder/PixelHead 全部可训 ──
    out["loss"].backward()
    for name, p in [("qformer.queries", model.qformer.queries),
                    ("qformer.layer0.self_attn.in_proj_weight",
                     model.qformer.layers[0].self_attn.in_proj_weight),
                    ("decoder.query_base", model.decoder.query_base),
                    ("decoder.layer0.cross_attn.in_proj_weight",
                     model.decoder.layers[0].cross_attn.in_proj_weight),
                    ("pixel_head.net.0.weight", model.pixel_head.net[0].weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    assert dino._p.grad is None, "冻结 DINO 不应有梯度"
    print(f"[ok] 梯度(冻结): QFormer/Decoder/PixelHead 可训, DINO 无梯度")

    # ── 2b. 解冻 DINO 梯度 ──
    dino2 = FakeDino(dim=D, num_patches=N)
    m2 = SRPhase1V3(dino2, num_patches=N, dim=D, num_queries=K,
                    qformer_depth=1, decoder_depth=1, head_hidden=128,
                    freeze_dino=False)
    out2 = m2(x)
    out2["loss"].backward()
    assert dino2._p.grad is not None and dino2._p.grad.abs().sum() > 0, \
        "解冻模式 DINO 应收梯度"
    print(f"[ok] 梯度(解冻): DINO 参数可训")

    # ── 3. eval 同路径 ──
    model.eval()
    with torch.no_grad():
        out_e = model(x)
    assert out_e["pixels"].shape == (2, N, PATCH_PX)
    print(f"[ok] eval 同路径: loss={out_e['loss'].item():.4f}")

    print("\nALL CHECKS PASSED")
