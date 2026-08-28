"""
SR-Diffusion Phase 1 v4 — register K 压缩版（2026-08-28）
=================================================================

实验动机（对照矩阵）:
    v3 (BLIP-2, K=64, DINO 冻结)      → 全量 L1 = 17.77（当前最优）
    register (K=N=576, DINO 可训)      → 全量 L1 = 24.14（编码侧修复无效）
    本实验 v4: **register 式编码（K=64 个 special token 进 DINO 序列, 由
    DINO 24 层直接算 z_s）+ K=64 固定压缩 + DINO 不冻结 + v3 式无时序解码**。
    与 v3 的变量只差两处: 编码器形态（QFormer → register specials）与
    DINO 冻结 → 可训。解码器/像素头/损失/数据口径与 v3 完全一致。
    判读: v4 ≥ 17.77 ⇒ register 编码不敌 QFormer（或 DINO 微调无益）;
    v4 < 17.77 ⇒ 深度 register 编码 + DINO 适配有增益。

架构:
    pixel_values (B,3,H,W)
      → dinov2.embeddings(x)           (B,1+N,D) [cls; patches] + PE
      → specials (B,K,D) 拼入序列: [cls; specials; patches]  (B,1+K+N,D)
      → DINO 24 层（全双向注意力, 默认不冻结）→ layernorm
      → z_s = seq[:,1:1+K]             (B,K,D)  ← K 固定压缩表示
      → OutputQueryDecoder（N 行查询 × K 键, 无时序）→ (B,N,D)
      → PixelHead（2 层 MLP）→ 像素 (B,N,588)
      → L = L1(pixels, target)（平权）

与 v2 register 版（K=N=576）的差异: K 固定为压缩数（默认 64, 与 v3 对齐）,
解码器换成 v3 式无时序输出查询（去掉采样步/kv_causal/渐进曲线）。
DINO 不冻结时 HF 位置编码/注意力全程参与反向（与 v2 同口径）。

用法
----
    model = SRPhase1V4(dinov2=dinov2_model, num_patches=576, dim=1024,
                       num_specials=64, freeze_dino=False)
    out = model(pixel_values)     # {"loss","recon","pixels","target","z_s"}
    loss = out["loss"]; loss.backward()
    model.eval()                  # 推理同路径

    自检: python model_v4.py（形状 / 梯度含 DINO / 冻结开关 / eval 同路径）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model_v3 import OutputQueryDecoder, PixelHead   # 复用 v3 解码器/像素头


# ═══════════════════════════════════════════════════════════════
# SpecialTokens — K 个 register special token（共享向量 + 逐位置 pos）
# ═══════════════════════════════════════════════════════════════

class SpecialTokens(nn.Module):
    """K 个 register token 的输入嵌入: 共享可学习向量 + 逐位置可学习 pos。

    与 v2 SpecialTokenBank 同构（仅数量参数化）: token (1,1,D) 对所有位置/
    所有图共享, pos (1,K,D) 提供位置区分。拼进 DINO 输入序列后由 24 层
    直接算出 z_s——输入本身无 patch 内容, 内容路由靠深层注意力（register
    式, Darcet et al.）。
    """

    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)

    def forward(self, B: int, device: torch.device) -> Tensor:
        return self.token.expand(B, -1, -1) + self.pos     # (B,K,D)


# ═══════════════════════════════════════════════════════════════
# SRPhase1V4 — register K 压缩主模型（DINO 默认不冻结）
# ═══════════════════════════════════════════════════════════════

class SRPhase1V4(nn.Module):
    """register 式 K 压缩重建模型。

    forward: pixel_values →
        dinov2.embeddings(x) → [cls; specials(K); patches] (1+K+N token)
        → DINO encoder.layer（全双向; freeze_dino=True 时 no_grad）→ layernorm
        → z_s (B,K,D) → OutputQueryDecoder(N 行 × K 键) → (B,N,D)
        → PixelHead → pixels (B,N,588); L = L1(pixels, target)
    """

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 576,
        dim: int = 1024,
        num_specials: int = 64,
        decoder_depth: int = 2,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        freeze_dino: bool = False,
        patch_px: int = 14 * 14 * 3,
        head_hidden: int = 2048,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.dim = dim
        self.patch_px = patch_px
        self.num_specials = num_specials
        self.freeze_dino = freeze_dino
        if freeze_dino:
            for p in dinov2.parameters():
                p.requires_grad_(False)

        self.specials = SpecialTokens(num_tokens=num_specials, dim=dim)
        self.decoder = OutputQueryDecoder(dim=dim, num_patches=num_patches,
                                          depth=decoder_depth, heads=heads,
                                          mlp_ratio=mlp_ratio)
        self.pixel_head = PixelHead(dim=dim, patch_px=patch_px,
                                    hidden=head_hidden)

    def forward(self, pixel_values: Tensor) -> dict:
        """pixel_values: (B,3,H,W) 归一化像素 → dict{loss, recon, pixels,
        target, z_s}"""
        x = pixel_values
        B, C, H, W = x.shape
        N = self.num_patches
        assert W % 14 == 0 and H % 14 == 0, "输入须为 14 的倍数"
        assert (W // 14) * (H // 14) == N, \
            f"输入 {W}x{H} 产生 {(W//14)*(H//14)} patches, 但模型 num_patches={N}"

        if self.freeze_dino:
            with torch.no_grad():
                emb = self.dinov2.embeddings(x)             # (B,1+N,D)
        else:
            emb = self.dinov2.embeddings(x)                 # (B,1+N,D)
        sp = self.specials(B, x.device)                     # (B,K,D)
        seq = torch.cat([emb[:, :1], sp, emb[:, 1:]], dim=1)  # (B,1+K+N,D)
        for layer in self.dinov2.encoder.layer:             # DINO 24 层（全双向）
            out = layer(seq)
            seq = out[0] if isinstance(out, (tuple, list)) else out
        seq = self.dinov2.layernorm(seq)                    # (B,1+K+N,D)
        z_s = seq[:, 1:1 + self.num_specials]               # (B,K,D) 压缩表示

        h = self.decoder(z_s)                               # (B,N,D) 无时序单次前向
        pixels = self.pixel_head(h)                         # (B,N,588) 归一化像素

        target = x.reshape(B, C, H // 14, 14, W // 14, 14) \
                   .permute(0, 2, 4, 1, 3, 5) \
                   .reshape(B, N, C * 14 * 14)              # (B,N,588)

        loss = F.l1_loss(pixels, target)
        return {"loss": loss, "recon": loss, "pixels": pixels,
                "target": target, "z_s": z_s}


# ═══════════════════════════════════════════════════════════════
# 自检（python model_v4.py）
#   1. 形状（z_s / pixels / loss）
#   2. 梯度（DINO 不冻结: 嵌入/层/layernorm/specials/decoder/pixel_head;
#      冻结开关生效）
#   3. eval 同路径
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from types import SimpleNamespace

    torch.manual_seed(0)

    class _FakeEmbeddings(nn.Module):
        def __init__(self, dim: int, num_patches: int):
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.patch_embeddings = nn.Conv2d(3, dim, kernel_size=14, stride=14)
            self.position_embeddings = nn.Parameter(
                torch.randn(1, num_patches + 1, dim) * 0.02)
            self.dropout = nn.Identity()

        def forward(self, pixel_values: Tensor, bool_masked_pos=None,
                    interpolate_pos_encoding=None) -> Tensor:
            B = pixel_values.shape[0]
            patch = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
            cls = self.cls_token.expand(B, -1, -1)
            return self.dropout(torch.cat([cls, patch], dim=1)
                                + self.position_embeddings)

    class _FakeDinoLayer(nn.Module):
        def __init__(self, dim: int, mlp_ratio: float = 4.0):
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

        def forward(self, hidden_states: Tensor, head_mask=None,
                    output_attentions=False):
            n = self.norm1(hidden_states)
            q = self.attention.attention.query(n)
            k = self.attention.attention.key(n)
            v = self.attention.attention.value(n)
            attn = F.softmax((q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5), dim=-1)
            h = hidden_states + self.attention.output.dense(attn @ v)
            h = h + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(h))))
            return (h, None)

    class FakeDino(nn.Module):
        def __init__(self, dim: int = 64, n_layers: int = 2, num_patches: int = 16):
            super().__init__()
            self.embeddings = _FakeEmbeddings(dim, num_patches)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList(
                [_FakeDinoLayer(dim) for _ in range(n_layers)])
            self.layernorm = nn.LayerNorm(dim)

        def forward(self, pixel_values: Tensor):
            raise NotImplementedError("v4 不使用 dino(x), 直接用 embeddings/层")

    N, D, K = 16, 64, 4
    PATCH_PX = 14 * 14 * 3
    x = torch.randn(2, 3, 56, 56)
    B, C, H, W = x.shape

    # ── 1. 形状（默认 DINO 不冻结）──
    dino = FakeDino(dim=D, num_patches=N)
    model = SRPhase1V4(dino, num_patches=N, dim=D, num_specials=K,
                       decoder_depth=1, head_hidden=128, freeze_dino=False)
    out = model(x)
    assert out["z_s"].shape == (2, K, D), out["z_s"].shape
    assert out["pixels"].shape == (2, N, PATCH_PX), out["pixels"].shape
    assert out["loss"].shape == () and out["recon"].shape == ()
    target = x.reshape(B, C, H // 14, 14, W // 14, 14) \
               .permute(0, 2, 4, 1, 3, 5).reshape(B, N, PATCH_PX)
    assert torch.isclose(out["loss"], F.l1_loss(out["pixels"], target))
    print(f"[ok] shapes: z_s{tuple(out['z_s'].shape)} (K={K}) "
          f"pixels{tuple(out['pixels'].shape)} loss={out['loss'].item():.4f}")

    # ── 2a. 梯度（DINO 不冻结）: 全链路可训 ──
    out["loss"].backward()
    for name, p in [("dino.embeddings.patch_embeddings.weight",
                     dino.embeddings.patch_embeddings.weight),
                    ("dino.embeddings.position_embeddings",
                     dino.embeddings.position_embeddings),
                    ("dino.encoder.layer.0.mlp.fc1.weight",
                     dino.encoder.layer[0].mlp.fc1.weight),
                    ("dino.layernorm.weight", dino.layernorm.weight),
                    ("specials.token", model.specials.token),
                    ("specials.pos", model.specials.pos),
                    ("decoder.query_base", model.decoder.query_base),
                    ("pixel_head.net.0.weight", model.pixel_head.net[0].weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    print(f"[ok] 梯度(不冻结): DINO 嵌入/层/layernorm + specials + decoder + pixel_head 全部可训")

    # ── 2b. 冻结开关: DINO 无梯度 ──
    dino_f = FakeDino(dim=D, num_patches=N)
    m_f = SRPhase1V4(dino_f, num_patches=N, dim=D, num_specials=K,
                     decoder_depth=1, head_hidden=128, freeze_dino=True)
    out_f = m_f(x)
    out_f["loss"].backward()
    assert dino_f.embeddings.patch_embeddings.weight.grad is None, \
        "冻结模式 DINO 不应有梯度"
    assert model.specials.token.grad.abs().sum() > 0
    print(f"[ok] 冻结开关: freeze_dino=True 时 DINO 无梯度")

    # ── 3. eval 同路径 ──
    model.eval()
    with torch.no_grad():
        out_e = model(x)
    assert out_e["pixels"].shape == (2, N, PATCH_PX)
    print(f"[ok] eval 同路径: loss={out_e['loss'].item():.4f}")

    print("\nALL CHECKS PASSED")
