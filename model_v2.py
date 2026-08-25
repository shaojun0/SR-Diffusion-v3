"""
SR-Diffusion Phase 1 v2 — 无预算简化版（YAGNI: 先不增实体）
=================================================================

沿革:
    v2 曾含"可学习前缀预算"机制（SelectHead 边界分布 + STE 门控 +
    率惩罚/目标铰链 + pad_token），统一 hard 后发现它是"先增的实体":
    与前向脱节、代码熵增、收益未验证。按 YAGNI 原则整块移除（git
    历史可找回），k 固定 = 全量 N——decoder 恒吃全部 z_s。
    后续若真需要预算（"少留也能重建"），再把 SelectHead/边界分布
    加回来。

当前架构（无选择、无预算）:
    DINOv2(冻结) → cls + patch 特征 (B,257,D)
    ReEncoder:    [cls; specials; patches] 因果 specials 块掩码 → z_cls, z_s
    FeatureDecoder: [z_cls; z_s(全量); <patch_token>×N] 块掩码
                    （z 因果链 + patch 全局）→ F_hat (B,N,D)
    L = L1(F_hat, patch)          ← 唯一损失

块状注意力掩码（两处一致的前缀链语义）:
    ReEncoder（causal_specials=True 默认）: [cls; specials; patches]
        cls 全局；specials 因果链（special i 只见 specials≤i + 全部
        patches）；patches 全局。
    FeatureDecoder: [z_cls; z_s; <patch_token>×N]
        z 部分因果链（z 行 i 只见 z≤i + 全部 patch 行）；patch 部分
        全局（见所有人、被所有人见）；输出 = patch 位置表示 = F_hat。
    掩码（build_prefix_mask）在 forward 内现算、按实际序列长度构建
    （牺牲一点速度，换可扩展性）。

踩坑记录（重要）:
    torch 2.x 的 bool 注意力掩码约定是 True=屏蔽（_canonical_mask:
    masked_fill_(mask, -inf)），与直觉相反。初版写成 True=允许导致
    输出不依赖输入、梯度为零（自检抓到），已按 True=屏蔽 实现。

用法
----
    model = SRPhase1V2(dinov2=dinov2_model, num_patches=256, dim=768)
    out = model(pixel_values)                 # {"loss","recon","F_hat"}
    loss = out["loss"]; loss.backward()
    model.eval()                              # 推理同路径

    自检: python model_v2.py（形状 / 两处块掩码结构 / 梯度 /
          init_reencoder_from_dino / eval 同路径）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════
# SpecialTokenBank — 特殊 token 池（输入相同，仅位置编码不同）
# ═══════════════════════════════════════════════════════════════

class SpecialTokenBank(nn.Module):
    """每个 patch 位置一个特殊 token：共享可学习向量 + 逐位置可学习位置编码。

    token (1,1,D) 对所有位置/所有图像完全相同；pos (1,N,D) 提供位置区分。
    与 patch 组合后输入 ReEncoder，输出中特殊 token 位置的表示 z_s 是
    decoder 的输入——图像原始 patch 编码不进入 decoder（只作为编码器
    输入参与聚合）。
    """

    def __init__(self, num_patches: int, dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)          # 共享向量
        self.pos = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)  # 位置编码

    def forward(self, B: int, device: torch.device) -> Tensor:
        return self.token.expand(B, self.num_patches, -1) + self.pos      # (B,N,D)


# ═══════════════════════════════════════════════════════════════
# build_prefix_mask — 块状前缀掩码（forward 内现算，可扩展）
# ═══════════════════════════════════════════════════════════════

def build_prefix_mask(seq_len: int, z_start: int, z_end: int,
                      device: torch.device = None) -> Tensor:
    """构建块状前缀注意力掩码（torch bool，True=屏蔽）。

    布局: [z 区域 (z_start..z_end-1) | 尾部 (z_end..seq_len-1)]
        · z 行 i: 屏蔽 z 列 (i+1..z_end-1)（因果链），可看尾部（全局）
        · 尾部行: 全开放（全局，见所有人、被所有人见）
    用于 ReEncoder（z=specials 1..N，尾部=patches）与 FeatureDecoder
    （z=z_cls+z_s 0..N，尾部=<patch_token>×N），传不同 z_start/z_end 即可。

    在 forward 内现算而非 __init__ 缓存: 牺牲一点速度，换可扩展性——
    之后支持变长序列、运行时切换掩码都只需改传参。
    """
    m = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(z_start, z_end):
        m[i, i + 1:z_end] = True          # 屏蔽 z 内部"后面的"
    return m


# ═══════════════════════════════════════════════════════════════
# ReEncoder — pass 2, 组合编码器 [cls; specials; patches]
# ═══════════════════════════════════════════════════════════════

class ReEncoder(nn.Module):
    """对 [cls; 特殊 token; patch] 组合序列做自注意力（带块状掩码）。

    输出中特殊 token 位置的表示 z_s 是 decoder 的输入；patch 位置的
    输出直接丢弃。输入始终是完整 2N+1 序列（训练/推理全量计算）。

    块状注意力掩码（causal_specials=True，默认）:
        cls(0)              → 全局（见所有人、被所有人见）
        specials(1..N)      → 只见 cls + specials≤i + 全部 patches（前缀链）
        patches(N+1..2N)    → 全局（图像无时序，全双向）
    即 M[i,j]=1 除 special 行 i 的 special 列 j>i 之外——special p 的编码
    z_s[p] 不依赖任何"后面的 special"（前缀稳定性：直接路径上）。
    掩码在 forward 内现算（build_prefix_mask），按实际序列长度构建。

    注意: patches 仍会关注 specials（M 的 1_{patch×special} 块）——
    若将来要裁 encoder 输入省算力，需同时把 patch→special 关注也屏蔽，
    否则 patch 编码会经 specials 间接变化。

    Input:  x (B, 2N+1, D) = [cls; special_1..N; patch_1..N]
    Output: z (B, 2N+1, D)
    """

    def __init__(self, dim: int = 768, num_patches: int = 256, depth: int = 4,
                 heads: int = 8, mlp_ratio: float = 4.0,
                 causal_specials: bool = True):
        super().__init__()
        self.num_patches = num_patches
        self.causal_specials = causal_specials
        L = 2 * num_patches + 1              # cls + N specials + N patches
        self.pos_embed = nn.Parameter(torch.randn(1, L, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        L = x.shape[1]
        # forward 内现算掩码（True=屏蔽; 可扩展: 变长/运行时切换只改传参）
        z_end = self.num_patches + 1 if self.causal_specials else 1
        self.attn_mask = build_prefix_mask(L, 1, z_end, device=x.device)
        x = x + self.pos_embed[:, :L, :]
        for layer in self.layers:
            x = layer(x, src_mask=self.attn_mask)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════
# FeatureDecoder — 块掩码解码器: [z_cls; z_s(全量); <patch_token>×N]
# ═══════════════════════════════════════════════════════════════

class FeatureDecoder(nn.Module):
    """块掩码解码器: 从 z_cls + z_s 解码出完整特征图 F_hat (B,N,D)。

    输入序列（恒长 2N+1）: [z_cls(1); z_s(N, 全量); <patch_token>×N]
    注意力掩码（True=屏蔽，torch 2.x bool 约定）:
        z 部分 (0..N)        → 因果链: z 行 i 只见 z≤i + 全部 patch 行
        patch 部分 (N+1..2N) → 全局: 见所有人、被所有人见
    掩码在 forward 内现算（build_prefix_mask），按实际序列长度构建。
    输出: patch 位置的最终表示 = F_hat（L1 监督 vs DINO patch 特征）。

    · 单栈自注意力 + 块掩码（无 query/cross-attention 两套机制）；
    · z 前缀链（因果）与 encoder 侧 causal specials 语义一致:
      每个 z 是"前缀 0..i 的表示"，patch 全局汇聚这些前缀；
    · 若不想让 z 看到 patch（更纯粹的"潜变量前缀→输出"语义），
      在 build_prefix_mask 调用处把 z 行的尾部列也置 True（屏蔽）。
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 heads: int = 8, depth: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_patches = num_patches
        L = 2 * num_patches + 1                # z_cls + N z + N patch
        self.patch_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # 共享输出 token
        self.pos_embed = nn.Parameter(torch.randn(1, L, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        B = z_cls.shape[0]
        N = self.num_patches
        patch_tokens = self.patch_token.expand(B, N, -1)   # (B,N,D)
        x = torch.cat([z_cls, z_s, patch_tokens], dim=1)   # (B,2N+1,D)
        L = x.shape[1]
        # forward 内现算掩码（True=屏蔽; z 区域 0..N 因果，尾部=patch 全局）
        self.attn_mask = build_prefix_mask(L, 0, N + 1, device=x.device)
        x = x + self.pos_embed[:, :L, :]
        for layer in self.layers:
            x = layer(x, src_mask=self.attn_mask)
        x = self.norm(x)
        return x[:, N + 1:]                                # (B,N,D) patch 输出


# ═══════════════════════════════════════════════════════════════
# SRPhase1V2 — 主模型（自包含；无预算，k 固定全量）
# ═══════════════════════════════════════════════════════════════

class SRPhase1V2(nn.Module):

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        dim: int = 768,
        reencoder_depth: int = 4,
        decoder_depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        causal_specials: bool = True,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.dim = dim

        self.special_bank = SpecialTokenBank(num_patches=num_patches, dim=dim)
        self.re_encoder = ReEncoder(dim=dim, num_patches=num_patches,
                                    depth=reencoder_depth, heads=heads,
                                    mlp_ratio=mlp_ratio,
                                    causal_specials=causal_specials)
        self.decoder = FeatureDecoder(dim=dim, num_patches=num_patches,
                                      depth=decoder_depth, heads=heads,
                                      mlp_ratio=mlp_ratio)

    def init_reencoder_from_dino(self, num_layers: int = 4):
        """用 DINO 编码器前 num_layers 层 warm-start ReEncoder（结构对齐
        nn.TransformerEncoderLayer ↔ HF DINO layer）。"""
        dino_layers = self.dinov2.encoder.layer
        depth = min(num_layers, len(self.re_encoder.layers), len(dino_layers))
        for i in range(depth):
            src, dst = dino_layers[i], self.re_encoder.layers[i]
            with torch.no_grad():
                dst.norm1.load_state_dict(src.norm1.state_dict())
                dst.norm2.load_state_dict(src.norm2.state_dict())
                dst.self_attn.in_proj_weight.data.copy_(torch.cat([
                    src.attention.attention.query.weight.data,
                    src.attention.attention.key.weight.data,
                    src.attention.attention.value.weight.data,
                ], dim=0))
                dst.self_attn.in_proj_bias.data.copy_(torch.cat([
                    src.attention.attention.query.bias.data,
                    src.attention.attention.key.bias.data,
                    src.attention.attention.value.bias.data,
                ], dim=0))
                dst.self_attn.out_proj.load_state_dict(src.attention.output.dense.state_dict())
                dst.linear1.load_state_dict(src.mlp.fc1.state_dict())
                dst.linear2.load_state_dict(src.mlp.fc2.state_dict())
        print(f"[init] ReEncoder layers 0..{depth-1} warm-started from DINO encoder")

    # ── forward（训练/推理同一路径，无分支）──
    def forward(self, pixel_values: Tensor) -> dict:
        """pixel_values: (B,3,224,224) → dict{loss, recon, F_hat}

        唯一损失: L1(F_hat, patch)。全量 z_s 进 decoder，无选择无预算。
        """
        x = pixel_values                                # (B,3,224,224)
        B = x.shape[0]
        N = self.num_patches

        out = self.dinov2(x)
        feats = out.last_hidden_state                   # (B,257,D)
        cls = feats[:, 0]                               # (B,D)
        patch = feats[:, 1:]                            # (B,N,D)

        # ── ReEncoder: [cls; specials; patches] → z_cls, z_s ──
        specials = self.special_bank(B, x.device)       # (B,N,D)
        enc_in = torch.cat([cls.unsqueeze(1), specials, patch], dim=1)  # (B,2N+1,D)
        z = self.re_encoder(enc_in)                     # (B,2N+1,D)
        z_cls = z[:, 0:1]                               # (B,1,D)
        z_s = z[:, 1:1 + N]                             # (B,N,D)

        # ── FeatureDecoder: [z_cls; z_s(全量); patch×N] → F_hat ──
        F_hat = self.decoder(z_cls, z_s)                # (B,N,D)

        recon = F.l1_loss(F_hat, patch)                 # 唯一损失
        return {"loss": recon, "recon": recon, "F_hat": F_hat}


# ═══════════════════════════════════════════════════════════════
# 自检（python model_v2.py）
#   1. 形状正确性
#   2. 两处块掩码结构（ReEncoder / FeatureDecoder）
#   3. 梯度流向（整模型可训: ReEncoder / Decoder / SpecialTokenBank）
#   4. eval 同路径
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from types import SimpleNamespace

    torch.manual_seed(0)

    # ── 假的 DINOv2：结构对齐 HF Dinov2Model（供 init_reencoder_from_dino）──
    def _mk_dino_layer(dim: int, mlp_ratio: float = 4.0) -> nn.Module:
        att = nn.Module()
        att.attention = nn.Module()
        att.attention.query = nn.Linear(dim, dim)
        att.attention.key = nn.Linear(dim, dim)
        att.attention.value = nn.Linear(dim, dim)
        att.output = nn.Module()
        att.output.dense = nn.Linear(dim, dim)
        mlp = nn.Module()
        mlp.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        mlp.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        layer = nn.Module()
        layer.norm1 = nn.LayerNorm(dim)
        layer.norm2 = nn.LayerNorm(dim)
        layer.attention = att
        layer.mlp = mlp
        return layer

    class FakeDino(nn.Module):
        def __init__(self, dim: int = 64, n_layers: int = 4, num_patches: int = 16):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([_mk_dino_layer(dim) for _ in range(n_layers)])
            self._np, self._dim = num_patches, dim
            self._feat = torch.randn(4, num_patches + 1, dim)   # 固定特征（B≤4）

        def forward(self, pixel_values: Tensor):
            B = pixel_values.shape[0]
            return SimpleNamespace(
                last_hidden_state=self._feat[:B].clone())

    N, D = 16, 64
    dino = FakeDino(dim=D, num_patches=N)
    model = SRPhase1V2(dino, num_patches=N, dim=D,
                       reencoder_depth=2, decoder_depth=2)
    model.init_reencoder_from_dino(2)
    x = torch.randn(2, 3, 224, 224)

    # ── 1. 形状 ──
    out = model(x)
    assert out["F_hat"].shape == (2, N, D), out["F_hat"].shape
    assert out["loss"].shape == () and out["recon"].shape == ()
    print(f"[ok] shapes: F_hat{tuple(out['F_hat'].shape)} loss={out['loss'].item():.4f}")

    # ── 2. 块状注意力掩码结构（ReEncoder: causal specials）──
    am = model.re_encoder.attn_mask                  # (2N+1, 2N+1) bool, True=屏蔽
    assert am.shape == (2 * N + 1, 2 * N + 1)
    assert not am[0].any()                           # cls 行无屏蔽（全局）
    assert not am[N + 1:].any()                      # patches 行无屏蔽（全双向）
    for i in range(1, N + 1):
        assert not am[i, :i + 1].any()               # special i 可见 cls + specials≤i
        assert am[i, i + 1:N + 1].all()              # 屏蔽后面的 special
        assert not am[i, N + 1:].any()               # special i 可见全部 patches
    # 关掉 causal 时掩码全 False（回退全双向；掩码 forward 内现算，先跑一次）
    m_free = SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                        causal_specials=False, reencoder_depth=2, decoder_depth=2)
    _ = m_free.re_encoder(torch.randn(2, 2 * N + 1, D))
    assert not m_free.re_encoder.attn_mask.any()
    print(f"[ok] ReEncoder block mask: causal_specials=True 结构正确, "
          f"causal_specials=False 全开放回退")

    # ── 3. FeatureDecoder 块掩码结构（[z_cls; z_s; patch×N]）──
    dm = model.decoder.attn_mask                     # (2N+1, 2N+1) bool, True=屏蔽
    assert dm.shape == (2 * N + 1, 2 * N + 1)
    for i in range(N + 1):                           # z 行 0..N: z 内部因果
        assert not dm[i, :i + 1].any()               #  可见 z≤i
        assert dm[i, i + 1:N + 1].all()              #  屏蔽后面的 z
        assert not dm[i, N + 1:].any()               #  可见全部 patch 行
    assert not dm[N + 1:].any()                      # patch 行全局
    print(f"[ok] Decoder block mask: z 因果链 + patch 全局, 结构正确")

    # ── 4. 梯度流向（整模型可训）──
    out["loss"].backward()
    for name, p in [("re_encoder.0", model.re_encoder.layers[0].linear1.weight),
                    ("decoder.0", model.decoder.layers[0].linear1.weight),
                    ("special_bank.pos", model.special_bank.pos),
                    ("decoder.patch_token", model.decoder.patch_token)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    print(f"[ok] 梯度: ReEncoder/Decoder/SpecialTokenBank/patch_token 全部可训 "
          f"(|grad|={model.re_encoder.layers[0].linear1.weight.grad.abs().sum().item():.4f})")

    # ── 5. 推理: 同一 forward（eval + no_grad）──
    model.eval()
    with torch.no_grad():
        out_e = model(x)
    assert out_e["F_hat"].shape == (2, N, D)
    print(f"[ok] eval 同路径: loss={out_e['loss'].item():.4f}")

    print("\nALL CHECKS PASSED")
