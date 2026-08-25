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
    后续该模型的编码器可想办法作为model.py的编码器，实现基于数据分布的极致token压缩。

当前架构（无选择、无预算）:
    DINOv2(冻结) → cls + patch 特征 (B,257,D)
    ReEncoder:    [cls; specials; patches] 因果 specials 块掩码 → z_cls, z_s
    FeatureDecoder: [z_cls; z_s(全量); <patch_token>×N] 块掩码
                    （z 因果链 + patch 全局）→ F_hat (B,N,D)
    L = L1(F_hat, patch)          ← 重建损失

    v2.1 新增（可选，构造传 vocab_size>0 启用）— TextDecoder，仅文字自回归:
        [z_cls; z_s; 文字×T] 块掩码（build_prefix_mask tail_causal=True）:
            z 行: 因果链且不看文字（防未来文字经 z 泄漏给前面的文字）;
            文字行: 见全部 z + 文字≤i（标准下三角）。
        文字 embedding 可复用 Qwen 预训练权重（init_text_from_qwen），
        L_text = 错位 CE（仅文字位置），与 L1 多任务联合。

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

    启用文字（可选）: 构造传 vocab_size>0；若复用 Qwen 预训练词表，
    再传 qwen_hidden（embedding 维度）并调 init_text_from_qwen(qwen_emb,
    qwen_lm_head=None) 加载权重（embedding 默认冻结）:
        out = model(pixel_values, text_ids)   # 多任务: loss = L1 + L_text
        ids = model.generate_text(pixel_values, prompt_ids)   # 自回归生成

    自检: python model_v2.py（形状 / 两处块掩码结构 / 梯度 /
          init_reencoder_from_dino / eval 同路径 / TextDecoder）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


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
                      device: torch.device = None,
                      tail_causal: bool = False,
                      z_see_tail: bool = True) -> Tensor:
    """构建块状前缀注意力掩码（torch bool，True=屏蔽）。

    布局: [z 区域 (z_start..z_end-1) | 尾部 (z_end..seq_len-1)]
        · z 行 i: 屏蔽 z 列 (i+1..z_end-1)（因果链）；z_see_tail=True 时
          可看尾部（全局），False 时屏蔽全部尾部（防尾部数据经 z 泄漏）
        · 尾部行: tail_causal=False → 全开放（全局）；True → 因果
          （只见 z 全部 + 尾部≤i，标准下三角）
    用于 ReEncoder（z=specials 1..N，尾部=patches）、FeatureDecoder
    （z=z_cls+z_s 0..N，尾部=<patch_token>×N）与 TextDecoder
    （z=z_cls+z_s 0..N，尾部=文字: tail_causal=True, z_see_tail=False），
    传不同 z_start/z_end + 开关即可。

    在 forward 内现算而非 __init__ 缓存: 牺牲一点速度，换可扩展性——
    之后支持变长序列、运行时切换掩码都只需改传参。
    """
    m = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(z_start, z_end):
        m[i, i + 1:z_end] = True          # 屏蔽 z 内部"后面的"
        if not z_see_tail:
            m[i, z_end:] = True           # 屏蔽尾部（文字场景防泄漏）
    if tail_causal:
        for i in range(z_end, seq_len):
            m[i, i + 1:] = True           # 尾部因果: 见 z 全部 + 尾部≤i
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

    输入序列: [z_cls(1); z_s(k, k∈[1,N]); <patch_token>×N]，长度 N+k+1
    注意力掩码（True=屏蔽，torch 2.x bool 约定）:
        z 部分 (0..k)        → 因果链: z 行 i 只见 z≤i + 全部 patch 行
        patch 部分 (k+1..N+k) → 全局: 见所有人、被所有人见
    掩码在 forward 内现算（build_prefix_mask），按实际序列长度构建。
    输出: patch 位置的最终表示 = F_hat（L1 监督 vs DINO patch 特征）。

    · 支持 z_s 前缀输入（k<N，"预算/渐进还原"场景）: 只喂前 k 个 z_s 仍
      输出全量 N 个 patch 重建；k=N 时与全量语义完全一致（向后兼容）。
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
        k = z_s.shape[1]                                   # z_s 前缀长度, 1≤k≤N
        patch_tokens = self.patch_token.expand(B, N, -1)   # (B,N,D)
        x = torch.cat([z_cls, z_s, patch_tokens], dim=1)   # (B,N+k+1,D)
        L = x.shape[1]
        # forward 内现算掩码（True=屏蔽; z 区域 0..k 因果，尾部=patch 全局）
        self.attn_mask = build_prefix_mask(L, 0, k + 1, device=x.device)
        x = x + self.pos_embed[:, :L, :]
        for layer in self.layers:
            x = layer(x, src_mask=self.attn_mask)
        x = self.norm(x)
        return x[:, k + 1:]                                # (B,N,D) patch 输出


# ═══════════════════════════════════════════════════════════════
# TextDecoder — 仅文字自回归: [z_cls; z_s; 文字×T] → 文字 logits
# ═══════════════════════════════════════════════════════════════

class TextDecoder(nn.Module):
    """从 z_cls + z_s 自回归生成文字（仅文字参与自回归与损失）。

    输入序列（长度 k+1+T）: [z_cls(1); z_s(k); 文字 token×T]，k∈[0,N]
    注意力掩码（True=屏蔽）:
        z 行 (0..k)    → z 内部因果链（z_s[i] 只见 z_cls + z_s≤i，与
                         ReEncoder 前缀链语义一致），且**不看文字**——
                         否则未来文字会经 z 的表示泄漏给前面的文字
        text 行        → 见全部已输入的 z + 文字≤i（标准下三角；错位
                         CE 在主模型 forward 里按 [1:]/[:-1] 计算）
    **z_s 支持前缀输入（k<N）**，如"预算/渐进条件"场景:
        按完整长度 M=N+1+T 现算掩码，再按保留索引（z_cls+z_s[:k]+文字）
        子采样掩码与位置编码；文字绝对位置恒为 N+1..N+T，不随 k 漂移，
        缺失的 z 行/列直接不存在（不会引用）。

    · 文字 embedding 维度可不同于 dim（qwen_hidden）: 经 embed_proj 投影；
    · 复用 Qwen 预训练词表: SRPhase1V2.init_text_from_qwen() 拷权重（可冻结）；
    · lm_head 输出 vocab 维，可被 qwen_lm_head 暖启动。
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 vocab_size: int = 0, qwen_hidden: Optional[int] = None,
                 max_text_len: int = 512, heads: int = 8, depth: int = 4,
                 mlp_ratio: float = 4.0, freeze_text_embed: bool = True):
        super().__init__()
        assert vocab_size > 0, "TextDecoder 需要 vocab_size>0"
        self.num_patches = num_patches
        self.max_text_len = max_text_len
        embed_dim = qwen_hidden if qwen_hidden is not None else dim
        self.text_embed = nn.Embedding(vocab_size, embed_dim)   # 词表
        self.embed_proj = nn.Linear(embed_dim, dim) if qwen_hidden is not None \
            else nn.Identity()
        self.lm_head = nn.Linear(dim, vocab_size)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1 + max_text_len, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        if freeze_text_embed:
            self.text_embed.weight.requires_grad_(False)

    def forward(self, text_ids: Tensor, z_cls: Tensor, z_s: Tensor) -> Tensor:
        """text_ids: (B,T) long（T≥1）；z_cls (B,1,D)；z_s (B,k,D)，k∈[0,N]。
        返回 text 位置 logits (B,T,V)。"""
        B, T = text_ids.shape
        k = z_s.size(1)
        N = self.num_patches
        assert 0 <= k <= N, f"z_s 长度 k={k} 超出 [0,{N}]"
        assert T + N + 1 <= self.pos_embed.size(1), \
            f"文字过长: T={T} 超过 max_text_len={self.max_text_len}"
        z = torch.cat([z_cls, z_s], dim=1)                     # (B,k+1,D)
        tok = self.embed_proj(self.text_embed(text_ids))       # (B,T,D)
        x = torch.cat([z, tok], dim=1)                         # (B,k+1+T,D)
        M = N + 1 + T                                          # 完整长度
        mask_full = build_prefix_mask(M, 0, N + 1, device=x.device,
                                      tail_causal=True, z_see_tail=False)
        kept = torch.cat([
            torch.arange(0, k + 1, device=x.device),           # z_cls + z_s[:k]
            torch.arange(N + 1, M, device=x.device),           # 文字（绝对位置）
        ])
        self.attn_mask = mask_full.index_select(0, kept).index_select(1, kept)
        self.kept = kept                                   # 供自检/调试
        x = x + self.pos_embed[:, kept, :]                     # 绝对位置，不随 k 漂移
        for layer in self.layers:
            x = layer(x, src_mask=self.attn_mask)
        h = self.norm(x)
        return self.lm_head(h[:, k + 1:])                      # (B,T,V) 文字 logits


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
        # ── 文字自回归（可选，vocab_size>0 时启用 TextDecoder）──
        vocab_size: int = 0,
        qwen_hidden: Optional[int] = None,
        text_decoder_depth: int = 4,
        max_text_len: int = 512,
        freeze_text_embed: bool = True,
        pad_token_id: int = -100,
        text_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.dim = dim
        self.pad_token_id = pad_token_id
        self.text_loss_weight = text_loss_weight

        self.special_bank = SpecialTokenBank(num_patches=num_patches, dim=dim)
        self.re_encoder = ReEncoder(dim=dim, num_patches=num_patches,
                                    depth=reencoder_depth, heads=heads,
                                    mlp_ratio=mlp_ratio,
                                    causal_specials=causal_specials)
        self.decoder = FeatureDecoder(dim=dim, num_patches=num_patches,
                                      depth=decoder_depth, heads=heads,
                                      mlp_ratio=mlp_ratio)
        self.text_decoder = None
        if vocab_size > 0:
            self.text_decoder = TextDecoder(
                dim=dim, num_patches=num_patches, vocab_size=vocab_size,
                qwen_hidden=qwen_hidden, max_text_len=max_text_len,
                heads=heads, depth=text_decoder_depth, mlp_ratio=mlp_ratio,
                freeze_text_embed=freeze_text_embed)

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

    def init_text_from_qwen(self, qwen_emb: Tensor,
                            qwen_lm_head: Optional[Tensor] = None):
        """用 Qwen 预训练词表 warm-start TextDecoder（结构对齐: 词表+头）。

        qwen_emb: (vocab_size, qwen_hidden) 输入/输出共享 embedding 权重；
        qwen_lm_head: 可选 (vocab_size, qwen_hidden)——经 embed_proj 投影后
        暖启动 lm_head（投影近似，非精确拷贝）；None 则 lm_head 随机。
        冻结与否由构造参数 freeze_text_embed 决定（默认冻结 embedding）。
        """
        td = self.text_decoder
        assert td is not None, "TextDecoder 未配置（构造需 vocab_size>0）"
        w = qwen_emb.detach()
        assert w.shape == td.text_embed.weight.shape, \
            f"qwen_emb 形状 {tuple(w.shape)} != text_embed {tuple(td.text_embed.weight.shape)}"
        td.text_embed.weight.copy_(w.to(dtype=td.text_embed.weight.dtype,
                                        device=td.text_embed.weight.device))
        if qwen_lm_head is not None:
            h = qwen_lm_head.detach()
            proj_w = getattr(td.embed_proj, "weight", None)
            with torch.no_grad():
                if proj_w is not None:      # qwen_hidden→dim 投影: 低秩近似
                    td.lm_head.weight.copy_(
                        (h @ proj_w.T).to(dtype=td.lm_head.weight.dtype,
                                          device=td.lm_head.weight.device))
                else:                       # 同维直连: 精确拷贝
                    td.lm_head.weight.copy_(h.to(dtype=td.lm_head.weight.dtype,
                                                 device=td.lm_head.weight.device))
        n_tr = sum(1 for p in td.text_embed.parameters() if p.requires_grad)
        print(f"[init] TextDecoder text_embed warm-started from Qwen "
              f"(vocab={w.shape[0]}, hidden={w.shape[1]}), "
              f"embedding 可训参数组数={n_tr}")

    def _encode_z(self, x: Tensor):
        """pixel_values → (z_cls (B,1,D), z_s (B,N,D), patch (B,N,D))。"""
        B = x.shape[0]
        N = self.num_patches
        out = self.dinov2(x)
        feats = out.last_hidden_state                   # (B,257,D)
        cls = feats[:, 0]                               # (B,D)
        patch = feats[:, 1:]                            # (B,N,D)
        specials = self.special_bank(B, x.device)       # (B,N,D)
        enc_in = torch.cat([cls.unsqueeze(1), specials, patch], dim=1)  # (B,2N+1,D)
        z = self.re_encoder(enc_in)                     # (B,2N+1,D)
        return z[:, 0:1], z[:, 1:1 + N], patch          # z_cls, z_s, patch

    # ── forward（训练/推理同一路径，无分支）──
    def forward(self, pixel_values: Tensor,
                text_ids: Optional[Tensor] = None,
                z_keep: Optional[int] = None) -> dict:
        """pixel_values: (B,3,224,224) → dict{loss, recon, F_hat, ...}

        重建损失: L1(F_hat, patch)（全量 z_s 进 decoder，无选择无预算）。
        text_ids 传入时（(B,T) long, T≥2）追加仅文字自回归:
            TextDecoder: [z_cls; z_s(全量); 文字] → 错位 CE（位置 i 预测
            i+1，ignore_index=pad_token_id），loss = L1 + text_loss_weight*L_text。
        z_keep: 可选（仅文字分支），只喂前 z_keep 个 z_s 作为文字条件
        （"预算/渐进"场景，k∈[0,N]；重建分支仍用全量 z_s）。
        """
        x = pixel_values                                # (B,3,224,224)
        z_cls, z_s, patch = self._encode_z(x)           # (B,1,D) (B,N,D) (B,N,D)

        # ── FeatureDecoder: [z_cls; z_s(全量); patch×N] → F_hat ──
        F_hat = self.decoder(z_cls, z_s)                # (B,N,D)
        recon = F.l1_loss(F_hat, patch)                 # 重建损失

        out = {"loss": recon, "recon": recon, "F_hat": F_hat}
        if text_ids is not None:
            assert self.text_decoder is not None, \
                "text_ids 传入但 TextDecoder 未配置（构造需 vocab_size>0）"
            assert text_ids.size(1) >= 2, "text_ids 至少 2 个 token（错位需要）"
            z_s_txt = z_s[:, :z_keep] if z_keep is not None else z_s
            logits = self.text_decoder(text_ids, z_cls, z_s_txt)   # (B,T,V)
            V = logits.size(-1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = text_ids[..., 1:].contiguous()
            text_loss = F.cross_entropy(
                shift_logits.view(-1, V), shift_labels.view(-1),
                ignore_index=self.pad_token_id)
            out["text_loss"] = text_loss
            out["text_logits"] = logits
            out["loss"] = recon + self.text_loss_weight * text_loss
        return out

    @torch.no_grad()
    def generate_text(self, pixel_values: Tensor, prompt_ids: Tensor,
                      max_new_tokens: int = 128,
                      temperature: float = 0.0, top_p: float = 1.0,
                      z_prefix: Optional[int] = None) -> Tensor:
        """图像条件 + 文字前缀 → 自回归续写（每步一次前向，追加一个 token）。

        prompt_ids: (B,T0) long。返回完整 token 序列 (B, T0+max_new_tokens)，
        由调用方 tokenizer.decode。temperature=0 → 贪心；>0 → 采样（可配 top_p）。
        z_prefix: 可选，只喂前 k 个 z_s（k∈[0,N]）作为条件（"预算/渐进"场景）；
        None = 全量 z_s。
        """
        assert self.text_decoder is not None, \
            "generate_text 需要 TextDecoder（构造时 vocab_size>0）"
        B = pixel_values.shape[0]
        z_cls, z_s, _ = self._encode_z(pixel_values)
        if z_prefix is not None:
            z_s = z_s[:, :z_prefix]                     # 只喂 z_s 前缀
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.text_decoder(ids, z_cls, z_s)[:, -1, :]   # (B,V)
            if temperature > 0:
                logits = logits / temperature
                if top_p < 1.0:
                    sorted_l, idx = logits.sort(dim=-1, descending=True)
                    cum = sorted_l.softmax(-1).cumsum(-1)
                    keep = cum - sorted_l.softmax(-1) <= top_p
                    sorted_l = sorted_l.masked_fill(~keep, float('-inf'))
                    logits = torch.zeros_like(logits).scatter_(1, idx, sorted_l)
                nxt = torch.multinomial(logits.softmax(-1), 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


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

    # ── 3b. FeatureDecoder z_s 前缀(k<N): 掩码 z_end=k+1, 输出仍全量 N ──
    z_cls_t = torch.randn(2, 1, D)
    k = 5
    F_hat_p = model.decoder(z_cls_t, torch.randn(2, k, D))
    assert F_hat_p.shape == (2, N, D), F_hat_p.shape       # 前缀仍出全量 N
    dm_p = model.decoder.attn_mask                         # (N+k+1, N+k+1)
    assert dm_p.shape == (N + k + 1, N + k + 1)
    for i in range(k + 1):                                 # z 行 0..k: 前缀内因果
        assert not dm_p[i, :i + 1].any()
        assert dm_p[i, i + 1:k + 1].all()                  # 屏蔽前缀内后面的 z
        assert not dm_p[i, k + 1:].any()                   # 仍可见全部 patch 行
    assert not dm_p[k + 1:].any()                          # patch 行全局
    print(f"[ok] Decoder z_s 前缀: k={k}<N 掩码 z_end=k+1 正确, 输出全量 {N} patch")

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

    # ── 6. TextDecoder: 仅文字自回归（[z_cls; z_s; 文字×T]）──
    V = 128
    tmodel = SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                        reencoder_depth=2, decoder_depth=2,
                        vocab_size=V, qwen_hidden=D, max_text_len=64,
                        text_decoder_depth=2)
    tmodel.init_text_from_qwen(torch.randn(V, D), qwen_lm_head=torch.randn(V, D))
    tid = torch.randint(0, V, (2, 20))
    out_t = tmodel(x, text_ids=tid)
    assert out_t["text_logits"].shape == (2, 20, V), out_t["text_logits"].shape
    assert out_t["text_loss"].shape == () and out_t["loss"].shape == ()
    assert abs(out_t["loss"].item()
               - (out_t["recon"].item() + out_t["text_loss"].item())) < 1e-5
    # 掩码（全量 z_s）: z 行 = z 内部因果 + 不看文字；文字行 = 见全部 z + 文字≤i
    tm = tmodel.text_decoder.attn_mask                       # (N+1+T, N+1+T)
    assert tm.shape == (N + 1 + 20, N + 1 + 20)
    for i in range(N + 1):
        assert tm[i, i + 1:N + 1].all()                      # z 内部因果
        assert tm[i, N + 1:].all()                           # z 不看文字（防泄漏）
    for i in range(N + 1, N + 1 + 20):
        assert not tm[i, :i + 1].any()                       # 见 z + 文字≤i
        assert tm[i, i + 1:].all()                           # 屏蔽后面文字
    assert not tmodel.text_decoder.text_embed.weight.requires_grad  # 冻结
    out_t["text_loss"].backward()
    for name, p in [("text.embed_proj", tmodel.text_decoder.embed_proj.weight),
                    ("text.lm_head", tmodel.text_decoder.lm_head.weight),
                    ("text.layer0", tmodel.text_decoder.layers[0].linear1.weight),
                    ("text.pos_embed", tmodel.text_decoder.pos_embed)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} 收不到梯度"
    print(f"[ok] TextDecoder: 掩码(文字下三角+z因果+z不看文字)/冻结/多任务 loss/梯度 正确, "
          f"text_loss={out_t['text_loss'].item():.4f}")

    # ── 7. z_s 前缀输入（k<N）: 绝对位置稳定，掩码不含缺失 z 列 ──
    z_cls_p, z_s_p, _ = tmodel._encode_z(x)                  # (2,1,D) (2,N,D)
    k = 5
    out_p = tmodel.text_decoder(tid, z_cls_p, z_s_p[:, :k])  # 只喂前 5 个 z_s
    assert out_p.shape == (2, 20, V), out_p.shape
    tm_p = tmodel.text_decoder.attn_mask
    assert tm_p.shape == (k + 1 + 20, k + 1 + 20), tm_p.shape  # 缺失 z 列不存在
    for i in range(k + 1):                                   # z 行（前缀内）
        assert tm_p[i, i + 1:k + 1].all()                    # 前缀内因果
        assert tm_p[i, k + 1:].all()                         # 不看文字
    for i in range(k + 1, k + 1 + 20):                       # 文字行
        assert not tm_p[i, :i + 1].any()                     # 见 z_cls+z_s[:k]+文字≤i
        assert tm_p[i, i + 1:].all()
    # 文字绝对位置: kept = [z_cls; z_s[:k]; 文字 N+1..N+T]，不随 k 漂移
    kept_exp = torch.cat([torch.arange(0, k + 1), torch.arange(N + 1, N + 1 + 20)])
    assert torch.equal(tmodel.text_decoder.kept.cpu(), kept_exp), \
        f"kept={tmodel.text_decoder.kept.cpu().tolist()} != {kept_exp.tolist()}"
    out_t2 = tmodel(x, text_ids=tid, z_keep=k)             # forward 内 z_keep 切片
    assert out_t2["text_logits"].shape == (2, 20, V)
    print(f"[ok] TextDecoder 前缀输入: k={k}<N 形状/掩码/绝对位置/forward z_keep 正确")

    with torch.no_grad():
        gen = tmodel.generate_text(x, torch.randint(0, V, (2, 3)), max_new_tokens=5)
        gen_k = tmodel.generate_text(x, torch.randint(0, V, (2, 3)),
                                     max_new_tokens=5, z_prefix=3)
    assert gen.shape == (2, 8) and gen_k.shape == (2, 8), (gen.shape, gen_k.shape)
    print(f"[ok] generate_text: 全量/前缀(z_prefix=3) 自回归生成形状正确 "
          f"{tuple(gen.shape)} / {tuple(gen_k.shape)}")

    print("\nALL CHECKS PASSED")
