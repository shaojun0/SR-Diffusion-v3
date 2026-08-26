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

    前缀课程（"预算/渐进"）: forward 传 z_keep=k（k∈[0,N]）即让**重建与
    文字两分支统一**只喂前 k 个 z_s（重建输出仍恒全量 N patch）；配合
    prefix_weight(k, N, ...) 给重建损失加权 w(k)（递减 ⇒ 梯度压力前置，
    见 DESIGN_prefix_weighting.md / MATH_mask_analysis.md §6）。
    **多 k 并行**: forward_prefix_set(x, ks=[...], text_ids=?, w_fn=?) 一次
    编码 + 对多个前缀 k 并行监督（重建/文字均对每个 k 前向后平均），
    一个训练步同时覆盖全量/大/中/小所有尺度（decoder 4 层小网络，
    成本可忽略）。
    自检: python model_v2.py（形状 / 两处块掩码结构 / 梯度 /
          init_reencoder_from_dino / eval 同路径 / TextDecoder /
          z_keep 统一前缀 / prefix_weight / sample_prefix_k /
          forward_prefix_set 多 k 并行）
"""

import random

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
# prefix_weight — 前缀重建损失权重 w(k)（随 k 递减，见 DESIGN §3 / MATH §6）
# ═══════════════════════════════════════════════════════════════

def prefix_weight(k: int, N: int, w_shape: str = "inv",
                  w_p: float = 1.0, w_floor: float = 0.05) -> float:
    """前缀长度 k 的重建损失权重 w(k)，要求 w_0 ≥ w_1 ≥ … ≥ w_N ≥ 0。

    w_shape:
        inv   : 1/(k+1)          （默认；MATH §6.4 梯度压力示例形状）
        power : (N/k)^w_p        （k=1 → N^p 最大，k=N → 1；w_p 控陡度）
        none  : 恒 1.0           （不加权，仅靠 k 采样频率的前置偏置 N−i+1）
    w_floor: 权重下限 max(w, w_floor)（防全量分支权重趋零；默认 0.05，
             见 DESIGN §4 选项 (c)）。
    k=0（仅 z_cls）为退化边界: inv → 1.0（最大），power 发散故返回 w_floor。

    用途: 训练时按前缀课程采样 k 后，重建损失取 w(k)·L1_k——梯度压力
    Σ_{j≥i} w_j 随 i 递减 ⇒ 越早的 z_s 压力越大 ⇒ 信息前置
    （正确论证见 MATH_mask_analysis.md §6.4，非草稿 §3 的 telescoping 恒等式）。
    """
    assert 0 <= k <= N, f"k={k} 超出 [0,{N}]"
    if w_shape == "inv":
        w = 1.0 / (k + 1)
    elif w_shape == "power":
        w = (N / max(k, 1)) ** w_p if k > 0 else w_floor   # k=0: 幂发散, 取地板
    elif w_shape == "none":
        w = 1.0
    else:
        raise ValueError(f"未知 w_shape={w_shape!r} (可选 inv|power|none)")
    return max(w, w_floor)


def sample_prefix_k(k_min: int, N: int, dist: str = "uniform",
                    rng: Optional[random.Random] = None) -> int:
    """按分布采样前缀长度 k ∈ [k_min, N]（prefix_curriculum 训练用）。

    uniform     : [k_min, N] 整数均匀。天然自带前置偏置——token i 出现的
                  前缀数 ∝ N−i+1（i=1→N, i=N→1，精确线性递减，
                  MATH §6.4），均匀采样本身即梯度压力前置。
    log_uniform : log 空间均匀 → 小 k 在整数轴上更密（DESIGN §4 选项 b）。
    """
    rng = rng or random
    if dist == "uniform":
        return rng.randint(k_min, N)               # 含两端 [k_min, N]
    if dist == "log_uniform":
        import math
        lo, hi = math.log(k_min), math.log(N)
        return max(k_min, min(N, int(round(math.exp(rng.uniform(lo, hi))))))
    raise ValueError(f"未知 dist={dist!r} (可选 uniform|log_uniform)")


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
# FeatureDecoderAllK — 全家桶解码器: 一次前向输出所有前缀 k 的重建
# （"k 轴并入输出位置"，用户选定的三维方案 C）
# ═══════════════════════════════════════════════════════════════

def build_allk_mask(num_patches: int, k_list: list,
                    device: torch.device = None) -> Tensor:
    """全家桶掩码: [z_cls; z_s(1..N); out 块×K（每块 N 个位置）]。

    True=屏蔽（torch 2.x 约定）。设计目标: out 块 k 的输出**只依赖**
    {z_cls, z_s[:k], 同块 out} —— "任意前缀重建"一次前向全家桶，无跨 k 泄漏:
        z_cls 行  : 只见自己（透传常量；若看任何 z_s 会被所有块共享 → 泄漏）
        z_s[i] 行 : 只见 z_cls + z_s[:i]（内部因果链），**不看任何 out**
                    （若看 out 块 k'，则块 k≥i 经 z_s[i] 间接拿到 k' 信息 → 泄漏）
        out 块 k  : 见 z_cls + z_s[:k] + 同块全部 out；屏蔽 z_s[k:] 与其他块
    """
    N = num_patches
    K = len(k_list)
    assert K > 0 and all(0 <= kk <= N for kk in k_list), \
        f"k_list={k_list} 须非空且 kk∈[0,{N}]"
    L = 1 + N + N * K
    m = torch.zeros(L, L, dtype=torch.bool, device=device)
    z_end = N + 1                 # z_s 区域 1..N
    out0 = N + 1
    m[0, 1:] = True               # z_cls 行: 只见自己
    for i in range(1, z_end):     # z_s 行: 内部因果 + 不看任何 out
        m[i, i + 1:z_end] = True
        m[i, out0:] = True
    for k, kk in enumerate(k_list):          # out 行: 块 k 见 z_cls+z_s[:kk]+同块
        b0 = out0 + k * N
        b1 = b0 + N
        for r in range(b0, b1):
            m[r, 1 + kk:z_end] = True        # 屏蔽 z_s[kk+1..N]
            m[r, out0:b0] = True             # 屏蔽前面块
            m[r, b1:] = True                 # 屏蔽后面块
    return m


class FeatureDecoderAllK(nn.Module):
    """全家桶解码器: 输入 [z_cls; z_s(全量); out 块×K] → (B,N,K,D)。

    F_hat[:, :, k] = "只用前 k_list[k] 个 z_s 的重建" —— k 是显式结构维度，
    一个前向同时给出所有 K 个前缀的重建，训练一步可监督全部（并行原生）。
    与单 k FeatureDecoder 的差异（防泄漏所必需）:
        · z_s 行不看输出（纯读出源；单 k 版 z_s 看 patch）;
        · z_cls 行只见自己（透传常量条件）;
        · 块内 out 保留全局双向（patch 间协作），块间全屏蔽。
    注意: 推理时 k' ∉ k_list 只能取最近邻（K 集受限，见 DESIGN §10）。
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 k_list: Optional[list] = None,
                 heads: int = 8, depth: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        assert k_list, "FeatureDecoderAllK 需要 k_list"
        self.num_patches = num_patches
        self.k_list = sorted(set(k_list))          # 去重排序，块序 = 此序
        self.K = len(self.k_list)
        L = 1 + num_patches + num_patches * self.K
        self.out_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # 共享输出 token
        self.pos_embed = nn.Parameter(torch.randn(1, L, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        B, N, D = z_s.shape
        assert N == self.num_patches, f"z_s={N} != num_patches={self.num_patches}"
        outs = self.out_token.expand(B, N * self.K, -1)   # (B, N*K, D)
        x = torch.cat([z_cls, z_s, outs], dim=1)          # (B, 1+N+NK, D)
        L = x.shape[1]
        self.attn_mask = build_allk_mask(N, self.k_list, device=x.device)
        x = x + self.pos_embed[:, :L, :]
        for layer in self.layers:
            x = layer(x, src_mask=self.attn_mask)
        x = self.norm(x)
        # 展平顺序 = [块0(N), 块1(N), ...]（k 慢变）→ reshape (B,K,N,D)
        # 再 permute 为 (B,N,K,D): F_hat[:,:,k] = 块 k。
        # ⚠️ 不能直接 reshape(B,N,K,D): 那是 n 慢变 k 快变（跨块交错采样），
        # 会把块边界打乱导致 F_hat[:,:,k] 不是"只用前 k_list[k] 个 z_s 的重建"。
        return x[:, N + 1:].reshape(B, self.K, N, D).permute(0, 2, 1, 3)


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
        k_list: Optional[list] = None,
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
        # 单 k（默认）: FeatureDecoder 一次前向一个前缀；全家桶: k_list 传入
        # 时用 FeatureDecoderAllK，一次前向输出 (B,N,K,D) 全部前缀重建。
        self.k_list = sorted(set(k_list)) if k_list else None
        if self.k_list:
            assert all(0 <= kk <= num_patches for kk in self.k_list), self.k_list
            self.decoder = FeatureDecoderAllK(
                dim=dim, num_patches=num_patches, k_list=self.k_list,
                heads=heads, depth=decoder_depth, mlp_ratio=mlp_ratio)
        else:
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

        重建损失: L1(F_hat, patch)。z_keep=None 时全量 z_s 进 decoder
        （无选择无预算，默认路径）；z_keep=k 时**重建与文字两分支统一**
        只喂前 k 个 z_s（前缀课程/预算场景，k∈[0,N]，FeatureDecoder 输出
        恒为全量 N 个 patch）。
        text_ids 传入时（(B,T) long, T≥2）追加仅文字自回归:
            TextDecoder: [z_cls; z_s(与重建同一前缀); 文字] → 错位 CE
            （位置 i 预测 i+1，ignore_index=pad_token_id），
            loss = L1 + text_loss_weight*L_text。
        """
        assert self.k_list is None, \
            "全家桶模式（构造传了 k_list）请用 forward_all_k()，forward() 仅单 k 模式"
        x = pixel_values                                # (B,3,224,224)
        z_cls, z_s, patch = self._encode_z(x)           # (B,1,D) (B,N,D) (B,N,D)
        z_s_in = z_s[:, :z_keep] if z_keep is not None else z_s   # 统一前缀

        # ── FeatureDecoder: [z_cls; z_s(前缀); patch×N] → F_hat（输出恒全量 N）──
        F_hat = self.decoder(z_cls, z_s_in)             # (B,N,D)
        recon = F.l1_loss(F_hat, patch)                 # 重建损失

        out = {"loss": recon, "recon": recon, "F_hat": F_hat}
        if text_ids is not None:
            assert self.text_decoder is not None, \
                "text_ids 传入但 TextDecoder 未配置（构造需 vocab_size>0）"
            assert text_ids.size(1) >= 2, "text_ids 至少 2 个 token（错位需要）"
            logits = self.text_decoder(text_ids, z_cls, z_s_in)    # (B,T,V)
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

    def forward_all_k(self, pixel_values: Tensor,
                      text_ids: Optional[Tensor] = None,
                      w_fn=None) -> dict:
        """全家桶（三维方案 C）: 一次前向输出**所有**前缀 k 的重建并监督。

        仅构造传 k_list 时可用。重建损失 = Σ_k w(k)·L1(F_hat[:,:,k], patch)/K
        —— 一个训练步同时监督全部 K 个前缀（"任意前缀重建"的原生并行，
        k 是显式结构维度，非循环展开）。
        text_ids 传入时: 文字 CE 对每个 k 也并行计算后平均（z_s[:k] 切片
        喂 TextDecoder，文字分支同样覆盖全部前缀；K 次小前向成本可忽略）。
        w_fn: k → 重建权重（默认 1/(k+1)，递减 ⇒ 梯度压力前置）。
        返回 {"loss","recon","F_hat"(B,N,K,D),"k_list"}（+text_loss）。
        """
        assert self.k_list, "forward_all_k 需要构造时传 k_list（全家桶模式）"
        z_cls, z_s, patch = self._encode_z(pixel_values)
        F_hat = self.decoder(z_cls, z_s)                 # (B,N,K,D)
        K = self.decoder.K
        if w_fn is None:
            w_fn = lambda k: 1.0 / (k + 1)
        l1_per_k = [F.l1_loss(F_hat[:, :, k], patch) for k in range(K)]
        recon = sum(w_fn(self.k_list[k]) * l1_per_k[k]
                    for k in range(K)) / K
        out = {"loss": recon, "recon": recon, "F_hat": F_hat,
               "l1_per_k": l1_per_k, "k_list": list(self.k_list)}
        if text_ids is not None:
            assert self.text_decoder is not None, \
                "text_ids 传入但 TextDecoder 未配置（构造需 vocab_size>0）"
            assert text_ids.size(1) >= 2, "text_ids 至少 2 个 token（错位需要）"
            V = self.text_decoder.lm_head.out_features
            tloss = 0.0
            for kk in self.k_list:
                logits = self.text_decoder(text_ids, z_cls, z_s[:, :kk])
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = text_ids[..., 1:].contiguous()
                tloss = tloss + F.cross_entropy(
                    shift_logits.view(-1, V), shift_labels.view(-1),
                    ignore_index=self.pad_token_id)
            tloss = tloss / K
            out["text_loss"] = tloss
            out["loss"] = recon + self.text_loss_weight * tloss
        return out

    def forward_prefix_set(self, pixel_values: Tensor,
                           ks: list,
                           text_ids: Optional[Tensor] = None,
                           w_fn=None) -> dict:
        """一次编码 + 多个前缀 k 的**并行监督**（前缀课程"多 k 并行"模式）。

        ks: 前缀长度列表（每步监督这些 k 的重建；可含 N=全量）。
        w_fn: k → 重建权重（默认 1/(k+1)，递减 ⇒ 梯度压力前置）。
        重建损失 = Σ_k w(k)·L1_k / |ks| —— 一个训练步同时覆盖
        大/中/小/全量所有尺度的前缀（用户要求"所有可能性且可并行"）;
        编码器仅前向一次（z_s 因果链 ⇒ z_s[:k] 是自足前缀，直接切片）。
        text_ids 传入时: 文字 CE 对**每个 k** 也并行计算后平均（文字分支
        同样覆盖所有前缀；TextDecoder 成本相对 DINO 编码器可忽略）。
        返回 {"loss", "recon", "text_loss"?}；loss = recon + text_weight·L_text。
        """
        assert ks and all(0 <= k <= self.num_patches for k in ks), \
            f"ks={ks} 须非空且 k∈[0,{self.num_patches}]"
        z_cls, z_s, patch = self._encode_z(pixel_values)
        if w_fn is None:
            w_fn = lambda k: 1.0 / (k + 1)
        K = len(ks)
        recon = sum(w_fn(k) * F.l1_loss(self.decoder(z_cls, z_s[:, :k]), patch)
                    for k in ks) / K
        out = {"loss": recon, "recon": recon}
        if text_ids is not None:
            assert self.text_decoder is not None, \
                "text_ids 传入但 TextDecoder 未配置（构造需 vocab_size>0）"
            assert text_ids.size(1) >= 2, "text_ids 至少 2 个 token（错位需要）"
            V = self.text_decoder.lm_head.out_features
            tloss = 0.0
            for k in ks:
                logits = self.text_decoder(text_ids, z_cls, z_s[:, :k])
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = text_ids[..., 1:].contiguous()
                tloss = tloss + F.cross_entropy(
                    shift_logits.view(-1, V), shift_labels.view(-1),
                    ignore_index=self.pad_token_id)
            tloss = tloss / K
            out["text_loss"] = tloss
            out["loss"] = recon + self.text_loss_weight * tloss
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
    z_cls_p, z_s_p, patch_p = tmodel._encode_z(x)            # (2,1,D) (2,N,D)
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
    # z_keep 现在统一作用于重建+文字两分支: recon 应为"前缀重建"（L1 与
    # 直接调 decoder(z_cls, z_s[:, :k]) 一致），输出仍全量 N patch
    F_hat_ref = tmodel.decoder(z_cls_p, z_s_p[:, :k])
    recon_ref = F.l1_loss(F_hat_ref, patch_p)
    assert abs(out_t2["recon"].item() - recon_ref.item()) < 1e-6, \
        (out_t2["recon"].item(), recon_ref.item())
    assert out_t2["F_hat"].shape == (2, N, D)
    print(f"[ok] TextDecoder 前缀输入: k={k}<N 形状/掩码/绝对位置/forward "
          f"z_keep(重建+文字统一) 正确, recon={out_t2['recon'].item():.6f}")

    # ── 8. prefix_weight: 递减 / 地板 / 三种形状 / 边界 ──
    Nw = 576
    ws = [prefix_weight(kk, Nw, "inv", w_floor=0.0) for kk in range(Nw + 1)]
    assert all(ws[i] >= ws[i + 1] for i in range(Nw)), "inv 必须非增"
    assert ws[0] == 1.0 and abs(ws[Nw] - 1.0 / (Nw + 1)) < 1e-12
    assert prefix_weight(1, Nw) > prefix_weight(Nw, Nw)      # 递减
    assert prefix_weight(Nw, Nw, w_floor=0.05) == 0.05       # 地板生效
    assert prefix_weight(10, Nw, "power", 1.0) == Nw / 10.0  # power
    assert prefix_weight(10, Nw, "none") == 1.0              # 恒 1
    assert prefix_weight(0, Nw, "power") == 0.05             # k=0 幂发散→地板
    wp = [prefix_weight(kk, Nw, "power", 0.5) for kk in range(1, Nw + 1)]
    assert all(wp[i] >= wp[i + 1] for i in range(Nw - 1)), "power 必须非增"
    print(f"[ok] prefix_weight: inv 非增(0→1.0, N→1/(N+1)) / power / none / "
          f"地板 / k=0 边界 正确")

    # ── 9. sample_prefix_k: 边界与分布特性 ──
    km = 8
    rng = random.Random(0)
    for dist in ("uniform", "log_uniform"):
        ks = [sample_prefix_k(km, Nw, dist, rng) for _ in range(20000)]
        assert all(km <= k <= Nw for k in ks), (dist, min(ks), max(ks))
    mu_u = sum(sample_prefix_k(km, Nw, "uniform", rng) for _ in range(20000)) / 20000
    mu_l = sum(sample_prefix_k(km, Nw, "log_uniform", rng) for _ in range(20000)) / 20000
    assert abs(mu_u - (km + Nw) / 2) < 5.0, f"uniform 均值 {mu_u:.1f} 偏离理论"
    assert mu_l < mu_u, f"log_uniform 应偏小 k: {mu_l:.1f} vs {mu_u:.1f}"
    print(f"[ok] sample_prefix_k: 边界正确, uniform≈{(km + Nw) / 2:.0f}, "
          f"log_uniform 偏小 k ({mu_l:.0f} < {mu_u:.0f})")

    # ── 10. forward_prefix_set: 多 k 并行 ≈ 逐 k 独立 forward 的平均 ──
    ks_test = [N, 5, 1]                            # 全量 + 中 + 小
    wf = lambda k: prefix_weight(k, N, "inv")      # 与 train 默认一致
    out_ps = tmodel.forward_prefix_set(x, ks=ks_test, w_fn=wf)
    ref_r = sum(wf(k) * tmodel(x, z_keep=k)["recon"] for k in ks_test) / len(ks_test)
    assert abs(out_ps["recon"].item() - ref_r.item()) < 1e-6, \
        (out_ps["recon"].item(), ref_r.item())
    assert out_ps["loss"].shape == () and out_ps["recon"].shape == ()
    out_ps_t = tmodel.forward_prefix_set(x, text_ids=tid, ks=ks_test, w_fn=wf)
    ref_t = sum(tmodel(x, text_ids=tid, z_keep=k)["text_loss"]
                for k in ks_test) / len(ks_test)
    assert abs(out_ps_t["text_loss"].item() - ref_t.item()) < 1e-6, \
        (out_ps_t["text_loss"].item(), ref_t.item())
    assert abs(out_ps_t["loss"].item()
               - (out_ps_t["recon"].item() + out_ps_t["text_loss"].item())) < 1e-5
    out_ps_t["loss"].backward()                    # 梯度经全部 ks 路径
    assert tmodel.decoder.layers[0].linear1.weight.grad is not None and \
        tmodel.text_decoder.layers[0].linear1.weight.grad is not None
    print(f"[ok] forward_prefix_set: 多 k={ks_test} 并行损失 = 逐 k 平均 "
          f"(recon 差<1e-6, text 差<1e-6), 梯度覆盖全部 ks 路径")

    # ── 11. 全家桶 FeatureDecoderAllK: 掩码结构 + 无泄漏（数值验证）──
    k_list_t = [16, 4, 1]                        # N=16, 全量+中+小
    am = SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                    reencoder_depth=2, decoder_depth=2, k_list=k_list_t)
    out_a = am.forward_all_k(x)
    assert out_a["F_hat"].shape == (2, N, 3, D), out_a["F_hat"].shape
    # forward() 在全家桶模式应拒绝（防误用）
    try:
        am(x)
        raise AssertionError("全家桶模式 forward 应拒绝")
    except AssertionError:
        pass
    # 掩码结构（True=屏蔽）
    amk = am.decoder.attn_mask
    L_a = 1 + N + N * 3
    assert amk.shape == (L_a, L_a)
    assert not amk[0, 0] and amk[0, 1:].all()                # z_cls 只见自己
    for i in range(1, N + 1):                                # z_s 行
        assert not amk[i, :i + 1].any()                      # 见 z_cls+z_s[:i]
        assert amk[i, i + 1:N + 1].all()                     # 屏蔽后面 z
        assert amk[i, N + 1:].all()                          # 不看任何 out
    out0 = N + 1
    kl = am.decoder.k_list                       # 排序去重后的块序 [1,4,16]
    assert kl == [1, 4, 16]
    for k, kk in enumerate(kl):                  # out 块 k
        b0 = out0 + k * N
        for r in range(b0, b0 + N):
            assert not amk[r, 0].any()                       # 见 z_cls
            assert not amk[r, 1:1 + kk].any()                # 见 z_s[:kk]
            assert amk[r, 1 + kk:N + 1].all()                # 屏蔽 z_s[kk+1..N]
            assert not amk[r, b0:b0 + N].any()               # 同块全允许
            assert amk[r, out0:b0].all() and amk[r, b0 + N:].all()  # 其他块屏蔽
    # 无泄漏数值验证: 扰动 z_s[:, kk+1:] 不得改变块 k 的输出。
    # ⚠️ 扰动必须各向异性（随机噪声）: LayerNorm 对均匀偏移/缩放不变
    # （LN(a·x+b)=LN(x)），+10 均匀扰动会被归一化完全抵消（坑，勿踩）。
    # 注意力屏蔽精确 ⇒ 噪声扰动下块 k 输出应逐位不变（<1e-6）。
    with torch.no_grad():
        z_c, z_s_a, _ = am._encode_z(x)
        F0 = am.decoder(z_c, z_s_a)                          # (2,N,3,D)
        noise = 3.0 * torch.randn_like(z_s_a)
        for k, kk in enumerate(kl):
            z_b = z_s_a.clone(); z_b[:, kk + 1:] += noise[:, kk + 1:]
            F1 = am.decoder(z_c, z_b)
            d_out = (F0[:, :, k] - F1[:, :, k]).abs().max().item()
            assert d_out < 1e-6, f"块 k={kk} 泄漏: 扰动 z_s[>k] 改变输出 {d_out}"
            z_b2 = z_s_a.clone(); z_b2[:, :kk] += noise[:, :kk]
            F2 = am.decoder(z_c, z_b2)
            d_in = (F0[:, :, k] - F2[:, :, k]).abs().max().item()
            assert d_in > 1e-2, f"块 k={kk} 应对 z_s[:k] 敏感, got {d_in}"
    print(f"[ok] FeatureDecoderAllK: 掩码块结构正确, 无泄漏(扰动 z_s[>k] 差<1e-6), "
          f"对 z_s[:k] 敏感, forward() 全家桶拒绝")

    # ── 11b. 全家桶 + 文字: 损失构成 / 梯度 / k_list 含 0 ──
    am_t = SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                      reencoder_depth=2, decoder_depth=2, vocab_size=V,
                      qwen_hidden=D, max_text_len=64, text_decoder_depth=2,
                      k_list=[16, 0])
    out_at = am_t.forward_all_k(x, text_ids=tid,
                                w_fn=lambda k: prefix_weight(k, N, "inv"))
    assert out_at["text_loss"].shape == () and out_at["recon"].shape == ()
    assert abs(out_at["loss"].item()
               - (out_at["recon"].item() + out_at["text_loss"].item())) < 1e-5
    out_at["loss"].backward()
    assert am_t.decoder.layers[0].linear1.weight.grad is not None and \
        am_t.text_decoder.layers[0].linear1.weight.grad is not None
    assert am_t.decoder.k_list == [0, 16]                    # 排序去重
    print(f"[ok] 全家桶+文字: k_list={am_t.decoder.k_list} 损失构成/梯度 正确, "
          f"recon={out_at['recon'].item():.6f} text={out_at['text_loss'].item():.6f}")

    with torch.no_grad():
        gen = tmodel.generate_text(x, torch.randint(0, V, (2, 3)), max_new_tokens=5)
        gen_k = tmodel.generate_text(x, torch.randint(0, V, (2, 3)),
                                     max_new_tokens=5, z_prefix=3)
    assert gen.shape == (2, 8) and gen_k.shape == (2, 8), (gen.shape, gen_k.shape)
    print(f"[ok] generate_text: 全量/前缀(z_prefix=3) 自回归生成形状正确 "
          f"{tuple(gen.shape)} / {tuple(gen_k.shape)}")

    print("\nALL CHECKS PASSED")
