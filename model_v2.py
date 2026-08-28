"""
SR-Diffusion Phase 1 v2 — 无预算简化版（YAGNI: 先不增实体）
=================================================================

⚠️ 项目目标（权威版 2026-08-28, 见 doc/2026-08-28/GOAL_compression_for_nlp.md）:
    本模型是 Phase 1 的**训练脚手架**——用像素重建当代理任务, 训练编码器
    （DINOv2 + ReEncoder + specials）的"token 压缩 + 联想"能力。训练完成后
    **冻结编码器**, 作为 model.py 的编码器接入 Qwen 做 NLP 解码（中文工地
    描述/隐患）。**验收标准是 Phase 2 文字生成质量, 不是像素 L1**。
    推论: 重建"糊"（L1=23.4, 边缘≈1/3）主要是块内高频纹理, 与语义无关,
    不追清晰度; t≥1 平台 = 联想能力已成立; 冗余（576 z_s 全是全局摘要副本）
    才是真问题, 压缩率 K 是练联想的真正杠杆。

沿革:
    v2 曾含"可学习前缀预算"机制（SelectHead 边界分布 + STE 门控 +
    率惩罚/目标铰链 + pad_token），统一 hard 后发现它是"先增的实体":
    与前向脱节、代码熵增、收益未验证。按 YAGNI 原则整块移除（git
    历史可找回），k 固定 = 全量 N——decoder 恒吃全部 z_s。
    后续若真需要预算（"少留也能重建"），再把 SelectHead/边界分布
    加回来。

当前架构（无选择、无预算）:
    DINOv2 → cls + patch 特征 (B,257,D)
    ReEncoder:    [cls; specials; patches] 因果 specials 块掩码 → z_cls, z_s
    OutputQueryDecoder: [z_cls; z_s] = 时序序列 A(S,H)；在采样时刻
                    T_sub（自然数平方计划: 前面密后面疏, 自动适配任意 N）
                    上输出 (N,D) 矩阵 = 全部 patch 的预测（输出查询注意力,
                    查询基行 k 对应 patch k）；KV 因果（前缀）
                    → F_hat = 采样步平均 (B,N,D)
    L = Σ_t w_t·L1(Y_t, patch) / Σ w_t   ← 全覆盖损失（默认密度补偿权重,
        即对全时刻无偏; w_t 可换 uniform / capability）

块状注意力掩码:
    ReEncoder（causal_specials=True 默认）: [cls; specials; patches]
        cls 全局；specials 因果链（special i 只见 specials≤i + 全部
        patches）；patches 全局。掩码（build_prefix_mask）在 forward
        内现算、按实际序列长度构建（牺牲一点速度，换可扩展性）。
    OutputQueryDecoder: [z_cls; z_s] 为时序序列 A=(S,H)，KV 因果——
        每步 t 只见前缀 s≤t；每步由输出查询注意力产生 (N,D) 矩阵 =
        全部 patch 的预测（每步全覆盖, 见类文档）。
    时刻采样（显存优化）: 查询只对 T_sub 构造（默认自然数平方计划,
        见 square_step_schedule）——Q 从 (S·N,D) 降到 (|T|·N,D)。

踩坑记录（重要）:
    torch 2.x 的 bool 注意力掩码约定是 True=屏蔽（_canonical_mask:
    masked_fill_(mask, -inf)），与直觉相反。初版写成 True=允许导致
    输出不依赖输入、梯度为零（自检抓到），已按 True=屏蔽 实现。
    注意该约定只对 nn.TransformerEncoderLayer/MultiheadAttention 成立；
    F.scaled_dot_product_attention 的 bool 掩码实测 True=允许（torch
    2.8 本机验证: mask=[True,False] 只 attend 第一个 key）。OutputQuery
    Decoder 因此统一用加法浮点掩码(-inf=屏蔽)规避歧义。

用法
----
    model = SRPhase1V2(dinov2=dinov2_model, num_patches=256, dim=768)
    out = model(pixel_values)                 # {"loss","recon","F_hat"}
    loss = out["loss"]; loss.backward()
    model.eval()                              # 推理同路径

    自检: python model_v2.py（形状 / 两处块掩码结构 / 梯度 /
          init_reencoder_from_dino / eval 同路径）

参考资料（防止遗忘参考来源）:
    [1] Hawthorne, C., Jaegle, A., Cangea, C., Borgeaud, S., Nash, C.,
        Malinowski, M., Dieleman, S., Vinyals, O., Botvinick, M.,
        Simon, I., Sheahan, H., Zeghidour, N., Alayrac, J.-B.,
        Carreira, J., & Engel, J. (2022).
        General-purpose, long-context autoregressive modeling with
        Perceiver AR. ICML 2022, PMLR 162, 8535-8558.
        解码器机制来源: 交叉注意力输出查询 + 因果掩码（Perceiver 家族;
        见 OutputQueryDecoder 文档）。
        https://mlanthology.org/icml/2022/hawthorne2022icml-generalpurpose/
    [2] Li, J., Li, D., Savarese, S., & Hoi, S. C. H. (2023).
        BLIP-2: Bootstrapping language-image pre-training with frozen
        image encoders and large language models. ICML 2023.
        可学习查询基（query_base）与 Q-Former 的可学习 query 设计同源;
        冻结视觉编码器 + 可训练查询桥接的思路。
        https://arxiv.org/abs/2301.12597
    [3] Fan, Y., Tong, J., Zhao, A., & Shen, X. (2026).
        What do visual tokens really encode? Uncovering sparsity and
        redundancy in multimodal large language models. arXiv:2603.00510.
        视觉 token 稀疏性/冗余分析——token 压缩与 specials 设计的背景。
        https://arxiv.org/abs/2603.00510
    [4] Apedo, Y., Poreba, M., Szczepanski, M., & Bouchafa, S. (2026).
        Beyond attention scores: SVD-based vision token pruning for
        efficient vision-language models (SVD-Prune). arXiv:2604.11530.
        视觉 token 剪枝——与已移除的"可学习前缀预算"机制相关的文献。
        https://arxiv.org/abs/2604.11530
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Sequence


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
    用于 ReEncoder（z=specials 1..N，尾部=patches）与旧版 FeatureDecoder
    （已被 OutputQueryDecoder 取代），传不同 z_start/z_end 即可。

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
# square_step_schedule — 时刻采样计划（自然数平方数, 自动适配任意 N）
# ═══════════════════════════════════════════════════════════════

def square_step_schedule(num_patches: int) -> list:
    """生成 decoder 的时刻采样计划 T_sub = {0} ∪ {k² ≤ N} ∪ {N}。

    采样时刻就是自然数平方数: 0, 1, 4, 9, 16, 25, …
    相邻间距是 (k+1)²−k² = 2k+1 的奇数: 1, 3, 5, 7, 9, …（线性递增）——
    所以"前面密、后面疏"，且比幂次计划（间距成倍拉开）温和得多。
    0 和 N 是额外补的: t=0 = 仅 z_cls 时刻; t=N = 全前缀（能力最强）。
    计划完全由 N 推导，自动适配任意序列长度（N=256→17 步,
    N=512→24 步, …），改 N 无需动代码。

    例: square_step_schedule(256)
        → [0,1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256]
    """
    steps = {0}
    k = 1
    while k * k <= num_patches:
        steps.add(k * k)
        k += 1
    steps.add(num_patches)          # 最后一步（全前缀, 能力最强）总是保留
    return sorted(steps)


# ═══════════════════════════════════════════════════════════════
# PixelHead — 特征 → 像素 patch 解码头（2026-08-27 目标改为像素）
# ═══════════════════════════════════════════════════════════════

class PixelHead(nn.Module):
    """把每 patch 特征 (B,N,D) 解码回像素 patch (B,N,14*14*3)。

    背景（重大 bug 修复）: 之前监督目标是 DINO patch 特征 —— 工地图
    的特征在空间上近常数（每图跨位置 std≈5e-5），模型学"输出质心"即
    达低 L1，是假收敛（像素解码 L1=64.9≈平均色）。正确目标是**原始
    像素** pixel_values：有真实空间结构，"输出质心"的损失压力会迫使
    模型保留空间信息。上一个智能体已实证真实 DINO 特征可线性解码回
    像素（L1=9.8），信息在编码器里存在，只需可学习解码头。

    输出不加激活: 像素已按 DINO_MEAN/STD 归一化（范围 ≈[-2,2]），
    L1 直接监督归一化空间，评估时再反归一化。
    """

    def __init__(self, dim: int, patch_px: int = 14 * 14 * 3):
        super().__init__()
        self.patch_px = patch_px
        self.proj = nn.Linear(dim, patch_px)   # 特征 → 588 维像素 patch

    def forward(self, feat: Tensor) -> Tensor:
        """feat: (..., N, D) → (..., N, patch_px)"""
        return self.proj(feat)


# ═══════════════════════════════════════════════════════════════
# OutputQueryDecoder — 输出查询注意力解码器（采样时刻上输出全部 patch）
# ═══════════════════════════════════════════════════════════════

class OutputQueryDecoder(nn.Module):
    """把 [z_cls; z_s] 当作时序序列 A=(S,H)，在采样时刻 T_sub 上每个
    时刻输出一个 (N,D) 矩阵 = 全部 patch 的预测，再沿时刻集成得到
    F_hat (B,N,D)。

    机制（一次前向，采样时刻 × 所有 patch 行完全并行）:
        A = [z_cls; z_s] + pos_embed      (B, S, D)     ← 上文的 A=(S,H)
        Q = W_q(A_t) + E                  (B, |T|, N, D)  E=查询基, 行 k↔patch k
        Y = SDPA(Q, K=V=A, mask)          (B, |T|·N, D)   一次 matmul
        Y = norm(ffn(Y))                  (B, |T|, N, D)  采样时刻 t = 全部 patch
        F_hat = mean_t(Y)                 (B, N, D)      采样步集成

    时刻采样（显存优化, 默认开启）:
        查询只对 T_sub 构造——Q 从 (S·N,D) 降到 (|T|·N,D)，显存与算力
        同比例下降（N=256: 全量 257 步 → 平方计划 17 步, 约 15×）。
        · 默认计划 = square_step_schedule(N)（自然数平方, 前面密后面疏,
          自动适配任意 N）;
        · 传 steps= 可自定义采样时刻列表（如 [0, 64, 128, 256]）;
        · 传 steps=list(range(N+1)) 即退化为全量不采样。
        注意: 未采样时刻不参与损失, 也不出现在 F_hat 集成里。

    损失权重（已移除，2026-08-27 用户要求"去掉加权体系"）:
        density/uniform/capability 加权机制整块删除（YAGNI），全部采样步
        平权 —— loss = mean_t L1(Y_t, target)。git 历史可找回。

    覆盖语义（每步全覆盖）:
        每个采样时刻 t 都预测全部 N 个 patch（查询基 E 的行 k 对应
        patch k, 对所有 t 相同）; 每个采样时刻都被监督还原全部 patch。

    KV 因果（kv_causal=True）:
        每步 t 只见前缀 s≤t。t 越小的键越少（t=0 只有 z_cls 一个键,
        N 行查询只能输出同一向量）, 重建越粗糙——"前缀越短越粗"是
        该设计的固有性质（渐进重建）。kv_causal=False 则每步见全部 z。

    掩码约定: 统一用加法浮点掩码(-inf=屏蔽)。实测 torch 2.8:
        SDPA 的 bool 掩码 True=允许, 与 TransformerEncoderLayer 的
        True=屏蔽 相反（见文件头踩坑记录）。

    参考: 交叉注意力输出查询范式源自 Perceiver 家族（文件头 [1]
        Perceiver AR / Perceiver IO）; 可学习查询基 query_base 的"行 k↔
        patch k"设计同源于 BLIP-2 Q-Former 的可学习 query（[2]）。
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 mlp_ratio: float = 4.0, kv_causal: bool = True,
                 steps: Optional[Sequence[int]] = None):
        super().__init__()
        self.num_patches = num_patches
        self.kv_causal = kv_causal
        if steps is None:
            steps = square_step_schedule(num_patches)
        steps = sorted(set(int(s) for s in steps))
        assert steps and all(0 <= s <= num_patches for s in steps), \
            f"steps 越界: {steps} (N={num_patches})"
        self.steps = steps
        S = num_patches + 1                                # z_cls + N z_s
        self.query_base = nn.Parameter(torch.randn(num_patches, dim) * 0.02)  # 行 k↔patch k
        self.W_q = nn.Linear(dim, dim, bias=False)         # z_t → 查询偏置
        self.pos_embed = nn.Parameter(torch.randn(1, S, dim) * 0.02)
        self.ffn = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)),
                                 nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        B, N, D = z_s.shape[0], self.num_patches, z_s.shape[-1]
        A = torch.cat([z_cls, z_s], dim=1) + self.pos_embed      # (B,S,D)
        A_t = A[:, self.steps]                                   # (B,|T|,D) 采样时刻
        Q = (self.W_q(A_t).unsqueeze(2) + self.query_base) \
            .reshape(B, len(self.steps) * N, D)                  # 展平 (t,k), t∈T_sub
        K = V = A
        mask = None
        if self.kv_causal:   # 行(t,k) 只允许 s≤t（前缀因果; -inf=屏蔽）
            tril = torch.tril(torch.ones(N + 1, N + 1, device=A.device)).bool()
            mask = torch.where(tril[self.steps], 0.0, float("-inf")) \
                .repeat_interleave(N, dim=0)                     # (|T|·N, S)
        self.attn_mask = mask                                    # 供自检
        Y = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)  # (B,|T|·N,D)
        Y = self.norm(self.ffn(Y)).reshape(B, len(self.steps), N, D)  # (B,|T|,N,D)
        self.last_Y = Y                                          # 采样步全部 patch 预测
        return Y.mean(dim=1)                                     # (B,N,D) 采样步集成


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
        heads: int = 8,
        mlp_ratio: float = 4.0,
        causal_specials: bool = True,
        decoder_steps: Optional[Sequence[int]] = None,
        patch_px: int = 14 * 14 * 3,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.dim = dim
        self.patch_px = patch_px

        self.special_bank = SpecialTokenBank(num_patches=num_patches, dim=dim)
        self.re_encoder = ReEncoder(dim=dim, num_patches=num_patches,
                                    depth=reencoder_depth, heads=heads,
                                    mlp_ratio=mlp_ratio,
                                    causal_specials=causal_specials)
        self.decoder = OutputQueryDecoder(dim=dim, num_patches=num_patches,
                                          mlp_ratio=mlp_ratio,
                                          steps=decoder_steps)
        self.pixel_head = PixelHead(dim=dim, patch_px=patch_px)

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
        """pixel_values: (B,3,H,W) 归一化像素 → dict{loss, recon, F_hat}

        目标（2026-08-27 重大修复）: 监督**原始像素**而非 DINO patch 特征。
        之前监督特征目标退化（工地图特征空间近常数, 学质心即低 L1）;
        像素目标有真实空间结构, 强制模型保留空间信息。
        H,W 须为 14 的倍数（DINO patch=14），patch_px = 14*14*3。

        损失: 每个采样时刻 t∈T_sub 都监督还原全部 patch 像素 —— 平权
        全覆盖损失 L = mean_t L1(Y_t_pix, target_pix)（去掉加权体系）。
        F_hat = 采样步平均 → PixelHead → 像素; recon 仅作监控。
        """
        x = pixel_values                                # (B,3,H,W)
        B, C, H, W = x.shape
        N = self.num_patches
        assert W % 14 == 0 and H % 14 == 0, "输入须为 14 的倍数"
        assert (W // 14) * (H // 14) == N, \
            f"输入 {W}x{H} 产生 {(W//14)*(H//14)} patches, 但模型 num_patches={N}"

        out = self.dinov2(x)
        feats = out.last_hidden_state                   # (B,257,D)
        cls = feats[:, 0]                               # (B,D)
        patch_feat = feats[:, 1:]                       # (B,N,D) 仅作编码输入, 不作监督目标

        # ── ReEncoder: [cls; specials; patches] → z_cls, z_s ──
        specials = self.special_bank(B, x.device)       # (B,N,D)
        enc_in = torch.cat([cls.unsqueeze(1), specials, patch_feat], dim=1)  # (B,2N+1,D)
        z = self.re_encoder(enc_in)                     # (B,2N+1,D)
        z_cls = z[:, 0:1]                               # (B,1,D)
        z_s = z[:, 1:1 + N]                             # (B,N,D)

        # ── OutputQueryDecoder: 采样时刻上每步 (N,D) 全覆盖 → F_hat ──
        F_hat = self.decoder(z_cls, z_s)                # (B,N,D) 采样步平均(特征)
        Y = self.decoder.last_Y                         # (B,|T|,N,D) 每步全部 patch

        # ── 像素目标: (B,3,H,W) 归一化像素 → (B,N,588) patch ──
        # 注意布局: DINO 的 patch 顺序是 row-major (先 y 后 x), 这里保持一致
        target_pix = x.reshape(B, C, H // 14, 14, W // 14, 14) \
                      .permute(0, 2, 4, 1, 3, 5) \
                      .reshape(B, N, C * 14 * 14)       # (B,N,588) 归一化像素

        # ── PixelHead: 特征 → 像素, 每个采样步全覆盖 ──
        Y_pix = self.pixel_head(Y)                      # (B,|T|,N,588)
        F_pix = self.pixel_head(F_hat)                  # (B,N,588) 采样步集成
        # 平权全覆盖损失: 每个采样时刻都还原全部 patch 像素
        per_step = F.l1_loss(Y_pix, target_pix.unsqueeze(1).expand_as(Y_pix),
                             reduction="none").mean(dim=(0, 2, 3))   # (|T|,)
        loss = per_step.mean()                          # 平权（去掉加权体系）
        recon = F.l1_loss(F_pix, target_pix)            # 集成重建（监控用, 归一化空间）
        return {"loss": loss, "recon": recon, "F_hat": F_pix}


# ═══════════════════════════════════════════════════════════════
# 自检（python model_v2.py）
#   1. 形状正确性
#   2. 两处块掩码结构（ReEncoder / OutputQueryDecoder KV 因果掩码）
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
                       reencoder_depth=2)
    model.init_reencoder_from_dino(2)
    # 像素目标绑定输入尺寸: N=16 patches ⇒ 输入须为 4×4 patch = 56×56 (14 的倍数)
    x = torch.randn(2, 3, 56, 56)
    B, C, H, W = x.shape

    # ── 1. 形状: 目标=像素, F_hat 是像素 patch (B,N,588) ──
    PATCH_PX = 14 * 14 * 3
    out = model(x)
    assert out["F_hat"].shape == (2, N, PATCH_PX), out["F_hat"].shape
    assert out["loss"].shape == () and out["recon"].shape == ()
    # 像素目标提取与模型内部一致
    target = x.reshape(B, C, H // 14, 14, W // 14, 14) \
               .permute(0, 2, 4, 1, 3, 5).reshape(B, N, PATCH_PX)
    assert torch.isclose(out["recon"],
                         F.l1_loss(out["F_hat"], target)), "recon 应为像素 L1"
    print(f"[ok] shapes: F_hat{tuple(out['F_hat'].shape)} (像素 {PATCH_PX}D) "
          f"loss={out['loss'].item():.4f}")

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
                        causal_specials=False, reencoder_depth=2)
    _ = m_free.re_encoder(torch.randn(2, 2 * N + 1, D))
    assert not m_free.re_encoder.attn_mask.any()
    print(f"[ok] ReEncoder block mask: causal_specials=True 结构正确, "
          f"causal_specials=False 全开放回退")

    # ── 3. OutputQueryDecoder: 平方采样计划 + KV 因果掩码 + 平权全覆盖损失 ──
    T_steps = model.decoder.steps
    assert T_steps == square_step_schedule(N), T_steps       # N=16 → [0,1,4,9,16]
    am = model.decoder.attn_mask                             # (|T|·N, N+1) float
    assert am is not None and am.shape == (len(T_steps) * N, N + 1), am.shape
    for ti, t in enumerate(T_steps):                         # 行(t,k): 只允许 s≤t
        row = am[ti * N]
        assert (row[:t + 1] == 0).all()
        assert (row[t + 1:] == float("-inf")).all()
    Y = model.decoder.last_Y                                 # (B,|T|,N,D) 特征
    assert Y.shape == (2, len(T_steps), N, D)
    Y_pix = model.pixel_head(Y)                              # (B,|T|,N,588) 像素
    assert Y_pix.shape == (2, len(T_steps), N, PATCH_PX)
    per = F.l1_loss(Y_pix, target.unsqueeze(1).expand_as(Y_pix),
                    reduction="none").mean(dim=(0, 2, 3))    # (|T|,) 每步
    assert torch.isclose(out["loss"], per.mean()), "loss 应为平权全覆盖像素 L1"
    # 计划自动适配任意 N（可扩展性）: 256→17 步, 512→24 步
    assert square_step_schedule(256) == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81,
                                         100, 121, 144, 169, 196, 225, 256]
    assert len(square_step_schedule(512)) == 24
    print(f"[ok] OutputQueryDecoder: 平方采样 {len(T_steps)} 步 + KV 因果掩码 "
          f"+ 平权全覆盖像素损失正确")

    # ── 4. 梯度流向（整模型可训, 含 PixelHead）──
    out["loss"].backward()
    for name, p in [("re_encoder.0", model.re_encoder.layers[0].linear1.weight),
                    ("decoder.W_q", model.decoder.W_q.weight),
                    ("decoder.query_base", model.decoder.query_base),
                    ("special_bank.pos", model.special_bank.pos),
                    ("pixel_head.proj", model.pixel_head.proj.weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    print(f"[ok] 梯度: ReEncoder/OutputQueryDecoder(W_q,query_base)/"
          f"SpecialTokenBank/PixelHead 全部可训 "
          f"(|grad|={model.re_encoder.layers[0].linear1.weight.grad.abs().sum().item():.4f})")

    # ── 5. 推理: 同一 forward（eval + no_grad）──
    model.eval()
    with torch.no_grad():
        out_e = model(x)
    assert out_e["F_hat"].shape == (2, N, PATCH_PX)
    print(f"[ok] eval 同路径: loss={out_e['loss'].item():.4f}")

    print("\nALL CHECKS PASSED")
