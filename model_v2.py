"""
SR-Diffusion Phase 1 v2 — 无预算简化版（YAGNI: 先不增实体）
=================================================================

⚠️ 项目目标（权威版 2026-08-28 修订 v2, 见 doc/2026-08-28/GOAL_compression_for_nlp.md）:
    本模型是 Phase 1 的**训练脚手架**——用像素重建当代理任务, 训练编码器
    （DINOv2 + ReEncoder + specials）的"token 压缩 + 联想"能力。训练完成后
    **冻结编码器**, 作为 model.py 的编码器接入 Qwen 做 NLP 解码（中文工地
    描述/隐患）。**验收标准是 Phase 2 文字生成质量**。
    像素重建 = **信息保持的直接探针**（语义是像素的函数: 能还原像素 ⇒ z
    携带整图信息 ⇒ 语义信息必然在）——重建质量决定 NLP 天花板, 与 NLP
    不构成对立。推论: 纹理级清晰度（块内高频）按文献 [3] 属"死信息", 不追;
    **Phase 1 中间验收 = K 压缩 × 重建质量**（K=32/64/128 下活信息=布局/
    物体/边界的保真, 压缩率 K 是练联想的真正杠杆）; t≥1 平台 = 联想能力
    已成立; 冗余（576 z_s 全是全局摘要副本）是真问题——2026-08-31 起
    specials 数 K 与 patch 数 N 解耦, 固定 K=128（默认, 可配）真正压缩。

沿革:
    v2 曾含"可学习前缀预算"机制（SelectHead 边界分布 + STE 门控 +
    率惩罚/目标铰链 + pad_token），统一 hard 后发现它是"先增的实体":
    与前向脱节、代码熵增、收益未验证。按 YAGNI 原则整块移除（git
    历史可找回），k 固定 = 全量 N——decoder 恒吃全部 z_s。
    后续若真需要预算（"少留也能重建"），再把 SelectHead/边界分布
    加回来。

    2026-08-31（用户需求, 两个修改）:
    ① specials 数 K 与 patch 数 N 解耦: 新增 num_specials（默认 128,
       可配）, SpecialTokenBank/ReEncoder/OutputQueryDecoder 全部按
       K 参数化——K≠N 时序列 = [cls(1); specials(K); patches(N)] =
       1+K+N token, decoder 的 KV 序列 S = 1+K（z_cls + K 个 z_s, 采样
       计划按 K 推导）, 查询基仍是 N 行（行 k↔patch k）。K=128/N=576
       即真压缩（2.6% 的 z_s 还原全部 patch）。
    ② 损失排除 t<loss_min_t 的早期采样步: 新增 loss_min_t（默认 5）,
       decode() 中 per_step 只对 t≥loss_min_t 的采样步求平均——排除
       {0,1,4} 这些"几乎无键"的粗重建步（t=0 只有 1 个键, 4 个键以内
       只能输出近乎同一向量, 是退化监督）。decoder 的 steps 不变
       （推理渐进曲线仍要全部步）, 只过滤损失; mask 为空时退化为
       全量平权并打印警告。

当前架构（无选择、无预算; K=num_specials 与 N=num_patches 解耦）:
    DINOv2 → cls + patch 特征 (B,1+N,D)
    ReEncoder:    [cls; specials(K); patches(N)] 因果 specials 块掩码
                    → z_cls, z_s (B,K,D)
    OutputQueryDecoder: [z_cls; z_s] = 时序序列 A(S=1+K,H)；在采样时刻
                    T_sub（自然数平方计划: 前面密后面疏, 按 K 自动适配）
                    上输出 (N,D) 矩阵 = 全部 patch 的预测（输出查询注意力,
                    查询基行 k 对应 patch k）；KV 因果（前缀）
                    → F_hat = 采样步平均 (B,N,D)
    L = mean_{t∈T_sub, t≥loss_min_t} L1(Y_t, patch)   ← 全覆盖损失,
        排除 t<loss_min_t 的"几乎无键"早期步（默认 loss_min_t=5）

register_specials=True（2026-08-28 新增, 修 F1/F2）:
    specials 不再由 ReEncoder 算, 而是作为额外 token 直接拼进 DINOv2 的
    输入序列 [cls; specials(K); patches(N)] (1+K+N token), 由 DINO 的
    24 层直接算出 z_s——register token 式（Darcet et al., "Vision
    Transformers Need Registers"）。special k 的输入仍共享 token+位置编码
    （SpecialTokenBank, 无 patch 内容）, 但深层网络直接做"内容路由",
    不再依赖 4 层 ReEncoder 在 1153 长序列上学路由——修 DIAGNOSIS_
    clarity.md 的 F1（special 无内容输入）与 F2（z_s 冗余全局摘要）。
    该模式下无 ReEncoder（省 51.6M 参数）。
    HF Dinov2Model 无 token 级注意力 mask API（见踩坑记录）, 故 DINO 内用
    全双向注意力（无掩码）——重建任务无时序因果需求, register 惯例亦然;
    解码器的 KV 因果仍提供渐进前缀语义。注意: 该模式让 z_s[k] 依赖全部
    patch（含 j>k）, "前缀稳定性"约束不再成立——渐进曲线语义由解码器
    kv_causal 提供, 与编码器无关。

块状注意力掩码:
    ReEncoder（causal_specials=True 默认）: [cls; specials(K); patches(N)]
        cls 全局；specials 因果链（special i 只见 specials≤i + 全部
        patches）；patches 全局。掩码（build_prefix_mask）在 forward
        内现算、按实际序列长度构建（牺牲一点速度，换可扩展性）。
    OutputQueryDecoder: [z_cls; z_s(K)] 为时序序列 A=(S=1+K,H)，KV 因果——
        每步 t 只见前缀 s≤t；每步由输出查询注意力产生 (N,D) 矩阵 =
        全部 patch 的预测（每步全覆盖, 见类文档）。
    时刻采样（显存优化）: 查询只对 T_sub 构造（默认自然数平方计划按 K
        推导, 见 square_step_schedule）——Q 从 (S·N,D) 降到 (|T|·N,D)。

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
    out = model(pixel_values)                 # {"loss","recon","F_hat","Y_pix","target_pix"}
    loss = out["loss"]; loss.backward()
    model.eval()                              # 推理同路径

    # 解耦参数: specials 数 K（默认 128, 与 patch 数 N 无关）与
    # 损失过滤阈值 loss_min_t（默认 5, 排除 t<5 的"几乎无键"早期步）:
    model = SRPhase1V2(dinov2=dinov2_model, num_patches=576, dim=1024,
                       num_specials=128, loss_min_t=5)
    # register 式（specials 合并进 DINO, 修 F1/F2）:
    model = SRPhase1V2(dinov2=dinov2_model, num_patches=256, dim=768,
                       register_specials=True)   # 无 ReEncoder

    自检: python model_v2.py（形状 K≠N 解耦 / 两处块掩码结构 / 梯度 /
          init_reencoder_from_dino / eval 同路径 / register 模式形状与梯度 /
          loss_min_t 过滤与退化）

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
    """K 个特殊 token（默认 128, 与 patch 数 N 解耦）: 共享可学习向量 +
    逐位置可学习位置编码。

    token (1,1,D) 对所有位置/所有图像完全相同；pos (1,K,D) 提供位置区分
    （K=num_specials）。与 patch 组合后输入编码器（ReEncoder 或 DINO
    register 序列），输出中特殊 token 位置的表示 z_s (B,K,D) 是 decoder
    的输入——图像原始 patch 编码不进入 decoder（只作为编码器输入参与聚合）。
    """

    def __init__(self, num_specials: int, dim: int):
        super().__init__()
        self.num_specials = num_specials
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)          # 共享向量
        self.pos = nn.Parameter(torch.randn(1, num_specials, dim) * 0.02)  # 位置编码

    def forward(self, B: int, device: torch.device) -> Tensor:
        return self.token.expand(B, self.num_specials, -1) + self.pos      # (B,K,D)


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
    """对 [cls; 特殊 token(K); patch(N)] 组合序列做自注意力（带块状掩码）。

    输出中特殊 token 位置的表示 z_s (B,K,D) 是 decoder 的输入；patch 位置
    的输出直接丢弃。输入始终是完整 1+K+N 序列（训练/推理全量计算）。
    K=num_specials 与 N=num_patches 解耦（默认 K=128, 生产配置 K≠N）;
    K=N 时退化为旧版 2N+1 语义。

    块状注意力掩码（causal_specials=True，默认）:
        cls(0)              → 全局（见所有人、被所有人见）
        specials(1..K)      → 只见 cls + specials≤i + 全部 patches（前缀链）
        patches(K+1..K+N)   → 全局（图像无时序，全双向）
    即 M[i,j]=1 除 special 行 i 的 special 列 j>i 之外——special p 的编码
    z_s[p] 不依赖任何"后面的 special"（前缀稳定性：直接路径上）。
    掩码在 forward 内现算（build_prefix_mask），按实际序列长度构建。

    注意: patches 仍会关注 specials（M 的 1_{patch×special} 块）——
    若将来要裁 encoder 输入省算力，需同时把 patch→special 关注也屏蔽，
    否则 patch 编码会经 specials 间接变化。

    Input:  x (B, 1+K+N, D) = [cls; special_1..K; patch_1..N]
    Output: z (B, 1+K+N, D)
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 num_specials: int = None, depth: int = 4,
                 heads: int = 8, mlp_ratio: float = 4.0,
                 causal_specials: bool = True):
        super().__init__()
        self.num_patches = num_patches
        self.num_specials = num_patches if num_specials is None else num_specials
        self.causal_specials = causal_specials
        L = 1 + self.num_specials + num_patches    # cls + K specials + N patches
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
        # z 区域 = specials 段: 下标 1..num_specials（z_end = K+1, 按实际 K 算）
        z_end = self.num_specials + 1 if self.causal_specials else 1
        self.attn_mask = build_prefix_mask(L, 1, z_end, device=x.device)
        x = x + self.pos_embed[:, :L, :]
        for layer in self.layers:
            x = layer(x, src_mask=self.attn_mask)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════
# square_step_schedule — 时刻采样计划（自然数平方数, 自动适配任意 N）
# ═══════════════════════════════════════════════════════════════

def square_step_schedule(max_prefix: int) -> list:
    """生成 decoder 的时刻采样计划 T_sub = {0} ∪ {k² ≤ max_prefix} ∪ {max_prefix}。

    采样时刻就是自然数平方数: 0, 1, 4, 9, 16, 25, …
    相邻间距是 (k+1)²−k² = 2k+1 的奇数: 1, 3, 5, 7, 9, …（线性递增）——
    所以"前面密、后面疏"，且比幂次计划（间距成倍拉开）温和得多。
    0 和 max_prefix 是额外补的: t=0 = 仅 z_cls 时刻; t=max_prefix = 全前缀
    （能力最强, KV 序列 S=1+K 的全部键）。
    计划完全由 max_prefix 推导，自动适配任意序列长度（K=128→13 步,
    N=256→17 步, N=512→24 步, …），改 K/N 无需动代码。

    注: 采样时刻 t 是 **KV 序列的前缀长度**（decoder 每步只见 [z_cls; z_s[:t]]）,
    因此 max_prefix 传 num_specials（K, 默认 128）而非 num_patches（N）——
    K=N 时二者一致（旧版行为不变）。

    例: square_step_schedule(256)
        → [0,1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256]
    """
    steps = {0}
    k = 1
    while k * k <= max_prefix:
        steps.add(k * k)
        k += 1
    steps.add(max_prefix)          # 最后一步（全前缀, 能力最强）总是保留
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

    K≠N 解耦（2026-08-31）: KV 序列长度 S = 1+K（z_cls + K 个 z_s,
    K=num_specials, 默认 128）；查询基 query_base 仍是 N=num_patches 行
    （行 k↔patch k）——K 只管"压缩了多少键", N 只管"输出多少 patch"。

    机制（一次前向，采样时刻 × 所有 patch 行完全并行）:
        A = [z_cls; z_s] + pos_embed      (B, S=1+K, D)  ← 上文的 A=(S,H)
        Q = W_q(A_t) + E                  (B, |T|, N, D)  E=查询基, N 行↔patch k
        Y = SDPA(Q, K=V=A, mask)          (B, |T|·N, D)   一次 matmul
        Y = norm(ffn(Y))                  (B, |T|, N, D)  采样时刻 t = 全部 patch
        F_hat = mean_t(Y)                 (B, N, D)      采样步集成

    时刻采样（显存优化, 默认开启）:
        查询只对 T_sub 构造——Q 从 (S·N,D) 降到 (|T|·N,D)，显存与算力
        同比例下降（N=256: 全量 257 步 → 平方计划 17 步, 约 15×）。
        · 默认计划 = square_step_schedule(num_specials)（自然数平方,
          前面密后面疏, 按 KV 长度 K 自动适配; K=N 时即旧版 N 计划）;
        · 传 steps= 可自定义采样时刻列表（如 [0, 64, 128, 256]）;
        · 传 steps=list(range(K+1)) 即退化为全量不采样。
        注意: 未采样时刻不参与损失, 也不出现在 F_hat 集成里。
        （2026-08-31: 采样步 t 是 KV 前缀长度, 上界 = K=num_specials——
        z_s 只有 K 个 token, t>K 无键可看, 因此计划与越界校验都按 K。）

    损失权重（已移除，2026-08-27 用户要求"去掉加权体系"）:
        density/uniform/capability 加权机制整块删除（YAGNI），全部采样步
        平权 —— loss = mean_t L1(Y_t, target)。git 历史可找回。
        （2026-08-31: SRPhase1V2.decode 在平权基础上再排除 t<loss_min_t
        的早期步——那是损失侧过滤, 与本解码器无关, steps 保持全量。）

    覆盖语义（每步全覆盖）:
        每个采样时刻 t 都预测全部 N 个 patch（查询基 E 的行 k 对应
        patch k, 对所有 t 相同——行数恒 = num_patches, 与 K 无关）;
        每个采样时刻都被监督还原全部 patch。

    KV 因果（kv_causal=True）:
        每步 t 只见前缀 s≤t。t 越小的键越少（t=0 只有 z_cls 一个键,
        N 行查询只能输出同一向量）, 重建越粗糙——"前缀越短越粗"是
        该设计的固有性质（渐进重建）。kv_causal=False 则每步见全部 z。
        K≠N 时键数上界是 K（全前缀 t=K 见全部 K+1 个键）, 语义不变。

    掩码约定: 统一用加法浮点掩码(-inf=屏蔽)。实测 torch 2.8:
        SDPA 的 bool 掩码 True=允许, 与 TransformerEncoderLayer 的
        True=屏蔽 相反（见文件头踩坑记录）。

    参考: 交叉注意力输出查询范式源自 Perceiver 家族（文件头 [1]
        Perceiver AR / Perceiver IO）; 可学习查询基 query_base 的"行 k↔
        patch k"设计同源于 BLIP-2 Q-Former 的可学习 query（[2]）。
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 num_specials: int = 256, mlp_ratio: float = 4.0,
                 kv_causal: bool = True,
                 steps: Optional[Sequence[int]] = None):
        super().__init__()
        self.num_patches = num_patches
        self.num_specials = num_specials
        self.kv_causal = kv_causal
        if steps is None:
            # 采样时刻是 KV 前缀长度, 默认计划按 K（z_s 个数）推导
            steps = square_step_schedule(num_specials)
        steps = sorted(set(int(s) for s in steps))
        # 上界是 K 而非 N: A = [z_cls; z_s] 只有 1+K 个 token, t>K 越界
        assert steps and all(0 <= s <= num_specials for s in steps), \
            f"steps 越界: {steps} (K={num_specials}, KV 序列长度 1+K)"
        self.steps = steps
        S = num_specials + 1                           # z_cls + K 个 z_s（KV 序列长度）
        self.query_base = nn.Parameter(torch.randn(num_patches, dim) * 0.02)  # N 行, 行 k↔patch k
        self.W_q = nn.Linear(dim, dim, bias=False)     # z_t → 查询偏置
        self.pos_embed = nn.Parameter(torch.randn(1, S, dim) * 0.02)
        self.ffn = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)),
                                 nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        B, N, D = z_s.shape[0], self.num_patches, z_s.shape[-1]
        K = self.num_specials
        A = torch.cat([z_cls, z_s], dim=1) + self.pos_embed      # (B,1+K,D) = (B,S,D)
        A_t = A[:, self.steps]                                   # (B,|T|,D) 采样时刻
        Q = (self.W_q(A_t).unsqueeze(2) + self.query_base) \
            .reshape(B, len(self.steps) * N, D)                  # 展平 (t,k), t∈T_sub
        keys = V = A
        mask = None
        if self.kv_causal:   # 行(t,k) 只允许 s≤t（前缀因果; -inf=屏蔽）
            tril = torch.tril(torch.ones(K + 1, K + 1, device=A.device)).bool()
            mask = torch.where(tril[self.steps], 0.0, float("-inf")) \
                .repeat_interleave(N, dim=0)                     # (|T|·N, K+1) = (|T|·N, S)
        self.attn_mask = mask                                    # 供自检
        Y = F.scaled_dot_product_attention(Q, keys, V, attn_mask=mask)  # (B,|T|·N,D)
        Y = self.norm(self.ffn(Y)).reshape(B, len(self.steps), N, D)  # (B,|T|,N,D)
        self.last_Y = Y                                          # 采样步全部 patch 预测
        return Y.mean(dim=1)                                     # (B,N,D) 采样步集成


# ═══════════════════════════════════════════════════════════════
# SRPhase1V2 — 主模型（自包含；无预算；K=num_specials 与 N 解耦）
# ═══════════════════════════════════════════════════════════════

class SRPhase1V2(nn.Module):
    """DINOv2(不冻结) → specials 编码 → OutputQueryDecoder → PixelHead。

    num_specials（K, 默认 128）= 特殊 token 数, 与 num_patches（N, patch
    数）解耦: K 决定编码器侧的"压缩键数"与 decoder 的 KV 长度 S=1+K,
    N 决定查询基行数与输出 patch 数。K=128/N=576 即 2.6% 键还原全部
    patch 的真压缩。
    loss_min_t（默认 5）= 损失过滤阈值: decode() 中只对 t≥loss_min_t 的
    采样步求平均（排除 {0,1,4} 这些"几乎无键"的粗重建步）, decoder 的
    steps 保持不变（推理渐进曲线要全部步）。
    """

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        num_specials: int = 128,
        dim: int = 768,
        reencoder_depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        causal_specials: bool = True,
        decoder_steps: Optional[Sequence[int]] = None,
        patch_px: int = 14 * 14 * 3,
        register_specials: bool = False,
        loss_min_t: int = 5,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.num_specials = num_specials
        self.dim = dim
        self.patch_px = patch_px
        self.register_specials = register_specials
        self.loss_min_t = loss_min_t

        self.special_bank = SpecialTokenBank(num_specials=num_specials, dim=dim)
        # register 式: specials 直接进 DINO 由深层网络算, 无 ReEncoder
        # （条件初始化, 避免 DDP find_unused_parameters=False 报未用参数）
        self.re_encoder = None
        if not register_specials:
            self.re_encoder = ReEncoder(dim=dim, num_patches=num_patches,
                                        num_specials=num_specials,
                                        depth=reencoder_depth, heads=heads,
                                        mlp_ratio=mlp_ratio,
                                        causal_specials=causal_specials)
        self.decoder = OutputQueryDecoder(dim=dim, num_patches=num_patches,
                                          num_specials=num_specials,
                                          mlp_ratio=mlp_ratio,
                                          steps=decoder_steps)
        self.pixel_head = PixelHead(dim=dim, patch_px=patch_px)

    def init_reencoder_from_dino(self, num_layers: int = 4):
        """用 DINO 编码器前 num_layers 层 warm-start ReEncoder（结构对齐
        nn.TransformerEncoderLayer ↔ HF DINO layer）。register 模式下无
        ReEncoder, 直接跳过。"""
        if self.register_specials:
            print("[init] register_specials 模式: 无 ReEncoder, 跳过 warm-start")
            return
        depth = min(num_layers, len(self.re_encoder.layers), len(self.dinov2.encoder.layer))
        dino_layers = self.dinov2.encoder.layer
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

    # ── encode: 输入 → 解码器输入 z_cls, z_s（按 register_specials 分派）──
    def encode(self, pixel_values: Tensor):
        """把输入编码成解码器输入 (z_cls, z_s)，训练/推理同一路径。

        register_specials=False: DINO → [cls; specials(K); patches(N)] → ReEncoder
        register_specials=True : specials 作为额外 token 拼进 DINO 输入序列,
            由 DINO 的 24 层直接算出 z_s（register token 式, Darcet et al.）——
            修 DIAGNOSIS_clarity.md F1（special 无 patch 内容输入）与 F2
            （z_s 全是冗余全局摘要）。无 ReEncoder。
        两种模式输出一致: z_cls (B,1,D), z_s (B,K,D), K=num_specials（默认 128）。
        """
        if self.register_specials:
            return self._encode_register(pixel_values)
        return self._encode_reencoder(pixel_values)

    def _encode_reencoder(self, pixel_values: Tensor):
        x = pixel_values                                # (B,3,H,W)
        out = self.dinov2(x)
        feats = out.last_hidden_state                   # (B,1+N,D)
        cls = feats[:, 0]                               # (B,D)
        patch_feat = feats[:, 1:]                       # (B,N,D) 仅作编码输入, 不作监督目标
        # ReEncoder: [cls; specials(K); patches(N)] → z_cls, z_s
        specials = self.special_bank(x.shape[0], x.device)      # (B,K,D)
        enc_in = torch.cat([cls.unsqueeze(1), specials, patch_feat], dim=1)  # (B,1+K+N,D)
        z = self.re_encoder(enc_in)                     # (B,1+K+N,D)
        return z[:, 0:1], z[:, 1:1 + self.num_specials]  # (B,1,D), (B,K,D)

    def _encode_register(self, pixel_values: Tensor):
        """register 式: specials 直接进 DINO 输入序列, 由 DINO 深层算 z_s。

        HF Dinov2Model 无 token 级注意力 mask API（见文件头踩坑记录）, 故
        DINO 内用**全双向注意力**（无掩码）——重建任务无时序因果需求,
        register token 惯例亦然; 解码器的 KV 因果仍提供渐进前缀语义。
        embeddings(pixel_values) 复用 HF 的 cls/patch 嵌入 + 位置编码
        （含其自动插值逻辑）, 与训练现状一致; specials 用 SpecialTokenBank
        （共享 token + 逐位置可学习 pos, K=num_specials 个）。
        """
        x = pixel_values                                # (B,3,H,W)
        emb = self.dinov2.embeddings(x)                 # (B,1+N,D) [cls; patches] + PE
        specials = self.special_bank(x.shape[0], x.device)   # (B,K,D) token+pos
        seq = torch.cat([emb[:, :1], specials, emb[:, 1:]], dim=1)   # (B,1+K+N,D)
        for layer in self.dinov2.encoder.layer:         # DINO 24 层（全双向）
            out = layer(seq)
            seq = out[0] if isinstance(out, (tuple, list)) else out
        seq = self.dinov2.layernorm(seq)                # (B,1+K+N,D)
        return seq[:, :1], seq[:, 1:1 + self.num_specials]  # (B,1,D), (B,K,D)

    # ── decode: 共享解码尾（Decoder → PixelHead → 像素损失）──
    def decode(self, z_cls: Tensor, z_s: Tensor, pixel_values: Tensor) -> dict:
        """解码器 + 像素头 + 像素 L1（两种模式共用）。

        z_cls: (B,1,D); z_s: (B,K,D)（K=num_specials, 与 N 解耦）。
        dict: {"loss", "recon", "F_hat"(像素 B,N,588), "Y_pix"(每采样步像素
        B,|T|,N,588), "target_pix"(B,N,588)} —— 训练取 loss; 推理取 F_hat /
        Y_pix / target_pix（全量 L1、渐进曲线、可视化同一路径）。
        loss = mean_{t∈T_sub, t≥loss_min_t} L1(Y_t_pix, target_pix):
        平权全覆盖, 但排除 t<loss_min_t 的"几乎无键"早期步（loss_min_t
        默认 5, 排除 {0,1,4}）; 全部步都 < loss_min_t 时退化为全量平权。
        """
        x = pixel_values
        B, C, H, W = x.shape
        N = self.num_patches
        # OutputQueryDecoder: 采样时刻上每步 (N,D) 全覆盖 → F_hat
        F_hat = self.decoder(z_cls, z_s)                # (B,N,D) 采样步平均(特征)
        Y = self.decoder.last_Y                         # (B,|T|,N,D) 每步全部 patch
        # 像素目标: (B,3,H,W) 归一化像素 → (B,N,588) patch
        # 注意布局: DINO 的 patch 顺序是 row-major (先 y 后 x), 这里保持一致
        target_pix = x.reshape(B, C, H // 14, 14, W // 14, 14) \
                      .permute(0, 2, 4, 1, 3, 5) \
                      .reshape(B, N, C * 14 * 14)       # (B,N,588) 归一化像素
        # PixelHead: 特征 → 像素, 每个采样步全覆盖
        Y_pix = self.pixel_head(Y)                      # (B,|T|,N,588)
        F_pix = self.pixel_head(F_hat)                  # (B,N,588) 采样步集成
        # 平权全覆盖损失: 每个采样时刻都还原全部 patch 像素
        per_step = F.l1_loss(Y_pix, target_pix.unsqueeze(1).expand_as(Y_pix),
                             reduction="none").mean(dim=(0, 2, 3))   # (|T|,)
        # 2026-08-31: 排除 t<loss_min_t 的"几乎无键"早期采样步（t=0 只有
        # z_cls 一个键, t∈{1,4} 键也很少, 只能输出近乎同一向量——是退化
        # 监督）。decoder 的 steps 不变（推理渐进曲线仍要全部步的 Y_pix）,
        # 这里只在损失侧按 t≥loss_min_t 过滤求平均。
        keep = torch.tensor([t >= self.loss_min_t for t in self.decoder.steps],
                            dtype=torch.bool, device=per_step.device)
        if keep.any():
            loss = per_step[keep].mean()                # 只对 t≥loss_min_t 的步
        else:
            # 所有采样步都 < loss_min_t（小自检配置可能如此）: 退化为全量
            loss = per_step.mean()
            print(f"[warn] loss_min_t={self.loss_min_t} 下无采样步满足 "
                  f"t>=loss_min_t (steps={self.decoder.steps}), "
                  f"退化为全量 per_step.mean()")
        recon = F.l1_loss(F_pix, target_pix)            # 集成重建（监控用, 归一化空间）
        return {"loss": loss, "recon": recon, "F_hat": F_pix,
                "Y_pix": Y_pix, "target_pix": target_pix}

    # ── forward（训练/推理同一路径，无分支）──
    def forward(self, pixel_values: Tensor) -> dict:
        """pixel_values: (B,3,H,W) 归一化像素 → dict{loss, recon, F_hat,
        Y_pix, target_pix}

        目标（2026-08-27 重大修复）: 监督**原始像素**而非 DINO patch 特征。
        之前监督特征目标退化（工地图特征空间近常数, 学质心即低 L1）;
        像素目标有真实空间结构, 强制模型保留空间信息。
        H,W 须为 14 的倍数（DINO patch=14），patch_px = 14*14*3。

        损失: 每个采样时刻 t∈T_sub 都监督还原全部 patch 像素 —— 平权
        全覆盖损失 L = mean_{t≥loss_min_t} L1(Y_t_pix, target_pix)
        （去掉加权体系; 2026-08-31 起排除 t<loss_min_t 的"几乎无键"
        早期步, 默认 loss_min_t=5; decoder 的 steps 仍全量输出）。
        F_hat = 采样步平均 → PixelHead → 像素; recon 仅作监控。
        """
        x = pixel_values                                # (B,3,H,W)
        B, C, H, W = x.shape
        N = self.num_patches
        assert W % 14 == 0 and H % 14 == 0, "输入须为 14 的倍数"
        assert (W // 14) * (H // 14) == N, \
            f"输入 {W}x{H} 产生 {(W//14)*(H//14)} patches, 但模型 num_patches={N}"
        z_cls, z_s = self.encode(x)
        return self.decode(z_cls, z_s, x)


# ═══════════════════════════════════════════════════════════════
# 自检（python model_v2.py）
#   1. 形状正确性（K=num_specials 与 N=num_patches 解耦, K≠N 小配置）
#   2. 两处块掩码结构（ReEncoder / OutputQueryDecoder KV 因果掩码）
#   3. OutputQueryDecoder: 平方计划按 K 推导 + 掩码尺寸 (|T|·N, K+1)
#   4. loss_min_t 过滤（t<loss_min_t 的步不参与 loss, 空 mask 退化）
#   5. 默认 num_specials=128（生产配置, K≠N 形状 + 掩码过滤生效）
#   6. 梯度流向（整模型可训: ReEncoder / Decoder / SpecialTokenBank）
#   7. eval 同路径
#   8. register 模式（K≠N 形状与梯度）; 9. register 默认 K=128 快速形状
#   10. 边界实验配置（K=64, decoder_steps=[32,64], register_specials）:
#       z_s 形状 / Y 两时刻 / loss 只在这两步平均（含小配置 K=8 验证）
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from types import SimpleNamespace

    torch.manual_seed(0)

    # ── 假的 DINOv2：结构对齐 HF Dinov2Model（供 init_reencoder_from_dino）──
    class _FakeDinoLayer(nn.Module):
        """HF Dinov2EncoderLayer 属性布局（供 init_reencoder_from_dino 拷贝）
        + 真实单头自注意力 forward（供 register 模式直接调用）。"""
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
            scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
            attn = F.softmax(scores, dim=-1)
            h = hidden_states + self.attention.output.dense(attn @ v)
            h = h + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(h))))
            return (h, None)

    class _FakeEmbeddings(nn.Module):
        """对齐 HF Dinov2Embeddings 的最小结构: conv patch + cls + PE(+插值)。"""
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
            patch = self.patch_embeddings(pixel_values)      # (B,D,h,w)
            patch = patch.flatten(2).transpose(1, 2)         # (B,N,D)
            cls = self.cls_token.expand(B, -1, -1)
            emb = torch.cat([cls, patch], dim=1)             # (B,1+N,D)
            pe = self.position_embeddings
            if emb.shape[1] != pe.shape[1]:                  # 对齐 HF 自动插值
                pe = F.interpolate(pe.transpose(1, 2).unsqueeze(0),
                                   size=(emb.shape[1],), mode="linear",
                                   align_corners=False).squeeze(0).transpose(1, 2)
            return self.dropout(emb + pe)

    class FakeDino(nn.Module):
        def __init__(self, dim: int = 64, n_layers: int = 4, num_patches: int = 16):
            super().__init__()
            self.embeddings = _FakeEmbeddings(dim, num_patches)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList(
                [_FakeDinoLayer(dim) for _ in range(n_layers)])
            self.layernorm = nn.LayerNorm(dim)
            self._np, self._dim = num_patches, dim
            self._feat = torch.randn(4, num_patches + 1, dim)   # 固定特征（B≤4）

        def forward(self, pixel_values: Tensor):
            B = pixel_values.shape[0]
            return SimpleNamespace(
                last_hidden_state=self._feat[:B].clone())

    # ── 小配置: N=16 patches, K=4 specials（K≠N 解耦验证）──
    N, K, D = 16, 4, 64
    dino = FakeDino(dim=D, num_patches=N)
    model = SRPhase1V2(dino, num_patches=N, dim=D, num_specials=K,
                       reencoder_depth=2)
    model.init_reencoder_from_dino(2)
    # 像素目标绑定输入尺寸: N=16 patches ⇒ 输入须为 4×4 patch = 56×56 (14 的倍数)
    x = torch.randn(2, 3, 56, 56)
    B, C, H, W = x.shape
    PATCH_PX = 14 * 14 * 3

    # ── 1. 形状（K≠N: z_s 是 K 维; Y/F_hat 仍以 N 为准; loss 标量）──
    out = model(x)
    assert out["F_hat"].shape == (2, N, PATCH_PX), out["F_hat"].shape
    assert out["loss"].shape == () and out["recon"].shape == ()
    z_cls, z_s = model.encode(x)
    assert z_cls.shape == (2, 1, D) and z_s.shape == (2, K, D), \
        (z_cls.shape, z_s.shape)
    assert model.special_bank.num_specials == K
    assert model.decoder.num_specials == K
    # 像素目标提取与模型内部一致
    target = x.reshape(B, C, H // 14, 14, W // 14, 14) \
               .permute(0, 2, 4, 1, 3, 5).reshape(B, N, PATCH_PX)
    assert torch.isclose(out["recon"],
                         F.l1_loss(out["F_hat"], target)), "recon 应为像素 L1"
    print(f"[ok] shapes (K={K}≠N={N}): F_hat{tuple(out['F_hat'].shape)} "
          f"(像素 {PATCH_PX}D), z_cls{tuple(z_cls.shape)}, "
          f"z_s{tuple(z_s.shape)}, loss={out['loss'].item():.4f}")

    # ── 2. ReEncoder 块掩码（布局 [cls(1); specials(K); patches(N)]）──
    am = model.re_encoder.attn_mask              # (1+K+N, 1+K+N) bool, True=屏蔽
    assert am.shape == (1 + K + N, 1 + K + N), am.shape
    assert not am[0].any()                       # cls 行无屏蔽（全局）
    assert not am[1 + K:].any()                  # patches 行无屏蔽（全双向）
    for i in range(1, K + 1):
        assert not am[i, :i + 1].any()           # special i 可见 cls + specials≤i
        assert am[i, i + 1:1 + K].all()          # 屏蔽后面的 special
        assert not am[i, 1 + K:].any()           # special i 可见全部 patches
    # 关掉 causal 时掩码全 False（回退全双向；掩码 forward 内现算，先跑一次）
    m_free = SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                        num_specials=K, causal_specials=False, reencoder_depth=2)
    _ = m_free.re_encoder(torch.randn(2, 1 + K + N, D))
    assert not m_free.re_encoder.attn_mask.any()
    print(f"[ok] ReEncoder block mask (1+{K}+{N}={1 + K + N} token): "
          f"causal_specials=True 结构正确, causal_specials=False 全开放回退")

    # ── 3. OutputQueryDecoder: 平方计划按 K 推导 + KV 因果掩码 (|T|·N, K+1) ──
    T_steps = model.decoder.steps
    assert T_steps == square_step_schedule(K) == [0, 1, 4], T_steps  # K=4 → [0,1,4]
    am = model.decoder.attn_mask                         # (|T|·N, K+1) float
    assert am is not None and am.shape == (len(T_steps) * N, K + 1), am.shape
    for ti, t in enumerate(T_steps):                     # 行(t,k): 只允许 s≤t
        row = am[ti * N]
        assert (row[:t + 1] == 0).all()
        assert (row[t + 1:] == float("-inf")).all()
    Y = model.decoder.last_Y                             # (B,|T|,N,D) 特征（N 为准）
    assert Y.shape == (2, len(T_steps), N, D), Y.shape
    Y_pix = model.pixel_head(Y)                          # (B,|T|,N,588) 像素
    assert Y_pix.shape == (2, len(T_steps), N, PATCH_PX)
    # 计划函数本身（按前缀长度推导, K=N 时即旧版 N 计划）: 256→17 步, 512→24 步
    assert square_step_schedule(256) == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81,
                                         100, 121, 144, 169, 196, 225, 256]
    assert len(square_step_schedule(512)) == 24
    print(f"[ok] OutputQueryDecoder: 平方计划按 K 推导 {len(T_steps)} 步 "
          f"{T_steps} + KV 因果掩码 {tuple(am.shape)} 正确")

    # ── 4. loss_min_t 过滤（2026-08-31 新增）──
    per = F.l1_loss(Y_pix, target.unsqueeze(1).expand_as(Y_pix),
                    reduction="none").mean(dim=(0, 2, 3))    # (|T|,) 每步
    # 4a. 全部步 < loss_min_t（默认 5, steps=[0,1,4]）→ 退化全量 + 打印警告
    assert torch.isclose(out["loss"], per.mean()), \
        "mask 为空应退化为全量 per_step.mean()"
    # 4b. loss_min_t=1: 排除 t=0, 只对 t∈{1,4} 求平均（对比手动计算）
    m1 = SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                    num_specials=K, loss_min_t=1, reencoder_depth=2)
    out1 = m1(x)
    per1 = F.l1_loss(out1["Y_pix"],
                     out1["target_pix"].unsqueeze(1).expand_as(out1["Y_pix"]),
                     reduction="none").mean(dim=(0, 2, 3))   # (|T|,)
    keep1 = torch.tensor([t >= 1 for t in m1.decoder.steps], dtype=torch.bool)
    assert keep1.tolist() == [False, True, True]
    assert torch.isclose(out1["loss"], per1[keep1].mean()), \
        "loss_min_t 应排除 t<loss_min_t 的采样步"
    print(f"[ok] loss_min_t: 全部步 < {model.loss_min_t} 时退化全量(带警告); "
          f"loss_min_t=1 时 loss=mean(per[t>=1]) 与手动一致")

    # ── 5. 默认 num_specials=128（生产配置 K≠N）: 形状 + 掩码过滤生效 ──
    dino_d = FakeDino(dim=D, num_patches=N)
    m_d = SRPhase1V2(dino_d, num_patches=N, dim=D, reencoder_depth=2)
    assert m_d.num_specials == 128 and m_d.loss_min_t == 5, \
        "默认值应为 num_specials=128, loss_min_t=5"
    out_d = m_d(x)
    z_cls_d, z_s_d = m_d.encode(x)
    assert z_cls_d.shape == (2, 1, D) and z_s_d.shape == (2, 128, D), z_s_d.shape
    Td = m_d.decoder.steps
    assert Td == square_step_schedule(128), Td               # 13 步, 含全前缀 128
    assert out_d["Y_pix"].shape == (2, len(Td), N, PATCH_PX)
    perd = F.l1_loss(out_d["Y_pix"],
                     out_d["target_pix"].unsqueeze(1).expand_as(out_d["Y_pix"]),
                     reduction="none").mean(dim=(0, 2, 3))   # (|Td|,)
    keepd = torch.tensor([t >= m_d.loss_min_t for t in Td], dtype=torch.bool)
    assert keepd.sum() == len(Td) - 3, "K=128 计划中 t<5 的步是 {0,1,4} 共 3 个"
    assert torch.isclose(out_d["loss"], perd[keepd].mean()), \
        "默认 K=128 配置下 loss 应排除 t<5 的步"
    print(f"[ok] 默认 K=128: z_s(2,128,D), 采样 {len(Td)} 步, "
          f"loss 排除 t<5 的 {int(len(Td) - keepd.sum())} 步")

    # ── 6. 梯度流向（整模型可训, 含 PixelHead）──
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

    # ── 7. 推理: 同一 forward（eval + no_grad）──
    model.eval()
    with torch.no_grad():
        out_e = model(x)
    assert out_e["F_hat"].shape == (2, N, PATCH_PX)
    print(f"[ok] eval 同路径: loss={out_e['loss'].item():.4f}")

    # ── 8. register 模式（K≠N: specials 进 DINO 序列 1+K+N token）──
    dino_r = FakeDino(dim=D, num_patches=N)
    m_reg = SRPhase1V2(dino_r, num_patches=N, dim=D, num_specials=K,
                       register_specials=True)
    assert m_reg.re_encoder is None, "register 模式不应有 ReEncoder"
    out_r = m_reg(x)
    assert out_r["F_hat"].shape == (2, N, PATCH_PX)
    assert out_r["Y_pix"].shape == (2, len(m_reg.decoder.steps), N, PATCH_PX)
    z_cls, z_s = m_reg.encode(x)
    assert z_cls.shape == (2, 1, D) and z_s.shape == (2, K, D), \
        (z_cls.shape, z_s.shape)
    # 梯度: DINO 嵌入(conv+PE+cls) / DINO 层 / special_bank / decoder / pixel_head
    out_r["loss"].backward()
    for name, p in [("dino.embeddings.patch_embeddings.weight",
                     dino_r.embeddings.patch_embeddings.weight),
                    ("dino.embeddings.position_embeddings",
                     dino_r.embeddings.position_embeddings),
                    ("dino.encoder.layer.0.mlp.fc1.weight",
                     dino_r.encoder.layer[0].mlp.fc1.weight),
                    ("special_bank.token", m_reg.special_bank.token),
                    ("decoder.query_base", m_reg.decoder.query_base),
                    ("pixel_head.proj", m_reg.pixel_head.proj.weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    m_reg.eval()
    with torch.no_grad():
        out_re = m_reg(x)
    assert out_re["F_hat"].shape == (2, N, PATCH_PX)
    print(f"[ok] register_specials (K={K}≠N={N}): specials 进 DINO 序列"
          f"({1 + K + N} token, 全双向) 无 ReEncoder; 形状/梯度(eval 同路径)正确")

    # ── 9. register 模式默认 K=128（生产配置, 快速形状）──
    dino_r2 = FakeDino(dim=D, num_patches=N)
    m_reg2 = SRPhase1V2(dino_r2, num_patches=N, dim=D, register_specials=True)
    z_cls, z_s = m_reg2.encode(x)
    assert z_cls.shape == (2, 1, D) and z_s.shape == (2, 128, D), z_s.shape
    out_r2 = m_reg2(x)
    assert out_r2["Y_pix"].shape == (2, len(m_reg2.decoder.steps), N, PATCH_PX)
    print(f"[ok] register_specials 默认 K=128: z_s(2,128,D), "
          f"Y_pix{tuple(out_r2['Y_pix'].shape)}")

    # ── 10. 边界实验配置（K=64, decoder_steps=[32,64], register_specials）──
    # 用户实验（2026-08-31）: 最大特殊 token 数 K=64, 只监督"前 32 / 前 64"
    # 两个前缀序列——decoder_steps=[32,64] 即 KV 前缀长度 32 和 64 两个
    # 采样时刻。压缩边界探索: 32-token 前缀 vs 64-token 全量前缀重建对比。
    dino_b = FakeDino(dim=D, num_patches=N)
    m_b = SRPhase1V2(dino_b, num_patches=N, dim=D, num_specials=64,
                     decoder_steps=[32, 64], register_specials=True,
                     loss_min_t=5)
    assert m_b.decoder.steps == [32, 64], m_b.decoder.steps
    z_cls_b, z_s_b = m_b.encode(x)
    assert z_cls_b.shape == (2, 1, D) and z_s_b.shape == (2, 64, D), \
        (z_cls_b.shape, z_s_b.shape)                # z_s (B,64,D): K=64
    out_b = m_b(x)
    assert out_b["Y_pix"].shape == (2, 2, N, PATCH_PX), out_b["Y_pix"].shape
    # loss 只在这两步算: per_step 只有 2 项, t=32/64 均 >= loss_min_t=5,
    # keep 全 True, 无退化警告
    perb = F.l1_loss(out_b["Y_pix"],
                     out_b["target_pix"].unsqueeze(1).expand_as(out_b["Y_pix"]),
                     reduction="none").mean(dim=(0, 2, 3))   # (2,)
    assert perb.shape == (2,), perb.shape
    keepb = torch.tensor([t >= m_b.loss_min_t for t in m_b.decoder.steps],
                         dtype=torch.bool)
    assert keepb.tolist() == [True, True], "K=64 边界配置两步都应满足 t>=5"
    assert torch.isclose(out_b["loss"], perb[keepb].mean()), \
        "边界配置 loss 应为 per_step 两步的平均（平权）"
    # 小配置 K=8, steps=[2,4]（非 register, 小步快验）: 同样两步采样,
    # loss_min_t=1 时两步均保留, 语义与 K=64 一致
    m_c = SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                     num_specials=8, decoder_steps=[2, 4], loss_min_t=1,
                     reencoder_depth=2)
    assert m_c.decoder.steps == [2, 4], m_c.decoder.steps
    out_c = m_c(x)
    per_c = F.l1_loss(out_c["Y_pix"],
                      out_c["target_pix"].unsqueeze(1).expand_as(out_c["Y_pix"]),
                      reduction="none").mean(dim=(0, 2, 3))   # (2,)
    keep_c = torch.tensor([t >= m_c.loss_min_t for t in m_c.decoder.steps],
                          dtype=torch.bool)
    assert keep_c.tolist() == [True, True]
    assert torch.isclose(out_c["loss"], per_c[keep_c].mean())
    print(f"[ok] 边界实验配置 K=64 steps=[32,64] (register): "
          f"z_s(2,64,D), Y_pix(2,2,{N},{PATCH_PX}), "
          f"loss=per_step 两步平均(无退化); 小配置 K=8 steps=[2,4] 同语义")

    print("\nALL CHECKS PASSED")
