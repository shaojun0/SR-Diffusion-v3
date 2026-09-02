"""
SR-Diffusion Phase 1 v2 — 训练脚手架（register 式, 无 ReEncoder）
=================================================================

项目目标（权威版见 doc/2026-08-28/GOAL_compression_for_nlp.md）:
    用像素重建当代理任务, 训练编码器（DINOv2 + register specials）的
    "token 压缩 + 联想"能力; 训练完成后冻结编码器接入 Qwen 做 NLP 解码
    （验收标准 = Phase 2 文字生成质量）。像素重建 = 信息保持的直接探针
    （能还原像素 ⇒ z 携带整图信息 ⇒ 语义信息必然在）。纹理级高频按文献
    [3] 属"死信息"不追; 压缩率 K 是练联想的真正杠杆（当前 k=N 全量零
    压缩, 尚未验证）。

沿革: v2 曾含 ReEncoder 路径与"可学习前缀预算"机制, 均已按 YAGNI 整块
    移除（git 历史可找回）; register 式（2026-08-28 起）为唯一路径。

架构（register 式, 无选择、无预算）:
    specials 作为额外 token 直接拼进 DINOv2 输入序列 [cls; specials; patches]
    (2N+1 token), 由 DINO 24 层直接算出 z_s（register token 式, Darcet et
    al.）——深层网络做"内容路由", 修 F1（special 无内容输入）与 F2（z_s
    冗余全局摘要）。DINO 内全双向注意力（HF 无 token 级 mask API）;
    z_s[k] 依赖全部 patch（含 j>k）, "前缀稳定性"不成立——渐进语义由
    解码器分块掩码提供。无 ReEncoder（省 51.6M 参数）。
    OutputQueryDecoder: [z_cls; z_s] 为时序序列; 在采样时刻 T_sub = 各 z_s
                    分块起点（平方数 1,4,9,…,⌊√N⌋², 自动适配任意 N）上
                    输出 (N,D) = 全部 patch 的预测（输出查询注意力, 查询基
                    行 k ↔ patch k）; 分块掩码: 每步只 attend 自己的 z_s 块
                    → F_hat = Σ_t Y_t（第 n 步结果 = 前 n 步之和）
    L = mean_n L1(cumsum_n(Y_t), patch)   ← 每步累积结果平权全覆盖损失
        （梯度按步解耦: carry 整体 detach + 自己的预测——每步恰收 1 份梯度,
        避免"t=0 收 |T| 份梯度"的三角失衡; 见 SRPhase1V2.decode）

踩坑记录（重要）:
    torch 2.x 的 bool 注意力掩码约定是 True=屏蔽（nn.TransformerEncoder
    Layer / MultiheadAttention / TransformerDecoderLayer）, 与直觉相反;
    F.scaled_dot_product_attention 的 bool 掩码实测 True=允许（torch 2.8
    本机验证）。掩码统一用加法浮点(-inf=屏蔽)规避歧义。

用法
----
    model = SRPhase1V2(dinov2=dinov2_model, num_patches=256, dim=768)
    out = model(pixel_values)                 # {"loss","recon","F_hat","Y_pix","target_pix"}
    loss = out["loss"]; loss.backward()
    model.eval()                              # 推理同路径

    自检: python model_v2.py

参考资料:
    [1] Hawthorne et al. (ICML 2022), Perceiver AR——输出查询 + 因果掩码解码器
        https://mlanthology.org/icml/2022/hawthorne2022icml-generalpurpose/
    [2] Li et al. (ICML 2023), BLIP-2——可学习查询基 / Q-Former 查询桥接
        https://arxiv.org/abs/2301.12597
    [3] Fan et al. (2026)——视觉 token 稀疏性/冗余分析（"死信息"依据）
        https://arxiv.org/abs/2603.00510
    [4] Apedo et al. (2026), SVD-Prune——视觉 token 剪枝（与已移除的预算机制相关）
        https://arxiv.org/abs/2604.11530
    [5] Vahdat & Kautz (NeurIPS 2020), NVAE——隐变量容量取舍对照
        https://proceedings.neurips.cc/paper/2020/hash/e3b21256183cf7c2c7a66be163579d37-Abstract.html
    [6] Gao & Shou (2025), D-AR——顺序扩散 tokenizer;"token 顺序定义渐进细化"参考
        https://arxiv.org/abs/2505.23660
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Sequence


# ═══ SpecialTokenBank — 特殊 token 池（输入相同, 仅位置编码不同）═══

class SpecialTokenBank(nn.Module):
    """每个 patch 位置一个特殊 token：共享可学习向量 + 逐位置可学习位置编码。

    token (1,1,D) 对所有位置/所有图像相同; pos (1,N,D) 提供位置区分。
    特殊 token 位置的表示 z_s 是 decoder 输入; 图像 patch 编码不进 decoder
    （只作为编码器输入参与聚合）。
    """

    def __init__(self, num_patches: int, dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)          # 共享向量
        self.pos = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)  # 位置编码

    def forward(self, B: int, device: torch.device) -> Tensor:
        return self.token.expand(B, self.num_patches, -1) + self.pos      # (B,N,D)


# ═══ build_prefix_mask — 块状前缀掩码（仅旧诊断脚本 gen_mask_gif.py 引用）═══

def build_prefix_mask(seq_len: int, z_start: int, z_end: int,
                      device: torch.device = None) -> Tensor:
    """块状前缀注意力掩码（torch bool, True=屏蔽）。

    布局: [z 区域 (z_start..z_end-1) | 尾部 (z_end..seq_len-1)]
        · z 行 i: 屏蔽 z 列 (i+1..z_end-1)（因果链）, 可看尾部（全局）
        · 尾部行: 全开放（全局）
    """
    m = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(z_start, z_end):
        m[i, i + 1:z_end] = True          # 屏蔽 z 内部"后面的"
    return m


# ═══ square_block_starts — 分块采样计划（块起点 = 平方数, 自动适配任意 N）═══

def square_block_starts(num_patches: int) -> list:
    """decoder 采样时刻 T_sub = {k² | 1 ≤ k² ≤ N}（块起点 = 平方数）。

    块 k = [k², min((k+1)²-1, N)]; 步数 = ⌊√N⌋, 自动适配任意 N
    （N=256→16 步, N=576→24 步）。例: square_block_starts(12) → [1,4,9]。
    """
    K = math.isqrt(num_patches)
    return [k * k for k in range(1, K + 1)]


# ═══ build_block_mask — 分块注意力掩码（每步只见自己的 z_s 块）═══

def build_block_mask(num_patches: int, steps: Sequence[int],
                     device: torch.device = None) -> Tensor:
    """分块注意力掩码（torch float, -inf=屏蔽, 0=允许）：每步只见自己的 z_s 块。

    块号 k = ⌊√t⌋, 块 k = [k², min((k+1)²-1, N)]（平方数边界铺满 1..N）。
    **第一个步特殊**: 能看到自己块之前的所有元素（含位置 0 = z_cls）——
    避免 skip_steps/max_steps 切片丢弃的前段信息完全不被使用; 其余步只
    允许自己的块, 位置 0 屏蔽。

    返回 (|T|·N, N+1)：每个采样步的 N 行 patch 查询共享同一掩码行。
    步值须 ∈ [0, N]（__init__ 已断言）; t=0 无块（仅显式传 0 时出现,
    默认计划不含 0）。
    """
    S = num_patches + 1                     # z_cls + N 个 z_s
    rows = []
    for i, t in enumerate(steps):
        k = math.isqrt(int(t))              # 块号 = 步值所在的平方块
        hi = min((k + 1) * (k + 1) - 1, num_patches)
        row = torch.full((S,), float("-inf"), device=device)
        if i == 0:
            row[:hi + 1] = 0.0              # 第一个步: 前面所有元素（含 z_cls）到块终点
        else:
            lo = max(k * k, 1)              # 位置 0 (z_cls) 屏蔽
            if lo <= hi:
                row[lo:hi + 1] = 0.0        # 只允许自己块内的 z_s
        rows.append(row)
    return torch.stack(rows).repeat_interleave(num_patches, dim=0)


# ═══ PixelHead — 特征 → 像素 patch 解码头 ═══

class PixelHead(nn.Module):
    """每 patch 特征 (B,N,D) → 像素 patch (B,N,14*14*3)。

    监督目标必须是原始像素而非 DINO 特征: 工地图的 DINO 特征在空间上近常数
    （跨位置 std≈5e-5）, 学"输出质心"即达低 L1, 是假收敛; 像素目标有真实
    空间结构, 强制模型保留空间信息。输出不加激活（像素已按 DINO_MEAN/STD
    归一化, 评估时再反归一化）。
    """

    def __init__(self, dim: int, patch_px: int = 14 * 14 * 3):
        super().__init__()
        self.patch_px = patch_px
        self.proj = nn.Linear(dim, patch_px)   # 特征 → 588 维像素 patch

    def forward(self, feat: Tensor) -> Tensor:
        """feat: (..., N, D) → (..., N, patch_px)"""
        return self.proj(feat)


# ═══ OutputQueryDecoder — 输出查询注意力解码器（采样时刻上输出全部 patch）═══

class OutputQueryDecoder(nn.Module):
    """把 [z_cls; z_s] 当时序序列 A=(S,H), 在采样时刻 T_sub 上每步输出
    (N,D) = 全部 patch 的预测, 沿时刻累加得 F_hat (B,N,D)。

    机制: A = [z_cls; z_s] + pos_embed; 查询 = A_t + 查询基 E（行 k↔patch k,
    BLIP-2 同源）→ 标准 nn.TransformerDecoder 堆叠（自注意力 + 交叉注意力
    (memory=A) + FFN）→ reshape (B,|T|,N,D), 返回 Y。
    · 分块掩码走 memory_mask（加法浮点, -inf=屏蔽; 见 build_block_mask）
    · 查询间自注意力无掩码全双向（Q-Former/Perceiver 惯例, 参考 [1][2]）——
      渐进语义只由 memory_mask 提供, 后步查询内容仍可经查询自注意力影响
      前步输出（有意为之, 标准模块的固有代价）
    · 结果沿采样步累加; **梯度按步解耦**（carry detach + 自己的预测, 见
      SRPhase1V2.decode）——每步 Y_t 只从自己那一步的损失收 1 份梯度
    · 时刻采样（显存优化, 默认开启）: 查询只对 T_sub 构造——默认计划 =
      square_block_starts(N); skip_steps/max_steps 可选切片挑选; steps= 可
      自定义。未采样时刻不参与损失, 也不出现在 F_hat 累加里
    · 覆盖语义: 每个采样时刻都预测全部 N 个 patch, 其累加结果都被监督
      还原全部 patch
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 mlp_ratio: float = 4.0,
                 steps: Optional[Sequence[int]] = None, heads: int = 8,
                 depth: int = 2, skip_steps: Optional[int] = None,
                 max_steps: Optional[int] = None):
        super().__init__()
        self.num_patches = num_patches
        # 采样计划: 显式 steps 原样使用（不切片）; 默认 square_block_starts(N)
        # = 全部 z_s 块; skip_steps/max_steps（默认 None）在默认计划上做
        # Python 切片（如 [4:9] 取中段, 每个保留步仍按自身步值归属分块）
        explicit = steps is not None
        if steps is None:
            steps = square_block_starts(num_patches)
        steps = sorted(set(int(s) for s in steps))
        assert steps and all(0 <= s <= num_patches for s in steps), \
            f"steps 越界: {steps} (N={num_patches})"
        if explicit:
            self.steps = steps                 # 显式列表原样（调用方负责）
        else:
            lo = 0 if skip_steps is None else int(skip_steps)
            hi = len(steps) if max_steps is None else int(max_steps)
            assert 0 <= lo < hi <= len(steps), \
                f"skip_steps/max_steps 越界: skip={lo} max={hi} " \
                f"(计划共 {len(steps)} 步 {steps})"
            self.steps = steps[lo:hi]
            assert self.steps, \
                f"切片后无采样步: steps[{lo}:{hi}] of {steps} 为空"
        S = num_patches + 1                                # z_cls + N z_s
        self.query_base = nn.Parameter(torch.randn(num_patches, dim) * 0.02)  # 行 k↔patch k
        # 标准解码器堆叠: nn.TransformerDecoder 内部按 num_layers 深拷贝同一
        # decoder_layer 并顺序执行（含逐层传掩码）, 无需手写循环。每层 =
        # 自注意力 + 交叉注意力(memory=A) + FFN + 残差, 各层共享同一 memory
        self.stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True),
            num_layers=depth,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, S, dim) * 0.02)

    def forward(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        B, N, D = z_s.shape[0], self.num_patches, z_s.shape[-1]
        A = torch.cat([z_cls, z_s], dim=1) + self.pos_embed      # (B,S,D)
        A_t = A[:, self.steps]                                   # (B,|T|,D) 采样时刻
        Y = (A_t.unsqueeze(2) + self.query_base) \
            .reshape(B, len(self.steps) * N, D)                  # 展平 (t,k), t∈T_sub
        # 分块掩码: 每步只见自己的 z_s 块（-inf=屏蔽, 0=允许）
        mask = build_block_mask(N, self.steps, device=A.device)  # (|T|·N, S)
        self.attn_mask = mask                                    # 供自检
        Y = self.stack(Y, A, memory_mask=mask)                   # (B,|T|·N,D)
        Y = Y.reshape(B, len(self.steps), N, D)                  # (B,|T|,N,D)
        self.last_Y = Y                                          # 采样步全部 patch 预测
        return Y


# ═══ SRPhase1V2 — 主模型（register 式; 无预算, k 固定全量）═══

class SRPhase1V2(nn.Module):

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        dim: int = 768,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        decoder_steps: Optional[Sequence[int]] = None,
        patch_px: int = 14 * 14 * 3,
        decoder_depth: int = 2,
        skip_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.dim = dim
        self.patch_px = patch_px

        self.special_bank = SpecialTokenBank(num_patches=num_patches, dim=dim)
        self.decoder = OutputQueryDecoder(dim=dim, num_patches=num_patches,
                                          mlp_ratio=mlp_ratio, heads=heads,
                                          steps=decoder_steps,
                                          depth=decoder_depth,
                                          skip_steps=skip_steps,
                                          max_steps=max_steps)
        self.pixel_head = PixelHead(dim=dim, patch_px=patch_px)

    # ── encode: 输入 → 解码器输入 z_cls, z_s ──
    def encode(self, pixel_values: Tensor):
        """输入 → 解码器输入 (z_cls, z_s), 训练/推理同一路径。

        register 式: specials 作为额外 token 拼进 DINO 输入序列, 由 DINO
        24 层直接算出 z_s（register token 式, Darcet et al.）——深层网络做
        内容路由, 修 F1（special 无内容输入）与 F2（z_s 冗余全局摘要）。
        """
        return self._encode_register(pixel_values)

    def _encode_register(self, pixel_values: Tensor):
        """specials 直接进 DINO 输入序列 [cls; specials; patches] (2N+1 token)。

        HF Dinov2Model 无 token 级注意力 mask API, 故 DINO 内全双向注意力
        （重建任务无时序因果需求, register 惯例亦然）; 渐进语义由解码器
        分块掩码提供。embeddings() 复用 HF 的 cls/patch 嵌入 + 位置编码
        （含自动插值逻辑）; specials 用 SpecialTokenBank（共享 token +
        逐位置可学习 pos, 不带 DINO PE——与 patch 的位置关系完全学出）。
        """
        x = pixel_values                                # (B,3,H,W)
        emb = self.dinov2.embeddings(x)                 # (B,1+N,D) [cls; patches] + PE
        specials = self.special_bank(x.shape[0], x.device)   # (B,N,D) token+pos
        seq = torch.cat([emb[:, :1], specials, emb[:, 1:]], dim=1)   # (B,2N+1,D)
        for layer in self.dinov2.encoder.layer:         # DINO 24 层（全双向）
            out = layer(seq)
            seq = out[0] if isinstance(out, (tuple, list)) else out
        seq = self.dinov2.layernorm(seq)                # (B,2N+1,D)
        return seq[:, :1], seq[:, 1:1 + self.num_patches]

    # ── decode: 共享解码尾（Decoder → PixelHead → 像素损失）──
    def decode(self, z_cls: Tensor, z_s: Tensor, pixel_values: Tensor) -> dict:
        """解码器 + 像素头 + 平权全覆盖像素 L1。

        **累加语义**: Y_cum[:, n] = Σ_{t≤n} Y_t（特征空间累加再统一过
        PixelHead——PixelHead 含 bias, 先投影再累加会重复加 bias）。
        **梯度按步解耦**: 数值上仍 Y_cum = cumsum(Y)（F_hat 不变）, 但
        carry = [0, cumsum(Y)[:-1]] 整体 detach + 自己的预测——每个 Y_t 只
        从自己那一步的损失收 1 份梯度（平权 1/|T|）, 不再从所有 ≥t 的
        累加位置收梯度（否则 t=0 有 |T| 份梯度动力, 学乱）。

        dict: {"loss", "recon", "F_hat"(像素 B,N,588), "Y_pix"(每采样步累加
        像素 B,|T|,N,588), "target_pix"(B,N,588)}——训练取 loss; 推理取
        F_hat / Y_pix / target_pix（全量 L1、渐进曲线、可视化同一路径）。
        """
        x = pixel_values
        B, C, H, W = x.shape
        N = self.num_patches
        Y = self.decoder(z_cls, z_s)                        # (B,|T|,N,D) 每步全部 patch
        # 像素目标: (B,3,H,W) 归一化像素 → (B,N,588) patch
        # 注意布局: DINO 的 patch 顺序是 row-major (先 y 后 x), 这里保持一致
        target_pix = x.reshape(B, C, H // 14, 14, W // 14, 14) \
                      .permute(0, 2, 4, 1, 3, 5) \
                      .reshape(B, N, C * 14 * 14)       # (B,N,588) 归一化像素
        # 特征空间累加 → 统一过 PixelHead（bias 只加一次）。
        # 梯度按步解耦: carry[n] = Σ_{t<n} Y_t（detach）; + Y[n] ⇒ 每步恰收
        # 1 份梯度（数值上 Y_cum[n] 仍 = Σ_{t≤n} Y_t, F_hat 不变）
        Y_cum = torch.cat([torch.zeros_like(Y[:, :1]),
                           Y.cumsum(dim=1)[:, :-1]], dim=1).detach() + Y   # (B,|T|,N,D)
        Y_pix = self.pixel_head(Y_cum)                  # (B,|T|,N,588) 累加像素
        F_pix = Y_pix[:, -1]                            # (B,N,588) 最终 = Σ_t Y_t
        # 平权全覆盖损失: 每个采样步的累加结果都还原全部 patch 像素
        per_step = F.l1_loss(Y_pix, target_pix.unsqueeze(1).expand_as(Y_pix),
                             reduction="none").mean(dim=(0, 2, 3))   # (|T|,)
        loss = per_step.mean()                          # 平权
        recon = F.l1_loss(F_pix, target_pix)            # 集成重建（监控用, 归一化空间）
        return {"loss": loss, "recon": recon, "F_hat": F_pix,
                "Y_pix": Y_pix, "target_pix": target_pix}

    # ── forward（训练/推理同一路径, 无分支）──
    def forward(self, pixel_values: Tensor) -> dict:
        """pixel_values: (B,3,H,W) 归一化像素 → dict{loss, recon, F_hat,
        Y_pix, target_pix}

        监督**原始像素**而非 DINO patch 特征（特征目标退化: 工地图特征
        空间近常数, 学质心即低 L1 是假收敛）。H,W 须为 14 的倍数。
        损失: 每个采样步的**累加结果**都监督还原全部 patch —— 平权全覆盖
        L = mean_n L1(cumsum_n Y_pix, target_pix)。F_hat = Σ_t Y_t → 像素;
        recon 仅作监控。
        """
        x = pixel_values                                # (B,3,H,W)
        B, C, H, W = x.shape
        N = self.num_patches
        assert W % 14 == 0 and H % 14 == 0, "输入须为 14 的倍数"
        assert (W // 14) * (H // 14) == N, \
            f"输入 {W}x{H} 产生 {(W//14)*(H//14)} patches, 但模型 num_patches={N}"
        z_cls, z_s = self.encode(x)
        return self.decode(z_cls, z_s, x)


# ═══ 自检（python model_v2.py）═══
#   1. 形状正确性（register 式全路径）
#   2. OutputQueryDecoder 分块采样计划 + 分块掩码结构
#   3. 梯度流向（整模型可训）+ 梯度按步解耦（Y_cum 数值==cumsum, 每步只收
#      自己那一步的梯度）
#   4. eval 同路径

if __name__ == "__main__":
    torch.manual_seed(0)

    # ── 假的 DINOv2: 结构对齐 HF Dinov2Model（register 式直接走层）──
    class _FakeDinoLayer(nn.Module):
        """HF Dinov2EncoderLayer 属性布局 + 真实单头自注意力 forward。"""
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
        """最小假 DINO: 只含 register 式用到的 embeddings / encoder.layer / layernorm。"""
        def __init__(self, dim: int = 64, n_layers: int = 4, num_patches: int = 16):
            super().__init__()
            self.embeddings = _FakeEmbeddings(dim, num_patches)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList(
                [_FakeDinoLayer(dim) for _ in range(n_layers)])
            self.layernorm = nn.LayerNorm(dim)

    N, D = 16, 64
    dino = FakeDino(dim=D, num_patches=N)
    model = SRPhase1V2(dino, num_patches=N, dim=D,
                       decoder_steps=square_block_starts(N))  # 显式传完整分块计划(不切片)
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

    # ── 2. OutputQueryDecoder: 分块采样计划 + 分块掩码 + 累加结果损失 ──
    T_steps = model.decoder.steps
    assert T_steps == square_block_starts(N), T_steps       # N=16 → [1,4,9,16]
    assert len(model.decoder.stack.layers) == 2, "默认 decoder_depth=2"
    am = model.decoder.attn_mask                             # (|T|·N, N+1) float
    assert am is not None and am.shape == (len(T_steps) * N, N + 1), am.shape
    # 分块掩码: 第一个步可见 0..hi（自己块 + 前面所有元素, 含位置 0 z_cls）;
    # 其余步只允许自己的块 [⌊√t⌋², min((⌊√t⌋+1)²-1, N)], 位置 0 屏蔽
    for ti, t in enumerate(T_steps):
        k = math.isqrt(t)
        hi = min((k + 1) * (k + 1) - 1, N)
        row = am[ti * N]
        if ti == 0:
            assert (row[:hi + 1] == 0).all()                 # 前面所有 + 自己块
            assert (row[hi + 1:] == float("-inf")).all()
        else:
            lo = max(k * k, 1)
            assert row[0] == float("-inf")                   # 其余步 z_cls 屏蔽
            assert (row[:lo] == float("-inf")).all()         # 块前屏蔽
            assert (row[lo:hi + 1] == 0).all()               # 块内允许
            assert (row[hi + 1:] == float("-inf")).all()     # 块后屏蔽
    Y = model.decoder.last_Y                                 # (B,|T|,N,D) 特征
    assert Y.shape == (2, len(T_steps), N, D)
    # 与模型 decode 相同的梯度解耦构造: carry 无梯度 + 自己的预测
    Y_cum = torch.cat([torch.zeros_like(Y[:, :1]),
                       Y.cumsum(dim=1)[:, :-1]], dim=1).detach() + Y
    Y_pix = model.pixel_head(Y_cum)                          # (B,|T|,N,588) 累加像素
    assert Y_pix.shape == (2, len(T_steps), N, PATCH_PX)
    # 数值不变性: 与朴素 cumsum 只差 float32 求和顺序噪声（<1e-4 级）
    assert torch.allclose(Y_cum, Y.cumsum(dim=1), atol=1e-4, rtol=1e-4), \
        "梯度解耦构造数值上应≈原 cumsum（float32 求和顺序噪声内）"
    assert torch.isclose(out["F_hat"], Y_pix[:, -1]).all(), "F_hat 应为各步像素累加和"
    per = F.l1_loss(Y_pix, target.unsqueeze(1).expand_as(Y_pix),
                    reduction="none").mean(dim=(0, 2, 3))    # (|T|,) 每步累加结果
    assert torch.isclose(out["loss"], per.mean()), "loss 应为累加结果平权 L1"
    # 计划自动适配任意 N（可扩展性）: 12→3 块, 256→16 块, 512→22 块
    assert square_block_starts(12) == [1, 4, 9], square_block_starts(12)
    assert len(square_block_starts(256)) == 16
    assert len(square_block_starts(512)) == 22
    # 用户给定示例核对（N=12）: 第一个步 t=1 可见 [0..3]; 步 2 (t=4) 只看 [4..8]
    m12 = build_block_mask(12, [1, 4, 9])
    assert m12.shape == (3 * 12, 13), m12.shape
    r1, r2 = m12[0], m12[12]                                 # 步 1 / 步 2 首查询行
    assert (r1[0:4] == 0).all() and (r1[4:] == float("-inf")).all()
    assert (r2[4:9] == 0).all() and (r2[:4] == float("-inf")).all() \
        and (r2[9:] == float("-inf")).all()
    # 可选挑选（skip_steps/max_steps, 默认 None = 全部分块）: [4:9] 切片取中段
    d_slice = OutputQueryDecoder(num_patches=256, dim=D, skip_steps=4, max_steps=9)
    assert d_slice.steps == square_block_starts(256)[4:9], d_slice.steps
    assert d_slice.steps == [25, 36, 49, 64, 81], d_slice.steps
    # 切片后: 第一个步 t=25 可见 0..35; 其余步只允许自己的块（t=81 → [81..99]）
    d_slice(torch.randn(1, 1, D), torch.randn(1, 256, D))
    sm = d_slice.attn_mask                                  # (5·256, 257)
    assert sm.shape == (5 * 256, 257), sm.shape
    r_s1, r_s5 = sm[0], sm[4 * 256]                         # t=25 (第一个) / t=81 首查询行
    assert (r_s1[0:36] == 0).all() and (r_s1[36:] == float("-inf")).all()
    assert (r_s5[81:100] == 0).all() and (r_s5[:81] == float("-inf")).all() \
        and (r_s5[100:] == float("-inf")).all()
    # 默认 None = 全部分块（等价不切片）; 验收: 第一个步 t=1 能看到 [0,1]
    d_full = OutputQueryDecoder(num_patches=256, dim=D)
    assert d_full.steps == square_block_starts(256), "默认 None 应为全部分块"
    assert d_full.steps[0] == 1, "默认计划第一个步应为 t=1"
    d_full(torch.randn(1, 1, D), torch.randn(1, 256, D))
    r_def1 = d_full.attn_mask[0]                            # 默认计划第一个步 (t=1)
    assert (r_def1[0:4] == 0).all(), "第一个步应可见位置 0 (z_cls)"
    assert (r_def1[1] == 0).all(), "第一个步应可见位置 1"
    print(f"[ok] OutputQueryDecoder: {len(model.decoder.stack.layers)} 层 "
          f"TransformerDecoder, 分块采样 {len(T_steps)} 步 {T_steps} "
          f"+ 分块掩码(示例核对, 第一个步可见[0,1]) + 可选挑选 [4:9]={d_slice.steps} "
          f"+ 累加结果像素损失正确")

    # ── 3. 梯度流向（整模型可训, 含 PixelHead; 首层自注意力 + 末层交叉/FFN）──
    out["loss"].backward()
    for name, p in [("dino.embeddings.patch_embeddings.weight",
                     dino.embeddings.patch_embeddings.weight),
                    ("dino.embeddings.position_embeddings",
                     dino.embeddings.position_embeddings),
                    ("dino.encoder.layer.0.mlp.fc1.weight",
                     dino.encoder.layer[0].mlp.fc1.weight),
                    ("decoder.stack.layers[0].self_attn.in_proj_weight",
                     model.decoder.stack.layers[0].self_attn.in_proj_weight),
                    ("decoder.stack.layers[-1].multihead_attn.in_proj_weight",
                     model.decoder.stack.layers[-1].multihead_attn.in_proj_weight),
                    ("decoder.stack.layers[-1].linear1.weight",
                     model.decoder.stack.layers[-1].linear1.weight),
                    ("decoder.query_base", model.decoder.query_base),
                    ("special_bank.pos", model.special_bank.pos),
                    ("pixel_head.proj", model.pixel_head.proj.weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    print(f"[ok] 梯度: DINO 嵌入/层 + OutputQueryDecoder({len(model.decoder.stack.layers)}×"
          f"TransformerDecoderLayer,query_base)/SpecialTokenBank/PixelHead 全部可训")

    # ── 3b. 梯度按步解耦 ──
    # 数值上 Y_cum == cumsum(Y)（F_hat/损失语义不变）; 但 dL/dY_t 只来自
    # 第 t 步自己的损失项——每步平权 1 份梯度, 不再有"t=0 收 |T| 份"的
    # 三角失衡。校验: (a) 数值恒等; (b) 全量梯度在位置 n == 仅第 n 步损失
    # 的梯度(÷|T|); (c) 第 n 步损失对其他位置 Y_{m≠n} 无梯度（结构上断开）。
    out_b = model(x)
    Y_b = model.decoder.last_Y                              # (B,|T|,N,D)
    Tb = Y_b.shape[1]
    Y_cum_b = torch.cat([torch.zeros_like(Y_b[:, :1]),
                         Y_b.cumsum(dim=1)[:, :-1]], dim=1).detach() + Y_b
    assert torch.allclose(Y_cum_b, Y_b.cumsum(dim=1), atol=1e-4, rtol=1e-4), \
        "梯度解耦后 Y_cum 数值上应≈原 cumsum（float32 求和顺序噪声内）"
    assert torch.isclose(out_b["F_hat"], out_b["Y_pix"][:, -1]).all()
    per_b = F.l1_loss(out_b["Y_pix"],
                      out_b["target_pix"].unsqueeze(1).expand_as(out_b["Y_pix"]),
                      reduction="none").mean(dim=(0, 2, 3))     # (|T|,) 每步损失
    g_total = torch.autograd.grad(out_b["loss"], Y_b, retain_graph=True)[0]
    for n in range(Tb):
        g_n = torch.autograd.grad(per_b[n] / Tb, Y_b,
                                  retain_graph=True)[0]          # 仅第 n 步损失
        others = [m for m in range(Tb) if m != n]
        assert torch.allclose(g_n[:, others],
                              torch.zeros_like(g_n[:, others]), atol=1e-5), \
            f"step {n} 的损失不应给其他步的预测梯度"
        assert torch.allclose(g_total[:, n], g_n[:, n], atol=1e-5), \
            f"step {n} 的 Y 梯度应只来自自己那一步的损失（平权 1/{Tb}）"
    print(f"[ok] 梯度按步解耦: Y_cum 数值==cumsum(F_hat 不变), "
          f"每步 Y_t 恰收 1/{Tb} 份梯度（不再三角失衡）")

    # ── 4. 推理: 同一 forward（eval + no_grad）──
    model.eval()
    with torch.no_grad():
        out_e = model(x)
    assert out_e["F_hat"].shape == (2, N, PATCH_PX)
    print(f"[ok] eval 同路径: loss={out_e['loss'].item():.4f}")

    print("\nALL CHECKS PASSED")
