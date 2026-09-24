"""ResNet + 循环网络的循环卷积解码器 (Recurrent CRNN auto-decoder)。

整体数据流（"循环输出 n 个向量 -> 循环卷积解码器循环解码出原图"）::

    pixel_values (B,3,H,W)
        │
        ├─ ResNetEncoder ──────────────► skips: [s0,s1,s2,s3]  (解码器用的 U-Net 跳连)
        │        └─ 最后一级特征 z (B,C,h,w)
        │
        ├─ Projection ─────────────────► latents (B,n,d)      ← 循环单元输出的 n 个向量
        │        （把 z 投影成 n 个 token，作为循环单元的输入序列）
        │
        ├─ RecurrentCore (RNN/LSTM/GRU) ► 在 n 个 token 上循环，逐步更新状态（BPTT）
        │
        └─ RecurrentConvDecoder ───────► recon_images (B,n,3,H,W)
                 （n 步循环卷积解码：每步读一次循环状态 + 上一帧图像做残差精修，
                   权重在所有步之间共享，并带自己的 GRU 时间记忆）

设计要点
--------
1. 两个"循环"是解耦的：
   - ``recurrent_core`` 决定 **n 个向量** 怎么产生（序列模型）；
   - ``recurrent_decoder`` 决定 **n 个图像** 怎么逐步解码出来（unrolled refinement）。
2. 共享权重 + 残差：解码器每步只预测一个 delta 加到上一帧上，
   第一步前馈输出为零，所以初始时刻等价于恒等映射，训练更稳。
3. 支持 t < n 的早停（推理时按比例减少循环步数）与可变步数（``num_steps`` 张量）。
4. 兼容 transformers：``BpttCrnnConfig`` / ``BpttCrnn.from_pretrained`` / ``save_pretrained``。

用法::

    from model import BpttCrnnConfig, BpttCrnn

    cfg = BpttCrnnConfig(
        depths=(2, 2, 2, 2),      # ResNet-18
        hidden_sizes=(64, 128, 256, 512),
        hidden_size=256,          # 循环状态维度
        num_recurrent_steps=8,    # n
        recurrent_type="lstm",    # rnn | lstm | gru
    )
    model = BpttCrnn(cfg)
    out = model(pixel_values=torch.randn(2, 3, 128, 128), target_images=...)
    out.loss, out.recon_image, out.latents
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from transformers import PreTrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

__all__ = [
    "BpttCrnnConfig",
    "BpttCrnn",
    "BpttCrnnOutput",
    "ResNetEncoder",
    "RecurrentCore",
    "RecurrentConvDecoder",
    "build_resnet_config",
]

# ---------------------------------------------------------------------------
# ResNet 预设：对应前面提到的各个"规模档位"
# ---------------------------------------------------------------------------
RESNET_PRESETS: dict[str, dict] = {
    #                    embedding, hidden_sizes,          depths,       layer_type
    "resnet-10": dict(embedding_size=16, hidden_sizes=(16, 32, 64, 128), depths=(1, 1, 1, 1)),
    "resnet-18": dict(embedding_size=64, hidden_sizes=(64, 128, 256, 512), depths=(2, 2, 2, 2)),
    "resnet-34": dict(embedding_size=64, hidden_sizes=(64, 128, 256, 512), depths=(3, 4, 6, 3)),
    "resnet-50": dict(
        embedding_size=64,
        hidden_sizes=(256, 512, 1024, 2048),
        depths=(3, 4, 6, 3),
        layer_type="bottleneck",
    ),
    "resnet-101": dict(
        embedding_size=64,
        hidden_sizes=(256, 512, 1024, 2048),
        depths=(3, 4, 23, 3),
        layer_type="bottleneck",
    ),
    "resnet-152": dict(
        embedding_size=64,
        hidden_sizes=(256, 512, 1024, 2048),
        depths=(3, 8, 36, 3),
        layer_type="bottleneck",
    ),
}


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
class BpttCrnnConfig(PreTrainedConfig):
    """``BpttCrnn`` 的配置。

    继承 ``PreTrainedConfig``（transformers>=5 的写法：纯 dataclass，字段即类属性）。
    所有字段都会写进 ``config.json``，因此可以直接 ``BpttCrnn.from_pretrained(dir)`` 还原。
    """

    model_type = "bptt_crnn"

    # ---- 输入 / 输出 -----------------------------------------------------
    num_channels: int = 3
    image_size: int = 128  # 只用于 dummy_inputs 与推理提示，模型本身对分辨率自适应

    # ---- ResNet 编码器 ---------------------------------------------------
    embedding_size: int = 64
    hidden_sizes: tuple[int, ...] = (64, 128, 256, 512)
    depths: tuple[int, ...] = (2, 2, 2, 2)
    layer_type: str = "basic"  # "basic" | "bottleneck"
    norm_type: str = "group"  # "batch" | "group" | "layer"
    num_groups: int = 32
    act_fn: str = "relu"  # "relu" | "gelu" | "silu"
    stem_type: str = "cifar"
    """stem 形式。

    - ``"cifar"``: 3x3 stride 1，不做空间下采样（总下采样 = 2**(len(depths)-1)，128 输入 -> 32 特征图）。
      **小图 / 生成任务推荐**：解码器要还原到原图分辨率，stem 下采样太狠会导致目标分辨率过低。
    - ``"imagenet"``: 7x7 stride 2 + maxpool，stem 下采样 4x（原版 ImageNet ResNet，
      需要输入 >= 128 才能让最后一级特征图 >= 2x2）。
    - ``"none"``: 完全不下采样（总下采样 = 1，特征图 = 原图）。
    """
    downsample_in_first_stage: bool = False
    """True: 第 1 个 stage 不下采样（论文里 CIFAR 版 ResNet-18/34 的变体）。"""
    downsample_in_bottleneck: bool = False

    # ---- 循环核心 (n 个向量) --------------------------------------------
    hidden_size: int = 256
    """循环单元状态维度，也是每个输出向量的维度 d。"""
    num_recurrent_steps: int = 8
    """循环步数 n：输出 n 个向量 / n 张重建图。"""
    recurrent_type: str = "lstm"  # "rnn" | "lstm" | "gru"
    recurrent_layers: int = 1
    recurrent_dropout: float = 0.0
    bidirectional: bool = False  # 仅 RNN 有效（用于演示，主流程用 causal 循环）

    # ---- 潜变量的空间组织形式 -------------------------------------------
    latent_grid: int = 8
    """把最后一级特征图聚合成 ``latent_grid x latent_grid`` 个 token（序列长度 = grid**2）。"""
    latent_from_global: bool = False
    """True 时只用全局池化得到的 1 个向量（序列长度 = n）。"""
    latent_project_hidden: bool = True
    """是否把投影后的 token 再过一个 1x1 卷积（更快收敛，可关掉做 ablation）。"""

    # ---- 循环卷积解码器 --------------------------------------------------
    decoder_hidden_size: int = 256
    decoder_num_blocks: int = 2
    decoder_recurrent: bool = True
    """True: 解码器内部也带循环记忆 (GRU)，跨步共享权重；False: 每步纯卷积。"""
    decoder_temporal_hidden: int = 32
    """解码器循环记忆的向量维度。"""
    decoder_temporal_pool: int = 4
    """解码器读上一帧图像时的池化尺寸（pool x pool）。"""
    decoder_dropout: float = 0.0
    decoder_use_skips: bool = True
    """是否使用 ResNet 各级特征做 U-Net 跳连。"""
    decoder_deep_supervision: bool = False
    """True: 每一步的输出都参与 loss；False: 只有最后一步。"""
    decoder_out_act: str = "sigmoid"  # "sigmoid" | "tanh" | "none"

    # ---- 损失 ------------------------------------------------------------
    recon_loss_type: str = "l1"  # "l1" | "l2" | "smooth_l1"

    def __post_init__(self, **kwargs):
        self.hidden_sizes = tuple(self.hidden_sizes)
        self.depths = tuple(self.depths)
        if len(self.hidden_sizes) != len(self.depths):
            raise ValueError(
                f"hidden_sizes 与 depths 长度必须一致，收到 {len(self.hidden_sizes)} vs {len(self.depths)}"
            )
        if self.layer_type not in ("basic", "bottleneck"):
            raise ValueError(f"layer_type 必须是 'basic' 或 'bottleneck'，收到 {self.layer_type}")
        if self.stem_type not in ("cifar", "imagenet", "none"):
            raise ValueError(f"stem_type 必须是 cifar/imagenet/none，收到 {self.stem_type}")
        if self.stem_type == "imagenet" and len(self.depths) > 4:
            raise ValueError("stem_type='imagenet' 假定 4 个 stage（解码器按此匹配通道与尺度）")
        if self.recurrent_type not in ("rnn", "lstm", "gru"):
            raise ValueError(f"recurrent_type 必须是 rnn/lstm/gru，收到 {self.recurrent_type}")
        if self.num_recurrent_steps < 1:
            raise ValueError("num_recurrent_steps 必须 >= 1")
        super().__post_init__(**kwargs)


def build_resnet_config(preset: str = "resnet-18", **overrides) -> dict:
    """返回某个 ResNet 档位的编码器超参，可直接解包进 ``BpttCrnnConfig``。

    >>> cfg = BpttCrnnConfig(**build_resnet_config("resnet-34", hidden_size=512))
    """
    if preset not in RESNET_PRESETS:
        raise KeyError(f"未知预设 {preset!r}，可选: {sorted(RESNET_PRESETS)}")
    merged = dict(RESNET_PRESETS[preset])
    merged.update(overrides)
    return merged


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------
_ACT_FNS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


def _resolve_norm(norm_type: str, num_channels: int, num_groups: int) -> nn.Module:
    """2D（图像特征图）归一化。"""
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "group":
        groups = min(num_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if norm_type == "layer":
        return nn.GroupNorm(1, num_channels)
    raise ValueError(f"未知 norm_type: {norm_type}")


def _resolve_norm_1d(norm_type: str, num_channels: int, num_groups: int) -> nn.Module:
    """1D（序列/向量，如循环核心输出的 token）归一化，语义与 2D 版对齐。"""
    if norm_type == "batch":
        return nn.BatchNorm1d(num_channels)
    return _resolve_norm(norm_type, num_channels, num_groups)


class ConvNormAct(nn.Module):
    """conv -> norm -> act 三联（ResNet 的标准零件）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        norm_type: str = "group",
        num_groups: int = 32,
        act_fn: str = "relu",
        use_norm: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_norm)
        self.norm = _resolve_norm(norm_type, out_channels, num_groups) if use_norm else nn.Identity()
        self.act = _ACT_FNS[act_fn]() if act_fn in _ACT_FNS else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResNetBasicBlock(nn.Module):
    """BasicBlock：3x3 -> 3x3，stride 放在第一个卷积上。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_type: str = "group",
        num_groups: int = 32,
        act_fn: str = "relu",
    ):
        super().__init__()
        self.conv1 = ConvNormAct(
            in_channels, out_channels, 3, stride=stride, norm_type=norm_type, num_groups=num_groups, act_fn=act_fn
        )
        self.conv2 = ConvNormAct(
            out_channels, out_channels, 3, stride=1, norm_type=norm_type, num_groups=num_groups, act_fn="none"
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _resolve_norm(norm_type, out_channels, num_groups),
            )
            if (stride != 1 or in_channels != out_channels)
            else None
        )
        self.act = _ACT_FNS.get(act_fn, nn.ReLU)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv2(self.conv1(x))
        return self.act(out + identity)

    @torch.no_grad()
    def zero_output(self) -> None:
        """残差块零初始化（BN gamma = 0 / 无 norm 时置零卷积权重），让 block 初始为恒等映射。"""
        if isinstance(self.conv2.norm, nn.BatchNorm2d):
            init.zeros_(self.conv2.norm.weight)
        elif isinstance(self.conv2.norm, nn.modules.batchnorm._NormBase):
            init.zeros_(self.conv2.norm.weight)
        else:
            init.zeros_(self.conv2.conv.weight)
            if self.conv2.conv.bias is not None:
                init.zeros_(self.conv2.conv.bias)


class ResNetBottleneckBlock(nn.Module):
    """Bottleneck: 1x1 -> 3x3 -> 1x1，用于 resnet-50 以上的档位。"""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_type: str = "group",
        num_groups: int = 32,
        act_fn: str = "relu",
        downsample_in_bottleneck: bool = False,
    ):
        super().__init__()
        mid_channels = out_channels // self.expansion
        if mid_channels * self.expansion != out_channels:
            raise ValueError(
                f"bottleneck 的 out_channels 必须能被 {self.expansion} 整除，收到 {out_channels}"
            )
        first_stride = stride if downsample_in_bottleneck else 1
        self.conv1 = ConvNormAct(
            in_channels, mid_channels, 1, stride=first_stride, norm_type=norm_type, num_groups=num_groups, act_fn=act_fn
        )
        self.conv2 = ConvNormAct(
            mid_channels, mid_channels, 3, stride=stride, norm_type=norm_type, num_groups=num_groups, act_fn=act_fn
        )
        self.conv3 = ConvNormAct(
            mid_channels, out_channels, 1, stride=1, norm_type=norm_type, num_groups=num_groups, act_fn="none"
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _resolve_norm(norm_type, out_channels, num_groups),
            )
            if (stride != 1 or in_channels != out_channels)
            else None
        )
        self.act = _ACT_FNS.get(act_fn, nn.ReLU)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        return self.act(out + identity)

    @torch.no_grad()
    def zero_output(self) -> None:
        if isinstance(self.conv3.norm, nn.modules.batchnorm._NormBase):
            init.zeros_(self.conv3.norm.weight)
        else:
            init.zeros_(self.conv3.conv.weight)
            if self.conv3.conv.bias is not None:
                init.zeros_(self.conv3.conv.bias)


def _make_block(config: BpttCrnnConfig, in_channels: int, out_channels: int, stride: int) -> nn.Module:
    cls = ResNetBottleneckBlock if config.layer_type == "bottleneck" else ResNetBasicBlock
    kwargs = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        norm_type=config.norm_type,
        num_groups=config.num_groups,
        act_fn=config.act_fn,
    )
    if config.layer_type == "bottleneck":
        kwargs["downsample_in_bottleneck"] = config.downsample_in_bottleneck
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# 1. ResNet 编码器
# ---------------------------------------------------------------------------
class ResNetStage(nn.Module):
    def __init__(self, config: BpttCrnnConfig, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        super().__init__()
        blocks = [_make_block(config, in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            blocks.append(_make_block(config, out_channels, out_channels, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    @torch.no_grad()
    def zero_output(self) -> None:
        # 只把每个 stage 的最后一个残差块置零，保证初始时刻整个 stage 是恒等映射
        self.blocks[-1].zero_output()


class ResNetEncoder(nn.Module):
    """ResNet stem + 4 个 stage，返回各级特征图（供解码器跳连）。

    stem 由 ``config.stem_type`` 决定下采样倍率（见 ``BpttCrnnConfig.stem_type``），
    之后每个 stage 再下采样 2 倍（第一级可被 ``downsample_in_first_stage`` 关掉）。
    解码器会自动跟随实际尺度，不做固定倍数假设。
    """

    def __init__(self, config: BpttCrnnConfig):
        super().__init__()
        self.config = config

        if config.stem_type == "imagenet":
            stem_layers = [
                ConvNormAct(
                    config.num_channels,
                    config.embedding_size,
                    7,
                    stride=2,
                    padding=3,
                    norm_type=config.norm_type,
                    num_groups=config.num_groups,
                    act_fn=config.act_fn,
                ),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        elif config.stem_type == "cifar":
            # CIFAR 风格：3x3 stride 1，stem 只做通道投影，下采样全交给 stage
            stem_layers = [
                ConvNormAct(
                    config.num_channels,
                    config.embedding_size,
                    3,
                    stride=1,
                    padding=1,
                    norm_type=config.norm_type,
                    num_groups=config.num_groups,
                    act_fn=config.act_fn,
                )
            ]
        else:  # "none"
            stem_layers = [
                nn.Conv2d(config.num_channels, config.embedding_size, 3, padding=1)
            ]
        self.stem = nn.Sequential(*stem_layers)

        stages = []
        in_channels = config.embedding_size
        self.stage_strides = []
        for index, (out_channels, depth) in enumerate(zip(config.hidden_sizes, config.depths)):
            stride = 2 if (index > 0 or config.downsample_in_first_stage) else 1
            stages.append(ResNetStage(config, in_channels, out_channels, depth, stride))
            self.stage_strides.append(stride)
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.out_channels = in_channels

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = False):
        """返回 ``(z, skips)``；``skips`` 为各级特征图列表（浅 -> 深）。"""
        x = self.stem(pixel_values)
        skips = [] if output_hidden_states else None
        for stage in self.stages:
            x = stage(x)
            if skips is not None:
                skips.append(x)
        return x, skips


# ---------------------------------------------------------------------------
# 2. 循环核心：在 token 序列上滚动，产出 n 个向量
# ---------------------------------------------------------------------------
class RecurrentCore(nn.Module):
    """把编码器特征整理成 token，再用 RNN/LSTM/GRU 滚动 n 步，输出 n 个向量。

    形状约定: ``latent (B, n_tokens, hidden_size)``，``masks`` 形状与 ``latent`` 的前两维一致。
    ``masks``（可选）用来做 **可变循环步数**：mask=0 的时刻状态被冻结、输出沿用上一步。
    每一步都会加上一个可学的 step embedding（类似 diffusion 的 timestep embedding），
    让解码器能区分"这是第几次精修"。
    """

    def __init__(self, config: BpttCrnnConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.recurrent_type = config.recurrent_type
        self.bidirectional = bool(config.bidirectional) and config.recurrent_type == "rnn"

        rnn_kwargs = dict(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.recurrent_layers,
            batch_first=True,
            dropout=config.recurrent_dropout if config.recurrent_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        if config.recurrent_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif config.recurrent_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            self.rnn = nn.RNN(**rnn_kwargs)

        self.step_embed = nn.Embedding(config.num_recurrent_steps, config.hidden_size)
        self.step_norm = _resolve_norm_1d(config.norm_type, config.hidden_size, config.num_groups)

    def forward(
        self,
        latent: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
    ):
        """``latent: (B, S, d)`` -> ``(outputs (B,S,d), hidden states)``。"""
        if self.bidirectional:
            raise NotImplementedError("bidirectional 循环核心与因果逐帧解码不兼容，请设置 config.bidirectional=False")

        outputs, state = self.rnn(latent)  # (B,S,d), (num_layers,B,d) 或 LSTM 的 tuple

        if masks is not None:
            outputs = self._apply_masks(outputs, latent, masks)

        # 加 step embedding，并做一次归一化，稳定循环状态的尺度
        steps = outputs.shape[1]
        outputs = outputs + self.step_embed.weight[:steps].unsqueeze(0)
        outputs = self.step_norm(outputs.transpose(1, 2)).transpose(1, 2)
        return outputs, state

    @staticmethod
    def _apply_masks(outputs: torch.Tensor, inputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """mask=0 的位置：输出冻结为上一时刻的输出（逐步累积，等价于"停止更新"）。"""
        masks = masks.to(dtype=outputs.dtype).unsqueeze(-1)  # (B,S,1)
        out = outputs
        prev = inputs
        collected = []
        for step in range(outputs.shape[1]):
            m = masks[:, step : step + 1]
            cur = m * out[:, step : step + 1] + (1.0 - m) * prev
            collected.append(cur)
            prev = cur
        return torch.cat(collected, dim=1)


# ---------------------------------------------------------------------------
# 3. 循环卷积解码器
# ---------------------------------------------------------------------------
class RecurrentConvBlock(nn.Module):
    """解码器里的残差卷积块。"""

    def __init__(self, channels: int, config: BpttCrnnConfig):
        super().__init__()
        self.conv1 = ConvNormAct(
            channels, channels, 3, norm_type=config.norm_type, num_groups=config.num_groups, act_fn=config.act_fn
        )
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _resolve_norm(config.norm_type, channels, config.num_groups)
        self.act = _ACT_FNS.get(config.act_fn, nn.ReLU)()
        self.dropout = nn.Dropout2d(config.decoder_dropout) if config.decoder_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.dropout(self.norm2(self.conv2(self.conv1(x)))) + x)


class RecurrentConvDecoder(nn.Module):
    """循环卷积解码器：反复"读向量 + 精修图像"来逐帧解码出原图。

    每个循环步 t：

    1. 用共享权重把状态 ``state_t`` 投到潜空间，得到一个粗解码图；
    2. 把它上采样（可带编码器跳连）拼到上一帧 ``image_{t-1}`` 上；
    3. 过一个共享的残差卷积块 + 1x1 输出头，得到增量 ``delta_t``；
    4. ``image_t = image_{t-1} + delta_t``，并用解码器自己的 GRU 更新时间记忆。

    由于所有步共享权重、且第一步输出为零，整条链路在 n 个循环步上构成一个
    "迭代精修算子"（unrolled optimization），而不是 n 个独立解码器。
    """

    def __init__(self, config: BpttCrnnConfig, latent_shape: tuple[int, int], encoder_out_channels: int):
        super().__init__()
        self.config = config
        self.latent_shape = latent_shape
        self.latent_hw = latent_shape[0] * latent_shape[1]

        self.proj = nn.Linear(config.hidden_size, self.latent_hw * config.decoder_hidden_size)

        # 解码器每级通道：由深到浅，与编码器倒序对齐
        self.stage_channels = list(reversed(list(config.hidden_sizes)))
        # 每级"来自上一级的上采样"通道数：第 0 级来自 proj
        upsample_in = [config.decoder_hidden_size] + self.stage_channels[:-1]
        # 每级跳连的原始通道数（与 self.stage_channels 同序）
        raw_skip = list(reversed(list(config.hidden_sizes)))
        # upsample_in / 跳连都会被投到本级通道数，bottleneck 的 stem 与 stage1 通道不同也能拼上
        self.skip_out = self.stage_channels if config.decoder_use_skips else [0] * len(self.stage_channels)

        self.up_proj = nn.ModuleList(
            [
                nn.Conv2d(upsample_in[i], channels, 1) if upsample_in[i] != channels else nn.Identity()
                for i, channels in enumerate(self.stage_channels)
            ]
        )
        self.skip_proj = nn.ModuleList(
            [
                nn.Conv2d(raw_skip[i], channels, 1) if raw_skip[i] != channels else nn.Identity()
                for i, channels in enumerate(self.stage_channels)
            ]
        )

        # 每级融合：上采样特征 + 编码器跳连 -> 本级通道
        self.fuse = nn.ModuleList()
        for index, channels in enumerate(self.stage_channels):
            self.fuse.append(
                nn.Sequential(
                    ConvNormAct(
                        channels + self.skip_out[index],
                        channels,
                        3,
                        norm_type=config.norm_type,
                        num_groups=config.num_groups,
                        act_fn=config.act_fn,
                    ),
                    *[RecurrentConvBlock(channels, config) for _ in range(config.decoder_num_blocks)],
                )
            )

        last_channels = self.stage_channels[-1]
        self.image_proj = nn.Conv2d(config.num_channels, last_channels, 3, padding=1)

        # 解码器自身的时间记忆：pool -> GRUCell -> 广播回每个像素
        self.temporal_hidden = config.decoder_temporal_hidden
        refine_in = last_channels
        if config.decoder_recurrent and self.temporal_hidden > 0:
            self.norm_t = _resolve_norm(config.norm_type, config.num_channels, config.num_groups)
            self.temporal_pool = nn.AdaptiveAvgPool2d(config.decoder_temporal_pool)
            self.gru = nn.GRUCell(
                config.num_channels + config.num_channels * config.decoder_temporal_pool**2,
                self.temporal_hidden,
            )
            self.temporal_proj = nn.Linear(self.temporal_hidden, last_channels)
            refine_in = last_channels * 2
        else:
            self.norm_t = nn.Identity()
            self.temporal_pool = None
            self.gru = None
            self.temporal_proj = None

        self.refine = nn.Sequential(
            ConvNormAct(
                refine_in,
                config.decoder_hidden_size,
                3,
                norm_type=config.norm_type,
                num_groups=config.num_groups,
                act_fn=config.act_fn,
            ),
            *[RecurrentConvBlock(config.decoder_hidden_size, config) for _ in range(config.decoder_num_blocks)],
        )
        self.out = nn.Conv2d(config.decoder_hidden_size, config.num_channels, 3, padding=1)

    # -- 单步解码 ---------------------------------------------------------
    def decode_step(
        self,
        state: torch.Tensor,
        prev_image: torch.Tensor,
        temporal_state: Optional[torch.Tensor] = None,
        skips: Optional[list[torch.Tensor]] = None,
        skip_hw: Optional[list[tuple[int, int]]] = None,
    ):
        batch = state.shape[0]
        h, w = self.latent_shape

        x = self.proj(state).reshape(batch, self.config.decoder_hidden_size, h, w)

        prev_channels = self.config.decoder_hidden_size
        for index, channels in enumerate(self.stage_channels):
            if x.shape[1] != channels:
                # 逐级解码：通道数自深到浅，空间自小到大，每次上采样固定 2 倍
                if x.shape[1] != prev_channels:
                    raise RuntimeError(
                        f"解码器通道不连续: 第 {index} 级期望输入 {prev_channels}，实际 {x.shape[1]}"
                    )
                factor = 2
                new_hw = (x.shape[-2] * factor, x.shape[-1] * factor)
                target_hw = tuple(skip_hw[index]) if skip_hw is not None and index < len(skip_hw) else new_hw
                x = F.interpolate(x, size=target_hw, mode="nearest")
            elif skip_hw is not None and index < len(skip_hw) and (x.shape[-2], x.shape[-1]) != tuple(skip_hw[index]):
                x = F.interpolate(x, size=tuple(skip_hw[index]), mode="nearest")

            if skips is not None and index < len(skips) and self.config.decoder_use_skips:
                x = torch.cat([self.up_proj[index](x), self.skip_proj[index](skips[index])], dim=1)
            else:
                x = self.up_proj[index](x)
            x = self.fuse[index](x)
            prev_channels = channels

        if (x.shape[-2], x.shape[-1]) != prev_image.shape[-2:]:
            x = F.interpolate(x, size=prev_image.shape[-2:], mode="nearest")

        # 低层特征 + 上一帧图像 + 时间记忆一起做残差精修
        features = x + self.image_proj(prev_image)
        if self.gru is not None:
            pooled = self.temporal_pool(prev_image).flatten(1)
            global_vec = F.adaptive_avg_pool2d(prev_image, 1).flatten(1)
            if temporal_state is None:
                temporal_state = torch.zeros(
                    prev_image.shape[0], self.temporal_hidden, dtype=prev_image.dtype, device=prev_image.device
                )
            temporal_state = self.gru(torch.cat([pooled, global_vec], dim=1), temporal_state)
            bias = self.temporal_proj(temporal_state).unsqueeze(-1).unsqueeze(-1)
            features = torch.cat([features, bias.expand_as(features)], dim=1)

        delta = self.out(self.refine(features))
        if self.config.decoder_out_act == "sigmoid":
            # 用 sigmoid 把每步增量压到 [-0.5, 0.5]，保证精修是"小步走"
            delta = torch.sigmoid(delta) - 0.5

        image = self.norm_t(prev_image + delta)
        return image, temporal_state

    # -- n 步循环 ---------------------------------------------------------
    def forward(
        self,
        states: torch.Tensor,
        skips: Optional[list[torch.Tensor]] = None,
        base_image: Optional[torch.Tensor] = None,
        output_all_steps: bool = False,
        num_steps: Optional[torch.Tensor] = None,
        skip_hw: Optional[list[tuple[int, int]]] = None,
    ):
        """``states: (B, n, d)`` -> 图像序列 ``(B, n, C, H, W)``。"""
        batch, steps = states.shape[0], states.shape[1]

        if base_image is None:
            target_hw = tuple(skip_hw[0]) if skip_hw else (self.config.image_size, self.config.image_size)
            prev_image = torch.zeros(
                batch, self.config.num_channels, *target_hw, dtype=states.dtype, device=states.device
            )
        else:
            prev_image = base_image

        temporal_state = None
        outputs = []
        for step in range(steps):
            step_mask = None
            if num_steps is not None:
                step_mask = (num_steps > step).to(states.dtype).view(batch, 1, 1, 1)
            image, temporal_state = self.decode_step(
                states[:, step], prev_image, temporal_state, skips, skip_hw
            )
            if step_mask is not None:
                image = step_mask * image + (1.0 - step_mask) * prev_image
            prev_image = image
            if output_all_steps:
                outputs.append(image)

        if output_all_steps:
            return torch.stack(outputs, dim=1), prev_image
        return prev_image


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
@dataclass
class BpttCrnnOutput(ModelOutput):
    """模型输出。

    - ``images``: ``(B, n, C, H, W)`` n 步循环解码的完整序列（deep supervision 用）
    - ``recon_image``: ``(B, C, H, W)`` 最后一步的重建图
    - ``latents``: ``(B, n, d)`` 循环核心输出的 n 个向量
    - ``vector``: ``(B, d)`` n 个向量聚合后的全局表达（下游 head / 分类 / 检索可用）
    """

    loss: Optional[torch.Tensor] = None
    recon_image: Optional[torch.Tensor] = None
    images: Optional[torch.Tensor] = None
    latents: Optional[torch.Tensor] = None
    vector: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[tuple] = None
    skips: Optional[list] = None
    hidden_states: Optional[list] = None
    recurrent_state: Optional[list] = None


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------
class BpttCrnn(PreTrainedModel):
    """ResNet 编码器 + 循环核心 + 循环卷积解码器（BPTT 端到端训练）。"""

    config_class = BpttCrnnConfig
    base_model_prefix = "bptt_crnn"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _supports_attn_implementation = False
    _no_split_modules = ["ResNetStage", "RecurrentConvBlock"]

    def __init__(self, config: BpttCrnnConfig):
        super().__init__(config)
        self.config = config

        # -- 1) ResNet 编码器 -------------------------------------------------
        self.encoder = ResNetEncoder(config)

        # -- 2) 投影：特征图 -> n 个 token ------------------------------------
        if config.latent_from_global:
            grid_h = grid_w = 1
            token_source_dim = self.encoder.out_channels
        else:
            grid_h = grid_w = config.latent_grid
            token_source_dim = self.encoder.out_channels
        self.token_source_dim = token_source_dim
        self.grid_hw = (grid_h, grid_w)

        self.token_embed = nn.Linear(token_source_dim, config.hidden_size)
        self.token_norm = _resolve_norm_1d(config.norm_type, config.hidden_size, config.num_groups)
        self.token_act = _ACT_FNS.get(config.act_fn, nn.ReLU)()
        self.token_proj = (
            nn.Conv2d(config.hidden_size, config.hidden_size, 1)
            if config.latent_project_hidden and not config.latent_from_global
            else None
        )

        # -- 3) 循环核心 ------------------------------------------------------
        #     在 token 序列上滚动出 n 个向量（内部自带 step embedding）
        self.core = RecurrentCore(config)

        # -- 4) 循环卷积解码器 ------------------------------------------------
        self.decoder = RecurrentConvDecoder(config, self.grid_hw, self.encoder.out_channels)

        # -- 权重初始化 -------------------------------------------------------
        self.post_init()

    # -- 工具 -------------------------------------------------------------
    def _feature_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """``z (B,C,h,w)`` -> 循环核心的输入序列 ``(B,S,d)``，``S = num_recurrent_steps``。"""
        if self.config.latent_from_global:
            pooled = F.adaptive_avg_pool2d(z, 1).flatten(1)  # (B,C)
            tokens = self.token_embed(pooled)  # (B,d)
            tokens = self.token_act(self.token_norm(tokens))  # 归一化后依然是 (B,d)
            tokens = tokens.unsqueeze(1)  # (B,1,d)
        else:
            pooled = F.adaptive_avg_pool2d(z, self.grid_hw)  # (B,C,gh,gw)
            pooled = pooled.flatten(2).transpose(1, 2)  # (B,S,C)
            tokens = self.token_embed(pooled)  # (B,S,d)
            tokens = self.token_act(self.token_norm(tokens.transpose(1, 2)).transpose(1, 2))
            if self.token_proj is not None:
                batch, seq, dim = tokens.shape
                grid = tokens.transpose(1, 2).reshape(batch, dim, *self.grid_hw)
                tokens = self.token_proj(grid).flatten(2).transpose(1, 2)

        # 序列长度对齐到 n：不足（如 global 池化只有 1 个 token）时用零向量补齐，
        # 多出来的 token 截断。真正的"第几步"信息由 RecurrentCore 的 step embedding 提供。
        steps = self.config.num_recurrent_steps
        if tokens.shape[1] < steps:
            pad = tokens.new_zeros(tokens.shape[0], steps - tokens.shape[1], tokens.shape[2])
            tokens = torch.cat([tokens, pad], dim=1)
        else:
            tokens = tokens[:, :steps]
        return tokens

    def _reconstruct_loss(self, images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_type = self.config.recon_loss_type
        if loss_type == "l1":
            return F.l1_loss(images, target)
        if loss_type == "l2":
            return F.mse_loss(images, target)
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(images, target)
        raise ValueError(f"未知 recon_loss_type: {loss_type}")

    # -- 前向 -------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        target_images: Optional[torch.Tensor] = None,
        num_steps: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BpttCrnnOutput:
        """前向。

        Args:
            pixel_values: ``(B, C, H, W)`` 输入图像。
            target_images: 可选，``(B, C, H, W)`` 重建目标；不传则不计算 loss。
            num_steps: 可选 ``(B,)`` 长整型张量，逐样本指定实际循环步数（早停）；其余步保持上一帧。
            output_hidden_states: 是否返回编码器各级特征。
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 1) ResNet 编码
        z, skips = self.encoder(pixel_values, output_hidden_states=True)

        # 2) 特征 -> n 个 token
        tokens = self._feature_tokens(z)

        # 3) 循环核心：滚动 n 步，得到 n 个向量
        latents, recurrent_state = self.core(tokens, masks=None)

        # 4) 循环卷积解码：n 步精修出原图
        #    解码器自深到浅走，因此跳连要倒序（skips[0] = 最深/最小分辨率）
        decoder_skips = list(reversed(skips))
        skip_hw = [tuple(s.shape[-2:]) for s in decoder_skips]
        base_image = torch.zeros(
            pixel_values.shape[0],
            self.config.num_channels,
            *pixel_values.shape[-2:],
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        images, recon_image = self.decoder(
            latents,
            skips=decoder_skips if self.config.decoder_use_skips else None,
            base_image=base_image,
            output_all_steps=True,
            num_steps=num_steps,
            skip_hw=skip_hw,
        )

        loss = None
        if target_images is not None:
            if target_images.dim() == 5:  # (B,n,C,H,W) 逐步监督
                loss = self._reconstruct_loss(images, target_images)
            elif self.config.decoder_deep_supervision:
                loss = self._reconstruct_loss(images, target_images.unsqueeze(1).expand_as(images))
            else:
                loss = self._reconstruct_loss(recon_image, target_images)

        vector = latents.mean(dim=1)  # (B,d) 聚合向量

        if not return_dict:
            items = (loss, recon_image, images, latents, vector)
            return tuple(item for item in items if item is not None)
        return BpttCrnnOutput(
            loss=loss,
            recon_image=recon_image,
            images=images,
            latents=latents,
            vector=vector,
            encoder_hidden_states=tuple(skips) if output_hidden_states else None,
            skips=skips,
            hidden_states=[latents],
            recurrent_state=recurrent_state,
        )

    # -- 只要向量的话，单独走编码 + 循环核心 -------------------------------
    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """返回 ``(B, n, d)`` 的 n 个向量（用于下游任务或可视化）。"""
        z, _ = self.encoder(pixel_values, output_hidden_states=False)
        latents, _ = self.core(self._feature_tokens(z))
        return latents

    def _default_skip_hw(self, size: int | tuple[int, int]) -> list[tuple[int, int]]:
        """按编码器实际下采样倍率推出解码器每级的空间尺寸（深 -> 浅）。

        以 1 为起点逐级累乘各 stage 的 stride（``stage_strides``），因此
        ``stem_type`` 与 ``downsample_in_first_stage`` 的任何组合都自动跟随。
        """
        if isinstance(size, int):
            size = (size, size)
        sizes = []
        scale = 1
        for index in range(len(self.config.depths)):
            scale *= int(self.encoder.stage_strides[index])
            sizes.append((max(size[0] // scale, 1), max(size[1] // scale, 1)))
        return list(reversed(sizes))

    def _empty_skips(self, skip_hw: list[tuple[int, int]], batch: int, dtype, device) -> list[torch.Tensor]:
        """生成与编码器跳连形状一致的零张量，用于纯向量解码（补齐 fuse 的通道数）。"""
        channels = list(reversed(list(self.config.hidden_sizes)))
        return [
            torch.zeros(batch, ch, h, w, dtype=dtype, device=device)
            for ch, (h, w) in zip(channels, skip_hw)
        ]

    # -- 从向量反向解码（配合 encode 使用） --------------------------------
    @torch.no_grad()
    def decode(self, latents: torch.Tensor, output_all_steps: bool = False):
        """只用 n 个向量做循环解码（不经过编码器），验证"向量 -> 图像"通路。

        跳连位置填 0，只保留形状对齐（``use_skips=False`` 需要单独建一个解码器，这里不做）。
        """
        batch = latents.shape[0]
        skip_hw = self._default_skip_hw(self.config.image_size)
        skips = self._empty_skips(skip_hw, batch, latents.dtype, latents.device)
        images, final = self.decoder(
            latents,
            skips=skips,
            base_image=None,
            output_all_steps=True,
            skip_hw=skip_hw,
        )
        return (images, final) if output_all_steps else final

    # -- 初始化 -----------------------------------------------------------
    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.modules.batchnorm._NormBase):
            init.ones_(module.weight)
            init.zeros_(module.bias)
            if module.running_mean is not None:
                init.zeros_(module.running_mean)
            if module.running_var is not None:
                init.ones_(module.running_var)
        elif isinstance(module, (nn.RNN, nn.LSTM, nn.GRU, nn.RNNCell, nn.LSTMCell, nn.GRUCell)):
            for name, param in module.named_parameters(recurse=False):
                if "weight_ih" in name:
                    init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    init.orthogonal_(param)
                elif "bias" in name:
                    init.zeros_(param)
                    # LSTM 遗忘门偏置置 1（标准做法）
                    hidden = param.shape[0] // 4 if module.__class__.__name__.startswith("LSTM") else 0
                    if hidden:
                        param[hidden : 2 * hidden].fill_(1.0)

    @torch.no_grad()
    def zero_init_decoder_output(self) -> None:
        """把循环解码器的输出头置零：初始时刻 n 步输出都等于 base_image（恒等精修算子）。"""
        init.zeros_(self.decoder.out.weight)
        if self.decoder.out.bias is not None:
            init.zeros_(self.decoder.out.bias)

    @torch.no_grad()
    def scale_decoder_output(self, std: float = 0.2) -> None:
        """放大解码器输出头（自测用：让未训练模型也能产生可见的逐步精修）。"""
        init.normal_(self.decoder.out.weight, std=std)
        if self.decoder.out.bias is not None:
            init.zeros_(self.decoder.out.bias)

    @property
    def dummy_inputs(self) -> dict[str, torch.Tensor]:
        return {"pixel_values": torch.randn(2, self.config.num_channels, self.config.image_size, self.config.image_size)}


# ---------------------------------------------------------------------------
# 自测：python model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    for preset in ("resnet-10", "resnet-18", "resnet-50"):
        cfg = BpttCrnnConfig(
            **build_resnet_config(preset),
            hidden_size=128,
            num_recurrent_steps=4,
            latent_grid=4,
            decoder_hidden_size=64,
            decoder_num_blocks=1,
            recurrent_type="lstm",
            image_size=64,
        )
        model = BpttCrnn(cfg)
        x = torch.randn(2, 3, 64, 64)
        out = model(pixel_values=x, target_images=x)
        out.loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        n_params = sum(p.numel() for p in model.parameters())
        # 相邻两步输出应当不同 => 循环解码真的在逐步精修
        step_deltas = (out.images[:, 1:] - out.images[:, :-1]).abs().amax(dim=(0, 2, 3, 4))
        print(
            f"[{preset}] params={n_params/1e6:.2f}M loss={out.loss.item():.4f} "
            f"latents={tuple(out.latents.shape)} images={tuple(out.images.shape)} "
            f"per-step delta={[round(float(d), 4) for d in step_deltas]} grads_finite={finite}"
        )

    # 零初始化输出头 => 循环解码器初始为恒等算子（n 步输出全为 0，训练更稳）
    cfg = BpttCrnnConfig(**build_resnet_config("resnet-18"), hidden_size=64, num_recurrent_steps=6, image_size=64)
    model = BpttCrnn(cfg)
    model.zero_init_decoder_output()
    x = torch.randn(2, 3, 64, 64)
    out0 = model(pixel_values=x)
    print(f"zero-init 输出头: 所有步输出为 0 = {bool((out0.images == 0).all())}")

    # 早停 / 可变步数：样本 0 只跑 1 步，之后的输出被冻结
    model.scale_decoder_output(0.2)  # 让未训练模型也能产生可见的逐步变化
    out = model(pixel_values=x, num_steps=torch.tensor([1, 6]))
    frozen = torch.allclose(out.images[0, 0], out.images[0, -1], atol=1e-6)
    still_running = not torch.allclose(out.images[1, 0], out.images[1, -1], atol=1e-6)
    print(f"early-stop: 样本0 冻结={frozen}, 样本1 继续循环={still_running}")

    hidden = model.encode(x)
    rec = model.decode(hidden)
    print(f"encode -> {tuple(hidden.shape)}, decode -> {tuple(rec.shape)}")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded = BpttCrnn.from_pretrained(tmp)
        with torch.no_grad():
            a = model(pixel_values=x).recon_image
            b = reloaded(pixel_values=x).recon_image
        print(f"save/load 一致性: {torch.allclose(a, b, atol=1e-5)}")

    # BPTT 端到端可训性：重新跑一次反传，检查每个参数都拿到梯度
    model.zero_grad(set_to_none=True)
    model(pixel_values=x, target_images=x).loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    print(f"反传后未收到梯度的参数: {missing if missing else '无（全部参数可训）'}")
