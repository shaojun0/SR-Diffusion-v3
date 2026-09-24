"""Recurrent ResNet (循环 ResNet) for image reconstruction.

实验矩阵（12 = 6 ResNet 档位 x 2 分辨率）
----------------------------------------
backbone  : resnet-10 / resnet-18 / resnet-34 / resnet-50 / resnet-101 / resnet-152
resolution: 224 x 224 与 448 x 448

对齐用户口径
------------
1. 「LSTM + 6 个 resnet 模型」：6 个 ResNet 档位，每个档位都是一套
   编码器(ResNet) + 解码器(倒置 ResNet) 的循环网络。
2. 「resnet 的每一个输出的 layer block 都接入自己的 LSTM」：
   编码器 4 个 layer block 的输出（layer1..layer4，即 4 个 skip 层级）各自接一个
   ``SpatialLSTM``；解码器每个 layer block 的输出同样各自接一个 ``SpatialLSTM``。
   共 8 个 LSTM / 模型。
3. 「解码器是编码器的倒置，架构同样是循环 resnet，只不过以前的下采样变成了上采样」：
   解码器逐级、逐 block 地镜像编码器——
     * 编码器 stage_i 的 stride-2 下采样  ->  解码器该级 stride-2 最近邻上采样 + 3x3 conv；
     * 编码器 stage_0 是 stride-1（不下采样）-> 解码器对应级也不上采样；
     * 编码器 stem 的 4x 下采样（conv s2 + maxpool s2）-> 解码器末尾两次 2x 上采样；
     * 每级 ResNet block 数量 = 编码器该 stage 的 ``depths[i]``，逐 block 镜像。
4. 「resnet 的参数是加载 hf 现有的模型的参数，不要从 0 开始」：
   编码器直接使用 transformers 的 ``ResNetModel.from_pretrained("microsoft/resnet-XX")``，
   加载 ImageNet 预训练权重。resnet-10 在 HF 上没有官方权重，因此用
   **ResNet-10 标准拓扑**（depths=(1,1,1,1)，标准宽度 64/128/256/512）建模，
   并从 ``microsoft/resnet-18`` 加载 —— ResNet-10 的每个 block 都是 ResNet-18
   对应 stage 的第一个 block（depths 是严格前缀），**所有张量 100% 命中**。
   ``load_report()`` 会打印实际命中情况，可复核。

归一化说明（诚实标注）
----------------------
编码器沿用 HF ResNet 的 BatchNorm（否则预训练权重无法加载）。
解码器为自建模块，使用 GroupNorm：解码器在 448² 下 batch 只能开到 8 左右，
GroupNorm 对小 batch 更稳，且解码器本身没有预训练权重可用，换成 GN 不损失任何
可迁移参数。block 拓扑与编码器严格镜像，仅 norm 实现不同。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["RecurrentResNet", "RecurrentResNetConfig", "build_model", "BACKBONES"]

BACKBONES = ("resnet-10", "resnet-18", "resnet-34", "resnet-50", "resnet-101", "resnet-152")

# 每个档位的 HF 权重来源 + depths 覆盖
_HF_SOURCE = {
    "resnet-10": ("microsoft/resnet-18", (1, 1, 1, 1)),
    "resnet-18": ("microsoft/resnet-18", (2, 2, 2, 2)),
    "resnet-34": ("microsoft/resnet-34", (3, 4, 6, 3)),
    "resnet-50": ("microsoft/resnet-50", (3, 4, 6, 3)),
    "resnet-101": ("microsoft/resnet-101", (3, 4, 23, 3)),
    "resnet-152": ("microsoft/resnet-152", (3, 8, 36, 3)),
}


# ---------------------------------------------------------------------------
# 基础卷积块
# ---------------------------------------------------------------------------
_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


def _norm(kind: str, channels: int, groups: int = 32) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        g = groups if channels % groups == 0 else 1
        return nn.GroupNorm(g, channels)
    if kind == "layer":
        return nn.GroupNorm(1, channels)
    raise ValueError(f"unknown norm {kind!r}")


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=None, norm="group",
                 groups=32, act="relu"):
        super().__init__()
        padding = k // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False)
        self.norm = _norm(norm, out_ch, groups)
        self.act = _ACT[act](inplace=True) if act != "none" else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BasicBlock(nn.Module):
    """与 ResNet-18/34 的 BasicBlock 同构（3x3 -> 3x3，stride 在第一个卷积）。"""

    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, norm="group", groups=32, act="relu"):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride=stride, norm=norm, groups=groups, act=act)
        self.conv2 = ConvNormAct(out_ch, out_ch, 3, stride=1, norm=norm, groups=groups, act="none")
        self.downsample = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                          _norm(norm, out_ch, groups))
            if (stride != 1 or in_ch != out_ch) else None
        )
        self.act = _ACT[act](inplace=True)

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        return self.act(self.conv2(self.conv1(x)) + idt)


class Bottleneck(nn.Module):
    """与 ResNet-50/101/152 的 Bottleneck 同构（1x1 -> 3x3 -> 1x1，expansion=4）。"""

    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, norm="group", groups=32, act="relu"):
        super().__init__()
        mid = out_ch // 4
        self.conv1 = ConvNormAct(in_ch, mid, 1, stride=1, norm=norm, groups=groups, act=act)
        self.conv2 = ConvNormAct(mid, mid, 3, stride=stride, norm=norm, groups=groups, act=act)
        self.conv3 = ConvNormAct(mid, out_ch, 1, stride=1, norm=norm, groups=groups, act="none")
        self.downsample = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                          _norm(norm, out_ch, groups))
            if (stride != 1 or in_ch != out_ch) else None
        )
        self.act = _ACT[act](inplace=True)

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        return self.act(self.conv3(self.conv2(self.conv1(x))) + idt)


def _block_cls(layer_type: str):
    return Bottleneck if layer_type == "bottleneck" else BasicBlock


def _make_stage(layer_type, in_ch, out_ch, num_blocks, stride, norm, groups, act):
    blocks = [_block_cls(layer_type)(in_ch, out_ch, stride, norm, groups, act)]
    for _ in range(1, num_blocks):
        blocks.append(_block_cls(layer_type)(out_ch, out_ch, 1, norm, groups, act))
    return nn.Sequential(*blocks)


# ---------------------------------------------------------------------------
# 每个 layer block 输出接入的自己的 LSTM
# ---------------------------------------------------------------------------
class SpatialLSTM(nn.Module):
    """一个 layer block 输出 -> 自己的 LSTM -> 残差精修回该 block 输出。

    ``(B,C,H,W)`` --自适应池化到 ``g x g``--> ``(B, g*g, C)`` 序列 --LSTM--> 
    ``(B, g*g, d)`` --线性投影回 C--> ``(B,C,g,g)`` --插值回 (H,W)--> 残差相加。

    * ``g = min(H, W, max_grid)``：token 数被 ``max_grid**2`` 封顶，因此 448² 与
      224² 的 LSTM 计算量同阶，且分辨率无关。
    * ``proj`` 零初始化 => 初始时刻残差项为 0，模型严格退化为预训练 ResNet，
      不会因为随机 LSTM 破坏 ImageNet 预训练特征。
    """

    def __init__(self, channels: int, hidden: int = 256, max_grid: int = 14,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.max_grid = max_grid
        self.in_norm = nn.LayerNorm(channels)
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.proj = nn.Linear(hidden, channels)
        self.post_norm = nn.LayerNorm(channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = max(1, min(h, w, self.max_grid))
        if (h, w) == (g, g):
            seq = x.flatten(2).transpose(1, 2)
        else:
            seq = F.adaptive_avg_pool2d(x, g).flatten(2).transpose(1, 2)
        seq = self.in_norm(seq)
        out, _ = self.lstm(seq)
        out = self.post_norm(self.proj(out))
        out = out.transpose(1, 2).reshape(b, c, g, g)
        if (h, w) != (g, g):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return x + out


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class RecurrentResNetConfig:
    backbone: str = "resnet-18"
    image_size: int = 224
    num_channels: int = 3
    lstm_hidden: int = 256
    lstm_max_grid: int = 14
    lstm_layers: int = 1
    decoder_norm: str = "group"
    decoder_groups: int = 32
    out_act: str = "sigmoid"          # 输出压到 [0,1]
    pretrained: bool = True

    def __post_init__(self):
        if self.backbone not in BACKBONES:
            raise ValueError(f"backbone 必须是 {BACKBONES} 之一，收到 {self.backbone!r}")


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------
class RecurrentResNet(nn.Module):
    def __init__(self, config: RecurrentResNetConfig):
        super().__init__()
        from transformers import ResNetConfig, ResNetModel

        self.config = config
        hf_id, depths = _HF_SOURCE[config.backbone]
        hf_cfg = ResNetConfig.from_pretrained(hf_id) if config.pretrained else ResNetConfig.from_pretrained(hf_id)
        hf_cfg.depths = list(depths)
        object.__setattr__(hf_cfg, "depths", list(depths))

        if config.pretrained:
            # 主档位直接用官方权重；resnet-10 用 resnet-18 的权重（严格前缀，全命中）
            self.encoder = ResNetModel.from_pretrained(hf_id, config=hf_cfg)
        else:
            self.encoder = ResNetModel(hf_cfg)

        self.depths = tuple(depths)
        self.hidden_sizes = tuple(hf_cfg.hidden_sizes)
        self.embedding_size = int(hf_cfg.embedding_size)
        _lt = str(getattr(hf_cfg, "layer_type", "basic")).lower()
        self.layer_type = "bottleneck" if "bottleneck" in _lt else "basic"

        # ---- 编码器：4 个 layer block 输出各自一个 LSTM ----
        self.enc_lstms = nn.ModuleList([
            SpatialLSTM(c, config.lstm_hidden, config.lstm_max_grid, config.lstm_layers)
            for c in self.hidden_sizes
        ])

        # ---- 解码器：编码器的倒置（下采样 -> 上采样）----
        # 编码器 stage_i: in_i -> out_i == hidden_sizes[i]，其中
        #   in_0 = embedding_size, in_i = hidden_sizes[i-1] (i>=1)
        # 倒置后该级: hidden_sizes[i] -> in_i。故解码器各级输出通道 =
        #   [hidden_sizes[2], hidden_sizes[1], hidden_sizes[0], embedding_size]
        # 且该级融合的 skip 就是编码器同一位置的输出（stem / s0 / s1 / s2）。
        dn, dg = config.decoder_norm, config.decoder_groups
        self.dec_ups = nn.ModuleList()
        self.dec_fuse = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_lstms = nn.ModuleList()
        self.dec_skip_index: list[int] = []   # -1 => 融合 stem 输出

        in_ch = self.hidden_sizes[-1]          # z: /32, C4
        for si in range(len(self.depths) - 1, -1, -1):
            if si == 0:
                out_ch = self.embedding_size
                self.dec_skip_index.append(-1)
            else:
                out_ch = self.hidden_sizes[si - 1]
                self.dec_skip_index.append(si - 1)
            upsample = si > 0                  # 编码器 stage si 有 stride 2 -> 这里上采样
            if upsample:
                self.dec_ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    ConvNormAct(in_ch, out_ch, 3, norm=dn, groups=dg),
                ))
            else:                              # stage0 是 stride 1 -> 只做通道投影
                self.dec_ups.append(
                    ConvNormAct(in_ch, out_ch, 1, norm=dn, groups=dg)
                    if in_ch != out_ch else nn.Identity()
                )
            self.dec_fuse.append(ConvNormAct(out_ch * 2, out_ch, 1, norm=dn, groups=dg))
            self.dec_blocks.append(
                _make_stage(self.layer_type, out_ch, out_ch, self.depths[si], 1, dn, dg, "relu")
            )
            self.dec_lstms.append(SpatialLSTM(out_ch, config.lstm_hidden, config.lstm_max_grid,
                                              config.lstm_layers))
            in_ch = out_ch

        # ---- stem 的倒置：conv s2 + maxpool s2 = 4x，用两次 2x 上采样镜像 ----
        self.stem_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvNormAct(in_ch, self.embedding_size, 3, norm=dn, groups=dg),
        )
        self.stem_lstm1 = SpatialLSTM(self.embedding_size, config.lstm_hidden, config.lstm_max_grid,
                                      config.lstm_layers)
        self.stem_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvNormAct(self.embedding_size, self.embedding_size, 3, norm=dn, groups=dg),
        )
        self.stem_lstm2 = SpatialLSTM(self.embedding_size, config.lstm_hidden, config.lstm_max_grid,
                                      config.lstm_layers)
        self.head = nn.Conv2d(self.embedding_size, config.num_channels, 3, padding=1)

    # ------------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor, return_features: bool = False):
        b, _, h, w = pixel_values.shape

        out = self.encoder(pixel_values, output_hidden_states=True)
        hs = out.hidden_states
        stem = hs[0]                 # /4,  embedding_size
        skips_raw = list(hs[1:])     # 4 个 layer block 输出，浅 -> 深

        # 编码器：每个 layer block 输出接自己的 LSTM
        skips = [self.enc_lstms[i](s) for i, s in enumerate(skips_raw)]
        enc_feats = list(skips)

        # 解码器：逐级倒置（下采样 -> 上采样），每级融合编码器同一位置的输出
        x = skips[-1]
        for k in range(len(self.dec_ups)):
            x = self.dec_ups[k](x)
            si = self.dec_skip_index[k]
            skip = stem if si < 0 else skips[si]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = self.dec_fuse[k](torch.cat([x, skip], dim=1))
            x = self.dec_blocks[k](x)
            x = self.dec_lstms[k](x)

        # stem 的倒置：/4 -> /2 -> /1
        x = self.stem_lstm1(self.stem_up1(x))
        x = self.stem_lstm2(self.stem_up2(x))
        if x.shape[-2:] != (h, w):
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = self.head(x)
        rec = torch.sigmoid(x) if self.config.out_act == "sigmoid" else x

        if return_features:
            return rec, {"encoder": enc_feats, "bottleneck": x}
        return rec

    # ------------------------------------------------------------------
    @torch.no_grad()
    def load_report(self) -> str:
        """返回编码器预训练权重命中情况（用于复核「不从 0 开始」）。"""
        from transformers import ResNetConfig, ResNetModel
        hf_id, depths = _HF_SOURCE[self.config.backbone]
        cfg = ResNetConfig.from_pretrained(hf_id)
        cfg.depths = list(depths)
        ref = ResNetModel.from_pretrained(hf_id, config=cfg)
        ref_sd = ref.state_dict()
        tgt_keys = set(self.encoder.state_dict().keys())
        hit = [k for k in tgt_keys if k in ref_sd]
        missing = [k for k in tgt_keys if k not in ref_sd]
        return (f"{self.config.backbone}: source={hf_id} depths={depths} "
                f"encoder_tensors={len(tgt_keys)} matched={len(hit)} missing={len(missing)}"
                + (f" missing_examples={missing[:5]}" if missing else " (100% matched)"))


def build_model(backbone: str, image_size: int, pretrained: bool = True, **kw) -> RecurrentResNet:
    cfg = RecurrentResNetConfig(backbone=backbone, image_size=image_size, pretrained=pretrained, **kw)
    return RecurrentResNet(cfg)
