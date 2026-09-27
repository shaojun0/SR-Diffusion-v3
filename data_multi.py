"""data_multi — 多数据集自然图像微调的数据管线（2026-09-26）

设计要点：
1. **不做 letterbox / 不压到固定画布**：编码器输入按「原生尺寸 + 长边上限 + 面积分桶」，
   得到固定的 patch 网格 (gh,gw) ⇒ 同形状样本可成批（HF Dinov2 无 token mask，
   padding 会被 register 看到，所以只能靠同尺寸分桶）。
2. 每个样本额外产出 **4 个固定分辨率的像素目标**（448×252 / 252×448 / 224×224 / 448×448），
   布局与 model_v2.decode 的 target_pix 完全一致（通道优先 (C,14,14)，row-major patch）。
3. 同时产出 best_idx = 原图最适配分辨率（长宽比最接近，平手比像素数）。

索引（path → 分桶尺寸 + best_idx）扫描一次后缓存 json，worker 不再重算。
"""
import glob
import json
import math
import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler

from model_multi import AREA_BUCKETS, PATCH, RESOS, best_fit_index

DINO_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
DINO_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0

DEFAULT_SOURCES = [
    ("/root/autodl-tmp/data/imagenet_val", (".JPEG", ".jpeg", ".jpg"), 50000),
    ("/root/autodl-tmp/data/coco/train2017", (".jpg",), 118000),
    ("/root/autodl-tmp/data/vimeo", (".png",), 92000),
    ("/root/autodl-tmp/data/div2k/DIV2K_train_HR", (".png",), 800),
    ("/root/autodl-tmp/data/flickr2k", (".png", ".jpg", ".jpeg"), 2650),
    ("/root/autodl-tmp/data/openimages", (".jpg", ".jpeg", ".png"), 15000),
]


def bucket_shape(W: int, H: int, long_side_cap: int = 1280,
                 buckets: Sequence[int] = AREA_BUCKETS) -> Tuple[int, int]:
    """原生尺寸 → (bw, bh)（14 的倍数）：长边截到 cap，再按 patch 数就近归桶，
    保持长宽比（在 patch 网格上取整）。"""
    m = max(W, H)
    if m > long_side_cap:
        s = long_side_cap / float(m)
        W, H = max(PATCH, int(round(W * s))), max(PATCH, int(round(H * s)))
    gw0, gh0 = max(1, W // PATCH), max(1, H // PATCH)
    n0 = gw0 * gh0
    B = min(buckets, key=lambda b: abs(math.log(n0 / b)))
    s = math.sqrt(B / n0)
    gw = max(1, int(round(gw0 * s)))
    gh = max(1, int(round(gh0 * s)))
    return gw * PATCH, gh * PATCH


def build_index(sources=DEFAULT_SOURCES, cache: str = "",
                long_side_cap: int = 1280, seed: int = 42, verbose: bool = True):
    """扫描各数据源 → [(path, bw, bh, best_idx), …]，可选 json 缓存。"""
    if cache and os.path.exists(cache):
        with open(cache) as f:
            idx = json.load(f)
        if verbose:
            print(f"[data] 读缓存 {cache}: {len(idx)} 条", flush=True)
        return [tuple(r) for r in idx]
    idx, rng = [], random.Random(seed)
    for root, exts, cap in sources:
        if not os.path.isdir(root):
            if verbose:
                print(f"[data] 跳过（不存在）: {root}", flush=True)
            continue
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(root, "**", f"*{e}"), recursive=True))
        files = sorted(set(files))
        rng.shuffle(files)
        files = files[:cap]
        kept, failed = 0, 0
        for p in files:
            try:
                with Image.open(p) as im:
                    W, H = im.size
                bw, bh = bucket_shape(W, H, long_side_cap)
                idx.append((p, bw, bh, best_fit_index(W, H)))
                kept += 1
            except Exception:
                failed += 1
        if verbose:
            print(f"[data] {root}: 可用 {kept}（cap={cap}, 读取失败 {failed}）", flush=True)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump([list(r) for r in idx], f)
        if verbose:
            print(f"[data] 写缓存 {cache}: {len(idx)} 条", flush=True)
    return idx


def _norm_tensor(img: Image.Image, w: int, h: int) -> torch.Tensor:
    arr = np.asarray(img.resize((w, h), Image.BICUBIC), np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    arr = (arr - DINO_MEAN) / DINO_STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()      # (3,h,w)


def _to_patches(img: Image.Image, w: int, h: int) -> torch.Tensor:
    """PIL → (N,588) 归一化像素 patch（布局 = model_v2.decode 的 target_pix）。"""
    t = _norm_tensor(img, w, h)                                     # (3,h,w)
    gh, gw = h // PATCH, w // PATCH
    return (t.reshape(3, gh, PATCH, gw, PATCH)
             .permute(1, 3, 0, 2, 4).reshape(gh * gw, 3 * PATCH * PATCH))


class MultiSourceDataset(Dataset):
    def __init__(self, index: List[tuple], resos: Sequence = RESOS, retries: int = 3):
        self.index = list(index)
        self.resos = [tuple(r) for r in resos]
        self.retries = retries

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        for k in range(self.retries):
            p, bw, bh, best = self.index[(i + k * 7919) % len(self.index)]
            try:
                img = Image.open(p)
                img = ImageOps.exif_transpose(img).convert("RGB")
                enc = _norm_tensor(img, bw, bh)                     # (3,bh,bw)
                tgt = [_to_patches(img, w, h) for (w, h) in self.resos]
                return {"enc": enc, "targets": tgt, "best": int(best), "path": p}
            except Exception:
                continue
        raise RuntimeError(f"连续 {self.retries} 张读失败, 起始 {self.index[i][0]}")


class MultiResCollator:
    """批内同形状（由 BucketBatchSampler 保证）⇒ 直接 stack。"""

    def __call__(self, batch: List[dict]) -> dict:
        return {
            "pixel_values": torch.stack([b["enc"] for b in batch]),
            "targets": [torch.stack([b["targets"][i] for b in batch])
                        for i in range(len(batch[0]["targets"]))],
            "best_idx": torch.tensor([b["best"] for b in batch], dtype=torch.long),
            "image_path": [b["path"] for b in batch],
        }


class BucketBatchSampler(Sampler):
    """按编码器输入形状 (bh,bw) 分桶成批；支持 DDP 按 rank 轮转分片。

    - 桶内打乱 → 组批；批间打乱（每 epoch 不同）
    - 末批不足 batch_size 也保留（drop_last=False）；drop_last=True 时丢弃
    - rank/world_size：批次级 round-robin（各 rank 批数可能差 1，用 drop 对齐）
    """

    def __init__(self, shapes: List[Tuple[int, int]], batch_size: int,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = False,
                 rank: int = 0, world_size: int = 1,
                 patch_budget: int = 0):
        self.shapes = list(shapes)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.patch_budget = int(patch_budget)
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = e

    def bs_for(self, shape: Tuple[int, int]) -> int:
        """按 patch 预算收缩 batch：patch 越大 batch 越小（≥1）。"""
        bs = self.batch_size
        if self.patch_budget:
            n = (shape[0] // 14) * (shape[1] // 14)
            bs = max(1, min(bs, self.patch_budget // max(1, n)))
        return bs

    def _batches(self):
        groups = {}
        for i, s in enumerate(self.shapes):
            groups.setdefault(s, []).append(i)
        rng = random.Random(self.seed + self.epoch)
        out = []
        for s, idxs in groups.items():
            if self.shuffle:
                rng.shuffle(idxs)
            bs = self.bs_for(s)
            for k in range(0, len(idxs), bs):
                b = idxs[k:k + bs]
                if len(b) == bs or not self.drop_last:
                    out.append(b)
        if self.shuffle:
            rng.shuffle(out)
        if self.world_size > 1:
            # DDP 必须每 rank 批数相同（否则先跑完的 rank 会卡在 allreduce）
            out = out[: (len(out) // self.world_size) * self.world_size]
            out = out[self.rank::self.world_size]
        return out

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        groups = {}
        for s in self.shapes:
            groups[s] = groups.get(s, 0) + 1
        n = 0
        for s, c in groups.items():
            bs = self.bs_for(s)
            n += c // bs + (0 if self.drop_last else (1 if c % bs else 0))
        if self.world_size > 1:
            n = len(range(self.rank, n, self.world_size))
        return n
