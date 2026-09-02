"""
make_sample_data.py — 生成小样本 parquet 数据（冒烟 / 新开发者试跑用）
=================================================
生成 N 张随机合成"工地风"图片 → out_dir 下 train-0000.parquet /
test-0000.parquet（image 列 = struct{bytes, path}, 与真实数据同构,
可直接喂 train_v2.py / infer_v2_test.py / visualize_recon_pixel.py）。

用法:
    python make_sample_data.py --out_dir sample_data --n_train 32 --n_test 8

冒烟训练（CPU 也能跑, 小模型小图）:
    python train_v2.py --data_dir sample_data \
        --dino_dir <你的 DINOv2 目录> \
        --output_dir output/smoke --smoke --limit 16 --max_steps 3 \
        --eval_every 1 --batch_size 2 --num_workers 0 --model_input 224x126
"""
import argparse
import io
import os

import numpy as np
from PIL import Image, ImageDraw
from datasets import Dataset, Features, Image as HFImage, Value


def make_image(rng, w=1600, h=900):
    """随机合成一张"工地风"图: 天空渐变 + 楼体 + 塔吊 + 噪点。"""
    arr = np.zeros((h, w, 3), np.uint8)
    top = np.array([rng.randint(60, 160), rng.randint(80, 180), rng.randint(140, 220)])
    bot = np.array([rng.randint(20, 80), rng.randint(20, 80), rng.randint(20, 60)])
    for y in range(h):
        t = y / h
        arr[y, :] = (top * (1 - t) + bot * t).astype(np.uint8)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    # 楼体
    for _ in range(rng.randint(3, 8)):
        x0 = rng.randint(0, w - 100)
        y0 = rng.randint(int(h * 0.3), h - 60)
        x1 = x0 + rng.randint(60, 260)
        y1 = min(h, y0 + rng.randint(80, 300))
        color = tuple(int(c) for c in rng.randint(40, 200, 3))
        d.rectangle([x0, y0, x1, y1], fill=color)
    # 塔吊
    for _ in range(rng.randint(1, 3)):
        x = rng.randint(100, w - 100)
        y0 = rng.randint(int(h * 0.4), h - 20)
        boom = rng.randint(120, 300)
        d.line([x, y0, x, y0 - boom], fill=(40, 40, 40), width=6)
        d.line([x, y0 - boom, x + rng.randint(60, 200), y0 - boom],
               fill=(40, 40, 40), width=4)
    # 噪点
    noise = rng.randint(0, 12, (h, w, 3)).astype(np.uint8)
    img = Image.fromarray(np.clip(np.asarray(img).astype(int) + noise, 0, 255).astype(np.uint8))
    return img


def main():
    p = argparse.ArgumentParser(description="生成合成工地样本 parquet（冒烟用）")
    p.add_argument("--out_dir", default="sample_data")
    p.add_argument("--n_train", type=int, default=32)
    p.add_argument("--n_test", type=int, default=8)
    p.add_argument("--size", default="1600x900", help="原图尺寸 WxH")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.RandomState(args.seed)
    W, H = (int(v) for v in args.size.lower().split("x"))

    def build_rows(n, tag):
        rows = {"image": [], "image_caption": [], "violations": []}
        for i in range(n):
            img = make_image(rng, W, H)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            rows["image"].append({"bytes": buf.getvalue(), "path": f"{tag}_{i:04d}.jpg"})
            rows["image_caption"].append(f"合成工地样本 {tag} #{i}")
            rows["violations"].append(None if i % 3 else "无安全帽")
        return rows

    os.makedirs(args.out_dir, exist_ok=True)
    feats = Features({
        "image": HFImage(),
        "image_caption": Value("string"),
        "violations": Value("string"),
    })
    for tag, n in (("train", args.n_train), ("test", args.n_test)):
        rows = build_rows(n, tag)
        ds = Dataset.from_dict(rows, features=feats)
        path = os.path.join(args.out_dir, f"{tag}-0000.parquet")
        ds.to_parquet(path)
        print(f"[ok] {path}: {n} 条 ({os.path.getsize(path) // 1024} KB)")
    print("完成: 可直接用 --data_dir sample_data 冒烟训练")


if __name__ == "__main__":
    main()
