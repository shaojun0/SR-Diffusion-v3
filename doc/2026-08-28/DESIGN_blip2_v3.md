# BLIP-2 式最小版 v3 — K 压缩 × 重建（2026-08-28）

> 状态：已实现（`model_v3.py` / `train_v3.py` / `infer_v3.py` / `visualize_v3.py`），
> 本地自检 + 迷你训练冒烟通过，待服务器全量训练验证。
> 起因：register_specials（specials 合并进 DINO）测试失败；用户拍板"完全按
> BLIP-2 来，先简单，直接输出固定维度（如 64），据此还原，不弄多余时序"。

---

## 1. 架构（一句话）

```
pixel_values (B,3,448,252)
  → DINOv2-large（默认冻结, no_grad）→ feats (B,257,D)
  → QFormer: K=64 个可学习 query, 自注意力 + 交叉注意力读 feats → z_s (B,64,D)
  → OutputQueryDecoder: N=576 个查询行（行 k ↔ patch k）交叉注意力读 64 个键
    → (B,576,D)   ← 单次前向, 无采样步/kv_causal/渐进曲线
  → PixelHead（2 层 MLP 1024→2048→588）→ 像素 (B,576,588)
  → L = L1(pixels, target)（平权）
```

对应 BLIP-2 三要素：**冻结视觉编码器 + 可学习查询桥接 + 输出喂下游**（本项目下游暂为像素头）。

## 2. 为什么这样回应两个担忧

| 担忧 | 设计回应 |
|---|---|
| 解码器参数量不够 → 欠拟合 | 解码器可配 `--decoder_depth`（默认 2 层交叉注意力 ≈ 25M）+ 像素头 2 层 MLP（`--head_hidden 2048`, 比 v2 单层线性 0.6M 大 3.5×）; 不够还可加大 `--qformer_depth` / `--mlp_ratio` |
| 数据不够（7009 张）→ 过拟合 | DINO **默认冻结**（可训练参数 367M → ~60M; BLIP-2 惯例; Phase 2 本来就要冻结编码器）; `--train_dino` 可解冻对照 |

另外：K=64 压缩实验**同时就是项目验收**（`GOAL_compression_for_nlp.md` §2: K 压缩 × 重建质量；旧 k-sweep 基于特征目标模型不可外推）。

## 3. 判据（跑完看什么）

| 指标 | 参照 | 目标/判读 |
|---|---|---|
| 全量像素 L1 (0-255) | v2=23.41; 平均色≈61; DINO 线性解码上限≈9.8 | ≤16 则解码链路基本成立 |
| **K 压缩探针**（infer_v3 自带）: z_s→像素线性 L1 | 同 9.8 口径 | ≈9.8-13 ⇒ 信息在 64 token 里, 差距在解码器（欠拟合/结构）; ≈23+ ⇒ QFormer 丢信息（K 太小或欠拟合） |
| eval 曲线 | v2 单调 0.53→0.42 | 单调下降无回升 = 未过拟合; 平台/回升 = 过拟合信号（调小容量/加正则/冻结 DINO） |
| 重建结构 | 边缘比 v2≈1/3 | 布局/物体/边界保真即可（纹理级清晰度非目标） |

**两个担忧怎么用数据区分**（infer_v3 的探针 + eval 曲线一起看）：
- 若 `z_s 线性解码 ≈ 9.8-13` 而全量 L1 高 ⇒ **解码器欠拟合** → 加 `decoder_depth`/`head_hidden`/lr；
- 若 `z_s 线性解码 ≈ 23+` ⇒ **压缩表示就没拿到信息** → 加 `qformer_depth`/`num_queries` 或解冻 DINO；
- 若 eval 先降后升 ⇒ **过拟合** → 冻结 DINO / 减小容量 / 加随机增强（远期）。

## 4. 命令

```bash
# 冒烟（先跑通: 500 步, 看 loss 趋势 + 显存）
accelerate launch --multi_gpu --num_processes 2 train_v3.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v3_smoke --smoke --max_steps 500 \
    --eval_every 250 --save_every 5000

# 全量（DINO 冻结, 预计比 v2 快: DINO 只前向不反传, ~1-1.5h）
accelerate launch --multi_gpu --num_processes 2 train_v3.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v3_blip2

# 推理（含 K 压缩探针）
python infer_v3.py --data_dir ... --dino_dir ... \
    --final_model output/phase1_v3_blip2/final_model.pt \
    --output output/phase1_v3_blip2/infer_test.json

# 可视化
python visualize_v3.py --data_dir ... --dino_dir ... \
    --final_model output/phase1_v3_blip2/final_model.pt \
    --out output/phase1_v3_blip2/recon_visual.png
```

> 调参开关：`--num_queries`(K) / `--qformer_depth` / `--decoder_depth` / `--head_hidden` /
> `--mlp_ratio` / `--train_dino`(解冻) / `--lr`(默认 3e-4, 只训新模块可以更高)。
> 推理/可视化时模型参数必须与训练一致（含 `--num_queries`、`--train_dino`）。

## 5. 相关文件

- `model_v3.py`（QFormer / OutputQueryDecoder(无时序) / PixelHead(2 层) / SRPhase1V3）
- `train_v3.py`（HF Trainer + 显式优化器只收可训练参数 + cosine）
- `infer_v3.py`（全量 L1 + K 压缩探针）/ `visualize_v3.py`（原图 vs 重建）
- 参考：`model_v2.py`（v2/register 路径, 保留作对照）; `DIAGNOSIS_clarity.md`（机制分析）;
  `GOAL_compression_for_nlp.md`（K 压缩验收）
