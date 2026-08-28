# register K 压缩版 v4 — DINO 不冻结对照实验（2026-08-28）

> 状态：已实现（`model_v4.py` / `train_v4.py` / `infer_v4.py` / `visualize_v4.py`），
> 本地自检 + 迷你训练冒烟通过（loss 下降、DINO 参数真实更新、分组 lr 生效），
> 待服务器全量训练验证。

---

## 1. 实验定位（对照矩阵）

| 实验 | 编码器 | K | DINO | 全量 L1 (0-255) |
|---|---|---|---|---|
| v2 像素目标 | ReEncoder 路由 | 576 | 可训(单 lr) | 23.41 |
| register (v2 模式) | specials 进 DINO 24 层 | 576 | 可训 | 24.14（无效） |
| **v3 BLIP-2** | QFormer 交叉注意力 | 64 | **冻结** | **17.77（当前最优）** |
| **v4（本实验）** | **specials 进 DINO 24 层** | **64** | **不冻结** | 待跑 |

v4 与 v3 的变量只有两处：编码器形态（QFormer → register specials）+ DINO
冻结 → 可训；解码器（v3 式无时序输出查询）、像素头（2 层 MLP）、损失、数据
口径全部一致。

**判读规则**：
- v4 ≥ 17.77 ⇒ register 深度编码不敌 QFormer，且 DINO 微调无益（或微调扰动特征）；
- v4 < 17.77 ⇒ "K 个 register token 由可训 DINO 直接算"能适配像素重建，有增益；
- 探针（h→线性解码 ≈ 全量 L1 ⇒ 解码链路通；z_s 均值→线性解码低 ⇒ 压缩表示
  均值无信息、信息在 token 间分工）定位剩余差距在哪一层。

## 2. 架构

```
pixel_values (B,3,448,252)
  → dinov2.embeddings(x) → (B,1+N,D) [cls; patches]
  → specials (B,K=64,D) 拼入: [cls; specials; patches]  (1+64+576 = 641 token)
  → DINO 24 层（全双向注意力, **不冻结**）→ layernorm
  → z_s = seq[:,1:1+K]   (B,64,D)   ← K 固定压缩表示
  → OutputQueryDecoder（576 行查询 × 64 键, 无时序）→ (B,576,D)
  → PixelHead（2 层 MLP 1024→2048→588）→ 像素 (B,576,588)
  → L = L1(pixels, target)（平权）
```

与 v2 register 版（K=N=576、时序解码器）的差异：K 固定为压缩数、解码器换成
v3 式无时序——v2 register 的 24.14 证明"576 个 register token + 时序解码器"
无效，本实验把 v3 有效的两件事（K=64 压缩压力 + 无时序解码）与 register 编码
合并，只看"编码器形态 + DINO 可训"两个变量。

## 3. 训练口径

- 全 fp32、bs16/卡×2、40 epochs / 8760 步、cosine + warmup 3%；
- **分组学习率**（v2 教训：单 lr 1.5e-4 对 304M DINO 可能扰动预训练特征）：
  DINO `--dino_lr 1.5e-4`（微调不宜过高）+ 新模块 `--lr 3e-4`；
- 显式优化器只收 `requires_grad` 参数（冻结时 DINO 参数不进 AdamW）；
- 时长预估：641 token（vs v2 register 1153 token），DINO 全反传，预计 ~1.5-2.5h。

## 4. 命令

```bash
# 冒烟
accelerate launch --multi_gpu --num_processes 2 train_v4.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v4_smoke --smoke --max_steps 500 \
    --eval_every 250 --save_every 5000

# 全量（默认 DINO 不冻结）
accelerate launch --multi_gpu --num_processes 2 train_v4.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v4_registerK

# 推理（全量 L1 + 双探针, 与 v3 同口径）
python infer_v4.py --data_dir ... --dino_dir ... \
    --final_model output/phase1_v4_registerK/final_model.pt \
    --output output/phase1_v4_registerK/infer_test.json --probe_limit 128

# 可视化
python visualize_v4.py --data_dir ... --dino_dir ... \
    --final_model output/phase1_v4_registerK/final_model.pt \
    --out output/phase1_v4_registerK/recon_visual.png
```

> 推理/可视化参数必须与训练一致（`--num_specials`、`--freeze_dino`）。
> 若想单独看"DINO 冻结 + register K=64"（把 v4 的 DINO 变量也消掉），加
> `--freeze_dino` 跑一版即可——那是 v3 的 register 孪生对照。

## 5. 相关文件

- `model_v4.py`（SpecialTokens + SRPhase1V4; 复用 `model_v3` 的解码器/像素头）
- `train_v4.py` / `infer_v4.py` / `visualize_v4.py`
- 参照：`REPORT_v3_blip2_experiment.md`（17.77, 双探针口径）、`REPORT_register_fp32_train.md`（register N=576 无效）
