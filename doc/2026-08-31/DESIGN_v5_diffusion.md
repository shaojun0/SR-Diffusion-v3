# SR-Diffusion v5 — 扩散式渐进细化（用户思想的正规实现）

> 日期：2026-08-31 ｜ 文件：`model_v5.py` / `train_v5.py` / `infer_v5.py` / `visualize_v5.py`
> 一句话：用户想法（t=0 粗结构 → 逐步补细节）本身正确；v2 失败是因为
> **机制没实现思想**（前缀截断 ≠ 渐进细化）。v5 用扩散把思想真正落地。

---

## 1. 为什么扩散能实现"渐进细化"（v2 缺的三个耦合）

| v2 缺的耦合 | v5 怎么补上的 |
|---|---|
| ① 步间耦合：`Y_{t+1}` 看不到 `Y_t`，每步从零独立预测整图 | DDPM 反向过程 `x_{t-1}` 的输入就是 `x_t`——每步在上一步基础上细化 |
| ② 每步目标不同：v2 每步监督同一张整图 → 信息全堆进第一个 token（悬崖+平台） | 不同 t 对应不同噪声水平，模型学"去该噪声"；高噪声步目标 = 粗结构 |
| ③ token 增量性：没有机制迫使 `z_s[t]` 装新信息 | **NVAE 式渐进解锁** `m(t)`：t 大（噪声大，粗）只给 z_cls+少量 token；t 小（细）逐步解锁全部 K=128 token——token 只在"细节该出现时"才可用，梯度被迫让后面的 token 装增量信息 |

参考：NVAE（Vahdat & Kautz 2020）的分层潜变量 / 渐进训练（先粗后细）思想，
映射为本模型的"噪声层级 ↔ token 解锁层级"。

## 2. 架构

```
pixel_values (B,3,448,252)
  → dinov2.embeddings(x)                     (B,1+N,D) [cls; patches]+PE
  → specials (B,K=128,D) 拼入: [cls; specials; patches] (B,1+128+576,D)
  → DINO 24 层（全双向, 默认不冻结）→ layernorm
  → z_cls (B,1,D), z_s (B,128,D)             ← K=128 固定压缩表示
  → 扩散解码器（DiT-lite, patch 级, ~40-50M 可训练）:
        x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·ε          (x0 = 归一化像素 patch, B,576,588)
        条件 ctx = [z_cls; z_s[:m(t)]]        (m(t) = 渐进解锁数)
        x̂0 = decoder(x_t, t_emb, ctx)         (x0 预测; CFG 可选)
        L = MSE(x̂0, x0) + 0.5·L1(x̂0, x0)
  推理: DDIM(σ=0) 确定性反向, 每步解锁更多 token → 渐进阶梯曲线
```

- 预训练结构 = DINOv2（项目已有，304M）；扩散解码器是 Phase 1 训练脚手架
  （与 v2/v3/v4 的 decoder 同性质，不需要额外预训练）。
- K=128 固定压缩（576 patch → 128 token）：冗余解容量不够 ⇒ token 被迫分工。

## 3. 探针语义（重要，与 v3/v4 不可直接对比）

v5 推理是从**纯噪声** DDIM 反向（条件生成）：结构/布局/边界（活信息）必须
来自 token；纹理（死信息，GOAL 不追）由扩散过程"编造"。因此：

- **L1 数字 ≠ v4 的 8.26**（那是确定性重建）。v5 的 L1 回答的问题是
  "token 能否驱动还原活信息"，不是"能否逐像素还原"。
- 若结构保真（布局/物体在）而纹理模糊/编造 ⇒ 设计按预期工作。

## 4. 判读方法（infer_v5.py 输出三个诊断，直接检验"思想是否成立"）

1. **渐进阶梯曲线**（staircase）：DDIM 每反向步的 (t, m, L1)。期望**阶梯下降**
   （m 越大 L1 越低）；若平台 ⇒ token 无增量性（v2 病复发）。
2. **m 扫描探针**（最直接的判据）：固定噪声水平 t、同一噪声 ε，逐量解锁
   token（m=0..128）看 L1。**L1 随 m 单调下降 = 每个 token 都在补新信息**。
3. **token 消融**：全程无条件（m=0）vs 有条件 DDIM 的 L1 差 = token 的价值
   （活信息保真度）。

可视化（visualize_v5.py）：原图 | 噪声起点 | 渐进快照（粗→细）| 最终重建，
直观验证"从背景长出结构"。

## 5. 训练口径（train_v5.py）

- 全 fp32、bs16/卡×2、40 epochs / 8760 步、cosine + warmup 3%；
- 分组 lr：DINO `--dino_lr 1.5e-4`（微调不宜过高）+ 新模块 `--lr 3e-4`；
- `--num_specials 128`、`--diffusion_steps 1000`（cosine 调度）、
  `--unlock linear|sqrt|none`（none = 不渐进，对照组）、`--cfg_drop 0.1`；
- eval 用固定 t=T//2（确定性 eval_loss）；
- DDP 用 `ddp_find_unused_parameters=True`（CFG 无条件路径随机跳过交叉注意力）。

## 6. 命令

```bash
# 冒烟（单卡小步数）
accelerate launch --multi_gpu --num_processes 2 train_v5.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v5_smoke --smoke --max_steps 500 \
    --eval_every 250 --save_every 5000

# 全量（默认 K=128, DINO 不冻结, unlock=linear）
accelerate launch --multi_gpu --num_processes 2 train_v5.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v5_diffusion

# 推理（全量 DDIM L1 + 阶梯曲线 + m 扫描 + 消融）
python infer_v5.py --data_dir ... --dino_dir ... \
    --final_model output/phase1_v5_diffusion/final_model.pt \
    --output output/phase1_v5_diffusion/infer_test.json

# 可视化（渐进快照蒙太奇）
python visualize_v5.py --data_dir ... --dino_dir ... \
    --final_model output/phase1_v5_diffusion/final_model.pt \
    --out output/phase1_v5_diffusion/recon_visual.png
```

> 推理/可视化参数必须与训练一致（--num_specials / --diffusion_steps /
> --decoder_depth / --freeze_dino / --unlock）。

## 7. 与 v2-v4 的对照

| 版本 | 解码机制 | K | DINO | 全量 L1 (0-255) | 渐进曲线 |
|---|---|---|---|---|---|
| v2 像素 | 前缀截断（每步独立预测整图） | 576 | 可训 | 23.41 | 悬崖+平台（t≥1 恒 22.7） |
| v3 BLIP-2 | 无时序交叉注意力 | 64 | 冻结 | 17.77 | 无 |
| v4 register | 无时序交叉注意力 | 64 | 可训 | 8.26 | 无 |
| **v5（本版）** | **扩散 + 渐进解锁** | **128** | 可训 | 待跑（口径不同） | **期望阶梯**（m 增 L1 降） |

v5 与 v4 的关系：v4 证明"K 压缩 + 可读出的解码器"能把确定性重建做到 8.26；
v5 在保留 K 压缩的同时，把解码器换成**真正实现渐进细化**的扩散过程——若
用户的"每个 token 补一层细节"思想成立，m 扫描与阶梯曲线会给出直接证据。

## 8. 已知限制 / 风险

1. **探针语义变化**：L1 为条件生成口径，与 8.26 不可直接对比；验收需同时看
   m 扫描/阶梯（增量性）与可视化（结构保真）。
2. **训练成本**：DINO 24 层全反传 + 每步随机采样一个 t，成本与 v4 同级
   （预估 ~2-3h / 8760 步）。
3. **x0 预测在高噪声步较难**（输入≈纯噪声），该步损失主导"从 token 生成粗
   结构"——这是设计意图（t=0 预测大体结构），但优化上比 ε 预测更费力；
   若训练不稳可改 `--l1_weight` 或换 unlock=sqrt（更早解锁更多 token）。
4. **CFG 默认关闭**（cfg_scale=1.0）；若 m 扫描显示 token 增量性存在但最终
   重建偏糊，可试 `--cfg_scale 2-3` 增强 token 作用（推理时）。
