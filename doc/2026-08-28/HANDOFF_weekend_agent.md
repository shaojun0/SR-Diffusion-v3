# SR-Diffusion-v3 — 周末交接文档（给家里智能体，2026-08-28 夜 / 周末使用）

> 本文件是**自包含交接**：项目目标、实验时间线与关键数字、根因分析与核心洞察、
> 当前代码状态、进行中实验、环境、待办、已知坑全部沉淀于此。家里的智能体
> **先读本文件**，再按需 clone 仓库读细节文档。代码已在 GitHub
> `shaojun0/SR-Diffusion-v3`（分支 `main`，最新 commit `309f419`，与 origin 同步），
> 本文件不携带代码，全部可 clone 获取。

---

## 0. 给家里智能体的一句话任务

接续推进 **SR-Diffusion Phase 1** 实验分析：
1. 若 v4 实验（register K=64 + DINO 不冻结）已跑完 → **分析结果**（判读规则见 §4.3）；
2. 若未跑 → 按 §4.2 命令在服务器跑（或先做离线分析/设计）；
3. 后续按 §6 待办推进（K 扫描、DINO 策略对照、验收实验）。
**先看文档、别重做已完成的实验**；中文交流；改动前读文件、改动后跑自检
（`python model_v3.py` / `python model_v4.py`）；完成 git commit + push（remote 已含凭据，分支 main）。

---

## 1. 项目目标（权威版 `doc/2026-08-28/GOAL_compression_for_nlp.md` 修订 v2）

- **一句话**：通过 token 压缩训练编码器的"联想能力"（把图像信息压进少量 special token z_s），
  训练完成后**冻结编码器**，作为 `model.py` 的编码器接 **Qwen 生成中文工地描述/隐患**（Phase 2，唯一验收）。
- **像素重建 = Phase 1 训练脚手架 + 信息保持直接探针**：语义是像素的函数 ⇒ 能还原像素 ⇒
  z 携带整图信息 ⇒ 语义信息在 ⇒ **重建质量决定 NLP 天花板**。重建与 NLP 不构成对立，
  对立只发生在**纹理级清晰度**（按文献 [3] 属死/冗余信息，不追）。
- **Phase 1 中间验收 = K 压缩 × 重建质量**（K=32/64/128 下"活信息"= 布局/物体/边界的保真度），
  而非 K=N=576 零压缩下的纹理锐度。
- **未决项**（GOAL §4，讨论中勿当已定）：① K 取值（候选 32/64/128）；② DINO 冻结 or 低 lr or 现状；
  ③ 是否叠加文字 CE 双目标；④ 验收实验设计（冻结→MLP→Qwen 小批量文字训练）。

---

## 2. 实验时间线与关键数字（最重要，别重做）

| 轮 | 配置 | 全量重建 L1 (0-255) | 结论 |
|---|---|---|---|
| A 08-26 | v2 **特征目标**（监督 DINO patch 特征） | 64.9 ≈ 平均色 61 | **假收敛**：工地 patch 特征跨位置 std≈5e-5，学"输出质心"即低 L1 |
| B 08-27 | v2 **像素目标**（监督原始像素）+ ReEncoder 路由 | **23.41** | 重大修复（2.6×）；渐进曲线 t=0 粗 60 / **t≥1 平台 22.7**（2 键≈577 键） |
| C 08-28 | **register_specials**（specials 进 DINO 24 层，K=N=576，DINO 可训） | 24.14 | **编码侧 F1/F2 修复无效**；t≥1 平台依旧；边缘保留仅 ~20% |
| D 08-28 | **v3 BLIP-2**（QFormer K=64 query，**DINO 冻结**，62.7M 可训，28 分钟） | **17.77 ± 6.45** | **当前最优**（比 v2 好 24%）；eval 0.509→0.3097 单调下降无过拟合 |
| E 08-29 | **v4 register K=64 + DINO 不冻结**（已实现，待服务器跑） | 待跑 | 与 v3 只差两变量：编码器形态 + DINO 可训性 |

**关键参照**：全图平均色 ≈61；**真实 DINO 特征线性解码像素上限 ≈9.8**（信息在特征里存在）。
产物目录（服务器）：`/root/autodl-tmp/sr-diffusion-v3-main/output/`（v3 在 `phase1_v3_blip2/`，register 在 `phase1_v2_reg/`）。

### 2.1 v3 双探针定位（infer_v3.py 口径，128 条 test）

| 探针 | L1 (0-255) | 解读 |
|---|---|---|
| z_s **均值** → 线性解码 | 60.83 | 压缩表示均值无像素信息（信息在 token 间分工，within-std=0.97） |
| **解码器输出 h** → 逐 patch 线性解码 | **17.03 ≈ 全量 17.77** | QFormer→解码器→h 链路信息充分，**瓶颈不在解码器** |
| 真实 DINO 特征 → 线性解码（历史） | 9.8 | 冻结 DINO 的信息上限 |

→ 剩余 7.2（17.77 vs 9.8）= **K=64 压缩本身的有损代价 + 冻结 DINO 的适配限制**。

### 2.2 核心洞察（v3 为什么好——已分析完，不要重做）

1. **有效信息通道**：v2/register 解码器输入端的有效独立 token ≈ **2 个**（t≥1 平台：2 键≈577 键，
   其余 574 个 z_s 全是"全局摘要副本"=冗余）；v3 的 64 个 token 彼此独立（within-std=0.97）→
   **~30× 有效带宽** → 这是 23.41→17.77 的第一主因。
2. **压缩压力 = 去冗余机制**：K=64 < N=576 强迫 token 分工（谁冗余谁浪费容量），而 v2 的 576 个
   special 无分工压力、全是全局摘要。**K=64 反而比 K=N=576 好 = 冗余才是敌人**（文献 [3] 的
   token 稀疏/冗余故事）。这直接支撑"K 压缩 × 重建质量"验收方向——压缩不是损失，是去冗余。
3. **冻结 DINO 保特征纯净**：9.8 探针证明原始特征信息足够；v2 单 lr=1.5e-4 微调 304M DINO 可能
   扰动预训练特征；且 v2 里解码器读不出（F3/F4），DINO 适配收益根本到不了像素损失——v3 解码器
   通了，"纯净特征"才第一次被真正读到。
4. **去时序释放容量**：v2 损失含 t=0 不可约项（单键，占损失 ~10%，恒定噪声梯度）+ kv_causal
   25 步梯度稀释 + 必须兼顾"每个前缀都能重建"；v3 单次前向，全部容量只服务最终重建。
5. 次要：PixelHead 单层线性 0.6M → 2 层 MLP 3.3M（探针证明像素头未丢信息，h 线性解码≈全量）。

### 2.3 失败实验的教训（防止重复）

- **register_specials（C 轮）无效** → 编码侧修复不影响清晰度；瓶颈在"解码器能读到的信息量/键的冗余度"，
  不在编码路由（DIAGNOSIS 的 F1/F2 是 F2 冗余键更致命，v3 用压缩压力结构性解决）。
- **特征目标假收敛（A 轮）** → 监督**必须**是原始像素。
- DIAGNOSIS_clarity.md（08-27 离线诊断）根因排序，机制分析仍成立：① 解码侧逐 patch 路由缺陷（F1-F4）
  ② DINO 信息上限 9.8 ③ PixelHead 线性+无空间偏置 ④ 纯 L1 ⑤ 数据量/训练口径。

---

## 3. 当前代码状态（GitHub main 分支，commit `309f419`）

| 文件 | 作用 |
|---|---|
| `model_v3.py` | **BLIP-2 式（当前最优 17.77）**：QFormer（K 个可学习 query 交叉注意力读 DINO 特征，默认冻结）→ 无时序 OutputQueryDecoder（N 行查询 × K 键）→ PixelHead（2 层 MLP）。自检 `python model_v3.py` |
| `model_v4.py` | **register K 压缩（进行中实验）**：K 个 special token 拼进 DINO 输入序列（1+K+N token）由 24 层全双向算 z_s，DINO 默认不冻结；解码器/像素头复用 v3。自检 `python model_v4.py` |
| `model_v2.py` | v2 路径（ReEncoder + 时序解码器）+ register_specials 模式（历史/对照，保留） |
| `train_v3.py` / `train_v4.py` | HF Trainer + 显式优化器 + cosine + 全 fp32；v3 只收可训练参数；v4 分组 lr（dino 1.5e-4 / 新模块 3e-4） |
| `infer_v3.py` / `infer_v4.py` | 全量 L1（归一化 + 0-255）+ **双探针**（z_s 均值 / 解码器 h 线性解码） |
| `visualize_v3.py` / `visualize_v4.py` | 原图 vs 重建蒙太奇 |
| `data_v2.py` | `fit_to_canvas` 确定性预处理（最优旋转角+等比缩放+1600:900 填充）+ ParquetImageDataset + V2Collator |
| `doc/2026-08-26/` | 第一轮：特征目标 + 泄露排查 + 历史 HANDOFF |
| `doc/2026-08-27/` | 像素目标报告 + **DIAGNOSIS_clarity.md**（清晰度根因诊断）+ trace_info_pixel.py（E1 探针） |
| `doc/2026-08-28/` | GOAL（权威目标）、REPORT_v3_blip2_experiment.md、REPORT_register_fp32_train.md、DESIGN_blip2_v3.md、DESIGN_v4_registerK.md、本文件 |

---

## 4. 进行中实验：v4（E 轮，register K 压缩 + DINO 不冻结）

### 4.1 设计
```
pixel_values → [cls; specials(64); patches(576)] = 641 token
  → DINO 24 层（全双向注意力，不冻结）→ layernorm
  → z_s = seq[:,1:1+K]  (B,64,D)   ← K=64 固定压缩
  → v3 式无时序解码器（576 行查询 × 64 键）→ PixelHead(2 层 MLP) → 像素
  → L = L1(pixels, target)（平权）
```
与 v3 只差两变量：**编码器形态（QFormer → register specials）** + **DINO 冻结 → 可训**；
解码器/像素头/损失/数据口径与 v3 完全一致。设计文档 `doc/2026-08-28/DESIGN_v4_registerK.md`。

### 4.2 命令（服务器）
```bash
# 冒烟
accelerate launch --multi_gpu --num_processes 2 train_v4.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v4_smoke --smoke --max_steps 500 \
    --eval_every 250 --save_every 5000
# 全量（默认 DINO 不冻结；--dino_lr 1.5e-4 / --lr 3e-4 分组）
accelerate launch --multi_gpu --num_processes 2 train_v4.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v4_registerK
# 推理（全量 L1 + 双探针；参数必须与训练一致）
python infer_v4.py --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --final_model output/phase1_v4_registerK/final_model.pt \
    --output output/phase1_v4_registerK/infer_test.json --probe_limit 128
# 可视化
python visualize_v4.py --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --final_model output/phase1_v4_registerK/final_model.pt \
    --out output/phase1_v4_registerK/recon_visual.png
```
可选项：`--freeze_dino` 再跑一版 = register K=64 + DINO 冻结（补齐四格矩阵：编码器形态 × DINO 可训性）。

### 4.3 判读规则（跑完看 infer_v4.py 输出）
- 全量 L1 **≥ 17.77** ⇒ register 深度编码不敌 QFormer，且 DINO 微调无益（或微调扰动特征）→ 冻结路线；
- 全量 L1 **< 17.77** ⇒ 可训 DINO + register 深度编码有增益 → 低 lr 微调路线成立；
- 双探针定位剩余差距：`h 线性解码 ≈ 全量 L1` ⇒ 解码链路通，瓶颈在压缩表示本身；
  `h 明显差于全量` ⇒ 解码器欠拟合（加 decoder_depth/head_hidden/lr）。
- 建议补 `verify_visual.py` 同口径测边缘保留（v3 报告没给，register/v2 都是 ~20%），确认清晰度来自活信息还是均值下降。

---

## 5. 环境

- **远程 GPU 服务器**（用户新实例）：`ssh -p 35056 root@connect.westd.seetacloud.com`（免密）；
  torch 2.12.1+cu130、transformers 5.15.1、accelerate 1.14、datasets 5.0.1、Python 3.12.3。
  ⚠️ 实例重启/更换后 **SSH 端口会变、pip 环境需重装**（数据在 autodl-tmp 持久化）。
- 数据：`/root/autodl-tmp/construction_site`（parquet：train 7009 + test 3004，含中文 image_caption/violations）
- DINOv2-large：`/root/autodl-tmp/models/dinov2-large`；Qwen：`/root/autodl-tmp/models/Qwen3.8-27B`（Phase 2）
- 代码目录（服务器）：`/root/autodl-tmp/sr-diffusion-v3-main`
- 训练时长参照：v3 冻结 DINO 28 分钟（8760 步）；v2 register ~2h40m；**v4 预估 1.5-2.5h**
- 网络：HF 官方不可达（hf-mirror 403）→ 模型走 ModelScope、pip 走阿里云镜像、训练/推理设 `HF_HUB_OFFLINE=1`
- 本地开发（工位 Windows `D:\Python_project\SR-Diffusion-v3`）：python 3.9 + torch 2.8.0(cpu) +
  transformers 4.57.6 —— 仅用于跑自检/冒烟；**服务器 transformers 5.15.1 的 Dinov2Model API 与 4.57 一致**，
  encoder layer 返回类型已做 tuple/张量兼容处理。

---

## 6. 待办/下一步（按优先级，用户拍板后执行）

1. **分析 v4 结果**（若已跑）：按 §4.3 判读；写 REPORT 到 `doc/2026-08-28/`；补边缘保留数值。
2. **K 扫描**：v3 的 `--num_queries` 64→128/256，看能否逼近 9.8-13（压缩代价削减）；每档跑 infer 探针对比。
3. **DINO 策略定案**（GOAL 未决项 #2）：v3 冻结 17.77 vs v4 不冻结 待跑 —— 两结果合起来定"冻结 or 低 lr"。
4. 若某实验探针显示 h 明显差于全量 L1 ⇒ 解码器欠拟合 → 加大 decoder_depth/head_hidden，或上
   P1-1 逐 patch 内容查询（`Q = W_q(A_t) + query_base + Linear(z_s[k])`，DIAGNOSIS §5 P1）。
5. **Phase 1 验收实验**：K=32/64/128 × 重建质量（活信息=布局/物体/边界保真，纹理不追）。
6. **Phase 2 预备**：冻结编码器（DINO+QFormer/register）→ z_s 经 MLP 接 Qwen，小批量文字训练
   （中文描述/隐患）——最终唯一验收。数据里已有中文标注（image_caption/violations）。

---

## 7. 已知坑（勿重蹈，代码注释里也有）

- **监督必须原始像素**：工地 DINO patch 特征跨位置 std≈5e-5，监督特征目标会假收敛（学质心即低 L1）。
- **torch 2.x 掩码约定**：`nn.TransformerEncoderLayer` 的 bool 掩码 **True=屏蔽**；`F.scaled_dot_product_attention`
  的 bool 掩码 **True=允许**（相反）——v2 解码器统一用加法浮点掩码（-inf=屏蔽）规避歧义。
- **HF Dinov2Model 无 token 级注意力 mask API** → register 模式 DINO 内全双向（无掩码）。
- **DINO mask_token**：加载后 `dino.config.use_mask_token=False; del dino.embeddings.mask_token`
  （否则 DDP 报"未用参数"）。
- **eval OOM（v2 已修）**：HF Trainer eval 会把全部 batch 预测累积 GPU → `prediction_step` ignore
  巨型键（Y_pix/target_pix）+ `eval_accumulation_steps=2`。v3/v4 输出 dict 也含这些键，若 eval OOM 同样处理。
- **探针口径**：z_s 是 K 个"整图"压缩 token（K≠N），与像素 (M,N,588) 无逐行对齐 → 线性解码用
  **z_s 均值池化复制 N 份**；解码器输出 h 是逐 patch 的，可直接对齐（infer_v3/v4 已实现）。
- **推理/可视化参数必须与训练一致**：`--num_queries`（v3）/`--num_specials`（v4）/`--freeze_dino`/`--train_dino`。
- **fp32 导出**：final_model.pt 必须 fp32（bf16 权重量化使重建 L1 劣化约 2 倍）。
- **"预期收益"都是推断**：v3 之前文档预期 ≤16，实测 17.77——以实测为准，文档结论按实测修订。

---

## 8. 工作方式

- 中文交流；改动前先读文件，改动后跑对应自检（`python model_v3.py` / `python model_v4.py`）；
  实现完 git commit + push（remote 已含凭据，分支 main）。
- 用户（shaojun0）是仓库 owner；遵循仓库风格（解耦、不造轮子、熵减、按日期归档文档）。
- 训练/推理命令参照 README 与 `doc/2026-08-28/` 各 DESIGN/REPORT。
- 服务器若关机，先确认实例/端口是否变化，再谈训练。
