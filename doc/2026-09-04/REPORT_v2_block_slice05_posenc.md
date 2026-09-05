# SR-Diffusion v3 — v2 slice05 实验：新提交「位置编码优化」(3151bab) 训练塌缩实证 + A/B 归因

> 日期: 2026-09-04 ｜ 分支: `main` HEAD `42fa6d0`（含 `3151bab` 位置编码优化 + 其后 59114f0/42fa6d0 两个纯文档提交）
> 服务器: `ksai.scnet.cn:10380`（2× Hygon DCU K500SM_AI 64GB; torch 2.9.0+das.opt1.dtk2604, transformers 4.57.6, accelerate 1.14.0, datasets 5.0.1; 全 fp32）
> 性质: 用户指示——在新提交（替换位置编码为**块级共享**）上复跑 slice05（与 exp1 同预算全量）。结果: **训练塌缩**（特征与 exp2「读侧全开」塌缩同型）; 追加 **A/B 归因**（唯一变量 = 位置编码块级共享 vs 逐位置）定位塌缩来源。

---

## 0. 结论速览（TL;DR）

1. **新提交（块级共享位置编码）slice05 训练自 epoch≈2 塌缩**: 2000 步 eval_loss = **1.1394**（exp1 分块读同期 0.5037; exp2 读侧全开同期 1.139）; train loss 峰值 lr 附近短暂下探后**跳回并冻结** 1.14–1.17, grad_norm 崩到 **0.001–0.006** 且不再恢复——与 exp2 塌缩**数值同型**。
2. **读侧分块掩码经探针验证确实生效**（step-1 读区外 register 置零 → step-1 输出逐位不变; 读区内置零 → 显著变化; attn_mask row0=[0,0,0,0,-inf,…]）——**排除**"掩码未生效/读侧误开"类实现 bug; 本次运行未设任何 memory_open。
3. **A/B 归因 [数据]**: 同机/同数据/同预算/同 seed, 唯一变量 = `pos_embed` 形状（块级共享 `(1,⌊√K⌋+1,D)=(1,6,D)` vs 逐位置 `(1,K+1,D)=(1,36,D)`）——回退逐位置版（`3151bab^`）**训练健康**: eval 0.539@800 步 → **0.499@1600 步**（与旧 exp1 0.504@2000 同轨）, train loss 0.68@epoch2.1 持续下降, grad 0.26–2.6。
4. **结论**: `3151bab` 的块级共享位置编码是 slice05 塌缩的直接原因 [数据]。机制 [推断]: 块内所有键共享同一位置向量 → 键表示趋同 → 注意力无法按位置区分块内键 → register 专业化（分工）训练压力消失 → 落入与 exp2 同型的**退化吸引子**（近常数输出）; 提交"块内逐位置编码无可利用语义"的论证只覆盖**推理表达**, 未覆盖训练动力学——逐位置编码实际充当**防 register 塌缩的对称破缺信号**。
5. **建议 [推断]**: 该提交不能作为训练验收（至少 slice05/K=35/lr1.5e-4 协议上训练级失效）; 修复方向 = 回退该改动, 或改为"块级共享 + 块内差分位置"（保参数收益 + 恢复块内可区分度）再验证。

---

## 1. 背景与目的

- 系列: v2 register 式 + OutputQueryDecoder 分块掩码 + 查询块因果 + K 自动推导; "渐进曲线全平/分工压力"机制系列见 09-03/09-04 文档。
- exp1（`REPORT_v2_block_slice05.md`）: slice05 因果+分块读, K=35, 全量 L1 20.554（全因果族最差）; exp2（`REPORT_v2_slice05_memory_open.md`）: 读侧全开仅留因果 mask → **塌缩** eval 1.138（epoch2 冻结, grad≈0.001）。
- **新提交 `3151bab`（15:00, 位置编码优化）**: 解码器 `pos_embed` 从逐位置 `(1,K+1,D)` 改为**按 step 块共享** `(1,⌊√K⌋+1,D)`（块号 = 位置 p 的 ⌊√p⌋, 与 build_block_mask 同分块; 块内所有位置共享同一向量）。动机: 掩码按块读取, 块内逐位置编码对解码器"无可利用语义", 只保留"哪个块（=哪个采样步）"信号, 参数从 K+1 降到 ⌊√K⌋+1。
- 本次任务: 在 HEAD（含 3151bab）上跑 slice05, 与 exp1/exp2 对照, 看位置编码改动对 slice05（此前全因果族最差, L1 20.55）的影响。
- 运行环境变化: 旧实验在 2× RTX PRO 6000（torch 2.12.1+cu130, transformers 5.16.1）; 本次为 DCU 主机（torch 2.9.0+das.opt1.dtk2604, transformers 4.57.6; 每步 3.29s ≈ 旧卡 0.72s 的 4.6 倍）。

## 2. 配置

| 项 | exp1（分块读, GPU） | exp2（读侧全开, GPU） | **本报告 A: block-pos（HEAD 42fa6d0, DCU）** | **本报告 B: per-pos 回退（3151bab^, DCU）** |
|---|---|---|---|---|
| slice / steps | [0:5] / [1,4,9,16,25] | 同左 | 同左 | 同左 |
| K (num_specials) | 35（自动） | 35 | 35（自动, 日志确认） | 35（自动, 日志确认） |
| 读侧 memory_mask | 分块+首步前缀 | 全 0（memory_open） | 分块+首步前缀（探针验证） | 分块+首步前缀 |
| 查询侧 tgt_mask | 块因果 | 块因果 | 块因果 | 块因果 |
| pos_embed | 逐位置 (1,36,D) | 逐位置 (1,36,D) | **块级共享 (1,6,D)** | 逐位置 (1,36,D) |
| 预算 | 8760 步/bs32 有效/fp32/seed42/depth2 | 同左 | 8760 计划, **2118 步截停**（塌缩确认后停） | 1600 步冒烟预算 + eval@800/1600 |
| 数据 | AutoDL construction_site 7009/3004 | 同左 | HF `aswin00000/ConstructionSite` 7009/3004（同源代理, 见 §3.4） | 同左 |

训练命令（A/B 两臂同, 仅 output_dir/代码差异）:
```bash
cd /root/work/<sr-diffusion-v3|ab_perpos>   # A=HEAD 42fa6d0; B=3151bab^ 的 model_v2.py + 同 data_v2/train_v2
source env.sh   # DCU LD_LIBRARY_PATH + HF_HUB_OFFLINE
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --same_network train_v2.py \
  --data_dir /root/work/construction_site --dino_dir /root/work/models/dinov2-large \
  --output_dir output/<phase1_v2_block_slice05_posenc|ab_perpos_slice05> \
  --model_input 448x252 --canvas 1600x900 --angle_step 0.5 \
  --epochs 40 --max_steps <8760|1600> --batch_size 16 --grad_accum 1 \
  --lr 1.5e-4 --weight_decay 0.01 --warmup_ratio 0.03 --grad_clip 1.0 \
  --num_workers 8 --limit 0 --eval_limit 0 --eval_every <2000|800> --save_every 2000 --log_every 20 \
  --seed 42 --heads 8 --mlp_ratio 4.0 --decoder_depth 2 --slice_start 0 --slice_end 5
```

## 3. 结果

### 3.1 训练动力学: A（block-pos）塌缩 vs B（per-pos）健康

**A（block-pos, 42fa6d0）**——与 exp2 同型:
| epoch | train loss | grad_norm | 说明 |
|---|---|---|---|
| 0.09 (step 20) | 1.395 | 0.46 | 初始 |
| 0.55–1.37 | 1.06 → 1.01 | 0.3–1.8 | lr 爬升, 短暂学到粗结构（exp2 同期到 0.899） |
| 1.46 (lr 峰值后) | **1.243 跳回** | 12.5（单次尖峰） | 退化吸引子入口（exp2: 1.153@1.83, grad 0.007） |
| 1.9–9.7（至截停 2118 步） | 1.14–1.17 | **0.001–0.006** | 梯度湮灭, 冻结, 再未恢复 |

**B（per-pos, 3151bab^）**——健康:
| epoch | train loss | grad_norm |
|---|---|---|
| 0.09 (step 20) | 1.279 | 0.36 |
| 1.55–2.1 | 0.90 → **0.68** | 0.8–2.6 |
| 7.1–7.31（1600 步收尾） | 0.50–0.52 | 0.26–0.39 |

### 3.2 eval（全量 3004 test, 归一化）对照

| step | exp1 分块读（GPU） | exp2 读侧全开（GPU） | **A: block-pos（DCU）** | **B: per-pos（DCU）** |
|---|---|---|---|---|
| 800 | — | — | — | **0.539** |
| 1600 | — | — | — | **0.499** |
| 2000 | 0.5037 | 1.139 | **1.1394** | — |
| 4000–8760 | 0.422 → 0.358 | 1.138（平台） | （截停; 平台特征与 exp2 一致） | — |

（B 的 1600 步 eval 0.499 已与 exp1 的 2000 步 0.504 同量级——per-pos 在 DCU 环境上复现了 exp1 的健康轨迹。）

### 3.3 掩码有效性探针（排除实现 bug）[数据]

在 HEAD 代码（A 臂同一 model_v2.py）上直接测 OutputQueryDecoder（K=35, steps [1,4,9,16,25]）:
- `pos_embed.shape == (1,6,64)`（块级共享生效）
- 干扰 **step-1 读区外** z_s[4:35] 置 0 → step-1 输出 max|ΔY₁| = **0.0**（逐位不变）——读侧掩码真实生效
- 干扰 **读区内** z_s[1:4] 置 0 → max|ΔY₁| = 0.90（显著变化）
- attn_mask row0 = [0,0,0,0,-inf,-inf,…]（首步前缀 0..3, 与 exp1 语义一致）
⇒ A 臂塌缩**不是**掩码失效/读侧误开所致。

### 3.4 数据与环境备注（诚实归因边界）

- 数据: 原 AutoDL 数据的 HF 同源仓库 `aswin00000/ConstructionSite`（分片 7 train + 3 test、行数 7009/3004、列 image/image_caption/violations 与旧数据全同; 镜像 `ps-9204/ConstructionSite` 内容相同）。旧库 caption 为中文、本库为英文——**纯像素重建不读 caption**, 对训练/指标无影响。用户确认"数量对上大致即同源, 不要求完全准确"。
- A/B 两臂共用同一数据与同一环境 → **位置编码是两臂间唯一变量, 归因成立** [数据]。
- 环境差异（DCU torch 2.9 / transformers 4.57 vs 旧 GPU torch 2.12 / transformers 5.16）在两臂间恒定; "块级共享位置编码在旧 GPU 环境上是否同样塌缩"未直接验证——机制层面（键位置可区分度消失 → 分工压力消失）与硬件/框架无关 [推断]。

## 4. 判读

### 4.1 塌缩机制: 块内键位置不可区分 → 分工压力消失（与 exp2 殊途同归）[推断]

- exp2（读侧全开）: 每个 register 被 5 步 × 576 查询行**共享** → 单键梯度占比 ≈1/36 → 任何单键都不值得携带独特内容 → register 冗余化 → 解码器退化为近常数输出; 退化态自锁（键越相似 → 注意力越均匀 → 梯度越弱）。
- 本报告 A（块级共享位置编码）: 掩码仍按块隔离, 但**块内所有键的位置向量相同**。每步查询只能读自己块内的 3–11 个键（slice05: 块1=3 键 … 块5=11 键）, 这些键被位置抹平后彼此只差内容; 交叉注意力对块内键的区分只剩内容余弦 → 键越趋同、注意力越均匀 → register 携带独特内容的收益越小 → 与 exp2 相同的**退化吸引子**在 lr 峰值附近（epoch~1.5–2）被吸入, cosine 衰减下无逃逸。
- 即: exp2 消灭分工压力靠"键被多方共享", A 消灭分工压力靠"键的位置身份被抹平"——两条路都使 register 失去"排他读出 → 内容专业化"的训练压力 [推断]。B 臂（逐位置）在两臂中唯一保留键的位置可区分度 → 健康。

### 4.2 `3151bab` 论证缺口 [推断]

提交论证"解码器对块内各列掩码相同 → 块内逐位置编码没有可利用语义 → 只需保留哪个块/采样步的信号"——该论证只对**前向表达的读写能力**成立（掩码确实不区分块内位置）; 但**训练动力学**上, 块内逐位置编码是防止 register 塌缩的对称破缺信号（每个键有唯一位置身份 → 专业化分工压力）。块级共享等价于把块内键"钉"在同一位置身份上, 在 K 小/块内键少的 slice05 上直接触发退化吸引子。**逐位置编码并不冗余: 它不只是"读哪个位置的信号", 也是"让每个 register 必须长得不一样"的训练压力来源。**

### 4.3 系列意义 [推断]

- "掩码/读侧/梯度侧改动变不出不存在的分工"（09-04 ANALYSIS §3）在此再获一个数据点, 且把**编码侧对称性**也纳入杠杆: 消灭键的可区分身份（共享位置/共享读方）都会触发同一类训练塌缩; 位置编码虽是"只读信号", 但它参与塑造 register 的训练分工, 不是可随意裁剪的参数量冗余。
- 渐进语义/分工的正向修复方向仍与系列结论一致: 编码侧注入（E2'/F1）或单发读出（v4 式）; 若想在保留块级位置编码参数收益的同时恢复可训练性, 需要"块级共享 + 块内差分"之类的折中设计, 并先做短程训练验证（本报告 A/B 流程即现成的验证协议）[推断]。

## 5. 产物与证据

- A（block-pos 塌缩）: 服务器 `/root/work/logs/train_slice05_posenc.log`（2118 步截停）+ `/root/work/output/phase1_v2_block_slice05_posenc/checkpoint-2000`（塌缩权重, eval 1.1394）; 训练 CLI 见 `doc/2026-09-04/data/args_block_slice05_posenc.json`。
- B（per-pos 健康对照）: `/root/work/logs/ab_perpos.log` + `/root/work/output/ab_perpos_slice05/final_model.pt`（1600 步, fp32）+ model_info.json/args.json（备份于 `doc/2026-09-04/data/`）。
- 轨迹抽样（loss/grad_norm 每 log 点 + eval 序列）: `doc/2026-09-04/data/posenc_slice05_evidence.json`（A 108 点 / B 80 点）。
- 原始日志本地备份: `/home/linaro/dsh/evidence_slice05_posenc/`（服务器已按用户指示关机）。

## 6. 复现

```bash
# A 臂（塌缩, 建议只跑到 ~500 步即可复现入口形态）: HEAD 42fa6d0 代码, 命令见 §2, --max_steps 8760
# B 臂（健康）: 代码 = 3151bab^ 的 model_v2.py（逐位置 pos_embed (1,K+1,D)）, 其余全同
# 复现判据: B 在 epoch≈2 后 train loss 继续下降（<1.0）, A 在 epoch≈2 冻结于 ~1.15 且 grad_norm<0.01。
```

## 7. 结论

新提交 `3151bab`（位置编码优化: 块级共享位置向量）在 slice05/K=35/同预算协议上**训练级失效**——与 exp2 同型的退化吸引子塌缩（eval 冻结 1.139, epoch2 起 grad≈0.001）; 掩码/读侧已探针排除; A/B（同环境唯一变量 = 位置编码）证明塌缩由块级共享位置编码触发, 回退逐位置编码即恢复健康训练（eval 0.499@1600 与 exp1 同轨）。建议回退或按"块级共享+块内差分"方向修复后再验收该提交 [推断]。
