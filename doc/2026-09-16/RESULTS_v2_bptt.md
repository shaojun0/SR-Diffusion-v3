# RESULTS：循环 carry BPTT —— Phase-1 重建的当前最好结果（2026-09-16）

> 本文是**总账**（一份可直接引用的结果汇总）。
> 分实验详报：24 步全轨迹 → [`REPORT_v2_bptt_vs_detach.md`](REPORT_v2_bptt_vs_detach.md)；
> slice[0:5] → [`REPORT_v2_slice05_bptt.md`](REPORT_v2_slice05_bptt.md)。
> 原始数据：本目录 [`data/`](data/)。

## 0. 一句话

**去掉 `model_v2.py:528` 循环 carry 的 `.detach()`（开启完整 BPTT），是到目前为止 Phase-1 像素重建上
唯一被单变量对照验证有效的大杠杆**：24 步全轨迹下全量 test 像素 L1 **23.72 → 16.42（−30.8%）**，
并且 24 步逐步曲线**首次由"平"转"降"**（19.10 → 16.38）。

## 1. 实验矩阵

| # | 代号 | 配置 | 代码 | 步数 | K | 全局 batch | 显存/卡 | 耗时 |
|---|---|---|---|---|---|---|---|---|
| A | `24step-detach` | 全轨迹 | `033cb27` 原样 | 24 | 576 | 32 | 87.2 GB | 4h35m23s |
| B | `24step-BPTT` | 全轨迹 | `033cb27` + patch | 24 | 576 | 32 | 87.2 GB | 4h36m50s |
| C | `slice05-BPTT` | `--slice_start 0 --slice_end 5` | `033cb27` + patch | 5 | 35 | 32 | 34.6 GB | 1h32m26s |
| — | `slice05-baseline` | 同上（历史） | `d72e5eb`（**并行** + blockdiag + **累加损失**） | 5 | 35 | 32 | — | 3h25m |

patch = [`model_v2_bptt.patch`](model_v2_bptt.patch)（一行）：

```diff
-            Y = (self.query_base + Y).detach()     # 喂给下一步当查询
+            Y = self.query_base + Y                # [BPTT] 不 detach: 循环 carry 反传
```

**A vs B 是干净的受控对照**：两 run 的 `args.json` 除 `output_dir` 外逐字节相同（见详报附录 B）。

## 2. 主结果

### 2.1 A vs B：同代码、单变量（24 步全轨迹）

| step | A `eval_recon` | B `eval_recon` | 相对 |
|---|---|---|---|
| 2000 | 0.528403 | **0.458352** | −13.2% |
| 4000 | 0.458687 | **0.398887** | −13.0% |
| 6000 | 0.427281 | **0.325998** | −23.7% |
| 8000 | 0.414043 | **0.287951** | −30.4% |

差距随训练**持续拉大**；B 在 step 4000 的 0.3989 已优于 A 训满 8000 步的 0.4140（收敛效率约翻倍）。

**全量 test（3004 张，同一 `infer_v2_test.py`）**：

| 指标 | A detach | B **BPTT** | 相对 |
|---|---|---|---|
| `full_norm_l1` | 0.412697 | **0.285709** | **−30.8%** |
| 像素 L1 (0–255) | 23.72 ± 8.44 | **16.42 ± 5.86** | −30.8% |
| `train_loss`（全程均值） | 0.494211 | **0.413971** | — |

### 2.2 C：slice[0:5]（K=35, 5 步）

| step | `slice05-baseline`（历史，跨代） | C **BPTT** |
|---|---|---|
| 2000 | 0.4860 | **0.4515** |
| 4000 | 0.4199 | **0.3835** |
| 6000 | 0.3525 | **0.3318** |
| 8000 | 0.3325 | **0.3147** |
| 全量 test | 0.3318 / 19.02 | **0.314064 / 18.05 ± 6.50** |

⚠️ 这个 −5.3% **不能归因给 BPTT**：历史基线是**并行架构 + 累加损失口径**，与 C（循环 + 直接预测）
架构与损失口径两处都不同 ⇒ 只能算"没有变差、略有提示"。未跑当前代码的 slice05 + detach 控制臂。

## 3. 逐步曲线（本文最关键的一张表）

| 配置 | 逐步像素 L1 | step1→最好 |
|---|---|---|
| A `24step-detach` | 23.65 → 23.72（**平**，min 23.64） | **0%**（后步无贡献） |
| B `24step-BPTT` | 19.10 → **16.38** → 16.42（**降**） | **14.2%** |
| C `slice05-BPTT` | 18.27 → **18.03** → 18.05 | **1.3%**（基本不分工） |

B 的完整曲线（采样步 1 / 25 / 81 / 169 / 289 / 441 / 576）：

```
19.10  16.68  16.46  16.40  16.38  16.39  16.42
```

**两条并列的结论**：
1. 在 24 步全轨迹上，BPTT **打破了"零增量自锁"均衡**——后步第一次真的在改善重建（14.2%）。
2. 但这个效应**强烈依赖操作点**：slice[0:5] 上后步只改善 1.3%。不能拿 1 去回答 2。

## 4. 可写进论文的四个点

1. **单变量因果证据**：同一代码、同一超参、只改一行 `detach`，全量重建 L1 −30.8%，且差距随训练扩大（−13% → −30%）。这不是调参运气。
2. **诊断被证实的机制**：`doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md` 预测"切断跨步梯度 ⇒ 后步无活可干 ⇒ 输出≈0"；本次 BPTT 让后步输出量级与重建质量同时起来，**该因果链被正面验证**（去掉病因 → 症状消失）。
3. **算力免费**：A、B 显存与步速**逐位相同**（87153 MiB/卡、1.83 s/it）。原因：损失 `mean_t L1(PixelHead(Y_t), target)` 本就要求保留全部采样步的激活，`detach` 只切 carry 的反传路径、从不省显存 ⇒ 该增益**不是用算力换的**。
4. **操作点依赖**：同一杠杆在 slice[0:5]（K=35, 5 步）上几乎无效（1.3%）。结合仓库既有的"可及键数 → L1"单调链（4 键 20.55 / 16 键 19.88 / 36 键 19.15），指向**信息供给**（K / 每步读窗口 / 压缩率）是"后步能否分工"的另一半前提——BPTT 只解决了"梯度通路"这一半。

> 顺带：仓库旧的"~19–20 读出上限"是**旧口径**（并行 + 累加损失 + detach）下测的。本次 B（16.42）与 C（18.05）都已低于那条线，说明该上限是旧读出路线的性质，**不适用于当前代码**。

## 5. 方法学：一个被静默作废的判据（建议修）

`DESIGN_v2_recurrent.md` §6 判据①用 `step_px_scale` 判断后步是否坍缩。本次实测（见
[`data/step_px_scale_slice05_bptt.json`](data/step_px_scale_slice05_bptt.json)）：

| | `step_px_scale` |
|---|---|
| 本次（slice05 + BPTT） | `[1.03246, 1.03658, 1.03897, 1.03959, 1.03985]` ⇒ 全部 ≈1 |
| 旧值（2026-09-10，累加口径） | `[1.0273, 0.0395, 0.0349, 0.0342, 0.0335]` ⇒ 后步塌 |

**全部 ≈1 不代表"坍缩被打破"**。实测同一批数据：`mean|target_pix| = 1.1415`、
`mean|Y_pix| = 1.04487`（= 目标的 **91.5%**）。现行损失把**每一步**都监督成整图 ⇒ 每步输出量级必然 ≈
目标量级，**与是否分工无关**。旧值有判别力是因为它测的是**累加口径下 `Y_t` 作为"增量"**的量级。

⇒ 2026-09-15 把损失从"累加"改成"直接预测"时，这条判据被**静默作废**。
现行口径下唯一有判别力的是**逐步 L1 是否递降**（§3 用的就是它）。
另：`doc/2026-09-10/probe_step_collapse.py` 传已删除的 `query_mask_mode`，在当前代码上直接 `TypeError`。

## 6. 边界（诚实的部分）

1. **单 seed（42）**，各配置只跑一次，无误差棒；−30.8% 的幅度未做重复性验证（趋势与曲线形态变化很干净）。
2. **C 无同代码控制臂** ⇒ slice05 的 −5.3% 不可归因。
3. **A/B 的 −30.8% 可归因**（同代码单变量），但其**绝对水平**不可与 2026-09-10 之前的任何历史数字直接比（损失口径已变）。
4. **"后步为什么在 K=576 行、K=35 不行"尚未分离变量**：`slice` 同时改步数 / K / 块划分 / 每步读窗口 / 压缩率，且 K 与最大步被硬绑定（`K = derive_num_specials(N, steps)`）。候选机制与建议的分离实验见 [`REPORT_v2_slice05_bptt.md`](REPORT_v2_slice05_bptt.md) §4。**本轮未跑该消融。**
5. **未改仓库默认行为**：`model_v2.py` 仍是 detach 版，patch 只作为附件。

## 7. 复现

```bash
git checkout 033cb27
git apply doc/2026-09-16/model_v2_bptt.patch

# B: 24 步全轨迹
NUM_GPUS=2 ./run_v2_train.sh --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v2_bptt --epochs 40

# C: slice[0:5]
NUM_GPUS=2 ./run_v2_train.sh --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v2_slice05_bptt --slice_start 0 --slice_end 5 --epochs 40
```

## 8. 数据文件

| 文件 | 内容 |
|---|---|
| [`data/bptt24_detach_infer_test.json`](data/bptt24_detach_infer_test.json) | A：全量 test 推理原始 json（`full_norm_l1` / `step_pixel_l1_255` 等） |
| [`data/bptt24_bptt_infer_test.json`](data/bptt24_bptt_infer_test.json) | B：同上 |
| [`data/slice05_bptt_infer_test.json`](data/slice05_bptt_infer_test.json) | C：同上 |
| [`data/step_px_scale_slice05_bptt.json`](data/step_px_scale_slice05_bptt.json) | §5 的判据失效证据（含饱和性验证） |
| [`model_v2_bptt.patch`](model_v2_bptt.patch) | 唯一代码改动 |

权重（未入库，服务器 `autodl-tmp`）：
`output/phase1_v2_bptt/final_model.pt`（1.31 GB，建议作为 Phase-1 产物）、
`output/phase1_v2_run1_detach/final_model.pt`、`output/phase1_v2_slice05_bptt/final_model.pt`。

## 9. K-sweep 原始数据

BPTT K-sweep（k15 / k24 / k35c / k48 / k63 / k99 / k120）的原始 `infer_test.json` + `args.json` + md5 索引见 [`data/ksweep/`](data/ksweep/)（逐文件索引：[`data/ksweep/README.md`](data/ksweep/README.md)），对应 [`REPORT_ksweep_bptt.md`](REPORT_ksweep_bptt.md) 及 `REPORT_ksweep_<tag>.md` 分报告。
