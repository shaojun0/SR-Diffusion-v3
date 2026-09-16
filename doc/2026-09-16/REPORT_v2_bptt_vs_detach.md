# BPTT vs detach：循环 carry 反传对照实验（2026-09-16）

## 0. 结论（TL;DR）

| 指标 | run1 `detach`（当前默认） | run2 **BPTT**（不 detach） |
|---|---|---|
| 全量 test `full_norm_l1` | 0.412697 | **0.285709** |
| 全量 test 像素 L1 (0–255) | 23.72 ± 8.44 | **16.42 ± 5.86** |
| 24 步逐步 L1（首→末） | 23.65 → 23.72（**平**） | 19.10 → 16.42（**下降**） |
| 相对提升 | — | **−30.8%** |

- 只改了 `model_v2.py` 循环 carry 的**一行**（去掉 `.detach()`），其余逐项对齐 ⇒ 干净的对照实验。
- 在**本实验的操作点（24 步全轨迹）**上，`doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md` 描述的"后步输出≈0 / 零增量自锁"**被打破**：逐步曲线不再平（**该文命题本身是 5 步配置**，见下方操作点声明）。
- **显存与步速与 run1 逐位不变**（87153 MiB/卡、1.83 s/it），原因见 §4 ⇒ 这 −30.8% **不是拿算力换的**。
- ⚠️ **单 seed（42）各跑一次，无误差棒**；结论强度以此为前提。

> ### ⚠️ 操作点声明（读本文前必看）
>
> 本实验跑的是 **`--slice_start` / `--slice_end` 均为 `null` 的全轨迹配置**：
> `N = 576`、`K = 576`、**24 个采样步** `[1, 4, 9, …, 529, 576]`（`model_info.json` 可核）。
>
> 这**不是**仓库里"后步坍缩 / 零增量自锁"那条线用的配置。那条线全部跑在
> **slice[0:5] → `K = 35` → 5 个采样步 `[1, 4, 9, 16, 25]`** 上：
>
> - `doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md` 的命题原文是"v2 时序解码 **5 步**渐进曲线永远全平"；
> - `doc/2026-09-15/DESIGN_v2_recurrent.md` §3 的训练配方写死 `--slice_start 0 --slice_end 5`；
> - 其 §6 判据基线 = slice[0:5] K=35 的 `eval_recon` **0.3318**
>   （`doc/2026-09-10/REPORT_v2_blockdiag_slice05.md`，注意那是**并行架构** `d72e5eb` 的成绩，只能作跨代参照）。
>
> ⇒ **本文结论成立于「24 步全轨迹」这个操作点，不能直接外推到 slice05，也不能与 0.3318 相比。**
> slice[0:5] + BPTT 需单独跑一臂才算回答那个问题（见 §7 建议后续）。
>
> 附带数据点（由本次 run1 得到）：detach 版在 **24 步**下逐步曲线同样全平（23.65 → 23.72），
> 说明"后步坍缩"并非只存在于 5 步配置——这一点此前未在仓库里记录过。

## 1. 这次跑的是什么

| 项 | run1（detach） | run2（BPTT） |
|---|---|---|
| 代码 | `main` HEAD `033cb27`（原样） | 同 HEAD + `model_v2.py:528` 一行 patch |
| 远端目录 | `/root/autodl-tmp/SR-Diffusion-v3` | 同（同一工作区，只多这一行） |
| 输出 | `output/phase1_v2_run1_detach/final_model.pt` | `output/phase1_v2_bptt/final_model.pt` |
| 启动脚本 | `/root/launch_train_phase1_v2.sh` | `/root/launch_train_phase1_v2_bptt.sh` |
| 日志 | `/root/train_phase1_v2.log` | `/root/train_phase1_v2_bptt.log` |
| 起止 | 09-15 15:04 → 19:40 | 09-15 20:22 → 09-16 00:59 |
| 耗时 | 16523.30 s（4h35m23s） | 16609.87 s（4h36m50s） |
| 硬件 | 2× RTX PRO 6000 Blackwell（97.9 GB/卡） | 同 |
| 训练期显存 | 87153 MiB/卡 | 87153 MiB/卡（**逐位相同**） |
| 步速 | 1.82–1.83 s/it | 1.83 s/it |

**唯一变量**（`args.json` 逐项 diff 见附录 B，除 `output_dir` 外完全一致）：

```diff
@@ model_v2.py:528 @@
-            Y = (self.query_base + Y).detach()                    # 喂给下一步当查询
+            Y = self.query_base + Y                                # [BPTT] 不 detach: 循环 carry 反传
```

共同超参：2 卡 DDP、每卡 `bs=16`、`grad_accum=1`、40 epoch、**8760 步**、`lr 1.5e-4`、
`weight_decay 0.01`、`warmup_ratio 0.03`、`grad_clip 1.0`、`seed 42`、
`eval_every = save_every = 2000`、`limit=0`（全量 7009 训练 / 3004 测试）、
`num_specials` 自动推导 = **576**、采样步 24 个、`slice_start/end = null`（全轨迹监督）。

## 2. 结果

### 2.1 训练期评测（每 2000 步，全量 3004 test）

| step | epoch | run1 `eval_loss` | run2 `eval_loss` | run1 `eval_recon` | run2 `eval_recon` | recon 相对 |
|---|---|---|---|---|---|---|
| 2000 | 9.13 | 0.520586 | 0.451505 | 0.528403 | 0.458352 | −13.2% |
| 4000 | 18.26 | 0.455693 | 0.397184 | 0.458687 | 0.398887 | −13.0% |
| 6000 | 27.40 | 0.425319 | 0.325745 | 0.427281 | 0.325998 | −23.7% |
| 8000 | 36.53 | 0.412451 | 0.291301 | 0.414043 | 0.287951 | **−30.4%** |

差距随训练推进**持续拉大**。run2 在 step 4000 的 `eval_recon = 0.3989` 已优于
run1 训满 8000 步的 `0.4140` ⇒ **收敛效率约翻倍**。

训练末段：

| | run1 | run2 |
|---|---|---|
| `train_loss`（全程平均） | 0.494211 | **0.413971** |
| 末次 log `loss`（step 8760） | 0.4029 | **0.2856** |
| `grad_norm` | 0.2372 | 0.6307 |

### 2.2 全量 test 推理（epoch-40 `final_model.pt`，`infer_v2_test.py`，3004 张）

| 指标 | run1 detach | run2 BPTT | 相对 |
|---|---|---|---|
| `full_norm_l1` | 0.412697 | **0.285709** | **−30.8%** |
| 像素 L1 (0–255) | 23.72 ± 8.44 | **16.42 ± 5.86** | −30.8% |
| 前段（早步）/ 后段（晚步）均值 | 23.65 / 23.70 | **17.67 / 16.40** | — |

单次全量推理耗时 259 s（run1 为 254 s）。

## 3. 逐步曲线：后步坍缩被打破，但**明显饱和**

逐步像素 L1（0–255，`infer_test.json` 的 `step_pixel_l1_255`）：

| 步序号 | 采样步 | run1 detach | run2 BPTT |
|---|---|---|---|
| 1 | 1 | 23.65 | **19.10** |
| 5 | 25 | 23.65 | **16.68** |
| 9 | 81 | 23.65 | **16.46** |
| 13 | 169 | 23.64 | **16.40** |
| 17 | 289 | 23.66 | **16.38**（全程最低） |
| 21 | 441 | 23.68 | 16.39 |
| 24 | 576 | 23.72 | 16.42 |

- **run1**：整条曲线是平的（23.64–23.72），后几步甚至轻微变差 ⇒ 后步无贡献，与 `ANALYSIS_k3_why_later_steps_zero.md` 的诊断一致。
- **run2**：曲线**真的下降**，但**绝大部分增益在最前两步就拿到了**（19.10 → 16.68），之后是缓慢细化并迅速饱和（16.68 → 16.38），末步还略回升 0.04。

**准确表述**：BPTT 确实给后步补上了梯度通路、后步也真的在改善重建（"零增量自锁"这个**均衡**被打破）；
但**递进式分工仍远不充分**——第 2 步之后 22 个步合计只再贡献约 0.3 L1。
距离 README §4.1 判据③想要的"渐进阶梯 / 各步分工"还有很大距离。
⇒ 结论是"**从 0 到有**"，不是"**已经把渐进分工训出来了**"。

## 4. 为什么 BPTT 不增加显存与时间

反直觉结果：去掉 detach 后显存（87153 MiB/卡）与步速（1.83 s/it）与 run1 **逐位相同**。

原因：损失是 `mean_t L1(PixelHead(Y_t), target)`，**每一个采样步都直接进损失**，
所以 24 步解码器的激活本来就必须全部保留到 backward；`detach` 只切断了
"上一步输出 → 下一步查询"这条 **carry 的反传路径**，**从不省显存**（前向数值也与不 detach 逐位相同）。
额外代价仅是 carry 路径上那部分反向计算，被 DINOv2-large 主干的反向开销淹没。

⇒ 两 run **在算力开销上完全可比**，本次 −30.8% 不能用"训练更久/更贵"解释。

## 5. 怎么确认 patch 真的生效（一个自检陷阱，建议修复）

⚠️ 仓库自带的 `model_v2.py` 自检里有一条 `[ok] 梯度按步解耦 … 跨步恒为 0`，
**它检测不出这件事**：该检查算的是 `∂L/∂Y_b`（`Y_b = decoder.last_Y` 的 stack 结果），
而跨步路径是 `Y_total[m] → carry → …`，**不经过 `torch.stack` 节点** ⇒ 无论 detach 与否都恒为对角。
实测：**打 patch 前后**跑 `python model_v2.py` **都是 `ALL CHECKS PASSED`**，即该自检对此改动完全不敏感。

真正有效的判据（本次采用）：

| 判据 | run1 detach | run2 BPTT | 依据 |
|---|---|---|---|
| `inspect.getsource(OutputQueryDecoder.forward)` 含 `.detach()` | True | **False** | 实测 |
| **`dL(step3) / d(query_base)`** | **0** | **17.30** | detach 时为 0 由构造保证（`model_v2.py:523` 注释已写明）；17.30 为本轮实测 |
| `dL(step0) / d(query_base)` | ≠0 | 1.2655 | 实测（run2） |

`dL(step≥1)/d(query_base) 恒为 0` 是 carry 被 detach 的**充分特征**（步 t≥1 的查询含被 detach 的项）。
**建议把这条断言补进 `model_v2.py` 自检**，替换/补充现在那条对 detach 不敏感的检查。

## 6. 复现方式

```bash
git checkout 033cb27
git apply doc/2026-09-16/model_v2_bptt.patch    # 或手工改 model_v2.py:528 那一行
NUM_GPUS=2 ./run_v2_train.sh \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v2_bptt --epochs 40
```

## 7. 边界、兼容性与后续

**已知边界**
1. **单 seed（42）各一次，无误差棒**；−30.8% 的幅度需多 seed 复核（趋势 −13% → −13% → −24% → −30% 与逐步曲线形态变化都很干净，但幅度未做重复性验证）。
2. **权重兼容性**：`detach` 只作用于 autograd，前向数值逐位相同 ⇒ run2 权重在**未打 patch 的原版代码**上做**推理**结果一致；但两 run 参数形状完全相同（482 keys / 343.0M），**续训 run2 必须带 patch**，否则会按 detach 语义**静默**改变梯度口径（`load_state_dict` 不会报错）。
3. **本实验没有改仓库默认行为**：`model_v2.py` 仍是 detach 版；patch 仅作为本文档附件保存。
4. **操作点 ≠ slice05**：本实验是 `K = 576` / **24 步**全轨迹；仓库此前的坍缩刻画与 `0.3318` 基线都在 **slice[0:5] / `K = 35` / 5 步**上。跨操作点外推不成立，slice05 + BPTT 需单独跑（见 §0 操作点声明）。
5. **训练日志里的 detach 提示已过时**：`train_v2.py` 启动时打印的"上一步输出（**detach**）喂回当下一步查询"是**写死的说明文字**，不是从代码推导的；打上本 patch 后该行**照旧打印**，读日志时**不要**据此判断 detach 是否开启。是否开启请用 §5 的判据（`dL(step≥1)/d(query_base)`）。

**建议后续**
- 多 seed 复核 −30.8% 的幅度。
- **补跑 slice[0:5]（`K=35`, 5 步）+ BPTT 臂**：这才是仓库那条线真正的问题所在。对照物有两类——历史 slice05 数字（`eval_recon` 0.3318，但是**并行架构** `d72e5eb`，只能作跨代参照），以及 `doc/2026-09-10/probe_step_collapse.py` 的 `step_px_scale` 判据（`DESIGN_v2_recurrent.md` §6 判据①，比只看 `eval_recon` 更贴题）。若要严格归因，还需在当前代码上补一个 slice05 + detach 的控制臂。
- 既然 BPTT 免费且后步已开始贡献，可**重测 README §4.1 的 P1 分区掩码损失**（`doc/2026-09-07/DESIGN_v2_region_loss.md`）：当时"只给私有目标无效"的结论是在 detach（后步无梯度）前提下得到的，**前提已经变了**。注意该结论**来自 slice05**，重测也应在 slice05 上做。
- 是否把这一行落成仓库默认（或重新做成 `--recurrent_detach` 开关）待定。

## 附录 A：patch

见同目录 [`model_v2_bptt.patch`](model_v2_bptt.patch)（`git apply` 可直接用）：

```diff
diff --git a/model_v2.py b/model_v2.py
index 913d6bc..c9481a5 100644
--- a/model_v2.py
+++ b/model_v2.py
@@ -525,7 +525,7 @@ class OutputQueryDecoder(nn.Module):
             # doc/2026-09-15/DESIGN_v2_recurrent.md §2.4 记的默认
             # （recurrent_detach=False = 整条循环反传 BPTT）**不一致**: 那些开关
             # 已随并行路径删除, 本行是唯一路径; 需要 BPTT 的口径只能改这里。
-            Y = (self.query_base + Y).detach()                    # 喂给下一步当查询
+            Y = self.query_base + Y                                # [BPTT] 不 detach: 循环 carry 反传
         Y = torch.stack(Y_total, dim=1)                          # (B,|T|,N,D) 沿步
         self.last_Y = Y                                          # 采样步全部 patch 预测
         return Y
```

## 附录 B：两个 run 的 `args.json` 差异

**除 `output_dir` 外逐字节相同**（`diff` 输出仅一行）：

```diff
4c4
<   "output_dir": "output/phase1_v2",
---
>   "output_dir": "output/phase1_v2_bptt",
```
