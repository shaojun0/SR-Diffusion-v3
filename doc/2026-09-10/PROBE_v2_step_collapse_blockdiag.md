# 探针：step1~5 是否还坍缩？（blockdiag 新默认 vs causal 历史）

> 日期 2026-09-10 ｜ 探针 `probe_step_collapse.py` ｜ 512 张 test ｜ 单卡推理，每模型约 10s
> 结论：**仍然坍缩，而且 blockdiag 比 causal 更彻底。**

## 1. 结论

**没有变化 —— 5 步阶梯依旧是"step-1 一肩扛"，且新默认 blockdiag 下后步比 causal 更接近零。**

| 判据 | causal（2026-09-04 产物） | **blockdiag（本次新默认）** | 判读 |
|---|---|---|---|
| `step_px_scale`（各步自身输出像素量级） | `[1.0072, 0.0630, 0.0584, 0.0572, 0.0563]` | `[1.0273, 0.0395, 0.0349, 0.0342, 0.0335]` | 两臂都塌 |
| 后 4 步 / step-1 | **5.6% – 6.3%** | **3.3% – 3.8%** | blockdiag **更塌（约减半）** |
| 渐进曲线（0-255, 累积）| `[21.3069, 21.2949, 21.2912, 21.2927, 21.2989]` | `[19.8614, 19.8425, 19.8393, 19.8423, 19.8498]` | 极差 0.016 / 0.022，**都是全平** |
| 整图 L1（0-255） | 21.2989 | **19.8498** | blockdiag 重建更好（Δ1.45） |
| `z_s` within-std | 0.0541 | **0.0951** | 见 §3 |
| 逐块 register 平均 cos | `[0.619, 0.854, 0.990, 0.998, 0.999]` | `[0.734, 0.876, 0.918, 0.916, 0.926]` | 见 §3 |

第二份独立证据 —— 区域×步 0-255 L1 矩阵（行 = step，列 = 5 个 patch 区）：

```
causal                              blockdiag
step1  17.21 20.57 24.97 22.39 21.38     step1  16.10 19.29 23.44 20.90 19.58
step2  17.19 20.56 24.96 22.39 21.37     step2  16.07 19.27 23.43 20.88 19.56
step3  17.19 20.56 24.96 22.38 21.37     step3  16.06 19.27 23.43 20.88 19.56
step4  17.20 20.56 24.96 22.38 21.36     step4  16.07 19.27 23.43 20.88 19.56
step5  17.21 20.57 24.97 22.38 21.37     step5  16.08 19.28 23.44 20.89 19.57
```

**五行几乎逐位相同** —— 第 2 步之后累积结果不再变化，即后 4 步对最终图**没有净贡献**。
真阶梯应该呈现"对角线下降"（每步把自己那片区域压低），这里完全没有。两臂一致。

## 2. 探针口径与校验

口径**逐行照抄** `doc/2026-09-07/ab_compare.py`：

```
step_px_scale[t] = mean |pixel_head(Y_t)|        ← Y_t 是未累加的原始增量
prog_curve[t]    = 0-255 空间第 t 步累积结果的整图 L1
E_px[t][r]       = 第 t 步累积结果在第 r 个目标区(rows)上的 0-255 L1
区域边界 = [(0,115),(115,230),(230,345),(345,460),(460,576)]，与 history 同
```

因为 `region_slices()` 随 region_loss 一起被还原掉了，边界按 `ab_compare.py` 注释里记录的同组值内联。

**校验（关键）**：拿 causal 旧 checkpoint 跑本探针，与 `doc/2026-09-04/probe_slice05_exp1_results.json` 逐位对比 ——

| 量 | 历史 json | 本探针 | |
|---|---|---|---|
| `step_px_scale` | `[1.00716477, 0.06299830, 0.05844265, 0.05723453, 0.05630243]` | `[1.0072, 0.0630, 0.0584, 0.0572, 0.0563]` | ✅ |
| `full_pixel_l1_255` | `21.298902571201324` | `21.2989` | ✅ |

两条主判据都**精确复现**，所以上表的 blockdiag↔causal 差异是真实差异，不是口径漂移。
（注：`REPORT_v2_block_slice05.md` 里的 20.5587 与本探针的 21.2989 不同源，那是另一次测量的数；
本文的比较全部在**同一次探针运行内**完成，故可比。）

## 3. 两个值得注意的点

**① 重建变好，但不是靠"后步干活"。**
blockdiag 的整图 L1 19.85 明显优于 causal 的 21.30（与训练期 `eval_recon` 0.3318 vs 0.3584 同向），
可是它后步的输出量级**只有 causal 的一半**。⇒ 这 1.45 的增益来自
**"step-1 在一个更干净的设置里独自把活干完"**，不是 5 步分工的收益。
换句话说：blockdiag 把"5 步串行 refinement"实质退化成了**单发模型 + 零增量累加**。
这与 README 的战略判断（v4 单发 8.26 已是仓库最佳、渐进阶梯不在 GOAL 验收内）是同一方向。

**② register 的 cos 更"健康"了，输出反而更塌 —— 实证了 cos 不能当判据。**
blockdiag 的逐块 cos 从 causal 的 `[0.619, 0.854, 0.990, 0.998, 0.999]` 变成
`[0.734, 0.876, 0.918, 0.916, 0.926]`（后三块 0.99→0.92），`within-std` 也从 0.054 升到 0.095 ——
按早期那套"cos 跌破 0.9 就是键塌缩被打破"的说法，这该算**改善**。
但同一批权重下后步输出反而**更接近零**。

这正好是 README §4.1 判据修正所预言的：**键相似度既不必要也不充分**。原因也说得通 ——
blockdiag 让后区 register 不必再当 step-1 的"中继"（`ANALYSIS_k3` §2.4 那条补给线被切断），
样本上就更发散一点；而"后步该输出什么"这件事由**损失结构**决定，mask 改不动它。
⇒ 瓶颈确认在**目标/损失侧**，不在键内容侧。

## 4. 与唯一真正破过该自锁的手段对照

仓库里唯一让后步真干活的是 **P1 分区域掩码损失**（`REPORT_v2_region_loss_AB.md`，slice27 K=63）：

| 配置 | `step_px_scale` | tail cos | within-std |
|---|---|---|---|
| B 旧全图损失 | `[1.05, 0.09–0.11 ×4]` | 0.78 | 0.20 |
| **A 分区域损失** | `[1.34, 0.61, 0.30, 0.46, 0.42]` | **0.49** | **0.56** |
| 本次 blockdiag（slice05 K=35） | `[1.03, 0.039, 0.035, 0.034, 0.034]` | 0.926 | 0.095 |

后步量级从 6% 拉到 45%（A 臂）靠的是**给每步私有目标**；只换 query mask（本次）把它从 6% 压到 3%。
**掩码开关不是这条线的杠杆**，与 `DESIGN_query_mask_mode.md` §5 的"blockdiag 不修 register 塌缩"一致。

（注：A 臂是 slice27/K=63，与本次 slice05/K=35 不同配置，只能作方向性对照，不是同配置 A/B。）

## 5. 对上一轮结果的影响

`doc/2026-09-10/REPORT_v2_blockdiag_slice05.md` 里那个 −0.0266 的 `eval_recon` 提升**依然成立**，
但现在归因要改：它**不是**"5 步渐进终于工作"带来的，而是 blockdiag 去掉跨步干扰后
单发通路收敛更好。若要把这条写进报告，措辞应是
"blockdiag 提升单发重建质量"，而不是"blockdiag 激活了后步"。

## 6. 复现

```bash
cd /root/autodl-tmp/sr-diffusion-v3-qmask
export PATH=/root/miniconda3/bin:$PATH HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
# 新默认 blockdiag 产物
python probe_step_collapse.py \
    --ckpt output/phase1_v2_block_slice05_blockdiag/final_model.pt \
    --tag blockdiag_new --limit 512 --out probe_blockdiag.json
# 历史 causal 对照（无 model_info 字段 → 自动 fallback causal）
python probe_step_collapse.py \
    --ckpt /root/autodl-tmp/sr-diffusion-v2-k99/output/phase1_v2_block_slice05/final_model.pt \
    --tag causal_0904 --limit 512 --out probe_causal.json
```

结果 json：远端 `/root/train_logs/probe_blockdiag.json`、`/root/train_logs/probe_causal.json`；
仓库副本 `doc/2026-09-10/data/probe_blockdiag_slice05.json`、`doc/2026-09-10/data/probe_causal_0904_slice05.json`。
