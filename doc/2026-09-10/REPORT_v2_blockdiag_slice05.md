# blockdiag slice[0:5] 单卡复跑结果（2026-09-10）

## 1. 这次跑的是什么

| 项 | 值 |
|---|---|
| 代码 | SR-Diffusion-v3 `main` HEAD `d72e5eb`（query_mask_mode 默认 causal→blockdiag） |
| 远端目录 | `/root/autodl-tmp/sr-diffusion-v3-qmask`（全新目录，未覆盖任何历史产物） |
| 输出 | `output/phase1_v2_block_slice05_blockdiag/final_model.pt`（1.37 GB, fp32） |
| 启动脚本 | `/root/run_blockdiag_slice05.sh`（仓库副本 `doc/2026-09-10/run_blockdiag_slice05.sh`） |
| 日志 | `/root/train_logs/blockdiag_slice05.log` |
| 起止 | 14:29:36 → 17:55:08，**3h25m**，`TRAIN_EXIT=0` |
| 硬件 | 单卡 RTX PRO 6000 Blackwell（97.9 GB），训练期显存 60.2 GB，功耗 ~580 W |

**与历史 `run_train_slice05.sh` 的唯一差别**（控制变量）:

```
num_processes  2 -> 1        (降配置只剩一张卡)
batch_size    16 -> 32       (保持全局 batch = 32)
```

其余逐项对齐：`max_steps 8760`、`lr 1.5e-4`、`warmup_ratio 0.03`、`seed 42`、
`slice[0:5]`、K=35（自动推导）、采样步 `[1,4,9,16,25]`、`eval_every 2000`。
落地校验：本次 `warmup 262` 步、`219 步/epoch`、总 8760 步，与历史 2 卡 bs=16 的输出**逐项一致**，
证明全局 batch 确实还原成 32，而不是只改了名义值。

新默认已在两处留痕：训练日志打印 `query_mask_mode=blockdiag`，`model_info.json` 记录
`"query_mask_mode": "blockdiag"`（消费方 `infer_v2_test.py` / `visualize_recon_pixel.py` 据此解析）。

## 2. 结果：匹配步 eval 全面更低

| step | epoch | **blockdiag**（本次） | causal（2026-09-04 基线） | 差值 |
|---|---|---|---|---|
| 2000 | 9.13 | **0.4860** | 0.5048 | −0.0188 |
| 4000 | 18.26 | **0.4199** | 0.4230 | −0.0031 |
| 6000 | 27.40 | **0.3525** | 0.3928 | −0.0403 |
| 8000 | 36.53 | **0.3325** | 0.3597 | −0.0272 |
| 8760 | 40.00 | **0.3318** | **0.3584** | **−0.0266（−7.4%）** |

辅助口径：终值 `eval_loss` 0.3316 vs 0.3583；全程 `train_loss` 0.4379 vs 0.4656。

## 3. 结论强度：**只能算强提示，不能算结论**

按 `doc/2026-09-10/DESIGN_query_mask_mode.md` §7 自定判据，本次至少缺三样：

1. **n=1 单 seed**。基线本身不单调（step 6000 的 0.3928 → step 8000 的 0.3597），
   单点差异里分不开 seed 噪声。
2. **硬件与数据分片顺序混淆**。两臂一个 2 卡 bs=16/卡、一个 1 卡 bs=32。
   全局 batch 虽同为 32、梯度数学期望相同，但**样本进网顺序不同**，轨迹并非逐位一致；
   所以"mask 模式"与"单卡/分片"两个变量目前**没有分离**。
3. **eval 粒度不足**。§7 要求 `eval_every` 加密到 step 250–450 每 20 步，
   用来观察早期塌缩迹象；本次沿用历史 2000 步间隔以保证可比，看不到早期区段。

另外 §7 对"blockdiag 显著更好"的预判是：**跨步耦合本身在贡献不稳定**，属"一条新线索"，
而不是 §5 预测的"对塌缩中性、收益只在可归因性"。真要这么说，必须先补下面第 4 条。

## 4. 建议的下一步（按性价比排序）

**P0（必做，一趟 ~3h25m）**：用**完全相同的单卡 bs=32 配置**补跑 `--query_mask_mode causal`。
这是唯一能把"掩码模式"从"单卡/分片顺序"里分离出来的控制，直接决定上面 ±0.027 该不该信。
两臂除该 flag 外逐项对齐即可。

**P1（要写进报告再做）**：按 §7 标准协议补 seed（≥3/臂）+ 密集 eval（250–450 每 20 步）+
记录 `‖Δθ‖` 与 Adam `v` 范数，判据用**匹配步 train loss + grad_norm** 而非端点 eval。
单卡下每趟 3h25m，可串行挂夜里。

**P2（可选）**：用 `infer_v2_test.py` 对 `final_model.pt` 出全量 test 指标 + 重建可视化，
做像素层面的直观核验（`model_info.json` 会自动带对 `blockdiag`，无需手传 flag）。

## 5. 环境与自检留痕

- 远端**无 GitHub 网络**（`git ls-remote` rc=124 超时），代码经 tar over ssh 同步，
  `model_v2.py` / `train_v2.py` / `README.md` 的 md5 与本地 HEAD 逐项一致
- `python model_v2.py` → `ALL CHECKS PASSED`
- `python doc/2026-09-10/smoke_query_mask_mode.py` → `ALL CHECKS PASSED`
  （默认=blockdiag、逐块隔离性矩阵、跨步梯度恰好 `0.0e+00` 均实测复现）
- 短冒烟（3 步, limit 64）先验证了新代码训练路径，再启动正式跑
- torch 2.12.1+cu130 / accelerate 1.14.0 / transformers 5.16.1
