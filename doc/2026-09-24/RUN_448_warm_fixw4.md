# RUN — 448×252 主线 warm_steps + 「固定4」采样计划（工地数据集）

> 2026-09-24 起，远程 `ssh -p 25934 root@connect.westd.seetacloud.com`（2× RTX PRO 6000 96GB）
> 用户口径：「448*252 的 warm_step + 固定4 time step 实验，在工地数据集上」
> 追问确认：臂 = `--step_plan fixed --block 4`（步值自然数、每步固定读 4 个 z_s）；
> 预算 = **A 相 16 epoch + B 相 40 epoch**。

## 1. 一次运行卡

| 项 | 值 |
|---|---|
| 代码 | `/root/autodl-tmp/SR-Diffusion-v3-main`（= 本地 `SR-Diffusion-v3` @ `9376900` + 本次 2 处修补） |
| 数据 | `/root/autodl-tmp/construction_site`（train 7009 / test 3004 parquet） |
| 编码器 | `dinov2-large`（343.0M 可训练，不冻结），序列 1+576+576 = 1153 token |
| 采样计划 | `--step_plan fixed --block 4` ⇒ |T| = **144** 步 [1..144]，窗口宽恒 4（首步 5 含 z_cls），**K = N = 576** |
| 两相配方 | `--warm_steps 3504`：前 3504 优化步 A 相 `decoder.steps=[576]`（单步全读），到点自动切回 144 步 B 相 |
| 预算 | A 16 epoch（3504 步）+ B 40 epoch（8760 步）= **56 epoch / 12264 优化步** |
| 批/LR | 每卡 bs **4** × 2 卡 × grad_accum **4** = 全局 **32**（对齐历史主线）；lr 1.5e-4 cosine + warmup 3%（367 步）/ wd 0.01 / clip 1.0 / seed 42 / depth 2 |
| 精度 | **纯 fp32**（fp16+8bit 已实测抹掉 BPTT 精修，故不采用） |
| 输出 | `/root/autodl-tmp/construction_site/out_448_main_warm_fixw4/` |
| 训练日志 | `/root/train_logs/448_main_warm_fixw4.log`（driver: `..._driver.log`） |
| 启动 | `tools/run_448_main_warm_fixw4.sh`（setsid nohup，已脱离 ssh） |
| 启动时刻 | 2026-09-24 09:56:55 |

## 2. 唯一变量（对照历史主线臂 `output/phase1_v2_bptt`）

历史主线臂 = square 计划 24 步 / 40 epoch / 全局 bs32 / fp32 / 同 seed。
本臂 = **采样计划 square 24 步 → fixed 144 步** + **前 16 epoch 单步全读热启动**。
其余（架构、K=N、损失、LR、batch、精度、数据切分）全同。

## 3. 可行性实测（本机测得，决定 batch）

| 配置（448×252 / dinov2-large / d2 / fp32 / fixw4） | 峰值显存 | 单微批 fwd+bwd |
|---|---|---|
| A 相（单步全读）bs4 | 10.7 GB | 0.24 s |
| B 相（144 步 BPTT）bs1 | 18.6 GB | 1.29 s |
| B 相 bs2 | 38.3 GB | 1.19 s |
| **B 相 bs4** | **75.2 GB**（DDP 训练实测 87.3 GB） | 1.86 s |
| B 相 bs8 | **OOM** | — |

⇒ bs4 是 96GB 卡上限；用 grad_accum 4 保住全局 batch 32。
⇒ 优化步/epoch = 7009//(4×2)//4 = **219**（与历史逐位一致）。
⇒ 预计墙钟：A 相 ≈1.0 h（实测 1.06 s/步）+ B 相 ≈18 h（≈7.4 s/步）+ eval（每 1752 步，共 7 次）≈1 h ⇒ **总计 ≈ 20 h**。

## 4. 本次代码改动（已同步到远程 `SR-Diffusion-v3-main`）

| 文件 | 改动 | 为什么 |
|---|---|---|
| `train_v2.py` | 新增 `--step_plan {square,fixed}` / `--block`，fixed 时按 `fixed_block_starts(N,block)` 显式算步集再传 `decoder_steps`（`select_steps` 只认 square 基表）；`model_info.json` 记 `step_plan`/`block`；**修正 warmup/total_steps 按优化步计**（grad_accum>1 时 `--warmup_ratio` 才正确）；打印改简要 | 主入口原本只有 sweep 工具支持 fixed 计划；warmup 原按微批步算，在 accum=4 下会变成 12% |
| `infer_v2_test.py` | 从 `model_info.json` 读 `step_plan`/`block` 并传给模型；新增 `step_win` / `step_cum_read` / `step_bpp_actual_beta1`（实际累计读入 = hi+1 列），打印 `bppA` 列 | fixed 产物若按 square 解释 `[1..144]` 会**静默读错窗口**；且名义 bpp=(t+1) 比实际小 ~4×（末步 1.32 vs **5.23**），画 RD 必须用 actual |
| `tools/run_448_main_warm_fixw4.sh` | 新增启动脚本（自算 epoch→step、打印预算） | 复现入口 |

> `model_v2.py` 未改（fixed 计划 `step_windows`/`K=N` 逻辑在 `9376900` 已具备）。
> 这些改动**尚未 commit/push**；远程以文件方式同步。

## 5. 监控与自动收尾（无人值守；用户明确要求「跑完 → 提交 GitHub → 关机」）

**双保险**，即使 DSH 会话/本地进程中断也会关机：

1. **本地后台看门狗** `srdiff_448_watchdog.sh`（每 20 min 轮询 → status log；训练进程结束后调用
   `finish_448_warm_fixw4.sh`）：
   - 跑全量 test 推理（`infer_v2_test.py`，3004 张，≈25 min；已用随机权重 ckpt 端到端验证过该路径：plan=fixed/block=4/|T|=144/bppA=5.234 都对）；
   - 拉回 `args/model_info/infer_test/trainer_state.json` + 训练日志（清洗去进度条）；
   - `gen_report_448_warm_fixw4.py` 生成 `doc/2026-09-24/REPORT_448_warm_fixw4.md`（含全量指标、逐步曲线、两相训练曲线、与历史 square24/slice05 对照）；
   - `git commit && git push origin main`（本地 repo 有 push 权限，已验证 `ls-remote` 可达）；
   - 最后向 25934 发 `shutdown`，并写 `SRDIFF_448_SHUTDOWN.md` 记录 push 成败。

2. **远程兜底看门狗** `/root/srdiff_autoshutdown.sh`（pid 已确认在跑，`setsid` 脱离 ssh）：
   训练结束后 4 h 无条件 `shutdown`。本地流水线正常时会先关机，它随机器一起消失；
   本地若也挂了，它保证实例最终会关（结果在数据盘 `/root/autodl-tmp` 跨关机保留）。

> ⚠️ `/usr/bin/shutdown` 会 kill supervisord ⇒ sshd 一起被杀，**无法再从 ssh 验证**；
> 已提示用户到 AutoDL 控制台确认状态确为「已关机」。

查进度（任选）：
```bash
tail -3 ~/dsh/srdiff_448_warm_fixw4_status.log
ssh -p 25934 root@connect.westd.seetacloud.com 'bash /root/srdiff_status.sh'
ssh -p 25934 root@connect.westd.seetacloud.com 'tail -c 800 /root/train_logs/448_main_warm_fixw4.log | tr "\r" "\n" | tail -3'
```

## 6. 边界 / 待确认

1. 「40 epoch」按**历史全局 batch 32** 折算（219 优化步/epoch）——若本意是别的 batch 口径，需重跑。
2. 「固定4」= 每步读 4 个 token（|T|=144），不是「4 个时间步」；若本意是后者（slice[0:4]→4 步）请尽快叫停。
3. B 相 144 步 BPTT 显存占卡 87/95 GB（90%）；已开 `expandable_segments`，并每 1752 步存 checkpoint 以便 OOM/崩溃后 `--resume`。
4. 单 seed；固定计划的读窗口恒 4，按 `REPORT_fixw4_plan.md` §3 机制在**高 N 上预期偏弱**（336² 曾低于 DC 基线），本臂正是把该结论搬到主线 448×252 验证。
