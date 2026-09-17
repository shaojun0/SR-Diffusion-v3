# 方案A（GNN + 置换不变 Readout）分支备份 + 8760 步正式实验启动报告

- 报告时间：**2026-09-17 18:28 CST**（含 18:19 启动验证与 18:28 稳定性复核；服务器 `autodl-container-jnb93wme4w-bf7ae4cd`，本地时间与服务器一致）
- 范围：仅操作服务器 clone `/root/autodl-tmp/sr-diffusion-v3-gnnA`（分支 `feat-gnn-schemeA`）与本地 repo 的 `git fetch` / `git push origin feat-gnn-schemeA`
- 状态：**(A) 分支备份已完成并校验通过；(B) 三臂 8760 步正式实验已在两卡启动并验证稳定运行**（本报告为启动快照，训练仍在后台继续）

---

## (A) 分支备份到 GitHub — 完成

### A.1 服务器直推 origin 不可行（非认证问题，是网络不可达）

- `feat-gnn-schemeA` 的 HEAD = **`793e99a816ee32f41af6802c489ed3f294ce8bf2`**（含 `f24b863` 方案A 接入 + `793e99a` 小工具），工作区干净（`git status --porcelain` 空）。
- 该 clone 的 `origin` 是**本地路径** `/root/autodl-tmp/SR-Diffusion-v3`，不是 GitHub，因此 `git push origin` 目标不是 GitHub。
- 服务器出网到 GitHub 被阻断，实测：
  - `curl https://github.com/` → `curl_rc=124`（超时）
  - `git ls-remote https://github.com/shaojun0/SR-Diffusion-v3.git` → `lsr_rc=124`（超时）
  - 服务器无 `~/.git-credentials`、无 `~/.gitconfig`、无 credential helper
- 结论：走任务书第 2 条 **bundle 中转** 路径。

### A.2 bundle 中转 + 本地 push（实际执行）

服务器侧（只读 clone，除 bundle 文件外无改动）：

```bash
git -C /root/autodl-tmp/sr-diffusion-v3-gnnA bundle create /root/gnnA.bundle feat-gnn-schemeA
git -C /root/autodl-tmp/sr-diffusion-v3-gnnA bundle verify /root/gnnA.bundle
# → "The bundle contains this ref: 793e99a... refs/heads/feat-gnn-schemeA"
# → "The bundle records a complete history."
# 产物 /root/gnnA.bundle = 19,424,942 B, md5 = 27de55b75f2ef5cc95d9eff4d234bc7f
```

中转与推送：

```bash
scp -P 38024 root@connect.westb.seetacloud.com:/root/gnnA.bundle /tmp/gnnA.bundle   # 19424942 B, md5 27de55b75f2ef5cc95d9eff4d234bc7f（一致）
cd /home/linaro/dsh/SR-Diffusion-v3
git fetch /tmp/gnnA.bundle feat-gnn-schemeA:feat-gnn-schemeA   # 新分支，SHA=793e99a
git push origin feat-gnn-schemeA                              # → * [new branch] feat-gnn-schemeA -> feat-gnn-schemeA
```

### A.3 校验

| 项 | 值 |
|---|---|
| 服务器 `feat-gnn-schemeA` HEAD | `793e99a816ee32f41af6802c489ed3f294ce8bf2` |
| `git ls-remote origin feat-gnn-schemeA` | `793e99a816ee32f41af6802c489ed3f294ce8bf2 refs/heads/feat-gnn-schemeA` |
| **一致？** | **是（SHA 完全相同，且 bundle 记录完整历史）** |
| 本地 `main` SHA（推送前后不变） | `fb0bb62225a3ed6657c5dc93bd310db425d930d3` |
| 本地工作区状态 | `git status --porcelain` 空（干净） |
| 本地 stash | `stash@{0}: On main: gnnA schemeA 集成…` 原样保留 |

- **没有 push 到 main**：只新增远端分支 `feat-gnn-schemeA`。
- 本地 repo 只执行了 `git fetch`（bundle）与 `git push origin feat-gnn-schemeA`，未 checkout / 未改动 main 工作区（本报告文件为新增未跟踪文件，未 commit）。

---

## (B) 8760 步正式实验（方案A 三臂 vs K-sweep 锚点）

### B.1 显存探针（实测定档）

任务书要求：先探针，`>92GB` 则三臂改用 `bs16 + ga2`。

| 探针 | 配置 | 结果 | 峰值显存 |
|---|---|---|---|
| ① `probe_K99_bs32` | K=99 (slice 0:9)、**bs32 ga1**、20 步 | **CUDA OOM（第 0 步）** | 已用 94.70 GiB / 容量 94.97 GiB（PyTorch allocated 93.20 GiB，还需 288 MiB） |
| ② `probe_bs16ga2` | K=99 (slice 0:9)、**bs16 ga2**、20 步 | 成功，20/20，train_runtime 54.9 s（~2.74 s/step）、`eval_recon` 正常 | **59,457 MiB = 58.06 GiB = 62.3 GB** |

- 判定：bs32 ga1 在 K=99 直接 OOM（>92GB 阈值，也超过物理容量）；**三臂一律采用 `--batch_size 16 --grad_accum 2`**（等效批量同为 32，任务书指定的回退口径）。
- **口径差异（必须在最终对比中标注）**：锚点为 *单卡 bs32 ga1*；本次方案A 三臂为 *单卡 bs16 ga2*，等效批量相同但**微批大小不同**，梯度累加次数不同 ⇒ 数值非逐位可比，仅同批量语义可比。
- 探针实测峰值（bs16 ga2, K=99）= 59,457 MiB，距 95.6 GB 上限余量充足，无 OOM 风险。
- 探针产物目录：`/root/gnnA_run/probe_bs16ga2/`、`/root/gnnA_run/probe_K99_bs32/`（日志 `/root/train_logs/gnnA_probe_K99_bs16ga2.log`、`gnnA_probe_K99_bs32.log`）。

### B.2 启动方式（`setsid nohup` 脱离 ssh，可续跑）

调度器 `/root/gnnA_run/lane_runner.sh`（仿 `/root/ksweep/lane_runner.sh`：**train → infer_v2_test.py → .done 标记**，支持断点续跑跳过已完成 tag）。

```bash
# 服务器上执行（已 setsid 脱离）
setsid nohup /root/gnnA_run/lane_runner.sh 0 /root/gnnA_run/queue_gpu0.txt 16 2 > /root/gnnA_run/lane0.out 2>&1 < /dev/null &
setsid nohup /root/gnnA_run/lane_runner.sh 1 /root/gnnA_run/queue_gpu1.txt 16 2 > /root/gnnA_run/lane1.out 2>&1 < /dev/null &
# 显存采样（每 10 s）→ /root/gnnA_run/gpu_mem.csv
setsid nohup /root/gnnA_run/gpu_monitor.sh > /root/gnnA_run/gpu_monitor.out 2>&1 < /dev/null &
```

每个实验的等价命令行（以 K=35 为例，`lane_runner.sh` 逐 arm 展开）：

```bash
cd /root/autodl-tmp/sr-diffusion-v3-gnnA
export PATH=/root/miniconda3/bin:$PATH; export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python train_v2.py \
  --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
  --output_dir output/gnnA_sum_K35_8760 \
  --model_input 448x252 --canvas 1600x900 --angle_step 0.5 \
  --epochs 40 --max_steps 8760 --lr 1.5e-4 --weight_decay 0.01 --warmup_ratio 0.03 --grad_clip 1.0 \
  --num_workers 8 --limit 0 --eval_limit 0 --eval_every 2000 --save_every 2000 --log_every 20 \
  --seed 42 --heads 8 --mlp_ratio 4.0 --decoder_depth 2 \
  --batch_size 16 --grad_accum 2 --slice_start 0 --slice_end 5 \
  --gnn_mode sum --gnn_inject replace --gnn_hid 256 --gnn_layers 2 --gnn_conv gin --gnn_k 8
# 训练完成后同目录: python infer_v2_test.py --data_dir ... --dino_dir ... \
#   --final_model output/gnnA_sum_K35_8760/final_model.pt --output output/gnnA_sum_K35_8760/infer_test.json
```

> `infer_v2_test.py` 未显式传 `--slice_start/--slice_end/--gnn_*`，与锚点一致：由 `output_dir/model_info.json` 驱动（训练侧写入 `num_specials`/`decoder_steps`/`slice_*`/`gnn_*` 等，推理侧 `_pick` 优先采用），保证 K 与权重形状对齐。

**未跑** `gnn_mode off` / null 对照（本轮只要求方案A 臂 vs 历史锚点）。

### B.3 PID / 队列 / 输出路径

| lane | GPU | 队列（串行） | lane_runner PID | 当前 train PID | 输出目录 | 日志 |
|---|---|---|---|---|---|---|
| lane0 | GPU0 | `K35 0 5` → `K99 0 9` | **537252** | **537270**（K=35；其 8 个 DataLoader worker 为 537552-537560） | `/root/autodl-tmp/sr-diffusion-v3-gnnA/output/gnnA_sum_K35_8760/`，随后 `…_K99_8760/` | `/root/train_logs/gnnA_sum_K35_8760.log`、`gnnA_sum_K99_8760.log` |
| lane1 | GPU1 | `K48 0 6` | **537337** | **537355**（K=48） | `/root/autodl-tmp/sr-diffusion-v3-gnnA/output/gnnA_sum_K48_8760/` | `/root/train_logs/gnnA_sum_K48_8760.log` |

其它进程：显存采样 `gpu_monitor.sh` PID 536239（每 10 s 写 `/root/gnnA_run/gpu_mem.csv`）；状态文件 `/root/gnnA_run/status/{K35,K48,K99}.status`、`lane{0,1}.log`；完成标记 `/root/gnnA_run/status/<tag>.done`；lane 结束标记 `lane{0,1}.alldone`。

### B.4 启动后 ~3 分钟验证（2026-09-17 18:19）

- **两卡都在跑**：GPU0 `49,951 MiB / 100%`，GPU1 `52,423 MiB / 100%`（`nvidia-smi`）。
- **日志在动**：K35 `102/8760`、1.88 s/it；K48 `96/8760`、1.99 s/it。
- **无 OOM**：`grep -l "OutOfMemory\|CUDA error" gnnA_sum_*_8760.log` 无命中（探针阶段的 bs32 OOM 已记录在探针日志，不属于正式 run）。
- **`gnn_mode=sum` 生效**：两臂日志均有
  - `[model] ★ 方案A 开启: gnn_mode=sum inject=replace conv=gin L=2 hid=256 k=8 | GNN 参数 1.13M; z_s 槽位 35 个 (读窗口上界 25)`（K=48 为 48 槽 / 上界 36）
  - `[train] 7009 样本 | 每卡 bs=16 x 1 卡 | grad_accum=2 | ~438 步/epoch x 40 = 8760 步 | warmup 262 步`
- **步数/口径正常**：`max_steps=8760`、`seed=42`、`slice 0:5 / 0:6`、全量 7009 训练样本、`eval_limit 0`。

**再核查（2026-09-17 18:28，运行 ~13 min）**：

- GPU0 `49,951 MiB / 100%`，GPU1 `52,423 MiB / 100%`（显存峰值与 18:19 完全一致，无增长/泄漏）。
- K35 `403/8760`（1.88 s/it，ETA 4h22m）；K48 `379/8760`（1.99 s/it，ETA 4h38m）。
- 两日志 `Traceback|OutOfMemory` 命中次数 = **0 / 0**。
- 训练在收敛：K35 loss `0.6973 → 0.6505 → 0.6381`（grad_norm ~1.4）；K48 loss `0.7473 → 0.7296 → 0.6729`（grad_norm ~1.1），与 bs16ga2 2000 步短程曲线量级一致。
- 输出目录 `/root/autodl-tmp/sr-diffusion-v3-gnnA/output/gnnA_sum_{K35,K48}_8760/` 已创建（首个 checkpoint 在 step 2000，约 19:20 / 19:23 落盘）。

### B.5 显存峰值（实测，采样自 `gpu_mem.csv`，每 10 s）

| arm | 配置 | 峰值（MiB） | 峰值（GiB） | 上限占用 |
|---|---|---|---|---|
| 探针 K=99 | bs16 ga2 | 59,457 | 58.06 | 60.7% |
| **K=35（运行中）** | bs16 ga2、slice 0:5 | **49,951** | 48.78 | 51.0% |
| **K=48（运行中）** | bs16 ga2、slice 0:6 | **52,423** | 51.19 | 54.7% |
| K=99（待跑，按探针估） | bs16 ga2、slice 0:9 | ~59,457 | ~58.06 | ~60.7% |
| bs32 ga1 K=99 | — | **OOM**（94,700 MiB 时仍差 288 MiB） | — | 不可用 |

（nvidia-smi 总显存 97,887 MiB = 95.59 GiB；`memory.used` 含非 PyTorch 开销。）

### B.6 当前进度与 ETA（快照 2026-09-17 18:28）

| arm | 进度 | 速率 | 训练剩余 | 预计训练完成 | 预计全 arm 完成（含 infer） |
|---|---|---|---|---|---|
| K35 (slice 0:5) | 403 / 8760 | 1.88 s/it | ~4h22m | ~22:50 | — |
| K48 (slice 0:6) | 379 / 8760 | 1.99 s/it | ~4h38m | ~23:06 | ~23:12（lane1 收尾） |
| K99 (slice 0:9) | 排队（lane0 第二棒） | 探针 ~2.3–2.7 s/it | ~5h36m（估） | ~04:35（次日） | **~2026-09-18 04:45 CST 全部完成** |

> ETA 依据：K35/K48 的实测 tqdm 速率外推，K99 由 bs16ga2 探针速率（2.74 s/step 含启动）按步数线性缩放并考虑 9 步 vs 5 步的解码开销；每个 run 另有 5 次全量 eval（eval_every 2000、eval_limit 0，3004 条，实测 ~100 s/次）与末尾 `infer_v2_test.py`（~2–3 min）。
> 每个 tag 完成后自动落 `output/gnnA_sum_<tag>_8760/{final_model.pt,infer_test.json,model_info.json,args}`；lane 阻塞于 ssh 断开不影响（`setsid`）。

### B.7 待对比的锚点（同代码 `bptt@d76f0c5`、**单卡 bs32 ga1**、8760 步、seed42、全量 3004 test）

引用自 `doc/2026-09-16/REPORT_ksweep_bptt.md`：

| K | slice | 锚点 像素 L1 (px) | 锚点 norm L1 | 本次方案A（bs16 ga2） |
|---|---|---|---|---|
| K=35 | 0:5 | **14.1470 ± 5.2324** | 0.246021 | ⏳ 运行中 |
| K=48 | 0:6 | **13.1483 ± 4.9386** | 0.228721 | ⏳ 运行中 |
| K=99 | 0:9 | **12.8812 ± 4.7884** | 0.224071 | ⏳ 排队 |

参考（**非本口径**，bs16+ga2 2000 步短程）：`gnn_mode off` 26.2784 px（eval_recon 0.4576）/ `sum` 23.6203（0.4118）/ `proto` 24.0418 — 说明方案A `sum` 在短程已优于 off。

### B.8 合规与风险备注

- 未改 main：服务器只动 `sr-diffusion-v3-gnnA`（其 `origin` 本地路径未被 push）；本地只 `git fetch` + `git push origin feat-gnn-schemeA`，`main` SHA/工作区/stash 均未受影响。
- 未触碰他人 clone/数据：`/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep`（锚点产物）只读；`/root/autodl-tmp/construction_site`、`/root/autodl-tmp/models/dinov2-large` 只读；未杀任何其他智能体进程；未装包/未 pytest；未删数据。
- 磁盘：`/root/autodl-tmp` 已用 629G / 可用 422G（60%），3 个 run 的 checkpoint（save_every 2000）预计占用 ~50 GB，余量充足。
- **关于本报告自身的版本状态（18:32 补充，须知会）**：本 subagent 全程**未执行任何 main 的 commit / push**（只做了 `git fetch`(bundle) 与 `git push origin feat-gnn-schemeA`）。但编排方在 18:19:36 已把本报告的**上一版本**（167 行）commit 为 `94c8aea` 并 `push origin main`（`git ls-remote origin main` = `94c8aea`）；此后本报告又追加了 18:28 稳定性复核等约 +12/−4 行，**这部分仍是未提交的工作区改动**（`git status` → ` M doc/2026-09-17/REPORT_gnn_schemeA_8760_progress.md`）。是否再次入库由编排方决定；本地 main 的代码文件与 stash 未受影响。
- 需后续核对的点：① 本次为 **bs16 ga2**，与锚点 bs32 ga1 存在微批口径差，最终报告必须标注；② 每个 run 的 `infer_test.json` 出来后按 `full_pixel_l1_255`/`full_norm_l1` 与上表对齐；③ 关注 K=99 实际峰值是否与探针一致。
