# REPORT_gnn_schemeA_8760 — 方案A（GNN + 置换不变 Readout）8760 步正式实验（K=35/48/99，sum+replace）

> 报告时间：**2026-09-18 ~09:15 CST** ｜ 服务器：`autodl-container-jnb93wme4w-bf7ae4cd`（`ssh -p 38024 root@connect.westb.seetacloud.com`）
> 代码：服务器 clone `/root/autodl-tmp/sr-diffusion-v3-gnnA`，分支 `feat-gnn-schemeA`，HEAD **`793e99a`**（含 `f24b863`；已备份 `origin/feat-gnn-schemeA`）
> 锚点代码：`bptt@d76f0c5`（clone `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep`）
> 设计文档：`feature/DESIGN_graph_embedding_schemeA.md` §6 注 / §10 E3 ｜ 实现文档：`doc/2026-09-17/DESIGN_IMPL_gnn_schemeA.md`
> 启动/口径背景：`doc/2026-09-17/REPORT_gnn_schemeA_8760_progress.md`
> 本轮为**纯文档核对**：只通过只读 `ssh`（`cat`/`grep`/`ls`/`awk`）读取服务器产物，**未训练、未占用 GPU、未 commit/push**。

**一句话结论**：在 **bs16+grad_accum2 / 8760 步 / seed42 / 全量 3004 test** 下，`--gnn_mode sum --gnn_inject replace` 的 K=35/48/99 三臂像素 L1 分别比同 K 的 `off`（register 式 z_s，bs32+ga1）锚点**差 +13.25% / +17.10% / +21.90%**，且**逐步落差塌缩到 +0.04%～+0.13%**（锚点为 −1.72%/−7.89%/−6.75%）。但本轮**没有**同口径 `off` 对照、也**没有**跑设计文档 §10 E3 指定的 `proto`（k=K）臂，因此**不足以判定"方案A 本身"的优劣**，只能判定"**sum+replace 这一集成方式**在 8760 步下明显劣于现有 register 式 z_s，并把逐步精炼打平"。

---

## 1. 口径声明（**先读这一节再看任何数字**）

### 1.1 训练口径差：本次 bs16+ga2，锚点 bs32+ga1

| 项 | 本轮方案A 三臂 | 同 K 锚点（k35c/k48/k99） |
|---|---|---|
| 单卡微批 `--batch_size` | **16** | **32** |
| `--grad_accum` | **2** | 1 |
| 等效批量 | 32 | 32 |
| 每步前向样本数 | 16 | 32 |
| `max_steps` / epochs | 8760 / 40 | 8760 / 40 |
| `--seed` | 42 | 42 |
| 数据 | `construction_site` 全量 7009 train / 3004 test | 同 |
| lr / wd / warmup_ratio / grad_clip | 1.5e-4 / 0.01 / 0.03 / 1.0 | 同 |
| 代码 | `feat-gnn-schemeA @ 793e99a`（BPTT 基线 `d76f0c5`） | `bptt @ d76f0c5` |

- 依据（服务器实测 `args.json`）：三臂均为 `"batch_size": 16, "grad_accum": 2, "max_steps": 8760, "seed": 42`。
- **为什么用 bs16**：任务书要求先探针，>`92GB` 则回退。实测探针 `probe_K99_bs32`（K=99、slice 0:9、bs32 ga1、20 步）在**第 0 步 CUDA OOM**：`GPU 0 has a total capacity of 94.97 GiB of which 267.56 MiB is free ... this process has 94.70 GiB memory in use ... Tried to allocate 288.00 MiB`（即 **94.70 / 94.97 GiB，还差 288 MiB**）。同配置 bs16+ga2 探针 20/20 成功，`gpu_mem.csv` 该窗口峰值 **59,457 MiB = 58.06 GiB**。⇒ 三臂一律采用 bs16+ga2（任务书指定的回退口径）。
- **这给对比带来的不确定性（必须明说）**：
  1. 等效批量相同，但**微批大小与梯度累加次数不同** ⇒ 每步梯度的噪声结构、以及（若有）任何批内统计量都不同，**数值不是逐位可比**，只能按"同等效批量的语义"比。
  2. 锚点**没有**同 K 的 bs16+ga2 对照，因此**无法把"13–22% 的劣化"在"GNN 集成"与"微批口径差"之间做归因**。历史同类口径差（单卡 bs32 vs 2卡 DDP bs16×2，同 K=35）量级为 **3.9030 px / 21.62%**（`doc/2026-09-16/REPORT_ksweep_k35c.md` §口径校准）——与本轮 K=48/K=99 的劣化幅度同量级，**这是一个真实且不可忽视的混淆项**。
  3. `seed42`、n=1/臂：无误差棒，臂间几个百分点的差异不可用重复性验证。

### 1.2 **推断（test）口径：三臂与锚点相同（bs32 前向）——这一点与"训练口径差"要分开看**

- `infer_v2_test.py` 的 `--batch_size` 默认值是 **32**（`infer_v2_test.py:66`）。
- 本轮 lane 调度脚本 `/root/gnnA_run/lane_runner.sh` 与锚点调度脚本 `/root/ksweep/lane_runner.sh` 的 infer 段**都没有显式传 `--batch_size`** ⇒ 三臂与锚点的 test 指标**同属 bs32 前向口径**（推理无梯度、无 OOM；三臂 `INFER_RC=0`）。
- 即：**§2 的像素 L1 / norm L1 / per-step 差异，全部来自训练口径与模型本身，不来自推理批大小**。这是本轮对比中最"干净"的一面。
- 附带：`infer_test.json` 只含 0–255 空间的 per-step L1（`step_pixel_l1_255` / `step_pixel_std_255`），**不含** per-step 归一化 L1；`full_norm_l1` 只有总量。与锚点报告同口径，因此 per-step 对比只在 0–255 空间。

### 1.3 每个 run 的收尾状态

| 臂 | train 起止（`status`） | TRAIN_RC | INFER_RC | 产物落盘 |
|---|---|---|---|---|
| K35 | 09-17 18:15:39 → 23:05:02 | 0 | 0 | 23:06:50 |
| K48 | 09-17 18:15:41 → 23:21:51 | 0 | 0 | 23:23:47 |
| K99 | 09-17 23:06:50 → 09-18 04:59:09 | 0 | 0 | 09-18 05:01:32 |

日志内无 `Traceback` / `OutOfMemory`；三臂训练速度约 **1.9–2.0 s/step**。显存（每 10 s 采样的 `gpu_mem.csv`，**整卡**口径，可能含同卡其它租户）：K35 窗口峰值 **50,035 MiB**、K48 **52,723 MiB**、K99 **60,173 MiB**；探针 bs16+ga2 独立测得 **59,457 MiB**（18:11–18:16 窗口），与 K99 正式 run 同量级。

---

## 2. 主结果表（3 臂 × 指标 + 与锚点逐项差）

### 2.1 主指标（全量 3004 test；`infer_v2_test.json` 原值）

| 臂 | K | 解码步 | 像素 L1 (0–255) | `full_pixel_std_255` | `full_norm_l1` | 同 K 锚点 px（bs32 ga1, off） | Δpx | Δ% | 锚点 norm | Δnorm | Δ% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **gnnA sum K35** | 35 | 5 | **16.0213** | 5.7259 | **0.278903** | 14.1470 | **+1.8743** | **+13.25%** | 0.246021 | +0.032882 | **+13.37%** |
| **gnnA sum K48** | 48 | 6 | **15.3963** | 5.5856 | **0.267856** | 13.1483 | **+2.2480** | **+17.10%** | 0.228721 | +0.039135 | **+17.11%** |
| **gnnA sum K99** | 99 | 9 | **15.7020** | 5.6810 | **0.273134** | 12.8812 | **+2.8208** | **+21.90%** | 0.224071 | +0.049063 | **+21.90%** |

- 所有 Δ% 均以锚点为分母；两个独立度量（px / norm）在同 K 上给出**一致的相对差**（13.25 vs 13.37、17.10 vs 17.11、21.90 vs 21.90），不是单一口径伪影。
- 顺带一个形态：方案A 三臂内部**非单调**（K35 16.02 > K99 15.70 > K48 15.40），与锚点"K 越大越好"（14.15 → 13.15 → 12.88）方向相反；但 n=1，不排除 run variance。

### 2.2 per-step（0–255 空间）

| 臂 | step1 | step2 | step3 | step4 | step5 | step6 | step7 | step8 | step9 | 末−首 (px) | 落差% | 最小步 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gnnA K35 | 16.0105 | 15.9867 | **15.9852** | 15.9967 | 16.0213 | — | — | — | — | **+0.0109** | **+0.068%** | 第 3 步 |
| gnnA K48 | 15.3759 | 15.3649 | **15.3637** | 15.3687 | 15.3790 | 15.3963 | — | — | — | **+0.0204** | **+0.133%** | 第 3 步 |
| gnnA K99 | 15.6956 | 15.6791 | 15.6732 | **15.6694** | 15.6697 | 15.6718 | 15.6770 | 15.6867 | 15.7019 | **+0.0064** | **+0.041%** | 第 4 步 |
| 锚点 K35（k35c） | 14.3941 | 14.2071 | 14.1501 | **14.1374** | 14.1470 | — | — | — | — | −0.2471 | **−1.717%** | 第 4 步 |
| 锚点 K48 | 14.2742 | 13.5300 | 13.2341 | 13.1504 | 13.1346 | **13.1483** | — | — | — | −1.1259 | **−7.888%** | 第 5 步 |
| 锚点 K99 | 13.8133 | 13.2176 | 13.0068 | 12.9207 | 12.8820 | 12.8651 | **12.8625** | 12.8670 | 12.8812 | −0.9321 | **−6.748%** | 第 7 步 |

> ⚠️ **符号约定**：按锚点报告的同一约定 `(末步 − 首步)/首步`，方案A 三臂是 **正数**（末步比首步**略差**，+0.041%～+0.133%）；任务书草稿里写的是 −0.07%/−0.13%/−0.04%，**量级正确、符号应为正**（见 §8 差异记录）。

- 方案A 三臂**首步→最优步**的全部可回收增益也只有 **−0.079%～−0.167%**；锚点同一量（首步→最优步）为 **−1.78% / −7.98% / −6.88%**。

### 2.3 `eval_recon` 轨迹对照（全量 3004 test，`eval_every=2000`；日志点 2190/4161/6132/8103/8760）

| 臂 | 2190 | 4161 | 6132 | 8103 | 8760 (final) | final 与锚点差 |
|---|---|---|---|---|---|---|
| **gnnA sum K35** | 0.4358 | 0.3469 | 0.3014 | 0.2798 | **0.2789** | +0.0328（**+13.33%**） |
| 锚点 k35c（bs32 ga1, off） | 0.3894 | 0.2996 | 0.2591 | 0.2465 | **0.2461** | — |
| **gnnA sum K48** | 0.4289 | 0.3196 | 0.2830 | 0.2682 | **0.2678** | +0.0390（**+17.05%**） |
| 锚点 k48 | 0.3739 | 0.2865 | 0.2423 | 0.2293 | **0.2288** | — |
| **gnnA sum K99** | 0.4342 | 0.3281 | 0.2854 | 0.2734 | **0.2731** | +0.0490（**+21.87%**） |
| 锚点 k99 | 0.3536 | 0.2726 | 0.2382 | 0.2245 | **0.2241** | — |

- 方案A 三臂在**全部 5 个评测点都更差**；且差距并非只在训练早期：
  - K35：+0.0464 → +0.0473 → +0.0423 → +0.0333 → +0.0328（后期略有收窄但从未逼近）；
  - K48：+0.0550 → +0.0331 → +0.0407 → +0.0389 → +0.0390（持平）；
  - K99：+0.0806 → +0.0555 → +0.0472 → +0.0489 → +0.0490（收窄后持平）。
- final `eval_recon` 与 `full_norm_l1` 交叉一致（如 K35 0.2789 vs 0.278903；K48 0.2678 vs 0.267856；K99 0.2731 vs 0.273134）。
- 三臂训练**无塌缩/发散**：eval_recon 单调下降并收敛，`TRAIN_RC=0`。

---

## 3. 两个关键观察

### (a) 绝对质量全面差于锚点 13–22%（且 K 越大越差）

| K | 方案A px | 锚点 px | 劣化 | 方案A norm | 锚点 norm | 劣化 |
|---|---|---|---|---|---|---|
| 35 | 16.0213 | 14.1470 | **+1.8743 px (+13.25%)** | 0.278903 | 0.246021 | **+13.37%** |
| 48 | 15.3963 | 13.1483 | **+2.2480 px (+17.10%)** | 0.267856 | 0.228721 | **+17.11%** |
| 99 | 15.7020 | 12.8812 | **+2.8208 px (+21.90%)** | 0.273134 | 0.224071 | **+21.90%** |

劣化随 K 单调放大：锚点从 K=35→99 改善 1.27 px（−8.95%），方案A 只改善 0.32 px（−2.0%），即**方案A 基本吃不到"更多 special token"带来的容量收益**。注意与之相对，同代码 2000 步短程三臂里 `sum` 曾优于同口径 `off`（26.2784 → 23.6203，−10.12%，见 `DESIGN_IMPL_gnn_schemeA.md` §4.2）——**短程的优势在 8760 步长训后被反转为 13–22% 的劣势**。

### (b) 逐步落差塌缩为 ~0 —— 循环解码器的逐步精炼机制被打平

- 方案A 三臂 `(末−首)/首` = **+0.068% / +0.133% / +0.041%**（末步甚至比首步略差）；锚点 = **−1.717% / −7.888% / −6.748%**。**相差 1～2 个数量级**。
- 锚点在 8760 步下"最小步位置随 K 后移（4 → 5 → 7）、后段步持续拿增益"的形态（`ANALYSIS_ksweep_stepwise_refinement.md`）在方案A 三臂里**完全消失**：三臂的最小步固定在第 3/3/4 步，之后一路回吐，末步回到全曲线最差或接近最差。
- 这不是"没收敛"：三臂 eval_recon 到 8760 步已基本走平（末段 Δ ≤ 0.0006），说明模型稳定停在一个"各步输出几乎等价"的解上，而不是还在爬坡。
- **但必须同读 §5 的缺口**：本轮没有 bs16+ga2 的 `off` 8760 对照，而 2000 步时 **off 臂本身也近乎平**（26.199 → 26.278，末步最差，`DESIGN_IMPL_gnn_schemeA.md` §4.2）。因此"逐步精炼是被 GNN 集成打掉的"目前只是**与锚点对比得到的关联**，尚未与"微批口径差/训练量"分离。

---

## 4. 机制假设（**明确标注：以下均为未验证假设**）

### 4.1 先纠正一个实现事实（与任务书草稿的机制表述不符，代码核对见 §8）

`--gnn_mode sum --gnn_inject replace` 的实际语义（`model_v2.py:747-754`，@ `793e99a`）是：

```python
z = g["z"]                                             # (B,1,D) 每图 1 个全局向量
bank = self.special_bank(B, pixel_values.device)       # (B,K,D) K 个 register
z_slots = torch.cat([z, bank[:, :K - 1]], dim=1)       # (B,K,D)：槽位 0 = z，槽位 1..K−1 = register
```

即：**只把 K 个槽位中的第 0 个替换成 GNN 全局向量**，其余 `K−1` 个槽位仍是 `SpecialTokenBank`（共享 token + 逐位置可学习 pos）的 register。所以"用 1 个全局向量替换 K 个 special token、K 个槽位失去差异化"**不成立**——结构差异其实很小：相对 `off` 路径，序列长度、解码器、读窗口全部未变，差异集中在"槽位 0 的内容"以及新增的 GNN 分支（1.13M 参数 + 一次额外的 DINO 前向用于取节点特征）。

### 4.2 假设 H1（与读窗口几何相关，待验证）

解码器每步只读取 z_s 的一个**平方块** `A[:, lo:hi+1]`，块随步号扩大（K=35 时按代码注释 `model_v2.py:332`：步 1→`z_s[0:2]`、步 4→`z_s[0:7]`、步 9→`z_s[0:14]`、步 16→`z_s[0:23]`、步 25→`z_s[0:34]`）。**首步窗口最小，且一定包含槽位 0**（首步 `lo=0`）。假设：槽位 0 的单一 GNN 全局向量已经把"整图摘要"喂给了最小窗口，使后续窗口多读入 register 槽位不再带来边际信息 ⇒ 各解码步输出趋同、逐步精炼塌缩。

### 4.3 假设 H2（优化/尺度假设，待验证）

GNN 的 `z` 是 L2 归一化后的向量，而 register 槽位是未归一化的可学习向量；把槽位 0 换成不同尺度的输入，可能改变了 DINO 序列中该位置的注意/残差尺度，导致该槽位退化为"常数级"输入 ⇒ 各步读到的首块几乎不携带步相关信息。需要测训练后 `z` 的跨图余弦/方差（`DESIGN_IMPL` §3.2 已记录**随机初始化**下 `proto` 跨图余弦 = 1.0000 的坍塌现象；`sum` 训练后是否也有类似退化**未测**）。

### 4.4 本轮**不能**支持的结论

- 不能把差额归因于"GNN 图表示内容不好"：缺"同槽位 register 对照""随机图/`gnn_layers=0` 消融"（`DESIGN_IMPL` §6.2 已列为缺口）。
- 不能把差额归因于"微批口径差"：缺同口径 `off`。
- 不能把逐步塌缩归因于"sum 替换了 K 个槽位"：实现只替换 1 个槽位（§4.1）。

---

## 5. 未做的对照（**显式缺口清单**）

| # | 缺口 | 作用 | 现状 |
|---|---|---|---|
| 1 | **同口径 `off` 对照**：bs16+ga2、8760 步、K=35、seed42 | 隔离"bs16+ga2 vs bs32+ga1"的微批口径差，是判定 13–22% 归因的**唯一**办法 | **未做**（服务器 `output/` 下无任何 `*off*` 目录） |
| 2 | **`proto`（k=K）臂**：`--gnn_mode proto --gnn_inject replace`、8760 步（建议 K=35） | 设计文档 §6 注 / §10 E3 指定的"与 K 个 special token 对齐"的方式；只有它才能回答 E3 的"sum vs k 原型取舍" | **未做**（服务器 `output/` 下无任何 `*proto*` 目录） |
| 3 | `--gnn_inject concat`（sum + 保留 K 个 register + 追加 1 全局向量） | 分离"替换槽位 0"与"追加全局 token"两种注入效应 | **未做** |
| 4 | `--gnn_conv {gcn,sage}` | §3 只跑了 GIN | **未做**（训练侧；仅自检覆盖） |
| 5 | `--gnn_grid` 网格/超像素邻接（§10 E1 的"可选并入坐标/超像素亲和"） | 图结构本身的消融 | **未做**（本轮 `gnn_grid=None`，纯 kNN k=8） |
| 6 | 多尺度 Readout（各层 h 联合聚合，§5） | 提升区分度 | **未做** |
| 7 | 自监督目标（对比学习/边重建，§7） | 让图表示在无标签下更强 | **未做** |
| 8 | E1 文本侧（z 冻结后接 Phase-2） | §10 E1 的验收口径 | **未做** |
| 9 | 纯内容特征（不含 DINO PE 的 patch 卷积特征）重建 kNN 图 | 严格版置换不变口径 | **未做** |
| 10 | 同口径 bs32 复测（若未来显存放宽） / 多 seed | 排除微批口径与 run variance | **未做**（K=99 bs32 确认 OOM） |

> 因此：**本轮的证据只覆盖"§10 E3 两个臂中的一个（sum）、且用了一种特定的注入方式（replace）、且训练微批与锚点不同"**。它是对**该集成方式**的判决，不是对方案A 设计（GNN + 置换不变 Readout）的判决。

---

## 6. 可复现命令

### 6.1 本轮 8760 三臂（服务器 `autodl-container-jnb93wme4w-bf7ae4cd`，`feat-gnn-schemeA @ 793e99a`）

调度脚本 `/root/gnnA_run/lane_runner.sh`（train → infer → `<tag>.done`，断点续跑）；启动方式：

```bash
setsid nohup /root/gnnA_run/lane_runner.sh 0 /root/gnnA_run/queue_gpu0.txt 16 2 \
  > /root/gnnA_run/lane0.out 2>&1 < /dev/null &
setsid nohup /root/gnnA_run/lane_runner.sh 1 /root/gnnA_run/queue_gpu1.txt 16 2 \
  > /root/gnnA_run/lane1.out 2>&1 < /dev/null &
# queue_gpu0.txt:  "K35 0 5" / "K99 0 9"   （串行）
# queue_gpu1.txt:  "K48 0 6"
```

`lane_runner.sh` 为每个 arm 展开的等价命令（以 GPU0、K=35 为例；K=48 用 `slice 0:6`、K=99 用 `slice 0:9`）：

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
# 训练完成后（脚本自动执行；未传 --batch_size ⇒ 默认 bs32 前向）：
CUDA_VISIBLE_DEVICES=0 python infer_v2_test.py \
  --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
  --final_model output/gnnA_sum_K35_8760/final_model.pt \
  --output output/gnnA_sum_K35_8760/infer_test.json
```

- `infer_v2_test.py` 不显式传 `--slice_*/--gnn_*`：结构超参由 `output_dir/model_info.json` 恢复（训练侧写入 `num_specials`/`decoder_steps`/`slice_*`/`gnn_*`），保证 K 与权重形状对齐。
- 探针（定档 bs16+ga2 的依据）：`probe_K99_bs32`（bs32 ga1，CUDA OOM）与 `probe_bs16ga2`（20 步成功，峰值 59,457 MiB），日志 `/root/train_logs/gnnA_probe_K99_{bs32,bs16ga2}.log`。

### 6.2 锚点（`bptt@d76f0c5`，**bs32 ga1**，同 8760 步/seed42）

出处：脚本 `/root/ksweep/lane_runner.sh`（train 段硬编码 `--batch_size 32 --grad_accum 1`，其余与上同；infer 段同样不传 `--batch_size`），clone `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep`；数字出处 `doc/2026-09-16/REPORT_ksweep_k35c.md`（K=35，14.1470 / 0.246021）、`REPORT_ksweep_k48.md`（13.1483 / 0.228721）、`REPORT_ksweep_k99.md`（12.8812 / 0.224071）。

```bash
# 锚点等价命令（tag = k35c | k48 | k99；slice 0:5 | 0:6 | 0:9）
cd /root/autodl-tmp/sr-diffusion-v3-bptt-ksweep
export PATH=/root/miniconda3/bin:$PATH; export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=<0|1> python train_v2.py \
  --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
  --output_dir output/ksweep_<tag> \
  --model_input 448x252 --canvas 1600x900 --angle_step 0.5 \
  --epochs 40 --max_steps 8760 --batch_size 32 --grad_accum 1 \
  --lr 1.5e-4 --weight_decay 0.01 --warmup_ratio 0.03 --grad_clip 1.0 \
  --num_workers 8 --limit 0 --eval_limit 0 --eval_every 2000 --save_every 2000 --log_every 20 \
  --seed 42 --heads 8 --mlp_ratio 4.0 --decoder_depth 2 \
  --slice_start <ss> --slice_end <se>
CUDA_VISIBLE_DEVICES=<0|1> python infer_v2_test.py \
  --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
  --final_model output/ksweep_<tag>/final_model.pt --output output/ksweep_<tag>/infer_test.json
```

### 6.3 建议的最小补跑（见 §7；未执行）

```bash
# 缺口 1：同口径 off 对照（bs16+ga2、8760、K=35、seed42）
CUDA_VISIBLE_DEVICES=<gpu> python train_v2.py  <同上 COMMON，--batch_size 16 --grad_accum 2 --slice_start 0 --slice_end 5> \
  --output_dir output/gnnA_off_K35_8760 --gnn_mode off
# 缺口 2：设计文档 §10 E3 指定的 proto(k=K) 臂（bs16+ga2、8760、K=35、seed42）
CUDA_VISIBLE_DEVICES=<gpu> python train_v2.py  <同上 COMMON> \
  --output_dir output/gnnA_proto_K35_8760 \
  --gnn_mode proto --gnn_inject replace --gnn_hid 256 --gnn_layers 2 --gnn_conv gin --gnn_k 8
# 各自跑完后同样用 infer_v2_test.py（默认 bs32）出 infer_test.json
# 注：proto+concat 被实现显式拒绝（K 原型 + K register 会超出读窗口），E3 用法固定为 proto+replace。
```

---

## 7. 诚实结论与下一步

### 7.1 可以下的结论（有实测支撑）

1. 在 **bs16+ga2 / 8760 步 / seed42 / 同 bs32 推断口径**下，`sum+replace` 的 K=35/48/99 三臂比同 K（bs32+ga1）`off` 锚点**差 +13.25% / +17.10% / +21.90%**（px 与 norm 两个独立口径一致），且**全部 5 个 eval_recon 评测点都更差**。
2. 同一批三臂的**逐步落差塌缩为 +0.04%～+0.13%**（锚点 −1.72%/−7.89%/−6.75%），最小步固定在 3/3/4 步后回吐 ⇒ **"多读后段槽位换精炼"这条通路在 sum+replace 下没有工作**。
3. 训练稳定、无 OOM/发散（`TRAIN_RC=INFER_RC=0`），结果不是训练失败的伪影。

### 7.2 不能下的结论

- **不能**说"方案A（GNN + 置换不变 Readout）本身不如 register"：缺同口径 `off`、缺 `proto(k=K)`（§5 缺口 1/2），也缺"同槽位 register / 随机图 / `gnn_layers=0`"等消除"注入方式"效应的对照。
- **不能**把 13–22% 全部归因于 GNN：微批口径差（bs16+ga2 vs bs32+ga1）未隔离；同类历史口径差可达 21.62%（K=35，DDP vs 单卡），与本轮 K=99 的劣化同量级。
- **不能**说逐步塌缩由"1 个向量替换 K 个槽位"造成：实现只替换槽位 0（§4.1）。

### 7.3 下一步最小实验集（按信息量/成本排序）

| 优先级 | 实验 | 成本（估） | 回答什么 |
|---|---|---|---|
| **P0** | **同口径 `off`：bs16+ga2、8760、K=35、seed42** | ~4.5–5 h / 1 GPU | 把"微批口径差"从 13.25% 里扣出来；若 off 也在 16 px 量级，则本轮全部劣势归因于口径，而非 GNN |
| **P0** | **`proto`（k=K）+ replace：bs16+ga2、8760、K=35、seed42** | ~5 h / 1 GPU（显存≈sum） | §10 E3 指定的对齐方式；回答"sum 单向量 vs k 原型"的取舍，才构成对方案A 的公平检验 |
| P1 | 若 P0 的 off 与 sum 差距仍 >5%：加 `gnn_layers=0` / 随机 kNN 边 / 同槽位 register 对照 | 各 ~5 h | 分离"GNN 图语义"与"注入一个额外可学习向量"的效应 |
| P2 | `gnn_inject=concat`（sum） | ~5 h | 分离"替换槽位 0"与"追加全局 token" |
| P3 | 训练后测 `z` 的跨图余弦/方差 + 逐步注意力可视化 | 分钟级 | 验证 §4.3 的 H2 退化假设 |

> 在 P0 两项完成前，建议任何对外表述都只用 7.1 的三条，并附 §1.1 的口径警告。

---

## 8. 溯源与核对差异记录

### 8.1 数字溯源（全部经服务器只读核对）

| 数字 | 来源（服务器） |
|---|---|
| 三臂 px / norm / std / per-step / `time_s`（101.2 / 109.3 / 136.1） | `/root/autodl-tmp/sr-diffusion-v3-gnnA/output/gnnA_sum_{K35,K48,K99}_8760/infer_test.json` |
| 三臂 `eval_recon` 轨迹（5 点）、`TRAIN_RC/INFER_RC`、起止时间 | `/root/train_logs/gnnA_sum_{K35,K48,K99}_8760.log`、`/root/gnnA_run/status/{K35,K48,K99}.status` |
| 三臂超参（bs16/ga2/8760/seed42/slice/gnn_*）、K、decoder_steps | 各 `output/gnnA_sum_*_8760/{args.json,model_info.json}` |
| 显存峰值（50,035 / 52,723 / 60,173 MiB；探针 59,457 MiB） | `/root/gnnA_run/gpu_mem.csv`（每 10 s；整卡口径） |
| bs32 K99 OOM 证据（94.70/94.97 GiB，缺 288 MiB） | `/root/train_logs/gnnA_probe_K99_bs32.log` |
| 调度命令 / infer 未传 `--batch_size` | `/root/gnnA_run/lane_runner.sh`、`/root/ksweep/lane_runner.sh`、`infer_v2_test.py:66`（默认 32） |
| `sum+replace` 的实际注入语义（只替换槽位 0） | `/root/autodl-tmp/sr-diffusion-v3-gnnA/model_v2.py:747-754`（@ `793e99a`） |
| 锚点 K=35/48/99（px、norm、per-step、eval_recon、bs32 ga1、`bptt@d76f0c5`） | `doc/2026-09-16/REPORT_ksweep_bptt.md` §1/§2、`REPORT_ksweep_k35c.md`、`REPORT_ksweep_k48.md`、`REPORT_ksweep_k99.md` |
| 2000 步短程三臂（off 26.2784 / sum 23.6203 / proto 24.0418） | `doc/2026-09-17/DESIGN_IMPL_gnn_schemeA.md` §4.2 |
| 设计文档 §6 注（k 原型）/ §10 E3（k 原型 k=K vs sum 的取舍） | `feature/DESIGN_graph_embedding_schemeA.md:70`、`:107` |

### 8.2 核对差异（**以实测为准**）

1. **per-step 总落差符号**：任务书草稿给 −0.07% / −0.13% / −0.04%。按锚点报告同一约定 `(末−首)/首`，实测为 **+0.068% / +0.133% / +0.041%**（末步比首步**略差**）。量级一致，**符号已按实测更正**；"塌缩为 ~0"的结论不受影响。
2. **机制表述**：任务书草稿称 `sum+replace`"用单个全局向量替换 K 个 special token，K 个槽位失去差异化"。代码核对（`model_v2.py:752-754`）显示实际是 `z_slots = cat([z, bank[:, :K-1]])` —— **只替换槽位 0，其余 K−1 个仍是 register**。报告正文 §4.1 按实测实现描述，H1/H2 仅作为假设。
3. **推断口径**：任务书把"微批口径差"整体套在对比上；实测三臂 infer 未传 `--batch_size`（默认 **32**），与锚点相同 ⇒ **test 指标是同一推断口径**，口径差只存在于**训练微批**（bs16+ga2 vs bs32+ga1）。§1.2 已单列。
4. **主表/逐项差/eval_recon/per-step 数值**：与 `infer_test.json` 及训练日志**逐项一致，无差异**（K35 16.021334977 / 0.278903425；K48 15.396302793 / 0.267855541；K99 15.701978813 / 0.273134099；eval_recon K35 0.4358/0.3469/0.3014/0.2798/0.2789 等）。
5. **未做项状态**：服务器 `output/` 下**不存在**任何 `*off*` / `*proto*` 8760 目录（已 `ls` 确认），§5 缺口 1/2 为**真缺口**，不是"未找到"。
6. 仍 **UNKNOWN**：K=99 在 bs32+ga1 下的表现（OOM 未取得）；微批口径差在 K=35/48/99 上各自的真实 offset；`sum` 的 GNN z 训练后是否退化（跨图余弦未测）；逐臂训练 wall-clock 的精确 `train_runtime`（本报告用 `status` 时间戳推算，5h 内量级正确）。
