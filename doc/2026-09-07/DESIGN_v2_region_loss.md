# v2 分区掩码损失（per-step region-masked target loss）— 设计与实现记录

> 日期: 2026-09-07 ｜ 性质: 设计 + 实现记录（损失侧改造, P1）
> 作者: Qwen3.8-max（审计 + 实现 + 数值验证）
> 状态: **已实现并通过本地自检 + 服务器数值验证（GPU 冒烟, 小规模随机张量）; 训练 A/B 未跑（待主控批准 GPU 预算）**
> 分支/基线: `main` HEAD `2cb409e` 之上（工作区修改, 未 commit——提交由主控执行）
> 涉及: `model_v2.py` / `train_v2.py` / `infer_v2_test.py`
> 动机来源: `doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md` §4 **P1（首选, 若坚持渐进）**；旁证 `ANALYSIS_k3_slice_infodiff.md`、`REPORT_v2_E1_probe.md`

---

## 0. TL;DR

1. **首次实现声明**: 分区掩码损失（第 t 步只监督还原自己的 patch 区域）在 v2 历史上**从未被实现过**——Qwen 审计逐版核对确认, 历史每一版 `decode()` 都是 `target_pix.unsqueeze(1).expand_as(Y_pix)` 的"每步累加结果监督**整图**"（平权全覆盖）。本次为**首次实现**（= K3 分析报告的 P1 方案）。
2. **改动核心**（`model_v2.py`）: `SRPhase1V2` 新增构造参数 `region_loss: bool = True`（普通 python 属性, **不进 state_dict**, 旧 checkpoint 仍可 strict 加载）。`decode()` 里 `region_loss=True` 时第 t 步损失 = 只在 region t 的行上 `mean |Y_pix_cum[:, t, region_t] − target_pix[:, region_t]|`; `False` 时**逐位复现**旧"每步整图"损失（A/B 对照/逃生口）。前向输出 `F_hat / Y_pix / target_pix` 与开关**无关, 逐位不变**（自检断言）; `recon` = 全图 L1(F_pix, target) 监控口径不变; 返回 dict 键不变。
3. **区域划分**: T=|steps|（升序步索引 t=0..T−1）, N=num_patches; `region t = [⌊N·t/T⌋, ⌊N·(t+1)/T⌋)`（整数除法, 无浮点误差）——互不相交、连续覆盖 0..N−1、大小差 ≤1; N≥T 时每区非空（decode 内断言）。N=576, T=5 → `[0,115) [115,230) [230,345) [345,460) [460,576)`。
4. **评估口径变化（重要）**: 新默认下 Trainer 的 `eval_loss` 变为**分区口径**（各步区域损失的均值）, 与历史 eval 数值（0.5037/1.1394 等）**不可直接对比**; 历史对比一律用 **`eval_recon`**（= recon = 全图 F_pix L1, 归一化空间, 口径自始未变, `compute_metrics` 一直在输出）。`infer_v2_test.py` 的全部指标本来就是全图口径, 不变, 与历史推理数值直接可比。
5. **主要风险**: 累加语义下, 第 t 步在**非本步区域**行上的输出无任何直接监督, 可能漂移并经 F_pix=Σ_t Y_t 污染最终图（§6.1）——监控 = eval_recon + infer 渐进曲线（后步抬升即污染信号）; 后续手段已列（§6.1c）。
6. **验证结果**: 本地 CPU 自检 ALL CHECKS PASSED（区域损失 == 手工掩码均值; False == 旧全图均值; F_hat/Y_pix 逐位不变; 每步 Y_t 梯度恰收 1/T 份且只落在自己 region 的行）。服务器 GPU（RTX PRO 6000, torch 2.12.1+cu130）自检 + 双口径冒烟全部通过（数字见 §8）。

---

## 1. 动机（引 ANALYSIS_k3_why_later_steps_zero.md P1）

K3 第三轮机制分析的最短因果链（该文 §0/§1, 权重 ~40% 的 L1 层）: 旧损失把**每个采样步的累加结果都监督成整图**、步间平权、梯度按步解耦 ⇒ step-1 的子任务 = 完整重建（16 键真能做到 ~19.9, S25/S64 锚点）, 后步的子任务 = 预测"step-1 没做出来的残差"; 残差从后步可读的键里不可预测时, **L1 的最优预测 = 零**（条件中位数）⇒ 后步输出 ≈0（|W·Y_t| = [1.054, 0.015×4] 实测）⇒ 后区 register 收不到有效读出梯度 ⇒ 塌缩成 cos≈0.97 冗余簇 ⇒ 残差更不可预测——**零增量自锁**。

P1（该文 §4 排序第一, "首选, 若坚持渐进"）: **分区域掩码损失**——step t 只监督自己对应的 patch 行区（576 行均分 5 区）, 去掉整图平权累加损失; 保留块因果读。预期效果: 每步有了 step-1 抢不走的**私有目标**; 键/区比从 16 键:576 patch 改善到 ~10 键:~115 patch（低于 S25 锚点难度）; 损失本身产生 register 分工压力（不需要额外正则）。P1 原文同时预警: 最终整图 L1 未必优于 19.88（这是为"真阶梯"付的价）; F3/F4 读出上限仍在。

本实现 = P1 的最小落地（代码 ~20 行核心 + 开关/自检/文档）, **不**捆绑 P2（F1/E2' 编码侧注入）——P2 是信息侧前提, 独立改动, 若 P1 在 ~19 读出上限处 stall 再上（K3 §4 P2 行）。

## 2. 精确规格

### 2.1 区域划分定义

```
T = |steps|                      # decoder.steps 升序（select_steps 保证）, 步索引 t = 0..T-1
N = num_patches                  # patch 行数（row-major, 448x252 输入 → N=576）
region t = [lo_t, hi_t),  lo_t = ⌊N·t/T⌋ = (N*t)//T,  hi_t = ⌊N·(t+1)/T⌋ = (N*(t+1))//T
```

性质（`region_slices()` docstring + 自检 + 500 组随机 (N,T) 扫描验证）: 互不相交; 连续覆盖 `[0, N)`; 大小差 ≤ 1; N ≥ T 时每区非空（decode 内 `assert N >= T` 快速失败——空区的 `mean` = NaN 会静默毁掉训练）。例: N=576,T=5 → 115×4+116; N=16,T=4 → 每区 4 行; N=17,T=4 → 4,4,4,5; T=1 → 整图（退化 = 旧行为）。

### 2.2 损失公式

归一化像素空间（与旧损失同空间）, `reduction="none"` 再取均值（与旧损失同 reduction 风格）, 步间平权:

```
region_loss=True（默认）:
    per_step[t] = mean_{(b, r, p) ∈ B × region_t × [0,588)} | Y_pix_cum[b, t, r, p] − target_pix[b, r, p] |
    loss        = mean_t per_step[t]                                   # (|T|,) → 标量

region_loss=False（对照, 逐位复现旧行为）:
    per_step[t] = mean_{(b, r, p) ∈ B × [0,N) × [0,588)} | Y_pix_cum[b, t, r, p] − target_pix[b, r, p] |
    loss        = mean_t per_step[t]
```

其中 `Y_pix_cum = PixelHead(Y_cum)`, `Y_cum = [0, cumsum(Y)[:-1]].detach() + Y`（**保留不变**: 特征空间累加、梯度按步解耦 carry.detach()、PixelHead 共享、`F_pix = Y_pix_cum[:, -1]`、`recon = 全图 L1(F_pix, target_pix)` 监控口径、返回 dict 键 `loss/recon/F_hat/Y_pix/target_pix`）。

### 2.3 伪码（decode() 核心, 实际实现见 model_v2.py）

```python
Y     = decoder(z_cls, z_s)                       # (B,|T|,N,D)
Y_cum = cat([0, cumsum(Y)[:-1]], 1).detach() + Y  # 梯度按步解耦（不变）
Y_pix = pixel_head(Y_cum)                         # (B,|T|,N,588)（不变）
F_pix = Y_pix[:, -1]                              # 不变
if region_loss:                                   # ── 新增分支 ──
    assert N >= T
    per_step = stack([ l1(Y_pix[:, t, lo:hi], target_pix[:, lo:hi], none).mean()
                       for t, (lo, hi) in enumerate(region_slices(N, T)) ])
else:                                             # ── 旧行为, 逐位一致 ──
    per_step = l1(Y_pix, target_pix.unsqueeze(1).expand_as(Y_pix), none).mean((0,2,3))
loss  = per_step.mean()
recon = l1(F_pix, target_pix)                     # 全图口径监控（不变）
```

### 2.4 开关的边界（明确不做什么）

- `region_loss` 是**普通 python 属性**（非 buffer/parameter）⇒ 不进 `state_dict` ⇒ 旧 checkpoint `strict=True` 加载不受影响（已实测, §8.1）; 推理侧构造时可自由设定, 权重形状无关。
- 开关**只影响 `loss`**（及经 loss 的梯度/训练动力学、Trainer 的 eval_loss 口径）; `F_hat / Y_pix / target_pix / recon` 全部与开关无关, 同权重同输入下**逐位不变**（自检用 `torch.equal` 断言）。
- 不改读出结构: memory_mask 分块读、tgt_mask 块因果、query_base、累加语义、PixelHead 全部原样（P1 的"保留块因果读"）。

## 3. 配置与 model_info

| 项 | 值/行为 |
|---|---|
| `model_v2.SRPhase1V2(..., region_loss=True)` | 构造参数, 默认 **True**（新分区损失）; False = 旧"每步整图"损失 |
| `train_v2.py --region_loss` | CLI, 默认 True; `--region_loss false/0/no` 关闭（`_str2bool`, 支持 `--region_loss=false`、裸 flag=True）; 记录进 `args.json`（`vars(args)` 自动） |
| `model_info.json` | 新增 `"region_loss": bool`（与当年 memory_open 的写法一致——布尔开关记录, infer 侧按 model_info 对齐; 但 region_loss 不影响权重形状, 只影响 loss 口径） |
| `infer_v2_test.py` | 构造模型时按 `model_info.json["region_loss"]` 对齐; **缺字段 = 旧产物 → False**（分区损失此前从未实现, 历史每版都是整图损失——Qwen 审计结论）。推理指标（全量 L1/渐进曲线）全部由 F_hat/Y_pix 直接计算, 不经 out["loss"], 数值与开关无关 |
| 旧 checkpoint | `strict=True` 加载不变（开关不进 state_dict）; 用旧 ckpt 推理时 out["loss"] 按 model_info 对齐口径 |

## 4. 评估口径变化说明（怎么接的）

**机制现状**（train_v2.py, 未改动的部分）: `SRPhase1V2Trainer.can_return_loss=True` ⇒ eval 走 `compute_loss` 路径, **eval_loss = 模型 forward 返回的 "loss"**; `prediction_step` ignore `Y_pix/target_pix` 后, `compute_metrics` 从输出元组显式取 **recon** ⇒ **eval_recon = 全图 F_pix L1（归一化空间）**。

**变化与接法**:
1. `region_loss=True`（新默认）下 **eval_loss 变为分区口径**（各步"只在自己区域上"的 L1 的均值）。历史日志里的 eval 数值（exp1 0.5037@2000、posenc/exp2 1.1394/1.139@2000、slice27_v2 0.2490@8760 等）是旧"每步整图"损失——旧损失下 eval_loss ≈ 全图 L1（后步增量 ≈0, 每步累加结果都 ≈ 整图; 实证: posenc 证据 json `eval_loss 1.1394238 / eval_recon 1.1394248`, 几乎逐位相等）。**因此新旧 eval_loss 不可直接对比。**
2. **全图口径指标继续输出**: `eval_recon` 一直存在且口径未变——与历史 eval 数值对比**一律用 eval_recon**（旧 run 的 eval_loss ≈ eval_recon, 可直接拿历史 eval_loss 当 eval_recon 的近似基线; 严格对比时从历史日志的 eval_recon 列取值）。无需改 `compute_metrics` 代码——接法 = 保留现有 eval_recon 通路 + 在 train_v2.py 头部/Eval 段注释、[model] 启动打印里**显式声明口径**（启动即打印 "eval_loss 同口径(分区), 与历史数值对比请看 eval_recon=全图 F_pix L1"）。
3. **训练日志的 train loss** 同样是分区口径（Trainer 记录 forward "loss"）; grad_norm 语义随之改变（后步第一次收到真实梯度, 预期量级上升, 不是异常）。
4. **推理侧零变化**: `infer_v2_test.py` 的 `full_norm_l1 / full_pixel_l1_255 / step_pixel_l1_255 渐进曲线` 全部是全图口径、由 F_hat/Y_pix 直接算——与历史推理数值（14.284/19.878/20.55 等 0-255 空间）**直接可比**, 这是 A/B 的主判据口径。

## 5. 验证判据（引 K3 文档 P1 行, 训练 A/B 后核验）

K3 P1 的四条验证判据（`ANALYSIS_k3_why_later_steps_zero.md` §4 表）+ 本设计补充的污染判据:

1. **各步隔离 L1 在自己区域上相近**（不再 39 vs 61 那种 step-1 独好）——用 probe 脚本对每步 `Y_pix[:, t]` 只在 region t 上算 L1(0-255);
2. **累加曲线出现真阶梯**（覆盖区单调降）——`infer_v2_test.py` 的 step_pixel_l1_255: 预期每步在自己区域上单调改善; 全图口径的累加曲线见 §6.1 污染判据;
3. **后步 |W·Y_t| ≫ 0.015**（像素量级）——后步输出不再是零藏身处;
4. **z_s within-std 从 0.14 回升、register 16..63 两两 cos 明显跌破 0.9**（分工压力出现）——E1/probe 脚本复测。

**GO/NO-GO 提醒**（K3 原文）: 最终整图 L1 未必优于 19.88——"真阶梯"与整图最优可能冲突; 若四条判据全中但整图变差, 属于 P1 的预期代价而非实现 bug, 战略取舍交主控（K3 的战略推荐仍是 v4 单发路线, 本实现只负责把 P1 做成可用的单变量实验）。

## 6. 风险与失败判据

### 6.1 核心风险: 非本步区域行的输出无直接监督 → 漂移污染最终图

**机制**: 累加语义下 `F_pix[i] = PixelHead(Σ_t Y_t)[i]`（行 i 的最终像素含**所有步**在该行的贡献）。行 i ∈ region_s 只被第 s 步的损失监督, 且监督对象是 `cumsum_s[i] = Σ_{t≤s} Y_t[i]`; **对 t > s, `Y_t[i]` 没有任何直接监督项**（第 t 步只管 region_t 的行; 第 s 步的累加到 s 为止）。旧整图损失下这些项被"每步都监督整图"直接压住（压成 ≈0——正是零增量自锁的另一面）; 新损失下它们只受共享参数/注意力行耦合的间接影响, 可能漂移到非零, 经 `Σ_t Y_t` **污染 region_s 行的最终像素**——region 口径 loss 与 eval_recon 背离、渐进曲线后步抬升。

**监控**（每个 eval / 每次 infer 必看）:
- (a) **eval_recon（全图 F_pix L1）vs 分区口径反推**: 若 eval_recon 明显差于"各步区域损失拼回全图"的水平（差异 ≫ 批噪声, n=3004 时归一化 L1 的 SE ≈ 0.002 量级 / 0-255 像素 SE≈0.12）, 即污染信号;
- (b) **infer 渐进曲线**: `step_pixel_l1_255[t]` 应大致单调不升（后步只加自己区域的改善）; **若后步显著抬升（> ~1.0 像素, ≈8×SE）= 漂移污染实锤**;
- (c) 训练期 grad_norm 突跳 / loss 平台但 recon 恶化, 同为污染侧写。

**后续手段**（若污染被证实, 本次均**不**实现——超出 P1 最小 scope）:
1. 推理/累加端"区域化输出": F_hat 只累加各步自己区域的行（`F_hat[i] = cumsum_{s(i)}[i]`, s(i)=行 i 所属区域的步）——结构小改, 直接消掉污染项, 但改变 F_hat 语义, 须单独立项;
2. 损失端加弱全图正则: `loss = region_loss + λ·recon`（λ ~0.1）, 给非本步区域行一个"别乱动"的锚;
3. 显式残差目标（K3 P3）配区域掩码: `target_t = target − stopgrad(pixel(cumsum_{<t}))` 只在 region_t 上监督——P3 单独无效, 配 P1 才有意义（K3 §4 P3 行）;
4. 对非本步区域行的 Y_t 输出加 L2 惩罚（最保守的"零锚"）。

### 6.2 其余风险

| 风险 | 评估 | 缓解 |
|---|---|---|
| 整图 L1 不如 19.88（P1 预期代价） | 中——K3 原文预警; 每步只为自己 ~115 行负责, 无全局一致性压力 | §5 GO/NO-GO 交主控; B 组（--region_loss false）同预算对照随时可回退 |
| 区域边界 = 行区间（row-major 条带）, 非 2D 方块 | 低——规格如此（P1 "576 行均分 5 区"）; 条带 = 图像的水平带, 各带内容分布同质（工地场景） | 不改; 若将来要 2D 块划分, 只动 region_slices 一处 |
| 步间难度不均（首步前缀规则: step-1 读 0..hi 键, 其余步只读自己块） | 低——键/区比 ~10:115 已低于 S25 锚点（36 键:576 patch → 19.15） | K3 判据①（隔离 L1 相近）直接度量 |
| eval_loss 口径变化被误读为"变好/变坏" | 中——历史习惯看 eval_loss | §4 显式声明 + 启动打印 + 本文档; 对比一律 eval_recon / infer 全图 L1 |
| N < T 配置（空区 NaN） | 极低——默认计划 T=⌊√N⌋ ≪ N; 显式 steps 最多 N+1 个 | decode 内 assert N ≥ T 快速失败 |
| 旧 checkpoint 兼容 | 无——开关不进 state_dict, strict 加载已实测通过 | §3; infer 缺字段→False |

### 6.3 失败判据汇总（训练 A/B 后判定）

- **失败 F1（污染）**: §6.1(b) 渐进曲线后步抬升 > ~1.0 像素, 或 (a) eval_recon 与分区口径显著背离 → 走 §6.1c 后续手段;
- **失败 F2（无效, K3 判据不中）**: 后步 |W·Y_t| 仍 ≈0.015 量级 / 隔离 L1 仍悬殊 / z_s within-std 不回升 → "零最优"没被打破, 印证 K3 §4 "P1 若在 ~19 读出上限处 stall, 病灶在信息侧（P2）";
- **失败 F3（训练不稳）**: grad_norm 崩到 0.001 量级平台（exp2/posenc 式塌缩同型）或 loss NaN → 回退 B 组, 排查 lr（region 口径下梯度结构改变, 1.5e-4 或需重调——首次 A/B 不调, 保持单变量）。
- **成功 S（K3 判据①-④ 至少 ②③ 中）**: 曲线出真阶梯 + 后步像素量级 ≫0.015 → P1 生效, 再按 §5 评估整图代价与 P2 叠加。

## 7. 复现命令（slice27_v2 同预算协议 A/B）

同预算单变量协议 = `REPORT_v2_block_slice27.md` / `REPORT_v2_slice27_causal_mask.md`（slice [2:7] → steps=[9,16,25,36,49], K=63, 2 卡 × bs16, 8760 步, lr 1.5e-4, wd 0.01, warmup 3%, seed 42, depth 2, fp32）。**唯一变量 = --region_loss**。前提: 先把本实现同步到服务器工作副本（/root/autodl-tmp/sr-diffusion-v2-k99 = git 36cf777 历史副本, 勿直接污染, 建议另拷目录或用 git 工作区）。

```bash
cd /root/autodl-tmp/sr-diffusion-v2-k99    # 或同步后的工作副本

# A 组: region_loss=True（新默认, 分区掩码损失）
LOG=/root/train_logs/train_slice27_v2_regionloss.log \
NUM_GPUS=2 ./run_v2_train.sh \
  --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
  --output_dir output/phase1_v2_slice27_v2_regionloss --model_input 448x252 --canvas 1600x900 --angle_step 0.5 \
  --epochs 40 --max_steps 8760 --batch_size 16 --lr 1.5e-4 --weight_decay 0.01 \
  --warmup_ratio 0.03 --grad_clip 1.0 --num_workers 8 --eval_every 2000 --save_every 2000 \
  --seed 42 --heads 8 --mlp_ratio 4.0 --decoder_depth 2 --slice_start 2 --slice_end 7 \
  --region_loss true

# B 组: region_loss=False（对照, 逐位复现旧"每步整图"损失 = slice27_v2 历史行为）
#   同上, 仅改: --output_dir output/phase1_v2_slice27_v2_fullloss  --region_loss false
#   健全性预期: B 组轨迹应与历史 slice27_v2（eval_loss 0.2490@8760, 全量 L1 19.878）复现一致

# 推理（两组同一命令; 指标全图口径, 与历史直接可比; region_loss 由 model_info 自动对齐）
python infer_v2_test.py --data_dir /root/autodl-tmp/construction_site \
  --dino_dir /root/autodl-tmp/models/dinov2-large \
  --final_model output/phase1_v2_slice27_v2_regionloss/final_model.pt \
  --output output/phase1_v2_slice27_v2_regionloss/infer_test.json --slice_start 2 --slice_end 7

# 对比要点: A/B 的 infer full_pixel_l1_255（主判据, 全图口径）+ A 的渐进曲线是否出阶梯/后步是否抬升（§6.1 污染监控）
#           + eval 轨迹只看 eval_recon（§4）; K3 判据①③④需另跑 probe（E1/single_step 脚本族）
```

## 8. 数值验证记录（2026-09-07, 实现后立即执行）

### 8.1 本地（CPU, torch 2.8.0）

- `python3 model_v2.py` → **ALL CHECKS PASSED**（含: region(N=16,T=4)=[(0,4),(4,8),(8,12),(12,16)] 损失 == 手工区域掩码均值; region_loss=False 复现旧全图损失且 F_hat/Y_pix 逐位不变; 梯度按步解耦 region 口径——每步 Y_t 恰收 1/4 份梯度且只落在自己 region 的行）。
- state_dict 兼容: `region_loss` 不出现在 state_dict; 旧式 ckpt（region_loss=False 模型导出）→ 新默认模型 `strict=True` 加载 missing/unexpected 均空。
- `region_slices` 随机扫描 500 组 (N,T)（含非整除/退化）: 互不相交/覆盖/大小差≤1/非空 全部成立。
- `--region_loss` CLI: 默认 True; `false/False/0/no/--region_loss=false` → False; 裸 flag/`true/1` → True; 非法值 argparse 报错退出。

### 8.2 服务器（connect.westd.seetacloud.com:32298, 2× RTX PRO 6000, torch 2.12.1+cu130, python 3.12.3）

临时目录 `/root/qwen_region_loss_val/`（只拷入改后的 model_v2.py + 冒烟脚本, 不触碰 sr-diffusion-v2-k99; 验证完已删除）:
- (a) `CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python model_v2.py` → ALL CHECKS PASSED（自检为 CPU 路径, FakeDino 结构, 无外部数据依赖）;
- (b) GPU 冒烟（真实 Dinov2Model(/root/autodl-tmp/models/dinov2-large), N=576, slice[2:7] → T=5, K=63, 随机输入, fp32）: region_loss=True 的 loss == 手工按 [(0,115),(115,230),(230,345),(345,460),(460,576)] 掩码的均值（逐位）; region_loss=False 的 loss == 全图均值（旧公式, 逐位）; 两口径 backward 均通, decoder.query_base / special_bank.pos / pixel_head 梯度非零; **decoder.last_Y 在非本步区域行的梯度精确为 0**（region 口径, atol=0）; 同权重同输入下两口径 F_hat/Y_pix 逐位相等。具体数值见实现报告（loss 量级 ~1.0, 随机初始化预期）。

## 9. 与历史文档的关系

- `ANALYSIS_k3_why_later_steps_zero.md` §4 P1 = 本设计的方案来源（判据/风险/成本预估均引自该文）;
- `ANALYSIS_k3_slice_infodiff.md`（z_s within-std / 键数差）= 判据④的基线数值来源;
- `REPORT_v2_block_slice27.md` / `REPORT_v2_slice27_causal_mask.md` = A/B 协议（slice27_v2 同预算）来源;
- `DESIGN_v2_num_specials_from_max_steps.md` = K 推导（本设计不改 K 语义, region 划分只依赖 N 与 T=|steps|）。
