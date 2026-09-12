# SR-Diffusion v3 — 「每步独立 [cls] / step 身份注入」实验 Runbook（2026-09-12）

> 目的: 检验报告 `ANALYSIS_k3_deepening_no_late_stage.md` §2.1 的论断——
> 「后段分工的充要条件之一是『步身份进前向』（让 5 步*能*不同），而当前架构没有这个注入点」。
> 4 层（depth-4 / stack2x 配置）是为了对齐唯一给出过 step4 大补救（Δ4=+0.3512）的模型。
> 服务器: `ssh -p 38024 root@connect.westb.seetacloud.com`（2× RTX PRO 6000 Blackwell 95GB）
> 控制组（单变量对照）: `/root/autodl-tmp/sr-diffusion-v3-stack2x` 的
> `output/phase1_v2_stack2x_lr1e4_slice05`（2048/16/4/dropout0.05/lr1.0e-4/seed42/8760 步）

---

## 0. 状态

| 臂 | 目录 | 状态 |
|---|---|---|
| ① additive step embedding | `/root/autodl-tmp/sr-diffusion-v3-stepid` | **完成** 2026-09-12 16:25（全 `EXIT=0`）→ 结果见 §6 |
| ② query 侧 per-step [cls] | `/root/autodl-tmp/sr-diffusion-v3-stepid-cls` | **完成** 2026-09-12 21:35（全 `EXIT=0`） |

> **最终四臂对照结论见 `REPORT_step_identity_AB.md`**（baseline / stack2x / 臂① / 臂②）。
> 一句话：两种身份注入都有效（臂② 把 stack2x→baseline 差距填掉 50%，且首次让 Δ16/Δ25 双翻正），
> **但都没有换来实质「后段分工」**——后段增益仅 +0.0015~0.005 px，收益主体是 step1 单发读出变好。

配置（两臂与控制器逐项相同，仅架构补丁不同）：

`--decoder_depth 4 --stack_dim 2048 --heads 16 --decoder_dropout 0.05 --lr 1.0e-4
--weight_decay 0.01 --warmup_ratio 0.03 --grad_clip 1.0 --batch_size 16 --grad_accum 1
(--num_processes 2 → 全局 batch 32) --max_steps 8760 --seed 42 --slice_start 0 --slice_end 5
--model_input 448x252 --canvas 1600x900`（K=35，steps=[1,4,9,16,25]，blockdiag）

启动脚本: `/root/run_stepid_d4_slice05.sh` + `/root/post_stepid_d4_slice05.sh`（①）
　　　　　`/root/run_stepidcls_d4_slice05.sh` + `/root/post_stepidcls_d4_slice05.sh`（②，待命）
日志: `/root/train_logs/stepid_d4_slice05.log`、完成标记 `/root/train_logs/stepid_d4_POSTDONE`

---

## 1. 臂① `step_embed`（additive step embedding）

补丁（`OutputQueryDecoder`，对 pristine 的 diff）：

```diff
@@ __init__ (self.pos_embed 之后)
+        # ── 实验臂①(2026-09-12): 每步独立可学习身份向量 step identity ──
+        # 零初始化 ⇒ 默认路径与未打补丁实现逐位一致; 仅新增 |T|·stack_dim 参数。
+        self.step_embed = nn.Parameter(torch.zeros(len(self.steps), self.stack_dim))
+
@@ forward
-        Y = self.stack(self.stack_in(Y), self.stack_in(A),
+        Y = self.stack_in(Y)                                     # (B,|T|·N,stack_dim)
+        # 实验臂①: 该步身份向量加到本步全部 N 行查询（5 步各自独立）
+        Y = Y + self.step_embed.repeat_interleave(N, dim=0).unsqueeze(0)
+        Y = self.stack(Y, self.stack_in(A),
                        memory_mask=mask, tgt_mask=tgt_mask)     # (B,|T|·N,stack_dim)
```

**验证（已通过）**
- 与 pristine 同权重同输入：`max_abs_diff = 0.0`（零初始化，默认路径**逐位不变**）
- `step_embed` 形状 `(5, 2048)`；真实 backward 下 `grad_norm = 0.049476`（有梯度）
- 6 步单卡 smoke `SMOKE_EXIT=0`，`final_model.pt` 正常保存
- 参数量增量：`|T|·stack_dim = 5×2048 = 10240`（相对 581M 可忽略 ⇒ 测的是「条件/身份」不是容量）

## 2. 臂② `step_cls`（query 侧拼接独立 [cls]）

补丁（对 pristine 的 diff）：

```diff
@@ __init__ (self.pos_embed 之后)
+        # ── 实验臂②(2026-09-12): 每步一个独立 [cls] 查询槽 ──
+        self.step_cls = nn.Parameter(torch.randn(len(self.steps), dim) * 0.02)
+
@@ forward（查询序列每步由 N 行变 N+1 行）
-        Y = (A_t.unsqueeze(2) + self.query_base) \
-            .reshape(B, len(self.steps) * N, D)
+        Q = N + 1
+        step_cls = self.step_cls.view(1, len(self.steps), 1, D).expand(B, -1, -1, -1)
+        Y = torch.cat([A_t.unsqueeze(2) + self.query_base, step_cls], dim=2) \
+            .reshape(B, len(self.steps) * Q, D)
@@ 掩码尺寸 N → Q
-        mask = build_block_mask(..., num_queries=N, ...)
+        mask = build_block_mask(..., num_queries=Q, ...)
-        tgt_mask = build_causal_query_mask(len(self.steps), N, ...)
+        tgt_mask = build_causal_query_mask(len(self.steps), Q, ...)
@@ 输出：切掉 [cls] 行，保持对外契约不变
-        Y = Y.reshape(B, len(self.steps), N, D)
+        Y = Y.reshape(B, len(self.steps), Q, D)
+        self.last_step_cls = Y[:, :, N]
+        Y = Y[:, :, :N]
         self.last_Y = Y
         return Y
```

**验证（已通过，small config dim=256/N=36/depth2）**
- `out = (2,5,36,256)`（对外契约与未打补丁一致 ⇒ decode/infer/probe 无需改动）
- `last_step_cls = (2,5,256)`；`attn_mask=(185,36)`；`tgt_mask=(185,185)`
- blockdiag 隔离成立：step0 行看不到 step1 行（`tgt_mask[0,37]=-inf`），能看到自己块的 cls+patch（`=0`）
- `step_cls` 有梯度：`grad_norm = 0.103429`
- 注意：② **不是**零初始化、**不**保持默认路径逐位一致（它改变了序列结构）——这是设计使然，其对照仍是未打补丁的 stack2x。

---

## 3. 后处理流水线（训练后自动串联，两臂同构）

1. 全量 test(3004) 推理 `infer_v2_test.py --final_model …/final_model.pt`
2. 终版探针 `probe_step_collapse.py --limit 512`
3. 各 ckpt(2000/4000/6000/8000) 探针 `--limit 128`
4. `step_gain_analysis.py` 多臂对照（`--probe/--glob/--infer`），产出
   `STEP_GAIN_stepid.md/json`（② 为 `STEP_GAIN_stepidcls.md/json`，含 baseline / stack2x_lr1e4 /
   stepid_d4 / stepidcls_d4 四臂）

## 4. 判读口径（事先写死，避免事后解释）

- **主判据**：后段边际增益 `Δ16`、`Δ25` 是否从「三模型全负」翻转（probe + 全量 test 同向）。
- **副判据**：后段 `step_px_scale` 占比、tail 逐块 cos、`z_s` within-std 是否出现结构性变化
  （对照 §3#1：blockdiag 曾让 cos 变健康但后步更塌 ⇒ 若②/①只改 cos 不改 Δ，仍判「无效」）。
- **聚合判据**：`eval_recon` 与 stack2x 的 0.4174 对比（更低 = 更好）。
- **聚合口径注意**：train loss 与 eval_recon 同为归一化像素空间；逐 ckpt 曲线必须一起看，
  单看终点容易把「没收敛」误判成「结构性更差」。

## 5. 复现（臂①）

```bash
ssh -p 38024 root@connect.westb.seetacloud.com
# 训练（~5h，2 卡；已完成/进行中则勿重复启动）
nohup setsid bash -c 'bash /root/run_stepid_d4_slice05.sh; bash /root/post_stepid_d4_slice05.sh' \
  > /root/train_logs/stepid_wrapper.log 2>&1 < /dev/null &
# 进度
tail -c 300000 /root/train_logs/stepid_d4_slice05.log | tr '\r' '\n' | tail -3
# 完成标记 / 结果
cat /root/train_logs/stepid_d4_POSTDONE
cat /root/autodl-tmp/sr-diffusion-v3-stepid/STEP_GAIN_stepid.md
```

---

## 6. 臂① 实验结果（训练 16:19:55 完成；后处理 16:25:32 完成，全 `EXIT=0`）

证据（已入库 `doc/2026-09-12/data_stepid/`）：`STEP_GAIN_stepid.md/json`、
`probe/probe_stepid_d4_slice05*.json`（终版 512 + 各 ckpt 128）、`infer_test_stepid_d4.json`、
`args_stepid_d4.json`、`model_v2_stepid.py`、两脚本。

### 6.1 聚合重建

| | baseline | stack2x_lr1e4 | **arm① stepid_d4** |
|---|---|---|---|
| `full_norm_l1` | 0.33179 | 0.41729 | **0.40316** |
| `full_pixel_l1_255` | 19.018 | 23.776 | **22.999** |
| 相对 baseline | — | +25.8% | **+21.5%** |
| 相对 stack2x | — | — | **−0.0141 norm（−3.4% 相对）/ −0.78 px** |

逐 ckpt `eval_recon`：`@2000 0.5029 / @4000 0.4583 / @6000 0.4371 / @8000 0.4074`
（stack2x 同点：0.5112 / 0.4655 / 0.4376 / 0.4191）⇒ **四个 ckpt 全部更优或持平**，
但优势在 @6000 收窄到 0.0005、@8000 又回到 0.0117。

### 6.2 各 step 边际增益 Δ（正 = 该步降低误差）

probe 512：

| t | baseline | stack2x_lr1e4 | **arm①** |
|---|---|---|---|
| 4 | +0.0190 | +0.3512 | **+0.4165** |
| 9 | +0.0032 | +0.0107 | **+0.0178** |
| 16 | **−0.0030** | +0.0001 | **+0.0050** ← 翻正 |
| 25 | **−0.0075** | −0.0020 | **+0.0006** ← 翻正 |

全量 test 3004：

| t | baseline | stack2x_lr1e4 | **arm①** |
|---|---|---|---|
| 4 | +0.0148 | +0.4337 | +0.3645 |
| 9 | +0.0019 | +0.0109 | +0.0134 |
| 16 | **−0.0037** | **−0.0009** | **+0.0022** ← 翻正 |
| 25 | **−0.0083** | **−0.0038** | **−0.0016** ← 仍负，但幅度减半 |

### 6.3 机制侧观测量

- 后 4 步 `scale` 占比：baseline 3.3–3.8% → stack2x 7.1–8.0% → **arm① 9.9–12.0%**（后段更"响"）。
- `z_s` 逐块 cos：arm① `[0.625, 0.9992, 0.9990, 0.9987, 0.9985]`、`within_std 0.061`
  —— 比 blockdiag 基线 `[0.734,0.876,0.918,0.916,0.926]`/0.095 **更塌**，但后段反而更不有害
  ⇒ 再次否证"register 塌缩 → 后段归零"。
- ckpt 轨迹（probe 128，总落差）：`2000 −0.0971 → 4000 −0.0507 → 6000 +0.0079 → 8000 +0.2606 → 8760 +0.4665`。

### 6.4 按事先判据的判读

- **主判据（部分成立）**：`Δ16` 在 probe 与全量 test 上**都翻正**；`Δ25` 在 probe 上翻正、
  在全量 test 上仍为负但幅度减半。⇒ "步身份注入"确实把后段从"净有害"推到"≈中性/微正"。
- **但未达成"后段分工"**：所有增益仍集中在 step1→step4；后段绝对增量 ≤0.005 px，
  相对 ~24 px 的绝对水平可忽略。**没有出现"对角线式"逐区下降。**
- **收益的归属是"整体抬升"而非"分工"**：step1 自身从 25.29 改善到 24.47（probe）/
  `head_mean` 23.885→23.097（全量），即新增容量主要被**主导通路 step1** 吸收
  —— 与报告 §2.3 "新容量被 step1 吃掉"的预测一致。
- **聚合仍远差于 baseline**（0.40316 vs 0.33179，+21.5%）⇒ 身份注入**不能**修复
  容量/lr/目标结构造成的主体退化。

**结论**：臂① = **可检出但微弱的正效应**；它否证了"无注入点是唯一绑定约束"的强版本，
支持"绑定约束在目标/损失侧"。据此按用户规则**改跑臂②**（更强的身份注入：
每步私有 [cls] 查询槽），已 2026-09-12 16:3x 启动。
