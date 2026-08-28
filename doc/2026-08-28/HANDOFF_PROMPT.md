# SR-Diffusion-v3 — 新会话交接 Prompt（2026-08-28 最新版）

> 用途：在**新会话**中无缝继续本项目的全部工作。**先读本文件**，再按需读
> `doc/2026-08-26/HANDOFF_PROMPT.md`（历史脉络第一轮）与各日期报告；已完成的
> 实验与分析不要重做，结论都在文档里。对话背景摘要已全部沉淀进本文件，无需
> 再向用户反复询问。

## 项目一句话（权威目标 2026-08-28 修订 v2）
推进 GitHub 仓库 `shaojun0/SR-Diffusion-v3`（用户私有项目）phase1 v2：
**通过 token 压缩训练编码器的联想能力**（把图像信息压进少量 special token z_s），
训练完成后**冻结编码器**，作为 `model.py` 的编码器做 **NLP 解码训练**（Qwen 生成
中文工地描述/隐患）。

⚠️ **像素重建 = Phase 1 的训练脚手架 + 信息保持探针，不是最终目标**：
验收标准是 Phase 2 冻结编码器后的文字生成质量；但重建与 NLP 不构成对立
（能还原像素 ⇒ z 携带整图信息 ⇒ 重建质量决定 NLP 天花板）。Phase 1 中间验收 =
**K 压缩 × 重建质量**（K=32/64/128 下活信息=布局/物体/边界保真），不追纹理级
清晰度（死信息）。**README.md 已含此目标的完整表述**（其引用的
`doc/2026-08-28/GOAL_compression_for_nlp.md` 目前在 GitHub 仓库缺失——用户本地
`D:\Python_project\SR-Diffusion-v3` 有，需用户推送；交接时以 README §目标为准）。

## 环境（2026-08-28 更新：换服务器了！）
- **本地工作区**: `/home/linaro/dsh/sr-diffusion-v3`（git 仓库，remote 含 GitHub 凭据可直接 push，分支 main）
- **远程 GPU 服务器（当前）**: `ssh -p 35056 root@connect.westd.seetacloud.com`（免密）
  - ⚠️ 08-28 起**换实例**：旧 `32558` 端口租不到被占用；用户换了新实例并把
    `autodl-tmp` 数据拷贝过去。**新实例 pip 环境重新装过**（torch 2.12.1+cu130、
    transformers 5.15.1、accelerate 1.14、datasets 5.0.1，Python 3.12.3）。
  - 数据 `/root/autodl-tmp/construction_site`（parquet：image + image_caption 中文 + violations，4.2G）
  - DINOv2-large `/root/autodl-tmp/models/dinov2-large`；Qwen3.8-27B `/root/autodl-tmp/models/Qwen3.8-27B`
  - 代码目录：`/root/autodl-tmp/sr-diffusion-v3-main`（本次部署的 main 分支最新代码）
  - 旧实例产物仍在 `/root/autodl-tmp/sr-diffusion-v3`（旧版）与 `sr-diffusion-v3-test`（test 分支/像素目标版）
- **网络**: HF 官方站不可达（hf-mirror 403）→ 模型走 ModelScope；pip 走阿里云镜像；训练/推理设 `HF_HUB_OFFLINE=1`

## 代码结构（main 分支根目录，均为当前最新）
| 文件 | 作用 |
|---|---|
| `model_v2.py` | SRPhase1V2：DINOv2-large(不冻结) → ReEncoder(因果 specials) 或 **register_specials**(specials 拼进 DINO 序列, 无 ReEncoder) → OutputQueryDecoder(输出查询注意力+KV因果+平方采样25步) → PixelHead → **重建原始像素**；**平权全覆盖 L1**（去掉加权体系）；自检 `python model_v2.py` |
| `data_v2.py` | `fit_to_canvas`（最优旋转角+等比缩放+居中填充 1600:900）+ ParquetImageDataset + V2Collator；自检 `python data_v2.py` |
| `train_v2.py` | **HF Trainer** 风格（不再手写 DDP 循环）；**全 fp32**（无 bf16/autocast）；平权 mean；bs16/卡 默认；`--register_specials` 开关 |
| `run_v2_train.sh` | `NUM_GPUS=n ./run_v2_train.sh ...` 启动 |
| `infer_v2_test.py` | 推理测试：全量重建像素 L1 (fp32) + 每采样步渐进曲线 |
| `visualize_recon_pixel.py` | 重建可视化（原图 vs 各采样步蒙太奇） |
| `doc/2026-08-26/` | 第一轮：特征目标训练 + 泄露排查 + HANDOFF（历史） |
| `doc/2026-08-27/` | 像素目标训练报告 + **DIAGNOSIS_clarity.md**（清晰度根因诊断） |
| `doc/2026-08-28/` | register 训练报告 + 重建可视化 + 本交接文档 |

## 历史脉络（三轮对话摘要，2026-08-26 → 08-28）
1. **08-26 第一轮（test 分支）**：用户更新 test 分支（`2655fc4 注意力机制改写`：
   FeatureDecoder→OutputQueryDecoder + 平方采样 + 加权全覆盖损失）。按 main 文档标准
   （工地数据/1600×900/576 patches）跑通训练（40ep/17520步, bf16 算 fp32 存）。
   推理显示**渐进曲线全平**（t=0 仅 1 键也达全量精度）→ 用户怀疑 token 泄露。
   **排查结论：不是泄露**（掩码数值验证 ΔY≈1e-8），而是**目标退化**：DINO patch
   特征在工地图上空间变异性仅 ~5e-5，重建退化为"输出常数"，模型 L1 比质心
   baseline 还差 35-57×。详见 `doc/2026-08-26/REPORT_test_v2_train_and_leak.md`。
2. **08-27 第二轮（像素目标重大修复）**：用户本地修复——监督目标从 DINO patch
   特征改为**原始像素**（PixelHead 1024→588），修复"特征目标退化"假收敛
   （旧模型像素 L1=64.9≈平均色 61 → 像素目标 L1=23.41，改善 2.6×）。渐进曲线恢复
   （t=0 粗糙 60 / t≥1 精细 22.7），联想能力成立。训练口径改为**平权 + 全 fp32 +
   bs16**。清晰度诊断见 `doc/2026-08-27/DIAGNOSIS_clarity.md`（根因排序：解码侧
   F3/F4 逐 patch 信息路由缺陷第一，DINO 信息瓶颈 9.8 第二）。产物在
   `sr-diffusion-v3-test/output/phase1_v2_pixel_fp32/final_model.pt`。
3. **08-28 第三轮（本次）**：test 并入 main；main 重构（dfb993b `改进模型` 新增
   **register_specials**：specials 作为额外 token 直接进 DINO 输入序列 1153 token
   全双向，由 24 层直接算 z_s，修 DIAGNOSIS 的 F1/F2；无 ReEncoder，省 51.6M 参数，
   训练慢 2-3×）。按用户三项要求（平权 / 全 fp32 / bs16 提高）在新服务器跑通
   register 版 40ep/8760 步，并测试清晰度。

## 当前状态与结论（2026-08-28）
1. **register 版训练完成**：40ep/8760 步，eval_loss 最终 **0.4575**（归一化空间，
   旧版 0.4230，持平）；全量重建像素 L1 (0-255) = **24.14**（旧版 23.41，持平）。
   产物 `/root/autodl-tmp/sr-diffusion-v3-main/output/phase1_v2_reg/final_model.pt`
   （1.26GB fp32）+ infer_test.json + recon_visual.png。
2. **渐进曲线**：t=0 L1=85.5（粗糙）→ t≥1 L1=23.6（平台）→ **联想能力成立**
   （2 键≈全量）。t≥1 平台依旧存在。
3. **清晰度关键结论（同口径 RGB 对比）**：register 版与旧版几乎完全一样——
   边缘保留均 ~20%（低通签名依旧），L1 23.9 vs 22.9。→ **register_specials
   （编码侧 F1/F2 修复）不改变重建清晰度**，数值验证 DIAGNOSIS 预判：瓶颈在
   **解码侧 F3/F4**（查询身份只来自固定 query_base 模板、输出=共享键凸组合）。
4. **工程修复（已 push main）**：eval OOM 两处——`eval_accumulation_steps=2`
   （4e081e1）+ `SRPhase1V2Trainer.prediction_step` ignore `Y_pix/target_pix`
   （37e7b04，根治：eval 预测从 ~100GB 降到 ~4GB，eval 从 30 分钟崩溃 → 90 秒完成）。

## 下一步（按 DIAGNOSIS §7 优先序，用户拍板后执行）
1. **改解码器为"逐 patch 内容查询"**（推荐先做）：`Q = W_q(A_t) + query_base +
   Linear(z_s[k])`（一行改动）——预期 t≥1 平台消失、全量 L1 从 ~23 降到 ≤16
   （回收 12.9 的大头）。
2. PixelHead 换小 CNN/转置卷积恢复高频（0.6M→几 M）。
3. 叠加感知损失 LPIPS / 多尺度 L1（在修路由之后，否则逼模型编造假细节）。
4. **K 压缩实验**（Phase 1 中间验收）：K=32/64/128 下活信息（布局/物体/边界）保真。
5. 文字模式（Qwen 冻结编码器接 NLP 解码，Phase 2 最终验收）。

## 已知坑（勿重蹈，代码注释里也有）
- **eval OOM（已修）**：HF Trainer eval 会把全部 batch 预测（含 Y_pix B×25×576×588）
  累积 GPU → OOM/超时。解法：prediction_step ignore 巨型键 + eval_accumulation_steps。
- **register_specials 模式**：DINO 处理 1153 token，训练慢 2-3×、显存峰值高
  （bs16 单卡 ~53GB）；收敛起步慢（DINO 需要长热身），但最终持平旧版。
- **DINO mask_token**：权重带 use_mask_token=True 但本任务不传 bool_masked_pos →
  DDP 报"未用参数" → 加载后 `dino.config.use_mask_token=False; del dino.embeddings.mask_token`。
- **HF Dinov2Model 无 token 级注意力 mask API** → register 模式 DINO 内全双向
  （无掩码）；解码器 KV 因果仍提供渐进前缀语义。
- **fp32 导出**：final_model.pt 必须 fp32（bf16 权重量化使重建 L1 劣化约 2 倍）。
- **目标退化教训**：工地 DINO patch 特征空间近常数（std≈5e-5）——监督特征目标会
  假收敛；**必须监督原始像素**（有真实空间结构）。
- **服务器换实例**：SSH 端口会变（32558→35056）；实例重启/更换后 pip 环境需重装
  （数据在 autodl-tmp 持久化）。

## 工作方式
- 中文交流；改动前先读文件，改动后跑对应自检（`python model_v2.py` / `python data_v2.py`）；
  实现完提交 git 并 push（remote 已含凭据，分支 main）。
- 用户（shaojun0）是仓库 owner，代码与文档均中文，遵循仓库现有风格（解耦、不造轮子、熵减）。
- 训练/推理命令参考 README 快速开始与 `doc/2026-08-28/REPORT_register_fp32_train.md`。
- 服务器若关机，先确认实例/端口是否变化，再谈训练。
