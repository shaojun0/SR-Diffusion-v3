# SR-Diffusion-v3 phase1 v2 — 新会话交接 Prompt

> 用途：在**新会话**中无缝继续本项目的全部工作。先读本文件 + 下面列出的关键文档，再动手；
> 已完成的实验与数学分析不要重做，结论都在文档里。

## 项目一句话
推进 GitHub 仓库 `shaojun0/SR-Diffusion-v3`（用户私有项目）的 phase1 v2 模型：
**DINOv2-large（参数不冻结）→ ReEncoder → FeatureDecoder，L1 重建 DINO patch 特征**；
图像预处理为**旋转(最优角)+等比缩放+居中填充到 1600:900**（轮廓不变形、内容面积最大化）；
可选 **TextDecoder** 文字自回归（中文工地 caption）；当前重点是**前缀加权机制**（预算/渐进还原）的设计与训练验证。

## 环境
- **本地工作区**: `/home/linaro/dsh/sr-diffusion-v3`（git 仓库，remote 已含 GitHub 凭据可直接 push，分支 main）
- **远程 GPU 服务器**: `ssh -p 32558 root@connect.westd.seetacloud.com`（免密）
  - ⚠️ 2026-08-26 起连接被拒（Connection refused），**先确认实例是否还在/端口是否变了**；若实例重启，训练产物可能丢失，需与用户讨论（可能要重训）
  - 环境: 2× RTX PRO 6000（96GB）；miniconda3（`/root/miniconda3/bin`，PATH 需 export）；数据 `/root/autodl-tmp/construction_site`（parquet：image + image_caption 中文 + violations）；DINOv2-large `/root/autodl-tmp/models/dinov2-large`（ModelScope 下载）；Qwen3.8-27B `/root/autodl-tmp/models/Qwen3.8-27B`（多模态，text_config hidden=5120、vocab=248320、embedding 在 model-00003-of-00018.safetensors，bf16）
- **网络**: HF 官方站不可达（hf-mirror 403）→ 模型走 ModelScope（modelscope.cn）；pip 走阿里云镜像

## 代码结构（都在本地仓库根目录）
| 文件 | 作用 |
|---|---|
| `model_v2.py` | SRPhase1V2：DINOv2(不冻结)→ReEncoder(因果 specials 前缀链)→FeatureDecoder(L1)；可选 TextDecoder(vocab_size>0)；`build_prefix_mask` 三处块掩码；**FeatureDecoder 已支持 k<N 前缀**（掩码 z_end=k+1，输出恒全量 N）；自检 `python model_v2.py` |
| `data_v2.py` | `fit_to_canvas`（最优旋转角网格搜索+等比缩放+居中填充 1600:900）+ ParquetImageDataset + V2Collator（text 模式可选 tokenizer 编码中文 caption，动态 padding）；自检 `python data_v2.py` |
| `train_v2.py` | accelerate 多卡 DDP；显式 `torch.autocast(bf16)`；**fp32 导出 final**；`--text_decoder --qwen_dir` 多任务模式（Qwen 词表 embedding 分片直读、默认冻结、优化器只收可训练参数） |
| `run_v2_train.sh` | `NUM_GPUS=n ./run_v2_train.sh ...` 启动 |
| `infer_k_sweep.py` | 还原精度 vs k 扫描（`--kmax 32 --anchors "64,128,..."`，fp32 度量） |
| `DESIGN_prefix_weighting.md` | 前缀加权机制设计草稿（**§3 已数学修正**，看最新版） |
| `MATH_mask_analysis.md` | 掩码机制数学分析（必要不充分论证、恒等式修正、梯度压力机制、传递自洽局限） |
| `mask_mechanism.gif` / `gen_mask_gif.py` / `verify_weighting_math.py` | 可视化动图与数值验证脚本 |

## 当前状态（2026-08-26）
1. **重建训练已完成**（旧代码全量 k=N）：40 epochs/17520 步双卡 DDP；fp32 全量测试 L1≈0.00114。产物在服务器 `output/phase1_v2/`（final_model.pt 为 fp32，来自 ckpt-16000；另有 k_sweep_full.json）——**服务器不可达，需先确认产物是否还在**。
2. **TextDecoder 已适配未训练**（用户新提交的模块，等用户审核后决定是否开训）。
3. **k 扫描结论**：k=1..32 平台期（L1≈0.0033 = 全量 2.9×）；窗口探针显示编码器**信息摊匀**（任意 32-token 窗口还原能力相同）。
4. **数学分析结论**：掩码**有用（必要不充分）**，应保留；"只算前 k 的预算推理"（截断编码器）需注意**传递前缀自洽不成立**（patches 枢纽回渗，MATH §3.2）。
5. **加权机制**：w(k) 递减 ⇒ 梯度压力 Σ_{j≥i}w_j 递减 ⇒ 信息前置（**正确论证**，DESIGN §3 已修正；均匀采样精确 ∝ N−i+1）；需配 p_full 全量保底 / w 地板 / k_min 下限。
6. **前缀课程已实现（2026-08-26，待用户审核）**：`train_v2.py --prefix_curriculum`
   + `--prefix_k_min / --prefix_p_full / --prefix_dist / --prefix_w / --prefix_w_p / --prefix_w_floor`；
   `model_v2.py` 的 `z_keep` 统一作用于重建+文字两分支，新增 `prefix_weight` / `sample_prefix_k`；
   `infer_k_sweep.py` 新增 `--window` 滑窗探针。全部自检通过（`python model_v2.py` /
   `python train_v2.py`）。详见 `DESIGN_prefix_weighting.md` §10 实施记录。

## 下一步（用户计划，待用户拍板后执行）
1. **先确认服务器状态**（实例/端口/数据/产物）；不可达则向用户说明并讨论恢复方案。
2. **审核前缀课程实现**（DESIGN §10 实施记录 + git commit）——通过后上服务器开训：
   ```bash
   # 纯重建 + 前缀课程（示例默认参数，可调）
   accelerate launch --multi_gpu --num_processes 2 \
       train_v2.py --data_dir /root/autodl-tmp/construction_site \
       --dino_dir /root/autodl-tmp/models/dinov2-large \
       --prefix_curriculum --prefix_p_full 0.5 --prefix_k_min 8 \
       --output_dir output/phase1_v2_prefix
   # 文字模式同加 --prefix_curriculum（文字条件与重建共享同一前缀 k）
   ```
3. 训练后重跑 `infer_k_sweep.py`（k 扫描 + `--window 32` 滑窗探针）验证：预期 k=1..32 平台消失、前段窗口信息量>后段、全量 L1 退化 ≤1.5×（当前 0.00114）。
4. 验证成功后并入正式版（改进 v2），届时再打版本 tag。

## 已知坑（勿重蹈，代码注释里也有）
- **torch 2.13 混合设备 bug**：输入在 CPU、模型在 CUDA 且开 autocast 时 conv 报 "Input type (float) and bias type (BFloat16)" → 训练/评估显式 `.to(acc.device)` + 显式 `torch.autocast("cuda", bf16)`；不要用 accelerate 的 mixed_precision 自动包装（实测有同样的报错）
- **DINO mask_token**：权重带 `use_mask_token=True` 但本任务不传 bool_masked_pos → DDP 报"未用参数" → 加载后 `dino.config.use_mask_token=False; del dino.embeddings.mask_token`
- **Qwen3.8-27B 是多模态**：词表在 `model.language_model.embed_tokens.weight`，用 safetensors 分片直读（勿加载全模型）；冻结词表 1.27B 参数不能进 AdamW（按 requires_grad 过滤）
- **fp32 导出**：final_model.pt 必须 fp32（实测 bf16 权重量化使重建 L1 劣化约 2 倍：0.0011→0.0021）
- **训练日志 eval 是 bf16 损失略低估**（0.000995 vs fp32 真实 0.00114），对比时用 fp32 度量

## 工作方式
- 中文交流；改动前先读文件，改动后跑对应自检（`python model_v2.py` / `python data_v2.py`）；实现完提交 git 并 push（remote 已含凭据）
- 用户（shaojun0）是仓库 owner，代码与文档均中文，遵循仓库现有风格（解耦、不造轮子、熵减）
- 服务器若恢复，训练命令参考 README/run_v2_train.sh；数据/模型路径见上
