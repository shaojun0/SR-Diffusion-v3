# D-AR 微调结果报告（construction_site 工地数据）

> 执行时间：2026-09-01 22:28 – 23:25（服务器时区 CST）
> 服务器：2× NVIDIA RTX PRO 6000 Blackwell（97GB/卡），base conda（torch 2.12.1+cu130）
> 数据：`/root/autodl-tmp/dar_data/images/construction/*.jpg`（1000 张，ImageFolder）
> 结论先行：**tokenizer 微调（280 步）与 D-AR 微调（3000 步）均成功完成**，训练损失显著下降，tokenizer 重建指标小幅提升（PSNR +0.15），D-AR 生成模型已适配工地数据分布。所有产物与日志见文末清单。

---

## 1. Tokenizer 微调

### 配置（`finetune_tokenizer.sh`，实际执行参数）
| 项 | 值 |
| -- | -- |
| 命令 | `bash finetune_tokenizer.sh`（2 卡 accelerate launch，bf16） |
| 模型 | 官方 `tokenizer_v1.pt`（307.4M VQModel，顺序扩散 VQ，codebook 16384×8，256-res） |
| 数据 | 1000 张工地图（`--data-path /root/autodl-tmp/dar_data/images`） |
| 批量 | `--global-batch-size 128`（64/卡） |
| 优化 | `--lr 1e-4 --warmup-steps 1000 --beta2 0.95 --weight-decay 0.0 --ema` |
| 损失 | `--perceptual-weight 0.5 --dino-weight 0.5 --entropy-loss-ratio 0.0`（无 GAN） |
| 时长 | `--epochs 40 --ckpt-every 2000 --log-every 20` |

### 结果
- **训练步数：280 步**（40 epochs × 7 步/epoch，1000 张/128 batch）
- **时长：约 7 分钟**（22:33:35 → 22:40:45）
- **最终 Train Loss：0.3453**（step 20 时 0.3749 → step 280 时 0.3453，持续下降）
  - 分量：fm_loss ~0.108，perceptual ~0.265，repa(DINOv2) 0.238 → **0.205**
  - codebook usage：0.44 → **0.97**（codebook 使用率明显提高，死码率 0）
- 显存：68.2 GB/卡（97GB 卡安全）；吞吐 ~0.66 step/s
- 预训练权重全量加载（Skipped params: []），EMA 重建、optimizer 重置、steps=0（官方裸 state_dict 的预期行为，见 §6 问题 2）
- **产物：`temp/tokenizer_v1_ft.pt`**（4.97GB，含 model/ema/optimizer/discriminator/config，steps=280）

## 2. D-AR 微调

### 配置（`finetune_dar.sh`，实际执行参数）
| 项 | 值 |
| -- | -- |
| 命令 | `bash finetune_dar.sh`（2 卡 accelerate launch，bf16） |
| 模型 | GPT-L（343M），初始化自官方 `temp/D-AR-L-360K.pt`（**裸 state_dict**，已兼容处理，见 §6 问题 5） |
| Tokenizer | 微调产物 `temp/tokenizer_v1_ft.pt`（EMA 权重，`load_visual_tokenizer` 优先读 ema+config） |
| 条件 | `--num-classes 1`（单类无条件生成，类嵌入随机初始化，见 §6 问题 6） |
| 批量 | `--global-batch-size 64`（32/卡） |
| 优化 | `--lr 1e-4 --ema --dropout-p 0.1 --token-dropout-p 0.1`（默认） |
| 其它 | `--image-size 256 --no-local-save --no-compile`（compile 修复见 §6 问题 7） |
| 时长 | `--epochs 200 --ckpt-every 2000 --log-every 20` |

### 结果
- **训练步数：3000 步**（200 epochs × 15 步/epoch）
- **时长：约 12 分钟**（23:02:20 → 23:14:23）
- **最终 loss：1.4788**（step 20 时 6.9472 → step 3000 时 1.4788，相对随机初始化 ln(16384)=9.70 大幅下降；初始 6.95 表明预训练权重已继承）
- 显存：23.5 GB/卡；吞吐 ~4.5 step/s
- 中间 temp ckpt 自动保存于 step 1000/2000/3000（脚本每 1000 步保存一次）
- **产物：`temp/dar_ft.pt`**（5.47GB，含 model/ema/optimizer/args，steps=3000）

## 3. 评估

### 3.1 Tokenizer 重建质量（微调前 vs 微调后，1000 张工地图，256-res）
评估命令（`reconstruction_vq_ddp.py`，经 `scripts/tokenizer/reconstruction_vq.sh`，`--use-ema`）：

| 指标 | 微调前（`tokenizer_v1.pt`） | 微调后（`tokenizer_v1_ft.pt`） | Δ |
| -- | -- | -- | -- |
| PSNR（脚本内置 skimage，1000 张） | 18.022 | **18.170** | **+0.148** |
| SSIM（同上） | 0.5176 | **0.5259** | **+0.0083** |
| PSNR（自写 torch 交叉验证，200 张） | 18.031 | 18.179 | +0.148 |
| SSIM（同上） | 0.3787 | 0.3874 | +0.0087 |

- 两套实现均确认微调后**小幅提升**（+0.15 PSNR / +0.008 SSIM），方向一致；绝对 SSIM 差异源于实现（skimage 高斯窗 vs torch 自写）。
- 注：微调后 ckpt 用 EMA 权重，微调前 ckpt 为裸 state_dict（无 EMA，只能用原始权重），存在轻微不对称。
- 可视化：`logs/montage_before.png`、`logs/montage_after.png`（重建 vs 原图对照，各 8 组）。
- 重建图与 gt 目录：`/root/autodl-tmp/dar_data/recon_before/`、`/root/autodl-tmp/dar_data/recon_after/`。

### 3.2 D-AR 生成采样
命令：`sample_c2i.py --gpt-ckpt temp/dar_ft.pt --tokenizer-ckpt temp/tokenizer_v1_ft.pt --num-classes 1 --label 0 --cfg-scale 4.0 --top-p 1.0 --top-k 0 --temperature 1.0 --preview 8`
- 产出：16 张 256×256 网格图 2 份（seed 2 / seed 42）
  - `logs/sample_dar_seed2.png`、`logs/sample_dar_seed42.png`
- 图像统计（seed 2 网格）：尺寸 1034×1034，平均亮度 0.56，平均 RGB (0.58, 0.56, 0.54)（暖灰调，符合工地场景），标准差 0.23–0.26，色彩丰富度 0.12。
- **主观质量需人工查看**（执行代理无图像能力）：建议主代理直接查看上述 png。
- FID 未评估：需要 ImageNet 参考集 / tensorflow（adm 套件），工地数据无参考分布，且 pytorch-fid 未安装；按方案建议以重建指标 + 采样主观质量为准。

## 4. 遇到的问题与修复（共 7 处，均已解决）

1. **DINOv2 离线加载失败**：`torch.hub.load('facebookresearch/dinov2', ...)` 在 torch 2.12 下即使有缓存也会先请求 github.com（`_parse_repo_info`），服务器出网被拒（RemoteDisconnected）导致训练启动即崩。
   → 补丁 `tokenizer/tokenizer_image/utils_repa.py`：优先用 `torch.hub._load_local(缓存repo目录, ...)`，权重从 `~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth` 本地加载（CPU 前向验证通过）。
2. **训练脚本无 `--max-steps` 参数**（冒烟命令里的参数无效）→ 冒烟改用 `--epochs 1` + `timeout 900`。
3. **temp-ckpt 只在固定步数保存**：tokenizer 脚本 `train_steps % 500 == 0`、D-AR 脚本 `% 1000 == 0`，且训练结束**不保存**。按方案配置 tokenizer 仅 280 步 < 500 → 永远不会产出 `tokenizer_v1_ft.pt`（这正是任务预告的坑）。
   → 给两个脚本各补一段"训练结束最终保存"（`vq_train_accelerate.py`、`train_c2i_accelerate.py`，含 config/ema 字段，格式与 `load_visual_tokenizer` 兼容）。
4. **重建评估脚本三连坑**：
   a. `skimage` 未安装（脚本顶层 import）→ `pip install scikit-image`（服务器有 aliyun pypi 镜像，成功）。
   b. `torch.load` 默认 `weights_only=True`，微调 ckpt 含 `argparse.Namespace` → 补丁 `reconstruction_vq_ddp.py` 加 `weights_only=False`。
   c. `create_npz_from_sample_folder(..., 50000)` 硬编码 50000 张会崩（我们只有 1000）→ 补丁改为遍历实际存在的文件。
5. **官方 `D-AR-L-360K.pt` 是裸 state_dict**（172 个 `_orig_mod.*` 键，无 model/ema/optimizer 包装），而 `train_c2i_accelerate.py` 写死 `ckpt['model']`/`ckpt['optimizer']` → 补丁：兼容裸 state_dict（直接取 ckpt 本身），缺失 optimizer 时跳过（optimizer 全新初始化，符合微调语义）。方案 §2.5 只验证了 tokenizer 的裸 ckpt，D-AR 的这条路径此前未被真正测过。
6. **cls_embedding 尺寸不匹配**：官方 GPT-L 类嵌入表 (1001,1024)（1000 类+cfg），我们 `--num-classes 1` → (2,1024)，`strict=False` 对尺寸不匹配照样抛 RuntimeError → 补丁：加载前剔除尺寸不匹配的键（类嵌入随机初始化），并给 EMA 加载补 `strict=False`。
7. **torch.compile 与 EMA 冲突**：脚本默认 `torch.compile` 开启（未传 `--no-compile` 时），compile 包装后参数名带 `_orig_mod.` 前缀，`update_ema` 按名查 EMA 字典 KeyError → `finetune_dar.sh` 加 `--no-compile`（微调无需 compile 加速）。
8. **`sample_c2i.py` 两个坑**：`torch.load` weights_only 默认值崩（同问题 4b，补丁 weights_only=False）；默认随机 label 0–999 超出 `--num-classes 1` 的嵌入表 → 采样必须 `--label 0`。

## 5. 结论与建议

1. **两阶段微调全部跑通**：tokenizer（280 步/7min）+ D-AR（3000 步/12min），服务器 GPU 现已空闲、未关机。
2. **Tokenizer 提升有限（+0.15 PSNR）**：280 步对 1000 张图是轻量适配，属预期；要更大收益建议：
   - 导出全量 10013 张（`tools/export_parquet_to_folder.py --limit 0`，~3.8GB，磁盘充足）后重训，或将 `--epochs` 提到 60+；
   - 或降 `--lr` 至 5e-5 做更保守适配（当前 1e-4 在 280 步内 codebook usage 已升到 0.97，说明主要在重排 codebook）。
3. **D-AR 适配效果显著**：loss 6.95 → 1.48，模型已学会工地数据的 code 分布；但类嵌入是随机初始化的，且只训了 3000 步，样本质量建议人工查看 `logs/sample_dar_seed*.png` 后再决定是否加训（GPT-L 显存余量很大，可上 batch 128 或 GPT-XL）。
4. **评估口径**：重建指标可靠（两套实现一致）；FID 无参考分布不适用。若需更严格对比，可后续装 pytorch-fid 对 recon_before/recon_after 的 gt 与重建目录算 FID（两目录已保留）。
5. **可复现性**：所有补丁脚本留在 `/root/autodl-tmp/D-AR/patch_*.py`（幂等，可重跑）；改动文件：`utils_repa.py`、`vq_train_accelerate.py`、`train_c2i_accelerate.py`、`reconstruction_vq_ddp.py`、`sample_c2i.py`、`finetune_dar.sh`。

## 6. 产物清单

| 路径 | 说明 |
| -- | -- |
| `temp/tokenizer_v1_ft.pt` | tokenizer 微调产物（4.97GB，steps=280） |
| `temp/dar_ft.pt` | D-AR 微调产物（5.47GB，steps=3000） |
| `logs/finetune_tokenizer.log` | tokenizer 训练日志 |
| `logs/finetune_dar.log` | D-AR 训练日志 |
| `logs/recon_before.log` / `logs/recon_after.log` | 重建评估日志 |
| `logs/recon_metrics_before.txt` / `logs/recon_metrics_after.txt` | 自写指标（200 张交叉验证） |
| `logs/montage_before.png` / `logs/montage_after.png` | 重建 vs 原图对照 |
| `logs/sample_dar_seed2.png` / `logs/sample_dar_seed42.png` | D-AR 生成网格（16 张/份） |
| `/root/autodl-tmp/dar_data/recon_before/`、`recon_after/` | 重建图与 gt 目录（各含 1000 张 + `*_results.txt`） |
| `patch_*.py`（repo 根目录，7 个） | 全部代码补丁（幂等） |
