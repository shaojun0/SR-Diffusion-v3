# A/B: arm1 stepid deep8 (2048/16/8 + step_embed) vs 基线 blockdiag slice[0:5]

- 新模型: `output/phase1_v2_stepid_d8_slice05/infer_test.json`
- 基线:   `/root/autodl-tmp/sr-diffusion-v3-stack2x/output/phase1_v2_stack8x_slice05/infer_test.json`
- test 全量: n=3004 (基线 n=3004)，N=576, K=35, steps=[1, 4, 9, 16, 25]

## 1. 全量重建（同一推理脚本/口径）

| 指标 | 基线 blockdiag slice[0:5] | 本次 arm1 stepid deep8 (2048/16/8 + step_embed) | 变化 |
|---|---|---|---|
| 归一化空间 L1 (eval_recon 口径) | 0.4311 | 0.4051 | -0.0260 (-6.03%) |
| 0-255 像素 L1 | 24.65 | 23.14 | -1.5062 (-6.11%) |

## 2. 渐进曲线（每采样步累积结果, 0-255 像素 L1）

| 采样步 t | 基线 | arm1 stepid deep8 (2048/16/8 + step_embed) | 变化 |
|---|---|---|---|
| 1 | 24.65 | 23.15 | -1.50 |
| 4 | 24.65 | 23.14 | -1.50 |
| 9 | 24.64 | 23.14 | -1.50 |
| 16 | 24.64 | 23.14 | -1.51 |
| 25 | 24.65 | 23.14 | -1.51 |

- 基线 step1→末步 落差: +0.01
- 本次 step1→末步 落差: +0.02

## 3. 训练中 eval_recon 曲线

| checkpoint step | 基线 | arm1 stepid deep8 (2048/16/8 + step_embed) |
|---|---|---|
| 2000 | 0.5894 | 0.5074 |
| 3999 | 0.4886 | 0.4531 |
| 6001 | 0.4428 | 0.4231 |
| 8000 | 0.4321 | 0.4056 |
| 8760 | 0.4310 | 0.4050 |

