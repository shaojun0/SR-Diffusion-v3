# K-sweep 原始数据（BPTT，2026-09-16）

> **本目录是 `REPORT_ksweep_*.md` 的原始数据。**
> 逐 run 的 `infer_test.json`（test 集全量推理结果）、`args.json`（训练参数，固定口径）与 `model_info.json`（模型规模信息）原始拷贝，未做任何加工。

- 来源根目录（服务器）：`/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/`
- 每个 run 的源目录：`<来源根目录>/ksweep_<tag>/`
- tag → `num_specials`(K) 期望：k15→15, k24→24, k35c→35, k48→48, k63→63, k99→99, k120→120（7/7 已校验一致）
- `args.json` 中 `num_specials` 记录为 `0`，表示由 `slice_start:slice_end` 自动推导；实际 K 由 `infer_test.json` 的 `num_specials` 为准。

## 1. 逐文件索引（`infer_test.json`）

| tag | 文件 | K (num_specials) | slice | decoder_steps | full_norm_l1 | full_pixel_l1_255 | 字节 | md5 |
|---|---|---|---|---|---|---|---|---|
| k15 | `ksweep_k15__infer_test.json` | 15 | 0:3 | `[1, 4, 9]` | 0.26115881206192443 | 15.024677545824318 | 626 | `4d5d343396ae291b2c28735eca2ce53a` |
| k24 | `ksweep_k24__infer_test.json` | 24 | 0:4 | `[1, 4, 9, 16]` | 0.2609450129789614 | 15.012149348557392 | 683 | `a46aff110335c98cf5679107059d9fb8` |
| k35c | `ksweep_k35c__infer_test.json` | 35 | 0:5 | `[1, 4, 9, 16, 25]` | 0.24602130773699235 | 14.147014562045527 | 735 | `6b83d94c9b4931d436883d589aaf4b24` |
| k48 | `ksweep_k48__infer_test.json` | 48 | 0:6 | `[1, 4, 9, 16, 25, 36]` | 0.22872075577391765 | 13.148314586492098 | 788 | `179197345ec34e22cebe7c469287694a` |
| k63 | `ksweep_k63__infer_test.json` | 63 | 0:7 | `[1, 4, 9, 16, 25, 36, 49]` | 0.24448260391123602 | 14.058703243494351 | 844 | `369348c4078e9a683bf64a72d0d287d7` |
| k99 | `ksweep_k99__infer_test.json` | 99 | 0:9 | `[1, 4, 9, 16, 25, 36, 49, 64, 81]` | 0.2240712031543493 | 12.881244918477837 | 954 | `f2dc7e389a77d27488d59811ec92590d` |
| k120 | `ksweep_k120__infer_test.json` | 120 | 0:10 | `[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]` | 0.2174935417073703 | 12.509486415573507 | 1012 | `2559926227cdfc1e1828441942c085bb` |

## 2. `step_pixel_l1_255`（逐步）

| tag | step_pixel_l1_255 |
|---|---|
| k15 | 15.062194753105885, 15.019506675425605, 15.024677545824318 |
| k24 | 15.097978482709902, 15.017992895865408, 15.003671221980719, 15.012149348557392 |
| k35c | 14.394131585538943, 14.207148108755383, 14.150134334234043, 14.137433231115024, 14.147014562045527 |
| k48 | 14.274197930184883, 13.529952098780402, 13.23409252954069, 13.150354662208201, 13.13462652823579, 13.148314586492098 |
| k63 | 14.221299406374184, 14.096344817018064, 14.058650589497207, 14.043270522522704, 14.04020059029367, 14.045122667254208, 14.058703243494351 |
| k99 | 13.813297124423295, 13.217613911025534, 13.006751343667428, 12.920666660354554, 12.88196201045091, 12.865058035412419, 12.862511250055265, 12.86695471711546, 12.881244918477837 |
| k120 | 13.850296050984754, 12.88229902447778, 12.602767954494919, 12.524335005311928, 12.49780271405704, 12.489368565707963, 12.487685954046947, 12.490945727148958, 12.498365277774166, 12.509486415573507 |

## 3. 训练参数与模型信息（固定口径）

| tag | args 文件 | args md5 | args 字节 | model_info 文件 | model_info md5 | model_info 字节 |
|---|---|---|---|---|---|---|
| k15 | `ksweep_k15__args.json` | `d76ef1ca57b5dc23ab81bc15bf8d0d79` | 729 | `ksweep_k15__model_info.json` | `fe06a2a9e51402837e52db506e5f3b62` | 586 |
| k24 | `ksweep_k24__args.json` | `8443c1f85cca2a2499fe6fd6d7007297` | 729 | `ksweep_k24__model_info.json` | `1585b3d8221ac19e7c59ffed5b13e4a4` | 594 |
| k35c | `ksweep_k35c__args.json` | `3456f3ee50dc1a538d88a0bcceafc31c` | 730 | `ksweep_k35c__model_info.json` | `0fbbea34c01de5b5afc773bee47bca92` | 602 |
| k48 | `ksweep_k48__args.json` | `5ad6c9460f7d50e65d3df160349d1e25` | 729 | `ksweep_k48__model_info.json` | `154997ebe386fc8c53ec55ea798b1477` | 610 |
| k63 | `ksweep_k63__args.json` | `2c0b3cc0161b61b91ed308439c13274d` | 729 | `ksweep_k63__model_info.json` | `667e64c5a0380e12080e3dd6665f9c9d` | 618 |
| k99 | `ksweep_k99__args.json` | `b0a4272637f8e1a636b0ad36084ca208` | 729 | `ksweep_k99__model_info.json` | `e491c8b04ce58d68d2017bcc879b9d69` | 634 |
| k120 | `ksweep_k120__args.json` | `33f063a316f5eea662b84515dceef4d3` | 731 | `ksweep_k120__model_info.json` | `a5e3c329eb64570f8cf401a9878be52b` | 645 |

## 4. 来源路径（逐 run）

| tag | infer_test.json 源路径 | args.json 源路径 | model_info.json 源路径 |
|---|---|---|---|
| k15 | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k15/infer_test.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k15/args.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k15/model_info.json` |
| k24 | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k24/infer_test.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k24/args.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k24/model_info.json` |
| k35c | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k35c/infer_test.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k35c/args.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k35c/model_info.json` |
| k48 | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k48/infer_test.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k48/args.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k48/model_info.json` |
| k63 | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k63/infer_test.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k63/args.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k63/model_info.json` |
| k99 | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k99/infer_test.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k99/args.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k99/model_info.json` |
| k120 | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k120/infer_test.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k120/args.json` | `/root/autodl-tmp/sr-diffusion-v3-bptt-ksweep/output/ksweep_k120/model_info.json` |

## 5. 校验记录

- 全部 7 个 `infer_test.json` 均已用 Python `json.load` 成功解析（UTF-8）。
- `num_specials` 与 tag 期望值 7/7 一致：15 / 24 / 24→24 / 35 / 48 / 63 / 99 / 120（详见表 1）。
- 权重文件（`final_model.pt` / `checkpoint-*`，各约 1.3 GB）**未入库**，仅归档小 JSON。
- 未修改服务器上任何数据。
