"""
SR-Diffusion Phase 1 v2 — 训练（v2.1: 可选文字自回归）
=================================================
架构（model_v2.py）: DINOv2-large(参数不冻结) → ReEncoder → FeatureDecoder,
    L1 重建 DINO patch 特征；可选 TextDecoder（vocab_size>0）: 图像条件
    z_cls+z_s + 文字 → 错位 CE，多任务 loss = L1 + text_loss_weight*CE。

两种模式:
    纯重建:  vocab_size=0（默认，不加载 tokenizer）
    多任务:  --text_decoder + --qwen_dir —— 文字目标 = construction_site
             中文 caption（+隐患），Qwen tokenizer 编码；文字 embedding
             从 Qwen 预训练权重 warm-start（默认冻结，--unfreeze_text_embed 解冻）。

数据（data_v2.py）: 原图 → 旋转(最优角) → 等比缩放 → 居中填充 1600:900
    (16:9) 画布 → 16:9 模型输入 (448×252)。轮廓不变形、内容面积最大化。

解耦 / 不造轮子:
    分布式/检查点 → accelerate；调度 → transformers get_scheduler；
    优化器 → torch AdamW；分词 → transformers AutoTokenizer；
    数据 → datasets + DataLoader + 自定义 collate。

多卡: accelerate launch --multi_gpu --num_processes N
（bf16 在代码内显式 torch.autocast: 规避 torch 2.13 混合设备 conv
  dtype bug 与 accelerate prepare 自动包装; 权重保持 fp32）

用法:
    纯重建:
    accelerate launch --multi_gpu --num_processes 2 \
        train_v2.py --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --output_dir output/phase1_v2

    多任务(文字):
    accelerate launch --multi_gpu --num_processes 2 \
        train_v2.py --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --qwen_dir /root/autodl-tmp/models/Qwen3.8-27B --text_decoder \
        --output_dir output/phase1_v2_text
"""
import argparse
import glob
import json
import os
import time
from contextlib import nullcontext

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Dinov2Model, AutoConfig, AutoTokenizer, get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed

from data_v2 import ParquetImageDataset, V2Collator
from model_v2 import SRPhase1V2


# ═══════════════════════════════════════════════════════════════
# Qwen 辅助（不加载全模型）
# ═══════════════════════════════════════════════════════════════

def qwen_text_config(qwen_dir: str):
    """多模态 Qwen（如 Qwen3.8-27B）的 LM 配置在 text_config 下。"""
    cfg = AutoConfig.from_pretrained(qwen_dir)
    return getattr(cfg, "text_config", None) or cfg


def load_qwen_embedding(qwen_dir: str) -> torch.Tensor:
    """从 safetensors 分片读取 text embedding 权重（不加载全模型）。

    兼容单模态(model.embed_tokens.weight)与多模态
    (model.language_model.embed_tokens.weight)两种命名。
    """
    from safetensors import safe_open
    idx = json.load(open(os.path.join(qwen_dir, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    key = ("model.embed_tokens.weight"
           if "model.embed_tokens.weight" in wm
           else "model.language_model.embed_tokens.weight")
    shard = wm[key]
    with safe_open(os.path.join(qwen_dir, shard), framework="pt") as f:
        return f.get_tensor(key)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="train phase1 v2 (DINOv2-large unfrozen)")
    p.add_argument("--data_dir", required=True, help="parquet 目录(train-*.parquet)")
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--output_dir", default="output/phase1_v2")
    p.add_argument("--model_input", default="448x252", help="16:9 模型输入 WxH (14 的倍数)")
    p.add_argument("--canvas", default="1600x900", help="旋转+缩放+填充目标画布")
    p.add_argument("--angle_step", type=float, default=0.5, help="最优旋转角网格步长(度)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=0, help="0 = epochs 决定")
    p.add_argument("--batch_size", type=int, default=8, help="每卡 batch")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--reencoder_depth", type=int, default=4)
    p.add_argument("--decoder_depth", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本(调试)")
    p.add_argument("--eval_limit", type=int, default=0, help="eval 只用前 N 条(冒烟)")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="accelerate checkpoint 目录")
    p.add_argument("--smoke", action="store_true", help="3 步冒烟, 不存盘")
    # ── 文字自回归（可选）──
    p.add_argument("--text_decoder", action="store_true", help="启用 TextDecoder 多任务")
    p.add_argument("--qwen_dir", default="models/qwen3.5-4B",
                   help="Qwen 模型目录(tokenizer+embedding 权重)")
    p.add_argument("--max_text_len", type=int, default=256)
    p.add_argument("--text_loss_weight", type=float, default=1.0)
    p.add_argument("--text_decoder_depth", type=int, default=4)
    p.add_argument("--unfreeze_text_embed", action="store_true",
                   help="默认冻结 Qwen 词表 embedding")
    p.add_argument("--text_template", default="描述这张建筑工地图片：{caption}")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# Eval — 仅主进程, 全量 test 分片
# ═══════════════════════════════════════════════════════════════

def evaluate(acc: Accelerator, model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    sums = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(acc.device)   # eval loader 未 prepare, 显式搬设备
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, text_ids=batch.get("text_ids"))
            bs = x.shape[0]
            for k in ("loss", "recon", "text_loss"):
                if k in out:
                    sums[k] = sums.get(k, 0.0) + out[k].item() * bs
            n += bs
    line = ", ".join(f"{k} {v / max(n, 1):.6f}" for k, v in sums.items())
    acc.print(f"[eval] {line}  (n={n})")
    return sums.get("loss", 0.0) / max(n, 1)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    W, H = (int(v) for v in args.model_input.lower().split("x"))
    CW, CH = (int(v) for v in args.canvas.lower().split("x"))
    assert W % 14 == 0 and H % 14 == 0, "DINOv2-large patch=14, 输入须为 14 的倍数"
    num_patches = (W // 14) * (H // 14)

    # mixed_precision="no" + 显式 torch.autocast(bf16)：本环境（torch 2.13）
    # 在「输入在 CPU、模型在 CUDA」且开 autocast 时 conv 会报 dtype 错，
    # 显式 .to(acc.device) + 显式 autocast 规避；权重保持 fp32（bf16 只算不存）。
    acc = Accelerator(mixed_precision="no", step_scheduler_with_optimizer=False)

    # ── 文字模式: tokenizer + pad id ──
    tokenizer = pad_id = None
    if args.text_decoder:
        tokenizer = AutoTokenizer.from_pretrained(args.qwen_dir)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id          # Qwen 系 pad 常为空, 退化为 eos
        acc.print(f"[text] tokenizer {args.qwen_dir} pad_id={pad_id}")

    # ── 数据 ──
    train_files = sorted(glob.glob(os.path.join(args.data_dir, "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    assert train_files, f"无 train-*.parquet in {args.data_dir}"
    coll = V2Collator(model_size=(W, H), canvas=(CW, CH), angle_step=args.angle_step,
                      tokenizer=tokenizer, max_text_len=args.max_text_len,
                      pad_token_id=pad_id or 0, text_template=args.text_template)
    train_ds = ParquetImageDataset(train_files, limit=args.limit)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=coll,
                              drop_last=True, pin_memory=True)
    test_loader = None
    if test_files:
        test_ds = ParquetImageDataset(test_files, limit=args.eval_limit)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=coll,
                                 drop_last=False)
    else:
        acc.print("[warn] 无 test-*.parquet, 跳过 eval")

    # ── 模型: DINOv2-large 不冻结; 可选 TextDecoder ──
    dino = Dinov2Model.from_pretrained(args.dino_dir)
    # 权重带 mask_token(use_mask_token=True) 但本任务不传 bool_masked_pos,
    # 该参数从不参与前向 → DDP 报"未用参数"。移除并关掉 flag。
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token

    text_kw = {}
    if args.text_decoder:
        tcfg = qwen_text_config(args.qwen_dir)
        vocab_size = tcfg.vocab_size
        qwen_hidden = tcfg.hidden_size
        assert tokenizer.vocab_size == vocab_size, \
            f"tokenizer vocab {tokenizer.vocab_size} != config vocab {vocab_size}"
        text_kw = dict(vocab_size=vocab_size, qwen_hidden=qwen_hidden,
                       text_decoder_depth=args.text_decoder_depth,
                       max_text_len=args.max_text_len,
                       freeze_text_embed=not args.unfreeze_text_embed,
                       pad_token_id=pad_id, text_loss_weight=args.text_loss_weight)
        acc.print(f"[text] vocab={vocab_size} qwen_hidden={qwen_hidden} "
                  f"freeze_embed={not args.unfreeze_text_embed}")

    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size,
                       reencoder_depth=args.reencoder_depth,
                       decoder_depth=args.decoder_depth, **text_kw)
    model.init_reencoder_from_dino(args.reencoder_depth)
    if args.text_decoder:
        emb = load_qwen_embedding(args.qwen_dir)
        model.init_text_from_qwen(emb)               # embedding warm-start(默认冻结)

    # ── 优化: 只收可训练参数（冻结的 Qwen 词表 1.27B 不进 AdamW）──
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    acc.print(f"[model] 可训练参数 {n_train / 1e6:.1f}M (含 DINOv2-large, 不冻结); "
              f"输入 {W}x{H}, patches={num_patches}")

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    model, optimizer, train_loader = acc.prepare(model, optimizer, train_loader)

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    scheduler = get_scheduler("cosine", optimizer=optimizer,
                              num_warmup_steps=int(total_steps * args.warmup_ratio),
                              num_training_steps=total_steps)
    acc.print(f"[train] {len(train_ds)} 样本 | 每卡 bs={args.batch_size} "
              f"x {acc.num_processes} 卡 | grad_accum={args.grad_accum} | "
              f"~{steps_per_epoch} 步/epoch x {args.epochs} = {total_steps} 步")

    global_step = 0
    if args.resume:
        acc.load_state(args.resume)
        global_step = int(os.path.basename(args.resume).rsplit("-", 1)[-1])
        acc.print(f"[resume] 从 {args.resume} 恢复, 续训于 step {global_step}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ── 训练循环（手动梯度累积: 显式、无插件魔法）──
    ema_loss, micro, t0 = None, 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        if global_step >= total_steps:
            break
        for batch in train_loader:
            x = batch["pixel_values"].to(acc.device)
            text_ids = batch.get("text_ids")          # None = 纯重建
            ctx = acc.no_sync(model) if micro % args.grad_accum != 0 else nullcontext()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x, text_ids=text_ids)["loss"] / args.grad_accum
                acc.backward(loss)

            if micro % args.grad_accum == args.grad_accum - 1:
                acc.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                lv = loss.item() * args.grad_accum
                ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv

                if global_step % args.log_every == 0:
                    el = time.time() - t0
                    sps = global_step * args.batch_size * acc.num_processes / max(el, 1e-9)
                    acc.print(
                        f"[step {global_step}/{total_steps}] epoch {epoch + 1} | "
                        f"loss {ema_loss:.5f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                        f"{sps:.1f} 样本/s | 已用 {el / 60:.1f}min", flush=True)

                if test_loader is not None and global_step % args.eval_every == 0:
                    if acc.is_main_process:
                        evaluate(acc, model, test_loader)
                    model.train()

                if not args.smoke and global_step % args.save_every == 0:
                    acc.wait_for_everyone()
                    ckpt = os.path.join(args.output_dir, f"ckpt-{global_step}")
                    acc.save_state(ckpt)
                    acc.print(f"[save] {ckpt}", flush=True)

                if global_step >= total_steps:
                    break
            micro += 1

    # ── 收尾: 仅主进程导出推理权重 ──
    acc.wait_for_everyone()
    if acc.is_main_process:
        sd = acc.unwrap_model(model).state_dict()
        # 必须 fp32: 本模型重建精度极高(L1~0.001), bf16 导出会因权重量化
        # 使 L1 劣化约 2 倍(实测 0.0011 → 0.0021)。不省这个空间。
        final = os.path.join(args.output_dir, "final_model.pt")
        torch.save(sd, final)
        info = {"input_size": [W, H], "canvas": [CW, CH],
                "num_patches": num_patches, "dim": dino.config.hidden_size,
                "reencoder_depth": args.reencoder_depth,
                "decoder_depth": args.decoder_depth,
                "dino_dir": args.dino_dir,
                "text_decoder": bool(args.text_decoder),
                "qwen_dir": args.qwen_dir if args.text_decoder else None,
                "pad_token_id": pad_id if args.text_decoder else None,
                "dtype": "fp32"}
        with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        acc.print(f"[final] {final} 已保存 (fp32, 含 DINO 权重)")


if __name__ == "__main__":
    main()
