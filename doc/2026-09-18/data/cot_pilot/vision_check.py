#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vision_check.py — 修复权重后确认 Qwen3.8-27B 真的能看见图。"""
import io
import struct
import sys
import zlib

import torch
from PIL import Image
from transformers import AutoProcessor

MODEL = "/root/autodl-tmp/models/Qwen3.8-27B"


def png_solid(rgb, w=64, h=64):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def C(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + C(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + C(b"IDAT", zlib.compress(raw)) + C(b"IEND", b""))


proc = AutoProcessor.from_pretrained(MODEL)
try:
    from transformers import AutoModelForImageTextToText as Cls
    model = Cls.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
except Exception:
    from transformers import Qwen3_5ForConditionalGeneration as Cls
    model = Cls.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
model.eval()

vis = model.get_submodule("model.visual")
p = next(vis.parameters())
print("视觉塔首个参数 std=%.5f (应 > 0)" % p.float().std().item(), flush=True)

tests = [
    ("纯红图", Image.open(io.BytesIO(png_solid((220, 20, 20)))).convert("RGB"), "这张图是什么颜色？只答颜色词。"),
    ("木纹实拍", Image.open("/root/autodl-tmp/cot_pilot/iluvvatar__wood_surface_defects/images/0000.jpg").convert("RGB"),
     "这张图里是什么东西？一句话。"),
    ("安全帽实拍", Image.open("/root/autodl-tmp/cot_pilot/hf-vision__hardhat/images/0000.jpg").convert("RGB"),
     "这张图里有什么？一句话。"),
]
for name, im, q in tests:
    msgs = [{"role": "user", "content": [{"type": "text", "text": q}, {"type": "image", "image": im}]}]
    enc = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt", enable_thinking=False, preserve_thinking=False)
    with torch.inference_mode():
        out = model.generate(**{k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in enc.items()},
                             max_new_tokens=60, do_sample=False)
    txt = proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    print("[%s] %s" % (name, txt[:150].replace("\n", " ")), flush=True)
