#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""image_format_probe.py — 直连 transformers 验证 Qwen3.8-27B 能接受哪种图片传参。"""
import base64
import io
import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor

MODEL = "/root/autodl-tmp/models/Qwen3.8-27B"
IMG = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/cot_pilot/iluvvatar__wood_surface_defects/images/0000.jpg"

Q = "这张图里是什么？用一句话回答，只描述你真正看到的东西。"
print("图片:", IMG, Image.open(IMG).size, flush=True)

proc = AutoProcessor.from_pretrained(MODEL)
try:
    from transformers import AutoModelForImageTextToText as Cls
    model = Cls.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
except Exception as e:
    print("fallback:", str(e)[:80], flush=True)
    from transformers import Qwen3_5ForConditionalGeneration as Cls
    model = Cls.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
model.eval()
dt = base64.b64encode(open(IMG, "rb").read()).decode()
pil = Image.open(IMG).convert("RGB")
CK = {"enable_thinking": False, "preserve_thinking": False}

variants = {
    "image_url + data URL": [{"role": "user", "content": [
        {"type": "text", "text": Q},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + dt}}]}],

    "image_type + data URL": [{"role": "user", "content": [
        {"type": "text", "text": Q},
        {"type": "image", "image": "data:image/jpeg;base64," + dt}]}],

    "image_type + 本地路径": [{"role": "user", "content": [
        {"type": "text", "text": Q},
        {"type": "image", "image": IMG}]}],

    "image_type + PIL": [{"role": "user", "content": [
        {"type": "text", "text": Q},
        {"type": "image", "image": pil}]}],
}

for name, msgs in variants.items():
    try:
        enc = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt", **CK)
        has_px = "pixel_values" in enc and enc["pixel_values"] is not None
        with torch.inference_mode():
            out = model.generate(**{k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in enc.items()},
                                 max_new_tokens=80, do_sample=False)
        txt = proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        print("[%s] pixel_values=%s -> %s" % (name, has_px, txt[:160].replace("\n", " ")), flush=True)
    except Exception as e:
        print("[%s] FAILED: %s: %s" % (name, type(e).__name__, str(e)[:200]), flush=True)
