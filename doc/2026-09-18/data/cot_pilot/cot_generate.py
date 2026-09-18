#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cot_generate.py — 用 Qwen3.8-27B(VL) 为试点图生成思维链初稿。

输入：/root/autodl-tmp/cot_pilot/manifest.jsonl（含 12:9 图路径、真值 bbox/label）
输出：/root/autodl-tmp/cot_pilot/gen/<dataset>.jsonl（可续跑：已完成的 id 跳过）
只读图片，不写回任何原始数据。
"""
import argparse
import base64
import glob
import json
import os
import re
import sys
import time

import requests

PILOT = os.environ.get("COT_ROOT", "/root/autodl-tmp/cot_pilot")

SYS_DETECT = """你是施工现场/工业场景的视觉标注专家。输入是一张已经统一到 12:9 画布（1200×900）的图片，
以及该图的真值目标清单（类别 + 画布归一化坐标 0–1000，[x1,y1,x2,y2]）。
坐标是给你**定位目标**用的：#第 i 个目标就位于第 i 组坐标处。#坐标由系统保存，**你不需要输出坐标**。

请写"空间解构"的思维链，解释这些目标为什么在那个位置：
1) spatial：先解构整幅画面的空间结构——有哪些主要结构/区域/层次，分别在什么方位（用 左/右/上/下/中央/前景/背景 等词），以及结构之间的遮挡、依附、包含关系。
2) objects：按真值清单的**顺序**逐个说明该目标。**先用该目标的坐标在图上找到那个位置**，再说明：那里是什么结构/区域、该目标处于这个结构的什么方位、判断依据是什么（相对其它物体的位置、所依托的结构、被什么遮挡）。同一个类别出现多个目标时，务必靠坐标区分它们，不要把它们的方位混淆。

要求：
- objects 数组**长度必须等于真值目标个数，顺序必须一致**；每项不超过 45 字。
- 只描述真值清单里的目标，不要臆造新目标。
- **不要输出任何坐标数字**（坐标由系统拼接）。
- 不要重复同一句话；输出必须简洁。

只输出一行 JSON（不要 markdown、不要解释）：
{"spatial":"整幅画面的空间结构解构","objects":["目标1的方位与依据","目标2的方位与依据"]}"""

SYS_CAPTION = """你是施工现场视觉专家。输入是一张统一到 12:9 画布（1200×900）的图片和该图的真值描述。
请解释"为什么由图可以得到这句描述"：
1) 先解构画面空间：主要结构/区域/层次各在什么方位；
2) 再给出支持该描述的关键视觉证据（人物、机械、材料、动作、方位关系）；
3) 最后复述真值描述。
输出一行 JSON（不要 markdown）：{"spatial":"...","evidence":"...","cot":"连贯推理","answer":"真值描述"}"""

SYS_VQA = """你是建筑缺陷视觉问答专家。输入是一张统一到 12:9 画布（1200×900）的图片、问题与真值答案。
请解释"为什么由图可以得到这个答案"：
1) 先解构画面空间与结构；
2) 指出与问题相关的视觉证据（缺陷类型、位置、形态、范围）；
3) 最后复述真值答案。
输出一行 JSON（不要 markdown）：{"spatial":"...","evidence":"...","cot":"连贯推理","answer":"真值答案"}"""

SYS_CLS = """你是野火建筑损毁评估专家。输入是一张统一到 12:9 画布（1200×900）的图片与真值类别。
请解释"为什么由图可以得到这个类别"：
1) 先解构画面空间（建筑主体、周边环境、损毁区域各在什么方位）；
2) 指出支持该类别的关键视觉证据；
3) 最后复述真值类别。
输出一行 JSON（不要 markdown）：{"spatial":"...","evidence":"...","cot":"连贯推理","answer":"真值类别"}"""

SYS = {"detect": SYS_DETECT, "caption": SYS_CAPTION, "vqa": SYS_VQA, "classify": SYS_CLS}


def user_text(rec):
    t = rec["task"]
    if t == "detect":
        lines = ["真值目标清单（第 i 项位于第 i 组坐标处；坐标仅供定位，不要在输出里写坐标）："]
        for i, b in enumerate(rec["boxes"], 1):
            x1, y1, x2, y2 = b["bbox1000"]
            lines.append("%d. %s @ [%s, %s, %s, %s]" % (i, b["label"], x1, y1, x2, y2))
        cap = (rec.get("gt") or {}).get("caption")
        if cap:
            lines.append("\n该图原始描述：%s" % cap)
        lines.append("\n请输出 JSON：{\"spatial\":\"...\",\"objects\":[ %d 项 ]}" % len(rec["boxes"]))
        return "\n".join(lines)
    g = rec.get("gt") or {}
    if t == "caption":
        return "真值描述：%s\n\n请按系统要求输出 JSON。" % g.get("text", "")
    if t == "vqa":
        return "问题：%s\n真值答案：%s\n\n请按系统要求输出 JSON。" % (g.get("question", ""), g.get("answer", ""))
    return "真值类别：%s\n\n请按系统要求输出 JSON。" % g.get("label", "")


def strip_think(s):
    s = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", s or "", flags=re.S | re.I)
    return re.sub(r"</?think\b[^>]*>", "", s, flags=re.I)


def _repair_json(t):
    """修补常见生成瑕疵：键缺左引号、尾逗号、中文引号包裹。"""
    t = re.sub(r'(?<=[{,\s])([A-Za-z_][A-Za-z0-9_]*)"\s*:', r'"\1":', t)   # evidence":  -> "evidence":
    t = re.sub(r",\s*([}\]])", r"\1", t)                                     # 尾逗号
    t = t.replace("“", '"').replace("”", '"')
    return t


def _salvage_json(t):
    """JSON 整体不可解析时，按固定 schema 逐字段抽取。"""
    out = {}
    for key in ("spatial", "evidence", "cot", "answer"):
        m = re.search(r'"?%s"?\s*:\s*"(.*?)(?<!\\)"' % key, t, flags=re.S)
        if m:
            out[key] = m.group(1).replace("\\n", "\n").strip()
    mo = re.search(r'"objects"\s*:\s*\[(.*?)\]', t, flags=re.S)
    if mo:
        out["objects"] = [x.replace("\\n", "\n").strip()
                          for x in re.findall(r'"(.*?)(?<!\\)"', mo.group(1), flags=re.S)]
    return out or None


def parse_json(raw):
    s = strip_think(raw).replace("```json", " ").replace("```", " ").strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    t = m.group(0)
    for cand in (t, _repair_json(t)):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return _salvage_json(t)


def b64_image(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def call_batch(base_url, model, recs, max_tokens, timeout=600):
    prompts = []
    for r in recs:
        prompts.append([
            {"role": "system", "content": SYS[r["task"]]},
            {"role": "user", "content": [
                {"type": "text", "text": user_text(r)},
                {"type": "image_url", "image_url": {"url": b64_image(os.path.join(PILOT, r["image"]))}},
            ]},
        ])
    body = {"model": model, "prompts": prompts, "temperature": 0.0, "top_p": 1.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}}
    r = requests.post(base_url.rstrip("/") + "/chat/completions", json=body, timeout=timeout,
                      headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"})
    r.raise_for_status()
    d = r.json()
    out = [None] * len(recs)
    for pos, ch in enumerate(d.get("choices") or []):
        idx = ch.get("index", pos)
        if 0 <= idx < len(recs):
            txt = ch["message"]["content"]
            if isinstance(txt, list):
                txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
            out[idx] = strip_think(txt).strip()
    missing = [i for i, v in enumerate(out) if v is None]
    if missing:
        raise RuntimeError("missing choices %s" % missing[:3])
    usage = d.get("usage") or {}
    return out, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="逗号分隔的数据集名（前缀匹配）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(PILOT, "gen"))
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    want = [d for d in args.datasets.split(",") if d]
    mfs = sorted(glob.glob(os.path.join(PILOT, "manifest*.jsonl")))
    recs = []
    _seen = set()
    for mf in mfs:
        for l in open(mf, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if r["id"] not in _seen:
                _seen.add(r["id"]); recs.append(r)
    recs = [r for r in recs if any(r["dataset"].startswith(w) for w in want)]
    if args.limit:
        recs = recs[:args.limit]

    by_ds = {}
    for r in recs:
        by_ds.setdefault(r["dataset"], []).append(r)

    t0 = time.time()
    for ds, items in by_ds.items():
        out_path = os.path.join(args.out_dir, ds + ".jsonl")
        done = set()
        if os.path.exists(out_path):
            for line in open(out_path, encoding="utf-8"):
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
        todo = [r for r in items if r["id"] not in done]
        print("[gen] %s todo=%d (done=%d) bs=%d" % (ds, len(todo), len(done), args.batch_size), flush=True)
        fo = open(out_path, "a", encoding="utf-8")
        n_ok = n_bad = 0
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            raw, usage, err = None, {}, None
            for a in range(args.retries):
                try:
                    raw, usage = call_batch(args.base_url, args.model, chunk, args.max_tokens)
                    err = None
                    break
                except Exception as e:
                    err = str(e)[:160]
                    time.sleep(2 + 3 * a)
            if err:
                print("[gen]   batch %d FAILED: %s" % (i, err), flush=True)
            for k, r in enumerate(chunk):
                txt = raw[k] if raw else None
                obj = parse_json(txt) if txt else None
                cot, aligned, note = None, None, ""
                if obj is not None and r["task"] == "detect":
                    positions = obj.get("objects") or obj.get("object") or []
                    if not isinstance(positions, list):
                        positions = [str(positions)]
                    gtb = r["boxes"]
                    aligned = (len(positions) == len(gtb))
                    if aligned:
                        parts = [str(obj.get("spatial", "")).strip()]
                        for b, pos in zip(gtb, positions):
                            if isinstance(pos, dict):
                                pos = pos.get("position") or pos.get("desc") or json.dumps(pos, ensure_ascii=False)
                            x1, y1, x2, y2 = b["bbox1000"]
                            parts.append("%s：%s → 坐标 [%s, %s, %s, %s]"
                                         % (b["label"], str(pos).strip(), x1, y1, x2, y2))
                        cot = "\n".join(p for p in parts if p)
                    else:
                        note = "objects 数 %d != 真值 %d" % (len(positions), len(gtb))
                elif obj is not None:
                    cot = str(obj.get("cot") or "").strip() or None
                out = {"id": r["id"], "dataset": ds, "task": r["task"], "row_id": r["row_id"],
                       "image": r["image"], "rotated": r["rotated"],
                       "gt": r.get("gt"), "gt_boxes": r["boxes"],
                       "gen": obj, "cot": cot, "aligned": aligned, "align_note": note,
                       "parse_ok": obj is not None,
                       "raw": (txt or "")[:1500], "error": err or ""}
                fo.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_ok += obj is not None
                n_bad += obj is None
            fo.flush()
        fo.close()
        print("[gen] %s DONE parse_ok=%d parse_fail=%d elapsed=%.0fs" % (ds, n_ok, n_bad, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
