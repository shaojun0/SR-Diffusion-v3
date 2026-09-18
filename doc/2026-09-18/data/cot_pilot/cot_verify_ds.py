#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cot_verify_ds.py — 用 DeepSeek（真·视觉）复核 CoT 初稿。

输入：gen/*.jsonl（Qwen 生成）+ 同一张 12:9 图 + 真值 bbox
输出：verify.jsonl（逐条判决，可续跑）
判据（DS 能看图，所以是视觉核验而非文本自洽）：
  - spatial_ok：整幅画面的空间结构描述是否与图相符
  - align_ok  ：每个目标的"落在什么结构/什么方位"是否与图 + 真值坐标相符
  - invented  ：是否描述了图上不存在/真值之外的目标
  - coord_ok  ：系统拼上的坐标是否与描述一致（与真值比对，检查描述与坐标矛盾）
"""
import argparse
import base64
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

KEY_FILE = os.path.expanduser("~/.deepseek_key")
PILOT = "/home/linaro/dsh/judge_work/cot/pilot"   # 本地副本：pilot/images + manifest

SYSTEM = """你是严格的视觉标注质检员。给你：一张已统一到 12:9 画布（1200×900）的图片、该图的真值目标清单
（类别 + 归一化坐标 0–1000）、以及一段由另一个模型生成的"空间解构"思维链。
**你能看到图片**，请逐项核对这段思维链：

1) spatial_ok：它对整幅画面空间结构（有哪些结构/区域/层次、各自方位、遮挡/依附关系）的描述是否与图相符？
2) per_target：它对每个目标的"落在哪个结构、处于什么方位、依据"的描述是否与图 + 真值坐标相符？
3) invented：它是否描述了真值清单之外、图上也不存在（或明显不存在）的目标？
4) coord_ok：把坐标与方位描述对照——坐标位置与方位词是否矛盾（例如说"左下"但坐标在右上）？

以下**不算**问题：用词风格、描述详细程度、没有提到的次要细节、同义表述。
只有**与图片事实不符或与坐标矛盾**才算问题。

只输出一行 JSON（不要 markdown、不要解释）：
{"spatial_ok":true,"align_ok":true,"invented":false,"coord_ok":true,
 "v":"pass","issues":[{"type":"方位错","target":"...","problem":"不超过25字"}]}
v 取值：pass（无实质问题）/ minor（小瑕疵，不影响使用）/ fail（有与图片事实不符的实质错误）。
issues 最多 3 条。"""

_tls = threading.local()


def sess():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        _tls.s = s
    return s


def load_key():
    k = os.environ.get("DEEPSEEK_API_KEY") or open(KEY_FILE, encoding="utf-8").read().strip()
    if not k:
        raise SystemExit("no key")
    return k


def img_data(path):
    p = path if os.path.isabs(path) else os.path.join(PILOT, path)
    with open(p, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def build_user(rec):
    lines = ["真值目标清单："]
    for i, b in enumerate(rec.get("gt_boxes") or [], 1):
        lines.append("%d. %s @ [%s]" % (i, b["label"], ", ".join(str(v) for v in b["bbox1000"])))
    gt = rec.get("gt") or {}
    for k in ("caption", "text", "question", "answer", "label"):
        if gt.get(k):
            lines.append("%s：%s" % (k, gt[k]))
    lines.append("\n待核对的思维链：\n%s" % (rec.get("cot") or "(空)"))
    return "\n".join(lines)


def verify_one(key, rec, model="deepseek-flash", timeout=180, retries=3):
    body = {"model": model, "temperature": 0.0, "max_tokens": 400,
            "reasoning_effort": "none", "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": [
                             {"type": "text", "text": build_user(rec)},
                             {"type": "image_url", "image_url": {"url": img_data(rec["image"])}}]}]}
    last = None
    for a in range(retries):
        try:
            r = sess().post("https://api.deepseek.com/chat/completions", json=body, timeout=timeout,
                            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError("http %d" % r.status_code)
            r.raise_for_status()
            d = r.json()
            txt = d["choices"][0]["message"].get("content") or ""
            m = re.search(r"\{.*\}", txt, flags=re.S)
            obj = json.loads(m.group(0)) if m else None
            if obj is None:
                return {"id": rec["id"], "v": "parse_error", "raw": txt[:200],
                        "usage": d.get("usage") or {}}
            obj["id"] = rec["id"]
            obj["usage"] = d.get("usage") or {}
            return obj
        except Exception as e:
            last = e
            time.sleep(1.5 * (a + 1))
    return {"id": rec["id"], "v": "api_error", "error": str(last)[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    key = load_key()

    recs = []
    for fn in sorted(os.listdir(args.gen_dir)):
        if not fn.endswith(".jsonl"):
            continue
        for line in open(os.path.join(args.gen_dir, fn), encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("cot"):
                    recs.append(r)
    if args.limit:
        recs = recs[:args.limit]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [r for r in recs if r["id"] not in done]
    print("[verify] total=%d todo=%d (done=%d) conc=%d" % (len(recs), len(todo), len(done), args.concurrency), flush=True)

    t0 = time.time()
    counts, tok = {}, 0
    lock = threading.Lock()
    fo = open(args.out, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(verify_one, key, r, args.model) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            o = fut.result()
            with lock:
                fo.write(json.dumps(o, ensure_ascii=False) + "\n")
                counts[o.get("v")] = counts.get(o.get("v"), 0) + 1
                tok += (o.get("usage") or {}).get("total_tokens", 0)
                if i % 50 == 0:
                    print("[verify] %d/%d %.1f/s %s tok=%d" % (i, len(todo), i / (time.time() - t0), counts, tok), flush=True)
                if i % 100 == 0:
                    fo.flush()
    fo.close()
    print("[verify] DONE %s elapsed=%.0fs tok=%d" % (counts, time.time() - t0, tok), flush=True)


if __name__ == "__main__":
    main()
