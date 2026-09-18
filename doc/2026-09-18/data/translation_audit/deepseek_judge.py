#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deepseek_judge.py — 用 DeepSeek deepseek-flash 对译文字段做逐条质检。

特点：并发 + 断点续跑（按 id 去重）+ JSON 输出模式 + 重试退避。
只读输入、只写自己的产物，不修改任何译文。
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

KEY_FILE = os.path.expanduser("~/.deepseek_key")
ALLOWED_V = ("ok", "minor", "major")
ALLOWED_E = ("num", "neg", "term", "omit", "extra", "garble", "fluency", "other")

SYS = """你是中英翻译质检员。给定一条英文原文和它的中文译文，判断译文有没有**会改变含义的硬错误**。

硬错误只有这六类：
- num 数字或数量与原文不符
- neg 否定丢失或方向反转
- term 专业名词明显译错
- omit 原文的整句或整个主体在译文中缺失
- extra 译文出现原文没有的主体或事件
- garble 译文出现重复字、断句错乱、不成词的字串

以下都**不算**错误：用词不同、语序调整、同义替换、句式重组、术语风格不统一、可以更通顺。
**绝大多数译文没有硬错误。** 漏报可以接受，误报不可接受。
只有当你能在译文中逐字引用到出错的片段时，才判错。

以 JSON 输出（不要 markdown、不要解释）：
{"v":"ok","err":[]}
或
{"v":"major","err":[{"type":"num","en":"英文片段","zh":"中文片段"}]}
err 最多 3 条；en 与 zh 必须是英文原文和中文译文里真实存在的片段。"""

FEWSHOT = """下面是四个已标注的例子：

例1 英文: The rear view of a mobile crane.
    中文: 移动式起重机的后视图。
    答案: {"v":"ok","err":[]}

例2 英文: Six workers are pouring concrete at night.
    中文: 六名工人在夜间浇筑混凝土。
    答案: {"v":"ok","err":[]}

例3 英文: Eleven people are in a stone pit.
    中文: 七人位于一个石坑内。
    答案: {"v":"major","err":[{"type":"num","en":"Eleven people","zh":"七人"}]}

例4 英文: The back view of an excavator in a wood.
    中文: 一台木工钻床的背面。
    答案: {"v":"major","err":[{"type":"term","en":"excavator","zh":"木工钻床"}]}
"""

_tls = threading.local()


def session():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        _tls.s = s
    return s


def load_key():
    k = os.environ.get("DEEPSEEK_API_KEY")
    if not k and os.path.exists(KEY_FILE):
        k = open(KEY_FILE, encoding="utf-8").read().strip()
    if not k:
        sys.exit("no API key (env DEEPSEEK_API_KEY or %s)" % KEY_FILE)
    return k


def build_messages(r, fewshot):
    sysp = SYS + ("\n\n" + FEWSHOT if fewshot else "")
    return [
        {"role": "system", "content": sysp},
        {"role": "user", "content": "英文原文:\n%s\n\n中文译文:\n%s"
         % ((r.get("text") or "").strip(), (r.get("zh") or "").strip())},
    ]


def parse_verdict(raw):
    s = re.sub(r"```(?:json)?", " ", raw or "").replace("```", " ").strip()
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    v = str(d.get("v", "")).strip().lower()
    if v not in ALLOWED_V:
        return None
    errs = d.get("err") or d.get("e") or []
    if isinstance(errs, str):
        errs = []
    norm = []
    for x in (errs if isinstance(errs, list) else []):
        if isinstance(x, dict):
            t = str(x.get("type", "")).strip().lower()
            en = re.sub(r"\s+", " ", str(x.get("en", ""))).strip()[:40]
            zh = re.sub(r"\s+", " ", str(x.get("zh", ""))).strip()[:40]
        else:
            t, en, zh = str(x).strip().lower(), "", ""
        norm.append({"type": t if t in ALLOWED_E else "other", "en": en, "zh": zh})
    if norm and v == "ok":
        v = "minor"
    if not norm and v in ("minor", "major"):
        v = "ok"
    return {"v": v, "err": norm, "e": [x["type"] for x in norm]}


def judge_one(args, key, r, retries=4):
    body = {
        "model": args.model,
        "messages": build_messages(r, args.fewshot),
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    url = args.base_url.rstrip("/") + "/chat/completions"
    last = None
    for a in range(retries):
        try:
            resp = session().post(url, json=body, timeout=args.timeout,
                                  headers={"Authorization": "Bearer " + key,
                                           "Content-Type": "application/json"})
            if resp.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError("http %d: %s" % (resp.status_code, resp.text[:120]))
            resp.raise_for_status()
            d = resp.json()
            msg = d["choices"][0]["message"]["content"]
            v = parse_verdict(msg)
            usage = d.get("usage") or {}
            if v is None:
                return {"v": "parse_error", "err": [], "e": [], "raw": (msg or "")[:200],
                        "usage": usage}
            v["usage"] = usage
            return v
        except Exception as e:  # noqa
            last = e
            time.sleep(1.5 * (a + 1) + 0.5 * a)
    return {"v": "api_error", "err": [], "e": [], "error": str(last)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--reasoning-effort", default="none",
                    help="none=关闭推理（快/省）；留空=默认；minimal/low/high=开启")
    args = ap.parse_args()
    key = load_key()

    recs = []
    for line in open(args.inp, encoding="utf-8"):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    if args.limit:
        recs = recs[:args.limit]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
        print("[deepseek] resume: %d already done" % len(done), flush=True)

    todo = [r for r in recs if r.get("id") not in done]
    print("[deepseek] in=%d todo=%d concurrency=%d fewshot=%s model=%s"
          % (len(recs), len(todo), args.concurrency, args.fewshot, args.model), flush=True)

    t0 = time.time()
    n = 0
    counts = {}
    tok_in = tok_out = 0
    lock = threading.Lock()
    fo = open(args.out, "a", encoding="utf-8")

    def work(r):
        v = judge_one(args, key, r)
        rec = {
            "id": r.get("id"), "dataset": r.get("dataset"), "field": r.get("field"),
            "kind": r.get("kind"), "round": r.get("round"), "row_id": r.get("row_id"),
            "text": r.get("text"), "zh": r.get("zh"),
            "v": v.get("v"), "e": v.get("e", []), "err": v.get("err", []),
            "raw": v.get("raw", ""), "error": v.get("error", ""),
        }
        return rec, v.get("usage") or {}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for fut in as_completed(futs):
            rec, usage = fut.result()
            with lock:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                counts[rec["v"]] = counts.get(rec["v"], 0) + 1
                tok_in += usage.get("prompt_tokens") or 0
                tok_out += usage.get("completion_tokens") or 0
                if n % 100 == 0:
                    el = time.time() - t0
                    print("[deepseek] %d/%d rate=%.2f/s ok=%d minor=%d major=%d err=%d "
                          "tok_in=%d tok_out=%d"
                          % (n, len(todo), n / el, counts.get("ok", 0), counts.get("minor", 0),
                             counts.get("major", 0),
                             counts.get("parse_error", 0) + counts.get("api_error", 0),
                             tok_in, tok_out), flush=True)
                if n % 200 == 0:
                    fo.flush()
    fo.flush()
    fo.close()
    el = time.time() - t0
    print("[deepseek] DONE n=%d elapsed=%.1fs counts=%s tok_in=%d tok_out=%d"
          % (n, el, counts, tok_in, tok_out), flush=True)


if __name__ == "__main__":
    main()
