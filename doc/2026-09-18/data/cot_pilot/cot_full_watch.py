#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cot_full_watch.py — 全量 CoT 生成看护 + 自动收尾（2026-09-18 接手会话新增）。

背景：run_cot_full.sh 的双车道生成要跑 ~38 小时。会话随时可能中断，因此把
"盯进度 + 完成后收尾三连"做成服务器端常驻进程，与会话解耦。

每个 INTERVAL 秒做一次快照（追加到 LOG/watch.jsonl，摘要到 LOG/watch.log）：
  - 逐数据集 已生成 / manifest 计划
  - ALL_DONE 标记是否存在
  - 是否还有 cot_generate.py 在跑
  - :8100 / :8101 两路模型服务是否存活、健康、RSS 是否异常
  - GPU 利用率与显存
  - 近期速率与 ETA

自动动作：
  1) 某路模型服务进程消失 -> 按记录的命令重启（RESTART_SERVICES=1，默认开；
     同一路 15 分钟内只重启一次）。不做这一步的话，该车道剩余全部条目都会
     批量失败写空 —— 重启可把损失限制在重启窗口内的几十条。
  2) 生成**真正完成**（ALL_DONE 存在 且 逐数据集计数与 manifest 完全吻合 且
     已无 cot_generate 进程）-> 依次执行
        cot_full_repair.py  ->  cot_full_qa.py  ->  cot_full_aggregate.py
     全部 rc==0 时写 FINALIZE_DONE；任一步失败写 FINALIZE_FAILED 且**不自动重试**
     （写 FINALIZE_ATTEMPTED 标记，避免反复重写产物），留待人工处理。
  3) ALL_DONE 存在但逐数据集计数不吻合 -> 说明某车道中途崩过。**不自动收尾**
     （否则会把缺量当成完成），改为按 handoff §1.4 自动重启全量驱动续跑
     （幂等：已存在的 id 会跳过），最多 WATCH_RESUME_LIMIT 次（默认 2），
     并写 WARN_ALL_DONE_MISMATCH / RESUME_ATTEMPTED。
  4) 超过 STALL_SEC 无任何新产出且无 cot_generate 进程且未完成 -> 写 WARN_STALL。

用法（必须脱离会话，setsid nohup）：
  setsid nohup python3 /root/translate/cot_full_watch.py \
      >> /root/translate_logs/cot_full/watch_console.log 2>&1 </dev/null &
"""
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

ROOT = os.environ.get("COT_ROOT", "/root/autodl-tmp/cot_full")
OUT = os.environ.get("COT_OUTDIR", "/root/autodl-tmp/cot_out")
LOG = os.environ.get("COT_LOGDIR", "/root/translate_logs/cot_full")
SERVER = "/root/translate/openai_compat_server.py"
MODEL = "/root/autodl-tmp/models/Qwen3.8-27B"
PYTHON = "/root/miniconda3/bin/python3"

INTERVAL = int(os.environ.get("WATCH_INTERVAL", "300"))
STALL_SEC = int(os.environ.get("WATCH_STALL_SEC", "3600"))
RSS_LIMIT_GIB = float(os.environ.get("WATCH_RSS_LIMIT_GIB", "20"))
RESTART_SERVICES = os.environ.get("RESTART_SERVICES", "1") == "1"
RESUME_LIMIT = int(os.environ.get("WATCH_RESUME_LIMIT", "2"))   # 自动续跑上限
PORTS = [(8100, "0"), (8101, "1")]          # (port, CUDA_VISIBLE_DEVICES)

os.makedirs(LOG, exist_ok=True)

# 防止重复启动
PIDFILE = os.path.join(LOG, "watch.pid")
if os.path.exists(PIDFILE):
    try:
        old = int(open(PIDFILE).read().strip())
        if old != os.getpid() and os.path.exists("/proc/%d" % old):
            print("[watch] already running pid=%d, exit" % old)
            sys.exit(0)
    except Exception:
        pass
open(PIDFILE, "w").write(str(os.getpid()))

JSONL = os.path.join(LOG, "watch.jsonl")
TXT = os.path.join(LOG, "watch.log")


def log(msg):
    line = "[%s] %s" % (datetime.now().strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(TXT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def marker(name, text=""):
    with open(os.path.join(LOG, name), "w", encoding="utf-8") as f:
        f.write("%s %s\n" % (datetime.now().isoformat(timespec="seconds"), text))


def expected_counts():
    """manifest 逐数据集计划条数（按 id 去重，与 cot_generate.py 一致）。"""
    seen = set()
    c = Counter()
    for mf in sorted(glob.glob(os.path.join(ROOT, "manifest*.jsonl"))):
        for line in open(mf, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("id") in seen:
                continue
            seen.add(r.get("id"))
            c[r.get("dataset")] += 1
    return c


def actual_counts():
    c = Counter()
    for f in sorted(glob.glob(os.path.join(OUT, "*.jsonl"))):
        ds = os.path.basename(f)[:-len(".jsonl")]
        n = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        c[ds] = n
    return c


def proc_argv():
    """返回 [(pid, rss_bytes, argv)]。按 argv 精确判断，**不做子串匹配** ——
    否则 `bash -c "... cot_generate.py ..."` 这类包装进程会被误判成生成器
    （§4.2/§4.6 的教训），进而让"已完成"的判定永远无法成立。"""
    hits = []
    for d in glob.glob("/proc/[0-9]*"):
        pid = int(os.path.basename(d))
        try:
            raw = open(os.path.join(d, "cmdline"), "rb").read()
        except Exception:
            continue
        if not raw:
            continue
        args = [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]
        rss = 0
        try:
            for ln in open(os.path.join(d, "status")):
                if ln.startswith("VmRSS:"):
                    rss = int(ln.split()[1]) * 1024
                    break
        except Exception:
            pass
        hits.append((pid, rss, args))
    return hits


def generator_pids():
    """真正的生成器进程：argv[1] 就是 cot_generate.py。"""
    gen = "/root/translate/cot_generate.py"
    return [pid for pid, _, a in proc_argv() if len(a) >= 2 and a[1] == gen]


def service_alive(port):
    for pid, rss, a in proc_argv():
        if len(a) >= 2 and a[1] == SERVER and "--port" in a:
            i = a.index("--port")
            if i + 1 < len(a) and a[i + 1] == str(port):
                return pid, rss
    return None, 0


def service_health(port, timeout=8):
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % port, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def restart_service(port, dev):
    cmd = [PYTHON, SERVER, "--model", MODEL, "--served-name", "qwen3.8-27b",
           "--host", "0.0.0.0", "--port", str(port), "--max-new-tokens", "1024",
           "--device-map", "single", "--tp-plan", "none", "--dtype", "bfloat16",
           "--max-model-len", "32768", "--batch-size", "8"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev, PATH="/root/miniconda3/bin:" + os.environ.get("PATH", ""))
    lf = open(os.path.join(LOG, "serve_%d.log" % port), "a")
    subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True, cwd="/root")
    log("[ALERT] 重启模型服务 :%d (CUDA_VISIBLE_DEVICES=%s) -> %s/serve_%d.log" % (port, dev, LOG, port))


def launch_driver():
    """按 handoff §1.4 重启全量驱动。生成幂等：产物中已存在的 id 会被跳过，
    因此只会补上缺口，不会重复生成。"""
    lf = open(os.path.join(LOG, "driver_console.log"), "a")
    env = dict(os.environ, PATH="/root/miniconda3/bin:" + os.environ.get("PATH", ""))
    subprocess.Popen(["bash", "/root/translate/run_cot_full.sh"], env=env, cwd="/root",
                     stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
    log("[ALERT] 已按 handoff §1.4 重启全量驱动 run_cot_full.sh（幂等续跑）")


def gpu_snapshot():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return [x.strip() for x in o.stdout.strip().splitlines()]
    except Exception:
        return []


def finalize(expected, actual):
    """收尾三连。返回 True 表示全部成功。"""
    log("[FINALIZE] 开始收尾三连：repair -> qa -> aggregate")
    marker("FINALIZE_ATTEMPTED")
    steps = [("repair", "cot_full_repair.py"),
             ("qa", "cot_full_qa.py"),
             ("aggregate", "cot_full_aggregate.py")]
    env = dict(os.environ, PATH="/root/miniconda3/bin:" + os.environ.get("PATH", ""))
    rcs = {}
    fl = open(os.path.join(LOG, "finalize.log"), "a", encoding="utf-8")
    for name, script in steps:
        fl.write("\n===== %s %s %s =====\n" % (name, script, datetime.now().isoformat(timespec="seconds")))
        fl.flush()
        t0 = time.time()
        p = subprocess.run([PYTHON, "/root/translate/" + script], stdout=fl,
                           stderr=subprocess.STDOUT, cwd="/root", env=env)
        rcs[name] = p.returncode
        log("[FINALIZE] %s rc=%d elapsed=%.0fs" % (name, p.returncode, time.time() - t0))
    fl.close()
    ok = all(v == 0 for v in rcs.values()) and os.path.exists(os.path.join(OUT, "SUMMARY.json"))
    if ok:
        summary = {}
        try:
            summary = json.load(open(os.path.join(OUT, "SUMMARY.json"), encoding="utf-8"))
        except Exception:
            pass
        marker("FINALIZE_DONE", json.dumps({k: summary.get(k) for k in
               ("total", "planned_total", "parse_ok", "parse_fail", "detect_rows", "aligned", "rotated")},
               ensure_ascii=False) + " rcs=%s" % rcs)
        log("[FINALIZE] DONE %s" % json.dumps(summary, ensure_ascii=False)[:400])
    else:
        marker("FINALIZE_FAILED", "rcs=%s" % rcs)
        log("[ALERT] 收尾失败 rcs=%s —— 不自动重试，见 %s/finalize.log" % (rcs, LOG))
    return ok


def main():
    log("[watch] START pid=%d interval=%ds stall=%ds rss_limit=%.0fGiB restart=%s root=%s out=%s"
        % (os.getpid(), INTERVAL, STALL_SEC, RSS_LIMIT_GIB, RESTART_SERVICES, ROOT, OUT))
    exp = expected_counts()
    exp_total = sum(exp.values())
    log("[watch] manifest 计划 %d 条 / %d 数据集" % (exp_total, len(exp)))

    last_total, last_ts = None, time.time()
    t_start, total_start = time.time(), None
    restart_at = {}
    resume_count, resume_at = 0, 0.0
    finalized = os.path.exists(os.path.join(LOG, "FINALIZE_ATTEMPTED"))

    while True:
        tick = {}
        try:
            act = actual_counts()
            total = sum(act.values())
            if total_start is None:
                total_start = total

            if last_total is None or total != last_total:
                last_total, last_ts = total, time.time()
            stall = time.time() - last_ts

            gens = generator_pids()
            all_done = os.path.exists(os.path.join(OUT, "ALL_DONE"))

            svc = {}
            for port, dev in PORTS:
                pid, rss = service_alive(port)
                healthy = service_health(port) if pid else False
                svc[port] = {"pid": pid, "rss_gib": round(rss / 2**30, 2), "healthy": healthy}
                if pid is None and not all_done and RESTART_SERVICES:
                    t_prev = restart_at.get(port, 0)
                    if time.time() - t_prev > 900:
                        restart_at[port] = time.time()
                        log("[ALERT] 模型服务 :%d 进程消失（未完成，已生成 %d/%d）" % (port, total, exp_total))
                        restart_service(port, dev)
                elif pid and rss / 2**30 > RSS_LIMIT_GIB:
                    log("[ALERT] :%d RSS=%.1fGiB 超过阈值 %.0fGiB（handoff §1.6）" % (
                        port, rss / 2**30, RSS_LIMIT_GIB))

            missing = {ds: (act.get(ds, 0), exp[ds]) for ds in exp if act.get(ds, 0) != exp[ds]}
            elapsed = time.time() - t_start
            avg_rate = (total - total_start) / elapsed if elapsed > 0 and total_start is not None else 0.0
            eta_h = (exp_total - total) / avg_rate / 3600 if avg_rate > 0 else None

            tick = {"ts": datetime.now().isoformat(timespec="seconds"), "total": total,
                    "expected_total": exp_total, "all_done": all_done,
                    "generators": len(gens), "stall_sec": int(stall),
                    "avg_rate_per_s": round(avg_rate, 4),
                    "eta_hours": round(eta_h, 2) if eta_h is not None else None,
                    "by_dataset": {ds: {"got": act.get(ds, 0), "want": exp[ds]} for ds in sorted(exp)},
                    "services": svc, "gpu": gpu_snapshot()}
            with open(JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(tick, ensure_ascii=False) + "\n")

            log("total=%d/%d (%.1f%%) gens=%d all_done=%s stall=%ds rate=%.3f/s eta=%s h | svc=%s"
                % (total, exp_total, 100.0 * total / max(exp_total, 1), len(gens), all_done,
                   stall, avg_rate, ("%.1f" % eta_h) if eta_h is not None else "?",
                   {p: ("pid%s" % v["pid"] if v["pid"] else "DOWN") for p, v in svc.items()}))

            # ---- 完成判定 ----
            if all_done and not missing and not gens:
                if finalized:
                    log("[watch] 已尝试过收尾（FINALIZE_ATTEMPTED），保持观察，不重复执行")
                else:
                    finalized = True
                    finalize(exp, act)
            elif all_done and missing:
                marker("WARN_ALL_DONE_MISMATCH", json.dumps(missing, ensure_ascii=False))
                log("[ALERT] ALL_DONE 已出现但计数不吻合（缺口 %d 个数据集）：%s"
                    % (len(missing), list(missing.items())[:5]))
                if not gens and resume_count < RESUME_LIMIT and time.time() - resume_at > 600:
                    resume_count += 1
                    resume_at = time.time()
                    try:
                        os.remove(os.path.join(OUT, "ALL_DONE"))
                    except Exception:
                        pass
                    marker("RESUME_ATTEMPTED", "count=%d missing=%s"
                           % (resume_count, json.dumps(missing, ensure_ascii=False)))
                    launch_driver()
                elif not gens and resume_count >= RESUME_LIMIT:
                    log("[ALERT] 自动续跑已达上限 %d 次，停止续跑，需人工介入（handoff §1.4）" % RESUME_LIMIT)

            # ---- 停滞判定 ----
            if not all_done and stall > STALL_SEC and not gens:
                marker("WARN_STALL", "no progress %ds, no cot_generate alive, total=%d" % (stall, total))
                log("[ALERT] 停滞 %ds 且无 cot_generate 进程，total=%d —— 需人工按 handoff §1.4 续跑" % (stall, total))

            if all_done and finalized:
                log("[watch] 任务结束，退出")
                break
        except Exception as e:
            log("[watch][ERROR] tick 异常：%r" % e)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
