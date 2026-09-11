"""diag_collect.py — 汇总所有短程消融臂, 产出消融矩阵 + summary json。

数据源优先级:
  1) <output_dir>/checkpoint-*/trainer_state.json 的 log_history（**精确 step**,
     因为 stdout 被块缓冲, 日志里的 dict 会成块 flush, 顺序仍对但行号不可靠）
  2) 训练 log 的 dict 顺序 × log_every

判据（任务书口径）:
  塌缩 := 末段 grad_norm < 1e-2 且 末段 loss 高于该臂前 100 步均值
另外报告 step≈400/500 与"loss 最低点 + 之后是否回升"。
"""
import glob, json, os, re, sys

STEP_RE = re.compile(r"(\d+)/(\d+) \[[^]]*\]")
LOSS_RE = re.compile(r"\{'loss': '([0-9.eE+-]+)', 'grad_norm': '([0-9.eE+-]+)',"
                     r" 'learning_rate': '([0-9.eE+-]+)'")


def from_trainer_state(arm, root):
    best = None
    for p in glob.glob(os.path.join(root, arm, "checkpoint-*", "trainer_state.json")) + \
             glob.glob(os.path.join(root, arm, "trainer_state.json")):
        try:
            st = json.load(open(p))
        except Exception:
            continue
        h = [r for r in st["log_history"] if "loss" in r and "grad_norm" in r]
        if h and (best is None or len(h) > len(best[1])):
            best = (p, h)
    if best is None:
        return None
    p, h = best
    return p, {int(r["step"]): (float(r["loss"]), float(r["grad_norm"]),
                                float(r.get("learning_rate", float("nan"))))
               for r in h}


def from_log(arm, logdir):
    p = os.path.join(logdir, f"stack2x_diag_{arm}.log")
    if not os.path.exists(p):
        return None
    raw = open(p, errors="ignore").read().replace("\r", "\n")
    recs = [m.groups() for m in (LOSS_RE.search(l) for l in raw.split("\n")) if m]
    if not recs:
        return None
    ms = re.findall(r"--log_every[\"']?\s*[:=]?\s*(\d+)", raw)
    le = int(ms[0]) if ms else 10
    return p, {(i + 1) * le: (float(a), float(b), float(c))
               for i, (a, b, c) in enumerate(recs)}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else \
        "/root/autodl-tmp/sr-diffusion-v3-stack2x/output"
    logdir = sys.argv[2] if len(sys.argv) > 2 else "/root/train_logs"
    out = {}
    arms = sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, "diag_*"))
                  if os.path.isdir(d))
    for arm in arms:
        got = from_trainer_state(arm, root) or from_log(arm, logdir)
        if got is None:
            out[arm] = {"error": "no data"}
            continue
        src, s = got
        ks = sorted(s)
        last = max(ks)
        early = [s[k][0] for k in ks if k <= 100]
        tail = [k for k in ks if k >= last - 100]
        tail_loss = sum(s[k][0] for k in tail) / len(tail)
        tail_gn = sum(s[k][1] for k in tail) / len(tail)
        early_loss = sum(early) / len(early) if early else None
        best_k = min(ks, key=lambda k: s[k][0])
        def at(t):
            c = [k for k in ks if k >= t]
            return [c[0], list(s[c[0]])] if c else [None, None]
        collapsed = bool(tail_gn < 1e-2 and early_loss is not None
                         and tail_loss > early_loss)
        # 更稳健的塌缩判据: 出现过 loss 最低点后 loss 抬升 >10% 且 grad < 1e-2
        rebound = (s[last][0] - s[best_k][0]) / max(abs(s[best_k][0]), 1e-9)
        soft_collapse = bool(tail_gn < 3e-2 and rebound > 0.05)
        out[arm] = {
            "source": src, "n_records": len(ks), "last_step": last,
            "loss_first100_mean": early_loss,
            "loss_tail_mean": tail_loss, "grad_norm_tail_mean": tail_gn,
            "best_step": best_k, "best_loss": s[best_k][0],
            "loss_rebound_after_best": rebound,
            "step400": at(400), "step500": at(500),
            "loss_last": s[last][0], "gn_last": s[last][1],
            "min_grad_norm": min(v[1] for v in s.values()),
            "collapsed": collapsed, "soft_collapse": soft_collapse,
            "curve": {str(k): list(s[k]) for k in ks},
        }
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "curve"}
                      for k, v in out.items()}, indent=2, ensure_ascii=False))
    with open("/root/train_logs/stack2x_diag_summary.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n[table] arm | loss@400 | gn@400 | loss@500 | gn@500 | best_loss@step "
          "| rebound | tail_gn | collapsed")
    for arm, v in out.items():
        if "error" in v:
            print(f"  {arm}: {v['error']}")
            continue
        def f(t):
            return f"{v[t][1][0]:.4f}/{v[t][1][1]:.5f}" if v[t][1] else "n/a"
        print(f"  {arm:8s} | {f('step400')} | {f('step500')} | "
              f"{v['best_loss']:.4f}@{v['best_step']} | "
              f"{v['loss_rebound_after_best']:+.3f} | {v['grad_norm_tail_mean']:.5f} "
              f"| {v['collapsed']}/{v['soft_collapse']}")


if __name__ == "__main__":
    main()
