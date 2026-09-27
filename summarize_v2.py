"""汇总 v2（六源）训练日志：轨迹、分段均值、换算到 0-255 像素 L1。"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/autodl-tmp/output/multires_natural_v2/train_log.jsonl"
recs = [json.loads(l) for l in open(path) if l.strip()]
print(f"records={len(recs)} step {recs[0]['step']}..{recs[-1]['step']}")
NS = [r["step"] for r in recs]
print("\n-- 每 4000 步取样 --")
for r in recs[::200]:
    print(f"  step {r['step']:6d} loss={r['loss']:.4f} "
          f"dec={[round(x, 3) for x in r['per_decoder']]} "
          f"gnorm={r['grad_norm']:.3f} {r['sec_per_step']}s")


def avg(rs, k):
    return sum(r[k] for r in rs) / len(rs)


f, l = recs[:100], recs[-100:]
print(f"\nfirst100 loss={avg(f,'loss'):.4f} | last100 loss={avg(l,'loss'):.4f} | "
      f"best={min(r['loss'] for r in recs):.4f}")
print("last100 per_decoder:", [round(sum(r['per_decoder'][i] for r in l) / len(l), 4)
                               for i in range(4)])
print("last100 recon:", round(avg(l, "recon"), 4))
print("last100 sec/step:", round(avg(l, "sec_per_step"), 3),
      "peak GiB:", l[-1]["peak_gib"])

# 口径换算：历史报告里 65.59 px L1 ↔ 1.138 归一化 ⇒ 除数 57.6
DIV = 57.6
print(f"\n换算到 0-255 像素 L1（÷{DIV}，与历史 448×252 口径一致）:")
print(f"  last100 loss   {avg(l,'loss'):.4f} → {avg(l,'loss')*DIV:6.2f} px")
print(f"  last100 recon  {avg(l,'recon'):.4f} → {avg(l,'recon')*DIV:6.2f} px")
for i, nm in enumerate(["448x252", "252x448", "224x224", "448x448"]):
    v = sum(r['per_decoder'][i] for r in l) / len(l)
    print(f"  dec{i} {nm}  {v:.4f} → {v*DIV:6.2f} px")
print("\n参考：历史主线 square24 BPTT = 16.42 px；v4 register K=64 = 8.26 px；"
      "常量预测平台 = 1.138 归一化 ≈ 65.6 px")
