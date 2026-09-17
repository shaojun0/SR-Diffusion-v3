import json, sys, os
rows = []
for name in sys.argv[1:]:
    p = f"/root/train_logs/infer_gnnA_{name}.json"
    d = json.load(open(p))
    rows.append((name, d))
keys = ["full_norm_l1", "full_pixel_l1_255", "full_pixel_l1_norm", "full_pixel_mae_255"]
print("arm      " + "".join(f"{k:>22s}" for k in keys) + "   step_pixel_l1_255")
for name, d in rows:
    line = f"{name:9s}"
    for k in keys:
        v = d.get(k)
        line += f"{v:22.6f}" if isinstance(v, (int, float)) else f"{str(v):>22s}"
    steps = d.get("step_pixel_l1_255")
    line += "   " + (", ".join(f"{v:.4f}" for v in steps) if steps else "-")
    print(line)
print()
for name, d in rows:
    print(name, "keys:", sorted(d.keys()))
    break
