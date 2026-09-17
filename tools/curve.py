import re, sys
for name in sys.argv[1:]:
    path = f"/root/train_logs/gnnA_{name}_K35_2000.log"
    if name == "proto":
        path = "/root/train_logs/gnnA_proto_sum_K35_2000.log"
    txt = open(path, errors="ignore").read()
    tr = [float(m) for m in re.findall(r"\{.loss.: .([0-9.]+)", txt)]
    ev = [(float(a), float(b)) for a, b in
          re.findall(r"eval_loss.: .([0-9.]+)., .eval_recon.: .([0-9.]+)", txt)]
    print(f"== {name}: train loss (每 20 步记录一次, 步号=20*(i+1))")
    print("   ", " ".join(f"{20*(i+1)}:{v:.4f}" for i, v in enumerate(tr[:6])))
    print("   ...")
    print("   ", " ".join(f"{20*(i+1)}:{v:.4f}" for i, v in enumerate(tr[-3:], start=len(tr)-3)))
    print("   eval(step, eval_loss, eval_recon):",
          " ".join(f"({500*(i+1)},{a:.4f},{b:.4f})" for i, (a, b) in enumerate(ev)))
