import json, sys
d = json.load(open(sys.argv[1]))
sh, st = d["split_half"], d["full_permutation_S"]["stats"]
S = d["full_permutation_S"]["S"]
hdr = ("variant", "split_half max|d|", "rel_fro", "cos_min", "unit_l2_max", f"S={S} mean l2", f"S={S} max l2")
print("%-46s %16s %10s %10s %12s %14s %14s" % hdr)
for k in sh:
    v = sh[k]
    if "l2_unit_max" in v:
        s = st[k]["l2_unit_max"]
        print("%-46s %16.4e %10.3e %10.8f %12.3e %14.2e %14.2e" % (
            k, v["max_abs"], v["rel_fro"], v["cos_min"], v["l2_unit_max"], s["mean"], s["max"]))
    else:
        s = st[k]["hausdorff_max"]
        n = st[k]["nn_mean"]
        print("%-46s %16.4e %10.3e %10s %12s %14.2e %14.2e" % (
            k, v["hausdorff_max"], v["nn_mean"], "-", "-", s["mean"], s["max"]))
print()
print("meta:", json.dumps(d["meta"], ensure_ascii=False))
