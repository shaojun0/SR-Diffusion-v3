import json, glob, os, subprocess

root = "/root/autodl-tmp/mm-datasets"
dirs = sorted(os.listdir(root))

# filter decision per dataset: keep/drop + reason
decision = {
    "aswin00000__ConstructionSiteCleanedDataSet": ("KEEP", "标注(检测bbox+违规理由)+caption;单图;无对话"),
    "chandrabhuma__multi_building_defect_vqa":    ("KEEP", "标注(VQA单轮/分类式);单图;单轮"),
    "ZhiyaYang__sewer-defect-crack-dataset":      ("KEEP", "标注(3类分类);单图;无文本"),
    "Francesco__construction-safety-gsnvb":       ("KEEP", "标注(COCO检测);单图;无文本"),
    "jhboyo__ppe-dataset":                        ("KEEP", "标注(YOLO检测);单图;无文本(部分文件未下完)"),
    "keremberke__construction-safety-object-detection": ("KEEP", "标注(COCO检测);单图;无文本(图片在zip内)"),
    "DBCMLAB__Constructionsafety_QApairs":        ("DROP", "非图像(纯文本)"),
    "juungwon__LLava-ConstructionSafety_v6":      ("DROP", "多轮对话"),
    "lmms-lab__LLaVA-NeXT-Data":                  ("DROP", "多轮对话"),
    "physicl__indoor-safety-hazard-detection-and-work-zone-monitoring": ("DROP", "无标注"),
    "Voxel51__Construction-Site-Traversability":  ("DROP", "无标注"),
    "pyimagesearch__construction-safety-object-detection-paligemma": ("DROP", "无数据(BLOCKED/gated)"),
}

def du(p):
    return int(subprocess.check_output(["du", "-sb", p]).split()[0])

keep = drop = 0
print(f"{'dataset':<62} {'size':>10}  decision  reason")
print("-" * 130)
rows = []
for d in dirs:
    p = os.path.join(root, d)
    b = du(p)
    dec, why = decision.get(d, ("?", "unknown"))
    if dec == "KEEP": keep += b
    else: drop += b
    rows.append((d, b, dec, why))
for d, b, dec, why in sorted(rows, key=lambda x: -x[1]):
    print(f"{d:<62} {b/1e9:>9.3f}G  {dec:<7}  {why}")
print("-" * 130)
print(f"KEEP total = {keep/1e9:.3f} GB   DROP total = {drop/1e9:.3f} GB")
print(f"NEED to reach 140G = {max(0, 140e9 - keep)/1e9:.1f} GB")

# language check on text-bearing KEPT datasets
print("\n=== language samples (kept text-bearing datasets) ===")
def sample_parquet(pat, cols, n=3):
    fs = sorted(glob.glob(pat))
    if not fs: return None
    import pyarrow.parquet as pq
    t = pq.read_table(fs[0], columns=cols).slice(0, n)
    return t.to_pylist()

# aswin00000 captions + rule reasons
try:
    s = sample_parquet("/root/autodl-tmp/mm-datasets/aswin00000__ConstructionSiteCleanedDataSet/data/*.parquet",
                       ["image_caption"], 3)
    print("[aswin00000 caption]", [ (x.get("image_caption") or "")[:120] for x in s ])
except Exception as e:
    print("[aswin00000] ERR", e)

# chandrabhuma question/answer
try:
    s = sample_parquet("/root/autodl-tmp/mm-datasets/chandrabhuma__multi_building_defect_vqa/data/*.parquet",
                       ["question", "answer"], 5)
    for x in s: print("[chandrabhuma]", (x.get("question") or "")[:80], "|", (x.get("answer") or "")[:40])
except Exception as e:
    print("[chandrabhuma] ERR", e)

# rule reasons in aswin00000 (nested struct)
try:
    import pyarrow.parquet as pq
    f = sorted(glob.glob("/root/autodl-tmp/mm-datasets/aswin00000__ConstructionSiteCleanedDataSet/data/*.parquet"))[0]
    sch = pq.ParquetFile(f).schema_arrow
    rule_cols = [n for n in sch.names if "rule" in n]
    print("[aswin00000 rule cols]", rule_cols)
    if rule_cols:
        t = pq.read_table(f, columns=rule_cols[:1]).slice(0, 3)
        for x in t.to_pylist():
            print("[rule sample]", json.dumps(x, ensure_ascii=False)[:300])
except Exception as e:
    print("[aswin00000 rule] ERR", e)
