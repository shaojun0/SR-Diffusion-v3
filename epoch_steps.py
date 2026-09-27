"""算一个 epoch 的优化步数（每 rank），用于设定 --max_steps。"""
import json
import sys

from data_multi import BucketBatchSampler

cache = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/data/index_cache.json"
bs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
budget = int(sys.argv[3]) if len(sys.argv) > 3 else 6144
world = int(sys.argv[4]) if len(sys.argv) > 4 else 2

idx = json.load(open(cache))
shapes = [(bh, bw) for (_, bw, bh, _) in idx]
s = BucketBatchSampler(shapes, bs, shuffle=True, drop_last=True, rank=0,
                       world_size=world, patch_budget=budget)
n = len(s)
print("samples", len(idx), "steps_per_rank", n, "global_batches", n * world)
