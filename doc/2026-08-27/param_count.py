"""参数量统计（不需要预训练权重, 随机初始化即可）"""
import torch, torch.nn as nn
import sys
sys.path.insert(0, "/home/linaro/dsh/sr-diffusion-v3")

class FakeDino(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.config = type("C", (), {"hidden_size": dim, "use_mask_token": False})()
        self.embeddings = nn.Module()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Identity() for _ in range(12)])
    def forward(self, x):
        raise NotImplementedError

from model_v2 import SRPhase1V2
dino = FakeDino(dim=1024)
model = SRPhase1V2(dinov2=dino, num_patches=576, dim=1024, reencoder_depth=4)
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"=== 非 DINO 部分可训练参数 (SRPhase1V2, 不含 DINOv2 权重) ===")
print(f"总参数量: {total/1e6:.2f}M")
parts = {
    "special_bank": model.special_bank,
    "re_encoder": model.re_encoder,
    "decoder": model.decoder,
    "pixel_head": model.pixel_head,
}
for name, m in parts.items():
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  {name:15s}: {n/1e6:.3f}M  ({n/total*100:.1f}%)")
# 每部分明细
print("\n=== 明细 ===")
for name, m in parts.items():
    print(f"-- {name} --")
    for pn, p in m.named_parameters():
        if p.requires_grad:
            print(f"    {pn}: {tuple(p.shape)} = {p.numel():,}")
# DINOv2-large 本体（参考）
print("\n=== 参考: DINOv2-large 本体 ≈ 304M (官方) ===")
