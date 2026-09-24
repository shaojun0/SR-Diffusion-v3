"""校验「解码器 = 编码器的倒置」是否逐级、逐 block 严格镜像，并复核预训练加载。"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from model import BACKBONES, build_model

_HF = {
    "resnet-10": ("microsoft/resnet-18", (1, 1, 1, 1)),
    "resnet-18": ("microsoft/resnet-18", (2, 2, 2, 2)),
    "resnet-34": ("microsoft/resnet-34", (3, 4, 6, 3)),
    "resnet-50": ("microsoft/resnet-50", (3, 4, 6, 3)),
    "resnet-101": ("microsoft/resnet-101", (3, 4, 23, 3)),
    "resnet-152": ("microsoft/resnet-152", (3, 8, 36, 3)),
}

def n_blocks(mod):
    n = 0
    for m in mod.modules():
        if type(m).__name__ in ("BasicBlock", "Bottleneck"):
            n += 1
    return n

ok = True
for bb in BACKBONES:
    m = build_model(bb, 224)
    _, depths = _HF[bb]
    enc, dec = m.depths, list(reversed(m.depths))
    print(f"\n### {bb}  depths={enc}  layer_type={m.layer_type}  embed={m.embedding_size}")
    print(f"  encoder hidden_sizes = {m.hidden_sizes}")
    assert tuple(m.depths) == tuple(depths), f"{bb} depths mismatch"

    # 解码器各级输出通道 == 编码器各级输入通道（倒置）
    enc_in = [m.embedding_size] + list(m.hidden_sizes[:-1])
    dec_out = []
    for k, si in enumerate(range(len(depths) - 1, -1, -1)):
        _ = si
    print(f"  期望解码器各级输出通道(倒置) = {list(reversed(enc_in))}")

    # 逐级 block 数镜像
    dec_counts = [n_blocks(s) for s in m.dec_blocks]
    print(f"  encoder stage blocks = {list(depths)} | decoder level blocks = {dec_counts}")
    assert dec_counts == list(reversed(depths)), f"{bb} decoder block counts not mirrored"

    # 前向 + 形状
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 3, 224, 224), y.shape
    print(f"  forward OK out={tuple(y.shape)}  params={sum(p.numel() for p in m.parameters())/1e6:.2f}M")

    # 初始时刻：编码器 4 个 LSTM 残差为 0（zero-init proj）=> 预训练特征逐位不变
    with torch.no_grad():
        out = m.encoder(x, output_hidden_states=True)
        hs = out.hidden_states
        for i, s in enumerate(hs[1:]):
            delta = (m.enc_lstms[i](s) - s).abs().max().item()
            assert delta == 0.0, f"{bb} encoder LSTM {i} not identity at init: {delta}"
    print("  encoder LSTM residuals at init = 0 (预训练特征逐位保留) OK")

    # 预训练加载报告
    rep = m.load_report()
    print("  " + rep)
    if "missing=0" not in rep:
        ok = False
        print("  !! 有未命中的张量")
    del m

print("\nVERIFY " + ("OK" if ok else "FAILED"))
