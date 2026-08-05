import io

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset

from model import SRQwenVLv10

default_image_size = 1024
energy_threshold=0.99
max_eig = 256
data_path = "translated_dataset_with_new_fields"


def load_image_as_feature(image,
                          image_size: int = 1024) -> torch.Tensor:
    """Load single image → grayscale → patchify → (n_patches, patch_dim)."""
    if isinstance(image, Image.Image):
        img = image.convert("L")
    else:
        img = Image.open(io.BytesIO(image)).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)
    return t-t.mean()  # (n_patches, patch_dim)   → (1024, 1024) for default params

def build_svd(image):
    feature = load_image_as_feature(image, default_image_size)
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(feature, full_matrices=False)
    S_sq = S * S
    total_e = S_sq.sum(dim=1, keepdim=True)
    cumsum = torch.cumsum(S_sq, dim=1)
    n_per_img = (cumsum / total_e < energy_threshold).sum(dim=1) + 1
    n_per_img = n_per_img.clamp(min=32, max=max_eig).item()
    U_top = U[:, :n_per_img].T  # (n, 1024)
    Vh_top = Vh[:n_per_img, :]  # (n, 1024)
    mat = torch.cat([U_top, Vh_top], dim=0)  # (2n, 1024)
    return mat
dataset = load_dataset(data_path,cache_dir="./cache")
model = SRQwenVLv10.from_pretrained("output/sr_qwen_vl_v10_output/final/")
svd_mat = build_svd(load_image_as_feature(dataset[0]["image"]))
res = model.generate(dataset[0]["image"],prompt="描述这张建筑工地图片：")
print(res)
if __name__ == '__main__':
    pass