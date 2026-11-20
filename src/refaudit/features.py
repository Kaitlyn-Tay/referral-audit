"""Frozen-backbone feature extraction, for original images and every counterfactual variant.

Each image is passed through the backbone once per variant. Variants:
  orig                      the unchanged image
  <factor>@<severity>       a capture-condition counterfactual (see counterfactuals.FACTORS)
  tta@<i>                   random test-time augmentation draw i (baseline)
  rot@<deg>                 rotation (ASRS-style baseline)
Features are saved as one float16 array per variant in a single .npz file.

Requires the optional `torch` extra: pip install -e ".[torch]"
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from refaudit import counterfactuals as cf

DEFAULT_MODEL = "vit_base_patch14_dinov2.lvd142m"
ROTATIONS = (-30, -15, 15, 30)


def variant_names(
    factors: Iterable[str] | None = None, n_tta: int = 5, rotations: Iterable[int] = ROTATIONS
) -> list[str]:
    factors = list(cf.FACTORS) if factors is None else list(factors)
    names = ["orig"]
    names += [f"{f}@{s}" for f in factors for s in cf.SEVERITIES]
    names += [f"tta@{i}" for i in range(n_tta)]
    names += [f"rot@{d}" for d in rotations]
    return names


def make_variant(img: Image.Image, variant: str, seed: int) -> Image.Image:
    if variant == "orig":
        return img
    kind, arg = variant.split("@")
    if kind == "tta":
        return cf.random_augment(img, np.random.default_rng(seed * 1000 + int(arg)))
    if kind == "rot":
        return cf.rotate(img, float(arg))
    return cf.apply(img, kind, int(arg))


class _VariantDataset(Dataset):
    def __init__(self, paths: list[str], variant: str, tfm, seed: int = 0):
        self.paths, self.variant, self.tfm, self.seed = paths, variant, tfm, seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(make_variant(img, self.variant, self.seed + i))


def load_backbone(
    name: str = DEFAULT_MODEL, device: str | None = None, weights_file: str | None = None
):
    """Load a timm backbone with its classifier removed.

    `weights_file` loads pretrained weights from a local file instead of the Hugging Face Hub
    (useful when the Hub is unreachable).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"img_size": 224} if "vit" in name else {}
    if weights_file:
        kwargs["pretrained_cfg_overlay"] = {"file": weights_file}
    model = timm.create_model(name, pretrained=True, num_classes=0, **kwargs)
    model.eval().to(device)
    cfg = model.pretrained_cfg
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cfg.get("mean", (0.485, 0.456, 0.406)),
                             cfg.get("std", (0.229, 0.224, 0.225))),
    ])
    return model, tfm, device


@torch.inference_mode()
def extract(
    paths: list[str],
    out_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    variants: list[str] | None = None,
    batch_size: int = 64,
    num_workers: int = 2,
    weights_file: str | None = None,
) -> Path:
    """Extract features for every variant and save to `out_path` (.npz). Resumes per variant."""
    out_path = Path(out_path)
    variants = variants or variant_names()
    existing: dict[str, np.ndarray] = {}
    if out_path.exists():
        with np.load(out_path) as f:
            existing = {k: f[k] for k in f.files}
    model, tfm, device = load_backbone(model_name, weights_file=weights_file)
    for v in variants:
        if v in existing and len(existing[v]) == len(paths):
            continue
        dl = DataLoader(_VariantDataset(paths, v, tfm), batch_size=batch_size,
                        num_workers=num_workers)
        chunks = []
        for x in tqdm(dl, desc=v, leave=False):
            chunks.append(model(x.to(device)).float().cpu().numpy().astype(np.float16))
        existing[v] = np.concatenate(chunks)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **existing)  # save after each variant so a crash loses little
    return out_path
