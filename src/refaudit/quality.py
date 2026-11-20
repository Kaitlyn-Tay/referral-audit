"""Baseline photo-quality measures, to separate skin tone from how the photo was taken.

If darker-skin images in a dataset are already blurrier, darker or more compressed, a model may
refer them more for that reason alone. These measures let us (a) report quality by skin-tone group
and (b) compare referral rates on quality-matched subsets.

Measures are computed on the image AS DOWNLOADED (before our resize and re-save), downscaled to
512 px on the longest side. The downloader records them in the manifest; `compute_quality` is a
fallback for images that are already on disk (then JPEG quality reflects our re-save).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import laplace
from tqdm import tqdm

from refaudit.groups import ita_from_image

QUALITY_COLS = ["sharpness", "brightness", "dynamic_range", "clipped_frac", "jpeg_quality",
                "megapixels"]
EXTRA_COLS = ["ita"]  # measured in the same pass but NOT used for quality matching


def estimate_jpeg_quality(img: Image.Image) -> float:
    """Rough libjpeg quality estimate from the luminance quantisation table (NaN if not JPEG)."""
    q = getattr(img, "quantization", None)
    if not q:
        return float("nan")
    table = np.array(q[0], dtype=float)
    mean_q = table.mean()
    # inverse of libjpeg scaling for its standard table (mean ~57.6 at quality 50)
    scale = mean_q / 57.625 * 100
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return float(np.clip(quality, 1, 100))


def image_quality(path_or_img: str | Path | Image.Image) -> dict[str, float]:
    raw = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    jpeg_q = estimate_jpeg_quality(raw)
    w, h = raw.size
    img = raw.convert("RGB")
    img.thumbnail((512, 512), Image.BILINEAR)
    lum = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    p1, p99 = np.percentile(lum, [1, 99])
    return {
        "sharpness": float(np.log1p(laplace(lum).var() * 1e4)),  # log variance of Laplacian
        "brightness": float(lum.mean()),
        "dynamic_range": float(p99 - p1),
        "clipped_frac": float(((lum <= 0.02) | (lum >= 0.98)).mean()),
        "jpeg_quality": jpeg_q,
        "megapixels": w * h / 1e6,
        "ita": ita_from_image(np.asarray(img)),
    }


def compute_quality(paths: list[str]) -> pd.DataFrame:
    return pd.DataFrame([image_quality(p) for p in tqdm(paths, desc="quality")])


def quality_by_group(quality: pd.DataFrame, groups: np.ndarray, order: list[str]) -> pd.DataFrame:
    df = quality.copy()
    df["group"] = groups
    return df.groupby("group")[QUALITY_COLS].median().reindex([g for g in order
                                                                if g in set(groups)])


def matched_referral_gap(
    referred: np.ndarray,
    groups: np.ndarray,
    quality: pd.DataFrame,
    cols: tuple[str, ...] = ("sharpness", "brightness"),
    n_bins: int = 3,
    high: str = "dark",
    low: str = "light",
) -> tuple[float, float, int]:
    """Referral gap (high - low) after matching on photo quality.

    Images are binned into quantiles of each quality column (bins fitted on all images); within
    every bin holding both groups the gap is computed, then averaged with weights equal to the
    `high` group's share in that bin (i.e. the gap a `high`-group patient would see versus a
    `low`-group patient with an equally good photo).

    Returns (raw gap, matched gap, number of `high` images in usable bins).
    """
    groups = np.asarray(groups, dtype=object)
    referred = np.asarray(referred, dtype=bool)
    raw = referred[groups == high].mean() - referred[groups == low].mean()
    keys = np.zeros(len(groups), dtype=np.int64)
    for c in cols:
        v = quality[c].to_numpy(dtype=float)
        edges = np.nanquantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])
        keys = keys * n_bins + np.digitize(np.nan_to_num(v, nan=np.nanmedian(v)), edges)
    gaps, weights = [], []
    for k in np.unique(keys):
        m = keys == k
        hi, lo = m & (groups == high), m & (groups == low)
        if hi.sum() and lo.sum():
            gaps.append(referred[hi].mean() - referred[lo].mean())
            weights.append(hi.sum())
    if not weights:
        return float(raw), float("nan"), 0
    return float(raw), float(np.average(gaps, weights=weights)), int(sum(weights))
