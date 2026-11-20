"""Waterbirds (Sagawa et al. 2020): birds pasted onto land or water backgrounds. In training,
most waterbirds sit on water and most landbirds on land, so models learn the background shortcut.
Used here as a no-people sanity check: does the audit find that the shortcut-breaking groups
(waterbird on land, landbird on water) are deferred more, and attribute it to the background?
Source: the WILDS package (`pip install wilds`), which downloads the official release.

Official splits are kept: train -> train, val -> cal, test -> test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from wilds import get_dataset

from refaudit.data.download import save_pil
from refaudit.quality import EXTRA_COLS, QUALITY_COLS

SPLITS = {"train": "train", "val": "cal", "test": "test"}
BIRD = {0: "landbird", 1: "waterbird"}
PLACE = {0: "land", 1: "water"}


def build_manifest(root: str | Path, limit: int | None = None, seed: int = 0) -> pd.DataFrame:
    root = Path(root)
    ds = get_dataset("waterbirds", download=True, root_dir=str(root / "wilds"))
    meta = ds.metadata_array.numpy()  # columns: background, y, from_source_domain
    rows = []
    for wsplit, ours in SPLITS.items():
        sub = ds.get_subset(wsplit)
        idx = np.arange(len(sub))
        if limit:
            idx = np.random.default_rng(seed).choice(len(sub), size=min(limit, len(sub)),
                                                     replace=False)
        for i in tqdm(idx, desc=f"waterbirds {wsplit}"):
            img, y, _ = sub[int(i)]
            gi = int(sub.indices[int(i)])
            bg = int(meta[gi, 0])
            dest = root / "images" / f"{gi}.jpg"
            q = save_pil(img, dest)
            rows.append({
                "image_path": str(dest),
                "label": BIRD[int(y)],
                "group": f"{BIRD[int(y)]}_on_{PLACE[bg]}",
                "tone_raw": np.nan,
                "dataset": "waterbirds",
                "source_id": str(gi),
                "split": ours,
                **{c: q[c] for c in QUALITY_COLS + EXTRA_COLS},
            })
    out = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    out.to_csv(root / "manifest.csv", index=False)
    return out
