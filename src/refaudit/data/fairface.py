"""FairFace (Karkkainen & Joo 2021): ~108k face images balanced across 7 race groups, with age
and gender labels. Used here as a NON-MEDICAL domain for the same deferral audit.
Source: Hugging Face `HuggingFaceM4/FairFace` (CC BY 4.0). Needs `pip install datasets`.

Task: age group (9 classes; hard, so the model defers often). Group: the race label.
Race labels are perceived categories, not skin tone; ITA from pixels is recorded alongside.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from refaudit.data.download import save_pil
from refaudit.quality import EXTRA_COLS, QUALITY_COLS

HF_NAME = "HuggingFaceM4/FairFace"


def build_manifest(root: str | Path, limit: int | None = 20000, config: str = "1.25",
                   seed: int = 0) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for hf_split in ("train", "validation"):
        ds = load_dataset(HF_NAME, config, split=hf_split)
        age_names = ds.features["age"].names
        race_names = ds.features["race"].names
        n = len(ds)
        idx = np.arange(n)
        if limit:
            share = limit if hf_split == "train" else max(1, limit // 8)
            idx = np.random.default_rng(seed).choice(n, size=min(share, n), replace=False)
        for i in tqdm(idx, desc=f"fairface {hf_split}"):
            ex = ds[int(i)]
            dest = root / "images" / f"{hf_split}_{i}.jpg"
            q = save_pil(ex["image"], dest)
            rows.append({
                "image_path": str(dest),
                "label": age_names[ex["age"]],
                "group": race_names[ex["race"]],
                "tone_raw": np.nan,
                "dataset": "fairface",
                "source_id": f"{hf_split}_{i}",
                **{c: q[c] for c in QUALITY_COLS + EXTRA_COLS},
            })
    out = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    out.to_csv(root / "manifest.csv", index=False)
    return out
