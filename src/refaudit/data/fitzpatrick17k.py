"""Fitzpatrick17k (Groh et al. 2021): ~16.5k clinical images with Fitzpatrick skin type labels.

Images are hosted on third-party atlas sites, so some links are dead. We download what we can and
record what was lost. Labels: `three_partition_label` (benign / malignant / non-neoplastic) is the
default task; `nine_partition_label` is available for a harder task.
Source: https://github.com/mattgroh/fitzpatrick17k (CC BY-NC-SA 3.0, non-commercial).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from refaudit.data.download import fetch_many
from refaudit.groups import fst_group
from refaudit.quality import EXTRA_COLS, QUALITY_COLS

CSV_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"


def load_metadata(csv_path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path or CSV_URL)
    df = df[df["fitzpatrick_scale"].between(1, 6)].copy()  # -1 = unknown
    df = df.drop_duplicates("md5hash")
    return df


def build_manifest(
    root: str | Path,
    label_col: str = "three_partition_label",
    limit: int | None = None,
    workers: int = 16,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download images into root/images and write root/manifest.csv. Re-running resumes."""
    root = Path(root)
    df = load_metadata(csv_path)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0)
    df["image_path"] = [str(root / "images" / f"{h}.jpg") for h in df["md5hash"]]
    jobs = [(u, Path(p)) for u, p in zip(df["url"], df["image_path"], strict=True)
            if isinstance(u, str)]
    ok = fetch_many(jobs, workers=workers, desc="fitzpatrick17k")
    df["downloaded"] = [Path(p) in ok for p in df["image_path"]]
    lost = (~df["downloaded"]).sum()
    print(f"downloaded {df['downloaded'].sum()} / {len(df)} images ({lost} dead links)")
    df = df[df["downloaded"]]
    q = pd.DataFrame([ok[Path(p)] for p in df["image_path"]], index=df.index)
    out = pd.DataFrame({
        "image_path": df["image_path"],
        "label": df[label_col],
        "tone_raw": df["fitzpatrick_scale"].astype(int),
        "group": [fst_group(v) for v in df["fitzpatrick_scale"]],
        "dataset": "fitzpatrick17k",
        "source_id": df["md5hash"],
    })
    out[QUALITY_COLS + EXTRA_COLS] = q[QUALITY_COLS + EXTRA_COLS]
    root.mkdir(parents=True, exist_ok=True)
    out.to_csv(root / "manifest.csv", index=False)
    return out
