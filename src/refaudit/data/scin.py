"""SCIN (Ward et al. 2024, Google): crowdsourced PHONE photos with dermatologist condition labels,
Fitzpatrick labels and Monk Skin Tone labels. Public Google Cloud bucket, no sign-up.
Source: https://github.com/google-research-datasets/scin (SCIN Data Use License).

Task: the top-weighted dermatologist condition, restricted to the `top_k` most common conditions.
Group: Monk Skin Tone (US raters) binned into light / medium / dark; falls back to the
dermatologist Fitzpatrick label when Monk is missing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from refaudit.data.download import fetch_many
from refaudit.groups import fst_group, mst_group

BUCKET = "https://storage.googleapis.com/dx-scin-public-data"
CASES_URL = f"{BUCKET}/dataset/scin_cases.csv"
LABELS_URL = f"{BUCKET}/dataset/scin_labels.csv"


def _top_condition(weighted: str) -> str | None:
    if not isinstance(weighted, str) or not weighted:
        return None
    d = ast.literal_eval(weighted)
    return max(d, key=d.get) if d else None


def _fst_num(v) -> float:
    if isinstance(v, str) and v.startswith("FST"):
        return float(v[3:])
    return float("nan")


def load_metadata(top_k: int = 8, all_images: bool = False) -> pd.DataFrame:
    cases = pd.read_csv(CASES_URL, dtype={"case_id": str})
    labels = pd.read_csv(LABELS_URL, dtype={"case_id": str})
    df = cases.merge(labels, on="case_id")
    df["label"] = df["weighted_skin_condition_label"].map(_top_condition)
    df = df[df["label"].notna()]
    keep = df["label"].value_counts().index[:top_k]
    df = df[df["label"].isin(keep)].copy()

    df["mst"] = pd.to_numeric(df["monk_skin_tone_label_us"], errors="coerce")
    df["fst"] = df["dermatologist_fitzpatrick_skin_type_label_1"].map(_fst_num)
    df["group"] = [
        mst_group(m) if pd.notna(m) else fst_group(f) for m, f in zip(df["mst"], df["fst"],
                                                                       strict=True)
    ]
    cols = ["image_1_path", "image_2_path", "image_3_path"] if all_images else ["image_1_path"]
    rows = []
    for _, r in df.iterrows():
        for c in cols:
            p = r[c]
            if isinstance(p, str) and p:
                rows.append({
                    "case_id": r["case_id"], "rel_path": p, "label": r["label"],
                    "mst": r["mst"], "fst": r["fst"], "group": r["group"],
                    "shot_type": r.get(c.replace("_path", "_shot_type")),
                })
    return pd.DataFrame(rows)


def build_manifest(
    root: str | Path, top_k: int = 8, all_images: bool = False, limit: int | None = None,
    workers: int = 16,
) -> pd.DataFrame:
    root = Path(root)
    df = load_metadata(top_k=top_k, all_images=all_images)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0)
    df["image_path"] = [str(root / "images" / (Path(p).stem + ".jpg")) for p in df["rel_path"]]
    jobs = [(f"{BUCKET}/{p}", Path(d)) for p, d in zip(df["rel_path"], df["image_path"],
                                                        strict=True)]
    ok = fetch_many(jobs, workers=workers, desc="scin")
    df = df[[Path(p) in ok for p in df["image_path"]]]
    out = pd.DataFrame({
        "image_path": df["image_path"],
        "label": df["label"],
        "tone_raw": df["mst"],
        "group": df["group"],
        "dataset": "scin",
        "source_id": df["case_id"],  # split by case so photos of one person stay together
        "shot_type": df["shot_type"],
    })
    root.mkdir(parents=True, exist_ok=True)
    out.to_csv(root / "manifest.csv", index=False)
    return out
