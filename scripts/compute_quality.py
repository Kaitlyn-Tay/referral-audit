"""Compute baseline photo-quality measures for every image in a manifest (CPU only).

  python scripts/compute_quality.py --manifest data/scin/manifest.csv --out data/scin/quality.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from refaudit.quality import compute_quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    q = compute_quality(df["image_path"].tolist())
    q.to_csv(args.out, index=False)
    print(q.describe().round(3))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
