"""Download an open dataset and write data/<name>/manifest.csv with train / cal / test splits.

Examples:
  python scripts/download_data.py fitzpatrick17k --root data/fitzpatrick17k
  python scripts/download_data.py scin --root data/scin
  python scripts/download_data.py fitzpatrick17k --root data/f17k_small --limit 300   # quick test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from refaudit.data import fitzpatrick17k, scin
from refaudit.data.splits import assign_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["fitzpatrick17k", "scin"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only N images, for smoke tests")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-images", action="store_true",
                    help="SCIN: keep all 2-3 photos per case (enables the real-vs-synthetic check)")
    args = ap.parse_args()

    if args.dataset == "fitzpatrick17k":
        df = fitzpatrick17k.build_manifest(args.root, limit=args.limit, workers=args.workers)
    else:
        df = scin.build_manifest(args.root, limit=args.limit, workers=args.workers,
                                all_images=args.all_images)
    df = df[df["group"].notna()]
    df = assign_splits(df, seed=args.seed)
    path = Path(args.root) / "manifest.csv"
    df.to_csv(path, index=False)
    print(df.groupby(["split", "group"]).size().unstack(fill_value=0))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
