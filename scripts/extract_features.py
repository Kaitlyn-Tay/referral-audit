"""Extract frozen-backbone features for all images in a manifest, for every variant.

  python scripts/extract_features.py --manifest data/fitzpatrick17k/manifest.csv \
      --out features/f17k_dinov2b.npz

GPU strongly recommended for the full datasets (Kaggle: Settings > Accelerator > GPU).
Use --model vit_small_patch14_dinov2.lvd142m for a faster, smaller backbone.
"""

from __future__ import annotations

import argparse

import pandas as pd

from refaudit.features import DEFAULT_MODEL, extract, variant_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n-tta", type=int, default=5)
    ap.add_argument("--weights-file", default=None, help="local weights instead of the HF Hub")
    ap.add_argument("--factors", nargs="*", default=None, help="subset of capture factors")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    variants = variant_names(args.factors, n_tta=args.n_tta)
    print(f"{len(df)} images x {len(variants)} variants with {args.model}")
    extract(df["image_path"].tolist(), args.out, model_name=args.model, variants=variants,
            batch_size=args.batch_size, num_workers=args.workers,
            weights_file=args.weights_file)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
