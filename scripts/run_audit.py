"""Run the referral audit on stored features and save tables + figures.

  python scripts/run_audit.py --manifest data/fitzpatrick17k/manifest.csv \
      --features features/f17k_dinov2b.npz --out results/f17k --coverage 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from refaudit.pipeline import run_audit  # noqa: E402
from refaudit.quality import QUALITY_COLS  # noqa: E402

TONE_COLORS = {"light": "#E8B58F", "medium": "#A86B45", "dark": "#5A3825"}
OTHER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]


def color_for(group: str, j: int) -> str:
    return TONE_COLORS.get(group, OTHER_COLORS[j % len(OTHER_COLORS)])


def plot_referral(group_table: pd.DataFrame, scores: list[str], path: Path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    groups = list(pd.unique(group_table["group"]))
    w = 0.8 / len(groups)
    x = np.arange(len(scores))
    for j, g in enumerate(groups):
        vals = [group_table.query("score == @s and group == @g")["referral_rate"].mean()
                for s in scores]
        ax.bar(x + (j - (len(groups) - 1) / 2) * w, vals, w, label=g, color=color_for(g, j))
    ax.set_xticks(x, scores, rotation=30, ha="right")
    ax.set_ylabel("referral rate (test)")
    ax.legend(title="group", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_factors(factor_table: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    cols = list(factor_table.columns)
    x = np.arange(len(cols))
    w = 0.8 / len(factor_table)
    for j, (g, row) in enumerate(factor_table.iterrows()):
        ax.bar(x + (j - (len(factor_table) - 1) / 2) * w, row.values, w, label=g,
               color=color_for(g, j))
    ax.set_xticks(x, cols, rotation=30, ha="right")
    ax.set_ylabel("mean instability (KL)")
    ax.legend(title="group", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--quality", default=None,
                    help="quality.csv (default: quality columns already in the manifest)")
    ap.add_argument("--dedup", type=float, default=0.95,
                    help="cosine threshold for near-duplicate removal; 0 disables")
    ap.add_argument("--gap-high", default="dark", help="group expected to be deferred more")
    ap.add_argument("--gap-low", default="light", help="reference group")
    ap.add_argument("--no-fix", action="store_true", help="skip consistency-training fix")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    with np.load(args.features) as f:
        feats = {k: f[k] for k in f.files}
    if args.quality:
        quality = pd.read_csv(args.quality)
    elif set(QUALITY_COLS) <= set(manifest.columns):
        quality = manifest[QUALITY_COLS]
    else:
        quality = None
    res = run_audit(manifest, feats, coverage=args.coverage, seed=args.seed, n_boot=args.n_boot,
                    quality=quality, dedup_threshold=args.dedup or None,
                    consistency_training=not args.no_fix, gap_high=args.gap_high,
                    gap_low=args.gap_low)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.scores_table.to_csv(out / "scores.csv", index=False)
    res.group_table.to_csv(out / "groups.csv", index=False)
    res.factor_table.to_csv(out / "factors.csv")
    res.tone_table.to_csv(out / "tone_levels.csv", index=False)
    res.test_scores.to_csv(out / "test_scores.csv", index=False)
    if res.quality_table is not None:
        res.quality_table.to_csv(out / "quality_by_group.csv")
    pd.Series(res.extras).to_json(out / "extras.json", indent=1)
    plot_referral(res.group_table, list(res.scores_table["score"]), out / "referral_by_group.png")
    plot_factors(res.factor_table, out / "instability_by_factor.png")

    pd.set_option("display.width", 160)
    print(res.scores_table.round(3).to_string(index=False))
    print("\nmean instability per factor and group:\n", res.factor_table.round(4))
    print("\nper tone level:\n", res.tone_table.round(3).to_string(index=False))
    if "spearman_rho" in res.tone_table.attrs:
        print(f"spearman(instability, referral) across tone levels: "
              f"rho={res.tone_table.attrs['spearman_rho']:.2f} "
              f"p={res.tone_table.attrs['spearman_p']:.3f}")
    if res.quality_table is not None:
        print("\nmedian photo quality per group:\n", res.quality_table.round(3))
    print("\nextras:", {k: round(v, 3) if isinstance(v, float) else v
                        for k, v in res.extras.items()})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
