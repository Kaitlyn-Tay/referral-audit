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

from refaudit.groups import GROUP_ORDER  # noqa: E402
from refaudit.pipeline import run_audit  # noqa: E402

COLORS = {"light": "#E8B58F", "medium": "#A86B45", "dark": "#5A3825"}


def plot_referral(group_table: pd.DataFrame, scores: list[str], path: Path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    w = 0.8 / len(GROUP_ORDER)
    x = np.arange(len(scores))
    for j, g in enumerate(GROUP_ORDER):
        vals = [group_table.query("score == @s and group == @g")["referral_rate"].mean()
                for s in scores]
        ax.bar(x + (j - 1) * w, vals, w, label=g, color=COLORS[g])
    ax.set_xticks(x, scores, rotation=30, ha="right")
    ax.set_ylabel("referral rate (test)")
    ax.legend(title="skin tone", frameon=False)
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
        ax.bar(x + (j - 1) * w, row.values, w, label=g, color=COLORS.get(g, "grey"))
    ax.set_xticks(x, cols, rotation=30, ha="right")
    ax.set_ylabel("mean instability (KL)")
    ax.legend(title="skin tone", frameon=False)
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
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    with np.load(args.features) as f:
        feats = {k: f[k] for k in f.files}
    res = run_audit(manifest, feats, coverage=args.coverage, seed=args.seed, n_boot=args.n_boot)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.scores_table.to_csv(out / "scores.csv", index=False)
    res.group_table.to_csv(out / "groups.csv", index=False)
    res.factor_table.to_csv(out / "factors.csv")
    res.tone_table.to_csv(out / "tone_levels.csv", index=False)
    res.test_scores.to_csv(out / "test_scores.csv", index=False)
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
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
