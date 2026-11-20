"""Train / calibration / test splits, stratified by label x group, grouped by source id."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_splits(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    id_col: str = "source_id",
) -> pd.DataFrame:
    """Add a `split` column with values train / cal / test.

    Rows sharing `id_col` (e.g. several photos of one SCIN case) always land in the same split.
    Stratification is over (label, group) of the first row of each id.
    """
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    first = df.drop_duplicates(id_col)
    strata = first["label"].astype(str) + "|" + first["group"].astype(str)
    split_of: dict = {}
    for _, ids in first.groupby(strata)[id_col]:
        ids = ids.to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(fractions[0] * n))
        n_cal = int(round(fractions[1] * n))
        for i, sid in enumerate(ids):
            split_of[sid] = "train" if i < n_tr else ("cal" if i < n_tr + n_cal else "test")
    out = df.copy()
    out["split"] = out[id_col].map(split_of)
    return out
