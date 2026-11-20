"""Near-duplicate removal using backbone features.

Web-scraped datasets (Fitzpatrick17k in particular) contain the same photo several times, often
resized or re-compressed. Duplicates split across train and test inflate accuracy, and duplicates
inside test shrink confidence intervals unfairly. We link images whose feature cosine similarity
is at or above a threshold and keep one image per linked cluster.

Images that share a `source_id` (e.g. several shots of one SCIN case) are never linked here:
they are legitimately different photos and the splitter already keeps them together.
"""

from __future__ import annotations

import numpy as np


def _find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def duplicate_clusters(
    X: np.ndarray, threshold: float = 0.95, source_ids: np.ndarray | None = None,
    chunk: int = 2048,
) -> np.ndarray:
    """Return a cluster id per row; rows with the same id are near-duplicates."""
    X = np.asarray(X, dtype=np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    n = len(X)
    parent = np.arange(n)
    src = None if source_ids is None else np.asarray(source_ids, dtype=object)
    for start in range(0, n, chunk):
        sims = X[start:start + chunk] @ X.T
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows + start, cols, strict=True):
            if c <= r:
                continue
            if src is not None and src[r] == src[c]:
                continue
            a, b = _find(parent, r), _find(parent, c)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return np.array([_find(parent, i) for i in range(n)])


def keep_mask(clusters: np.ndarray, source_ids: np.ndarray | None = None) -> np.ndarray:
    """True for rows to keep: per duplicate cluster, keep every row of the first source id."""
    keep = np.zeros(len(clusters), dtype=bool)
    src = np.arange(len(clusters)) if source_ids is None else np.asarray(source_ids, dtype=object)
    chosen: dict[int, object] = {}
    for i, c in enumerate(clusters):
        if c not in chosen:
            chosen[c] = src[i]
        keep[i] = src[i] == chosen[c]
    return keep
