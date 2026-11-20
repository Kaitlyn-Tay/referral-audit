"""Counterfactual instability: how much a prediction moves under each capture-condition change.

For each image x and factor a with severities s = 1..S, given the model's probabilities on the
original image p(x) and on the changed images p(T_a,s(x)), we compute per-factor scores:

- kl:        mean over s of KL(p(x) || p(T_a,s(x)))
- flip:      fraction of severities where the predicted class changes
- conf_drop: mean over s of max(0, p(x)[y_hat] - p(T_a,s(x))[y_hat]), where y_hat is the original
             prediction

The per-factor scores form a vector (one entry per factor) that says WHICH capture condition the
model is sensitive to. A combined score is the mean over factors, or a learned weighting.

Convention in this module: higher instability = LESS trustworthy. To use as a confidence score in
selective.py (higher = more confident), negate it.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-12


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise KL(p || q). p, q: (..., k)."""
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(axis=-1)


def factor_scores(
    p_orig: np.ndarray, p_cf: np.ndarray, metric: str = "kl"
) -> np.ndarray:
    """Instability for one factor.

    p_orig: (n, k) probabilities on original images.
    p_cf:   (s, n, k) probabilities on counterfactuals, one slice per severity.
    Returns (n,) scores.
    """
    if p_cf.ndim != 3 or p_cf.shape[1:] != p_orig.shape:
        raise ValueError(f"p_cf must be (s, n, k) matching p_orig {p_orig.shape}, got {p_cf.shape}")
    if metric == "kl":
        return kl_divergence(p_orig[None], p_cf).mean(axis=0)
    yhat = p_orig.argmax(axis=1)
    if metric == "flip":
        return (p_cf.argmax(axis=2) != yhat[None]).mean(axis=0)
    if metric == "conf_drop":
        idx = np.arange(len(yhat))
        drop = p_orig[idx, yhat][None] - p_cf[:, idx, yhat]
        return np.maximum(drop, 0.0).mean(axis=0)
    raise ValueError(f"unknown metric {metric!r}")


def instability_matrix(
    p_orig: np.ndarray, p_cf_by_factor: dict[str, np.ndarray], metric: str = "kl"
) -> tuple[np.ndarray, list[str]]:
    """Stack per-factor scores into an (n, n_factors) matrix. Returns (matrix, factor names)."""
    names = list(p_cf_by_factor)
    cols = [factor_scores(p_orig, p_cf_by_factor[f], metric) for f in names]
    return np.stack(cols, axis=1), names


def combined_score(matrix: np.ndarray) -> np.ndarray:
    """Unweighted combination: mean of per-factor scores after rank-normalising each column."""
    n = matrix.shape[0]
    ranks = np.argsort(np.argsort(matrix, axis=0), axis=0) / max(n - 1, 1)
    return ranks.mean(axis=1)


class LearnedCombiner:
    """Logistic regression from [per-factor instability, optional confidence] to P(error).

    Fit on a calibration split with known correctness; apply to test. Its output, negated, is a
    confidence score for selective classification.
    """

    def __init__(self, C: float = 1.0):
        self.model = LogisticRegression(C=C, max_iter=1000)

    @staticmethod
    def _features(matrix: np.ndarray, confidence: np.ndarray | None) -> np.ndarray:
        feats = [np.log1p(np.maximum(matrix, 0.0))]
        if confidence is not None:
            feats.append(confidence[:, None])
        return np.concatenate(feats, axis=1)

    def fit(self, matrix: np.ndarray, correct: np.ndarray, confidence: np.ndarray | None = None):
        self.model.fit(self._features(matrix, confidence), (~correct.astype(bool)).astype(int))
        return self

    def error_prob(self, matrix: np.ndarray, confidence: np.ndarray | None = None) -> np.ndarray:
        return self.model.predict_proba(self._features(matrix, confidence))[:, 1]

    def confidence(self, matrix: np.ndarray, confidence: np.ndarray | None = None) -> np.ndarray:
        return -self.error_prob(matrix, confidence)
