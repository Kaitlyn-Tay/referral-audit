"""Selective classification: confidence scores, coverage thresholds, and per-group referral audit.

Convention: a HIGHER score means MORE confident. The model classifies a case when its score is
at or above the threshold and refers ("abstains") otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score

# ---------- confidence scores ----------


def msp(probs: np.ndarray) -> np.ndarray:
    """Maximum softmax probability. probs: (n, k)."""
    return probs.max(axis=1)


def neg_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return (p * np.log(p)).sum(axis=1)


def neg_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Negative energy (Liu et al. 2020): T * logsumexp(logits / T). Higher = more confident."""
    return temperature * logsumexp(logits / temperature, axis=1)


def ensemble_probs(member_probs: np.ndarray) -> np.ndarray:
    """Average probabilities over ensemble members. member_probs: (m, n, k)."""
    return member_probs.mean(axis=0)


def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    return softmax(logits, axis=1)


# ---------- thresholds and curves ----------


def threshold_for_coverage(scores: np.ndarray, coverage: float) -> float:
    """Score threshold so that a fraction `coverage` of cases is classified (score >= threshold).

    Fit this on a held-out calibration split, then apply it to test data.
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    return float(np.quantile(scores, 1.0 - coverage, method="lower"))


def risk_coverage_curve(scores: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (coverage, selective risk) when accepting cases in order of decreasing score."""
    order = np.argsort(-scores, kind="stable")
    errs = (~correct.astype(bool))[order].astype(np.float64)
    n = len(scores)
    k = np.arange(1, n + 1)
    return k / n, np.cumsum(errs) / k


def aurc(scores: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage_curve(scores, correct)
    return float(risk.mean())


def error_auroc(scores: np.ndarray, correct: np.ndarray) -> float:
    """AUROC for separating correct from wrong cases using the score (higher = better)."""
    correct = correct.astype(bool)
    if correct.all() or (~correct).all():
        return float("nan")
    return float(roc_auc_score(correct, scores))


# ---------- per-group audit ----------


@dataclass
class GroupAudit:
    group: str
    n: int
    coverage: float  # fraction classified (1 - referral rate)
    referral_rate: float
    selective_accuracy: float  # accuracy on classified cases
    full_accuracy: float  # accuracy if nothing were referred


def audit_groups(
    scores: np.ndarray,
    correct: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    group_order: list[str] | None = None,
) -> list[GroupAudit]:
    """Coverage, referral rate and selective accuracy per group at a fixed threshold."""
    groups = np.asarray(groups, dtype=object)
    correct = correct.astype(bool)
    accepted = scores >= threshold
    names = group_order or sorted({g for g in groups if g is not None})
    out = []
    for g in names:
        m = groups == g
        n = int(m.sum())
        if n == 0:
            continue
        acc_m = accepted[m]
        cov = float(acc_m.mean())
        sel_acc = float(correct[m][acc_m].mean()) if acc_m.any() else float("nan")
        out.append(GroupAudit(g, n, cov, 1 - cov, sel_acc, float(correct[m].mean())))
    return out


def referral_gap(audits: list[GroupAudit], high: str = "dark", low: str = "light") -> float:
    """Referral rate of `high` minus that of `low` (positive = `high` referred more)."""
    by = {a.group: a for a in audits}
    return by[high].referral_rate - by[low].referral_rate


def bootstrap_ci(
    stat_fn,
    n: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    stratify: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap. `stat_fn(idx)` computes the statistic on resampled indices.

    If `stratify` is given, resampling is done within each stratum (e.g. skin-tone group), which
    keeps small groups from vanishing in a resample.
    Returns (point estimate on all data, lower, upper).
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    point = stat_fn(all_idx)
    stats = []
    if stratify is None:
        strata = [all_idx]
    else:
        stratify = np.asarray(stratify, dtype=object)
        strata = [all_idx[stratify == s] for s in dict.fromkeys(stratify)]
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        v = stat_fn(idx)
        if np.isfinite(v):
            stats.append(v)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
