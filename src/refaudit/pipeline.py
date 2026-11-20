"""End-to-end audit on stored features.

Inputs: a manifest with columns label, group, tone_raw, split (train / cal / test), and a dict of
feature arrays keyed by variant name (see features.variant_names), rows aligned with the manifest.

Outputs (all on the TEST split, thresholds fitted on CAL):
- scores_table: error AUROC, AURC, and referral gap (dark - light) with bootstrap CI, per score
- group_table:  coverage / referral rate / selective accuracy per group, per score
- factor_table: mean per-factor instability per group (which capture condition hurts whom)
- tone_table:   per raw tone level, referral rate vs mean instability (label-free prediction check)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from refaudit import selective as sel
from refaudit.counterfactuals import FACTORS
from refaudit.groups import GROUP_ORDER
from refaudit.heads import HeadEnsemble
from refaudit.instability import LearnedCombiner, combined_score, instability_matrix, kl_divergence


@dataclass
class AuditResult:
    scores_table: pd.DataFrame
    group_table: pd.DataFrame
    factor_table: pd.DataFrame
    tone_table: pd.DataFrame
    test_scores: pd.DataFrame  # per-test-image scores, for plotting


def _variants_of(feats: dict[str, np.ndarray], prefix: str) -> list[str]:
    return sorted(k for k in feats if k.split("@")[0] == prefix)


def _cross_fit_combiner(mat_cal, correct_cal, conf_cal, mat_test, conf_test, seed=0):
    """Out-of-fold scores on cal (so its threshold is honest) and full-cal fit for test."""
    oof = np.zeros(len(correct_cal))
    y = (~correct_cal).astype(int)
    n_splits = 5 if min(np.bincount(y, minlength=2)) >= 5 else 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(mat_cal, y):
        c = LearnedCombiner().fit(mat_cal[tr], correct_cal[tr],
                                  None if conf_cal is None else conf_cal[tr])
        oof[te] = c.confidence(mat_cal[te], None if conf_cal is None else conf_cal[te])
    full = LearnedCombiner().fit(mat_cal, correct_cal, conf_cal)
    return oof, full.confidence(mat_test, conf_test)


def run_audit(
    manifest: pd.DataFrame,
    feats: dict[str, np.ndarray],
    coverage: float = 0.8,
    n_members: int = 5,
    seed: int = 0,
    n_boot: int = 1000,
    factors: list[str] | None = None,
) -> AuditResult:
    m = manifest.reset_index(drop=True)
    classes = np.array(sorted(m["label"].unique()))
    y = np.searchsorted(classes, m["label"].to_numpy())
    idx = {s: np.flatnonzero(m["split"].to_numpy() == s) for s in ("train", "cal", "test")}

    heads = HeadEnsemble(n_members=n_members, seed=seed).fit(feats["orig"][idx["train"]],
                                                             y[idx["train"]])
    factors = factors or [f for f in FACTORS if f"{f}@1" in feats]

    def probs(variant, rows):
        return heads.predict_proba(feats[variant][rows].astype(np.float32))

    out_scores: dict[str, dict[str, np.ndarray]] = {"cal": {}, "test": {}}
    correct: dict[str, np.ndarray] = {}
    mats: dict[str, np.ndarray] = {}
    for s in ("cal", "test"):
        rows = idx[s]
        X = feats["orig"][rows].astype(np.float32)
        member_p = heads.member_probs(X)
        p_ens = member_p.mean(axis=0)
        p_single = member_p[0]
        correct[s] = p_ens.argmax(1) == y[rows]
        sc = out_scores[s]
        sc["msp_single"] = sel.msp(p_single)
        sc["msp_ensemble"] = sel.msp(p_ens)
        sc["neg_entropy"] = sel.neg_entropy(p_ens)
        sc["energy"] = sel.neg_energy(heads.member_logits(X)[0])
        sc["ensemble_agreement"] = -kl_divergence(member_p, p_ens[None]).mean(axis=0)
        tta = _variants_of(feats, "tta")
        if tta:
            sc["random_tta"] = -np.mean([kl_divergence(p_ens, probs(v, rows)) for v in tta], axis=0)
        rot = _variants_of(feats, "rot")
        if rot:
            sc["rotation_asrs"] = -np.mean([kl_divergence(p_ens, probs(v, rows)) for v in rot],
                                           axis=0)
        p_cf = {f: np.stack([probs(f"{f}@{k}", rows) for k in range(1, 6)]) for f in factors}
        mats[s], fnames = instability_matrix(p_ens, p_cf, metric="kl")
        sc["cf_unweighted"] = -combined_score(mats[s])

    # learned combiners (cross-fitted on cal)
    oof, te = _cross_fit_combiner(mats["cal"], correct["cal"], None, mats["test"], None, seed)
    out_scores["cal"]["cf_learned"], out_scores["test"]["cf_learned"] = oof, te
    oof, te = _cross_fit_combiner(mats["cal"], correct["cal"], out_scores["cal"]["msp_ensemble"],
                                  mats["test"], out_scores["test"]["msp_ensemble"], seed)
    out_scores["cal"]["cf_learned+msp"], out_scores["test"]["cf_learned+msp"] = oof, te

    test_rows = idx["test"]
    groups_test = m.loc[test_rows, "group"].to_numpy(dtype=object)
    c_test = correct["test"]
    score_rows, group_rows = [], []
    for name, s_test in out_scores["test"].items():
        thr = sel.threshold_for_coverage(out_scores["cal"][name], coverage)
        audits = sel.audit_groups(s_test, c_test, groups_test, thr, GROUP_ORDER)
        have = {a.group for a in audits}

        def gap_stat(ix, s_test=s_test, thr=thr):
            a = sel.audit_groups(s_test[ix], c_test[ix], groups_test[ix], thr, GROUP_ORDER)
            by = {x.group: x for x in a}
            if "dark" not in by or "light" not in by:
                return float("nan")
            return by["dark"].referral_rate - by["light"].referral_rate

        if {"dark", "light"} <= have:
            gap, lo, hi = sel.bootstrap_ci(gap_stat, len(s_test), n_boot=n_boot, seed=seed,
                                           stratify=groups_test)
        else:
            gap = lo = hi = float("nan")
        score_rows.append({
            "score": name,
            "error_auroc": sel.error_auroc(s_test, c_test),
            "aurc": sel.aurc(s_test, c_test),
            "test_coverage": float((s_test >= thr).mean()),
            "selective_acc": float(c_test[s_test >= thr].mean()),
            "referral_gap_dark_minus_light": gap,
            "gap_ci_low": lo,
            "gap_ci_high": hi,
        })
        for a in audits:
            group_rows.append({"score": name, **a.__dict__})

    factor_df = pd.DataFrame(mats["test"], columns=fnames)
    factor_df["group"] = groups_test
    factor_table = factor_df.groupby("group")[fnames].mean().reindex(
        [g for g in GROUP_ORDER if g in set(groups_test)])

    # label-free check: does mean instability per tone level track referral rate per tone level?
    thr = sel.threshold_for_coverage(out_scores["cal"]["msp_ensemble"], coverage)
    tone = m.loc[test_rows, "tone_raw"].to_numpy()
    tdf = pd.DataFrame({
        "tone_raw": tone,
        "referred": out_scores["test"]["msp_ensemble"] < thr,
        "instability": -out_scores["test"]["cf_unweighted"],
        "correct": c_test,
    }).dropna(subset=["tone_raw"])
    tone_table = tdf.groupby("tone_raw").agg(
        n=("referred", "size"), referral_rate=("referred", "mean"),
        mean_instability=("instability", "mean"), accuracy=("correct", "mean")).reset_index()
    if len(tone_table) >= 3:
        rho, p = spearmanr(tone_table["mean_instability"], tone_table["referral_rate"])
        tone_table.attrs["spearman_rho"], tone_table.attrs["spearman_p"] = float(rho), float(p)

    test_scores = pd.DataFrame(out_scores["test"])
    test_scores["group"] = groups_test
    test_scores["correct"] = c_test
    for j, f in enumerate(fnames):
        test_scores[f"inst_{f}"] = mats["test"][:, j]

    return AuditResult(pd.DataFrame(score_rows), pd.DataFrame(group_rows), factor_table,
                       tone_table, test_scores)
