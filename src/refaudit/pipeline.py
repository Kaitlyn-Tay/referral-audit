"""End-to-end audit on stored features.

Inputs: a manifest with columns label, group, tone_raw, split (train / cal / test), and a dict of
feature arrays keyed by variant name (see features.variant_names), rows aligned with the manifest.

Outputs (all on the TEST split, thresholds fitted on CAL):
- scores_table: error AUROC, AURC, and referral gap (dark - light) with bootstrap CI, per score
- group_table:  coverage / referral rate / selective accuracy per group, per score
- factor_table: mean per-factor instability per group (which capture condition hurts whom)
- tone_table:   per raw tone level, referral rate vs mean instability (label-free prediction check)
- quality_table: median photo quality per group (if quality measures are given)
- extras:       dedup counts, quality-matched referral gap, real-vs-synthetic shift check
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from refaudit import selective as sel
from refaudit.counterfactuals import FACTORS
from refaudit.dedup import duplicate_clusters, keep_mask
from refaudit.groups import GROUP_ORDER, ita_group
from refaudit.heads import HeadEnsemble
from refaudit.instability import LearnedCombiner, combined_score, instability_matrix, kl_divergence
from refaudit.quality import matched_referral_gap, quality_by_group


@dataclass
class AuditResult:
    scores_table: pd.DataFrame
    group_table: pd.DataFrame
    factor_table: pd.DataFrame
    tone_table: pd.DataFrame
    test_scores: pd.DataFrame  # per-test-image scores, for plotting
    quality_table: pd.DataFrame | None = None
    extras: dict = field(default_factory=dict)


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
    quality: pd.DataFrame | None = None,
    dedup_threshold: float | None = 0.95,
    consistency_training: bool = True,
) -> AuditResult:
    """Run the full audit. `quality` (optional) must be row-aligned with `manifest`."""
    m = manifest.reset_index(drop=True)
    extras: dict = {"n_images": len(m)}
    if quality is not None:
        quality = quality.reset_index(drop=True)
    if dedup_threshold is not None:
        src = m["source_id"].to_numpy() if "source_id" in m else None
        clusters = duplicate_clusters(feats["orig"], dedup_threshold, src)
        keep = keep_mask(clusters, src)
        extras["n_near_duplicates_removed"] = int((~keep).sum())
        m = m[keep].reset_index(drop=True)
        feats = {k: v[keep] for k, v in feats.items()}
        if quality is not None:
            quality = quality[keep].reset_index(drop=True)
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

    def evaluate(name, s_cal, s_test, c_test, groups_test=groups_test):
        thr = sel.threshold_for_coverage(s_cal, coverage)
        audits = sel.audit_groups(s_test, c_test, groups_test, thr, GROUP_ORDER)
        have = {a.group for a in audits}

        def gap_stat(ix):
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
            "full_acc": float(c_test.mean()),
            "test_coverage": float((s_test >= thr).mean()),
            "selective_acc": float(c_test[s_test >= thr].mean()),
            "referral_gap_dark_minus_light": gap,
            "gap_ci_low": lo,
            "gap_ci_high": hi,
        })
        for a in audits:
            group_rows.append({"score": name, **a.__dict__})

    for name, s_test in out_scores["test"].items():
        evaluate(name, out_scores["cal"][name], s_test, c_test)

    # FIX: counterfactual-consistency heads, trained on original + mild/moderate (severity 1-3)
    # capture changes of the training images with the same labels. Evaluated as its own model.
    if consistency_training and factors:
        tr = idx["train"]
        Xs = [feats["orig"][tr]] + [feats[f"{f}@{k}"][tr] for f in factors for k in (1, 2, 3)]
        ys = np.concatenate([y[tr]] * len(Xs))
        fixed = HeadEnsemble(n_members=n_members, seed=seed).fit(
            np.concatenate(Xs).astype(np.float32), ys)
        fixed_scores, fixed_correct = {}, {}
        for s in ("cal", "test"):
            p_fix = fixed.predict_proba(feats["orig"][idx[s]].astype(np.float32))
            fixed_scores[s] = sel.msp(p_fix)
            fixed_correct[s] = p_fix.argmax(1) == y[idx[s]]
        evaluate("FIX_consistency_heads", fixed_scores["cal"], fixed_scores["test"],
                 fixed_correct["test"])

    # ATTRIBUTE: referral gap left after matching dark and light images on each factor's
    # instability. A factor whose matching shrinks the gap most is the biggest driver.
    thr_msp = sel.threshold_for_coverage(out_scores["cal"]["msp_ensemble"], coverage)
    referred_msp = out_scores["test"]["msp_ensemble"] < thr_msp
    if {"dark", "light"} <= set(groups_test):
        inst_df = pd.DataFrame(mats["test"], columns=fnames)
        attribution = {}
        for f in fnames:
            raw, matched, _ = matched_referral_gap(referred_msp, groups_test, inst_df, cols=(f,),
                                                   n_bins=4)
            attribution[f] = matched
        extras["referral_gap_raw_msp"] = float(raw)
        extras["gap_after_matching_on_factor"] = attribution

    # LABEL-FREE: replace tone labels with ITA measured from pixels and compare the gap estimate
    if "ita" in m:
        ita = m.loc[test_rows, "ita"].to_numpy(dtype=float)
        ok = ~np.isnan(ita)
        ita_groups = np.array([ita_group(v) if not np.isnan(v) else None for v in ita],
                              dtype=object)
        extras["ita_coverage_frac"] = float(ok.mean())
        extras["ita_vs_label_group_agreement"] = float(
            (ita_groups[ok] == groups_test[ok]).mean()) if ok.any() else float("nan")
        def _gap(grp):
            d, li = grp == "dark", grp == "light"
            if d.sum() == 0 or li.sum() == 0:
                return float("nan")
            return float(referred_msp[d].mean() - referred_msp[li].mean())
        extras["gap_true_labels"] = _gap(groups_test)
        extras["gap_ita_estimate"] = _gap(ita_groups)
        if ok.sum() >= 10:
            extras["spearman_ita_vs_tone_raw"] = float(spearmanr(
                ita[ok], m.loc[test_rows, "tone_raw"].to_numpy(dtype=float)[ok],
                nan_policy="omit")[0])

    factor_df = pd.DataFrame(mats["test"], columns=fnames)
    factor_df["group"] = groups_test
    factor_table = factor_df.groupby("group")[fnames].mean().reindex(
        [g for g in GROUP_ORDER if g in set(groups_test)])

    # label-free check: does mean instability per tone level track referral rate per tone level?
    thr = thr_msp
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

    # photo quality: by group, and referral gap on quality-matched subsets
    quality_table = None
    if quality is not None:
        q_test = quality.loc[test_rows].reset_index(drop=True)
        quality_table = quality_by_group(q_test, groups_test, GROUP_ORDER)
        if {"dark", "light"} <= set(groups_test):
            raw, matched, n_used = matched_referral_gap(
                out_scores["test"]["msp_ensemble"] < thr, groups_test, q_test)
            extras.update(referral_gap_raw=raw, referral_gap_quality_matched=matched,
                          matched_n_dark=n_used)

    # real vs synthetic: for cases with 2+ photos in test, does the real shift between photos of
    # the same case track our synthetic instability?
    if "source_id" in m:
        src_test = m.loc[test_rows, "source_id"].to_numpy()
        p_test = heads.predict_proba(feats["orig"][test_rows].astype(np.float32))
        raw_inst = mats["test"].mean(axis=1)
        real, synth = [], []
        for sid in pd.unique(src_test):
            ix = np.flatnonzero(src_test == sid)
            if len(ix) < 2:
                continue
            a, b = ix[0], ix[1]
            real.append(0.5 * (kl_divergence(p_test[a], p_test[b])
                               + kl_divergence(p_test[b], p_test[a])))
            synth.append(0.5 * (raw_inst[a] + raw_inst[b]))
        if len(real) >= 10:
            rho, p = spearmanr(synth, real)
            extras.update(real_shift_cases=len(real), real_vs_synth_spearman=float(rho),
                          real_vs_synth_p=float(p), real_shift_median_kl=float(np.median(real)),
                          synth_median_kl=float(np.median(synth)))

    test_scores = pd.DataFrame(out_scores["test"])
    test_scores["group"] = groups_test
    test_scores["correct"] = c_test
    for j, f in enumerate(fnames):
        test_scores[f"inst_{f}"] = mats["test"][:, j]

    return AuditResult(pd.DataFrame(score_rows), pd.DataFrame(group_rows), factor_table,
                       tone_table, test_scores, quality_table, extras)
