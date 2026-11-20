import numpy as np
import pandas as pd
import pytest
from PIL import Image

from refaudit import counterfactuals as cf
from refaudit import selective as sel
from refaudit.data.splits import assign_splits
from refaudit.dedup import duplicate_clusters, keep_mask
from refaudit.groups import (
    fst_group,
    ita_degrees,
    ita_from_image,
    ita_group,
    mst_group,
    skin_mask,
)
from refaudit.heads import HeadEnsemble
from refaudit.instability import (
    LearnedCombiner,
    combined_score,
    factor_scores,
    instability_matrix,
    kl_divergence,
)
from refaudit.pipeline import run_audit
from refaudit.quality import image_quality, matched_referral_gap


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    arr = (rng.random((64, 80, 3)) * 200 + 30).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# ---------- groups ----------

def test_tone_bins():
    assert [fst_group(i) for i in range(1, 7)] == ["light"] * 2 + ["medium"] * 2 + ["dark"] * 2
    assert mst_group(7) == "dark" and mst_group(4) == "medium" and mst_group(None) is None
    assert fst_group(float("nan")) is None


def test_ita_orders_light_above_dark():
    light = np.full((10, 3), [235, 200, 180])
    dark = np.full((10, 3), [90, 60, 45])
    assert ita_degrees(light) > ita_degrees(dark)
    assert ita_group(ita_degrees(light)) == "light"
    assert ita_group(ita_degrees(dark)) == "dark"


# ---------- counterfactuals ----------

@pytest.mark.parametrize("factor", list(cf.FACTORS))
def test_factors_keep_size_and_grow_with_severity(img, factor):
    base = np.asarray(img, dtype=float)
    diffs = []
    for s in cf.SEVERITIES:
        out = cf.apply(img, factor, s)
        assert out.size == img.size and out.mode == "RGB"
        diffs.append(np.abs(np.asarray(out, dtype=float) - base).mean())
    assert diffs[-1] > diffs[0]  # stronger severity changes the image more


def test_white_balance_direction(img):
    warm = np.asarray(cf.apply(img, "warm_wb", 5), dtype=float).mean((0, 1))
    cool = np.asarray(cf.apply(img, "cool_wb", 5), dtype=float).mean((0, 1))
    assert warm[0] - warm[2] > cool[0] - cool[2]


def test_background_swap_keeps_masked_pixels(img):
    mask = np.zeros((64, 80), dtype=bool)
    mask[20:40, 30:50] = True
    out = np.asarray(cf.background_swap(img, 5, mask))
    assert np.array_equal(out[mask], np.asarray(img)[mask])
    assert not np.array_equal(out[~mask], np.asarray(img)[~mask])


def test_bad_inputs(img):
    with pytest.raises(ValueError):
        cf.apply(img, "blur", 9)
    with pytest.raises(KeyError):
        cf.apply(img, "nope", 1)


# ---------- selective ----------

def test_threshold_hits_coverage():
    s = np.random.default_rng(0).random(1000)
    thr = sel.threshold_for_coverage(s, 0.8)
    assert abs((s >= thr).mean() - 0.8) < 0.01


def test_aurc_perfect_vs_random():
    rng = np.random.default_rng(0)
    correct = rng.random(500) < 0.7
    perfect = correct.astype(float) + rng.random(500) * 0.01
    random = rng.random(500)
    assert sel.aurc(perfect, correct) < sel.aurc(random, correct)
    assert sel.error_auroc(perfect, correct) > 0.99


def test_audit_groups_detects_gap():
    rng = np.random.default_rng(0)
    groups = np.array(["light"] * 500 + ["dark"] * 500, dtype=object)
    scores = np.concatenate([rng.normal(1, 1, 500), rng.normal(0, 1, 500)])
    correct = np.ones(1000, dtype=bool)
    thr = sel.threshold_for_coverage(scores, 0.8)
    audits = sel.audit_groups(scores, correct, groups, thr, ["light", "dark"])
    assert sel.referral_gap(audits) > 0.1


def test_bootstrap_ci_contains_point():
    x = np.random.default_rng(0).normal(size=300)
    point, lo, hi = sel.bootstrap_ci(lambda ix: x[ix].mean(), len(x), n_boot=300)
    assert lo <= point <= hi


# ---------- instability ----------

def test_kl_zero_for_same():
    p = np.array([[0.7, 0.3], [0.5, 0.5]])
    assert np.allclose(kl_divergence(p, p), 0)


def test_factor_scores_rank_unstable_higher():
    p = np.array([[0.9, 0.1], [0.9, 0.1]])
    cf_p = np.stack([np.array([[0.9, 0.1], [0.2, 0.8]])] * 3)  # image 1 flips
    for metric in ("kl", "flip", "conf_drop"):
        s = factor_scores(p, cf_p, metric)
        assert s[1] > s[0]


def test_matrix_and_combiners():
    rng = np.random.default_rng(0)
    n = 400
    p = rng.dirichlet([1, 1, 1], size=n)
    unstable = rng.random(n) < 0.3
    cfs = {}
    for f in ("a", "b"):
        q = np.stack([np.where(unstable[:, None], rng.dirichlet([1, 1, 1], size=n), p)
                      for _ in range(5)])
        cfs[f] = q
    mat, names = instability_matrix(p, cfs)
    assert mat.shape == (n, 2) and names == ["a", "b"]
    comb = combined_score(mat)
    assert comb[unstable].mean() > comb[~unstable].mean()
    correct = ~unstable
    lc = LearnedCombiner().fit(mat, correct)
    assert sel.error_auroc(lc.confidence(mat), correct) > 0.9


# ---------- heads, splits, pipeline ----------

def _synthetic(n=600, d=16, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    group = rng.choice(["light", "medium", "dark"], n)
    centers = rng.normal(size=(3, d)) * 2
    noise = np.where(group == "dark", 2.5, 1.0)[:, None]  # dark images noisier -> more referral
    X = centers[y] + rng.normal(size=(n, d)) * noise
    manifest = pd.DataFrame({
        "image_path": [f"img{i}.jpg" for i in range(n)],
        "label": np.array(["a", "b", "c"])[y],
        "group": group,
        "tone_raw": np.select([group == "light", group == "medium"], [2, 4], 6),
        "source_id": np.arange(n),
    })
    feats = {"orig": X.astype(np.float16)}
    for f in ("underexpose", "blur"):
        for s in range(1, 6):
            feats[f"{f}@{s}"] = (X + rng.normal(size=X.shape) * 0.2 * s * noise).astype(np.float16)
    for i in range(3):
        feats[f"tta@{i}"] = (X + rng.normal(size=X.shape) * 0.3).astype(np.float16)
    for r in (-15, 15):
        feats[f"rot@{r}"] = (X + rng.normal(size=X.shape) * 0.3).astype(np.float16)
    return manifest, feats


def test_splits_keep_ids_together():
    df = pd.DataFrame({"source_id": [1, 1, 2, 2, 3, 3] * 20,
                       "label": ["x"] * 120, "group": ["light"] * 120})
    df["source_id"] = df["source_id"] + np.repeat(np.arange(20), 6) * 10
    out = assign_splits(df)
    assert out.groupby("source_id")["split"].nunique().max() == 1
    assert set(out["split"]) == {"train", "cal", "test"}


def test_head_ensemble_shapes():
    manifest, feats = _synthetic()
    y = pd.factorize(manifest["label"], sort=True)[0]
    h = HeadEnsemble(n_members=3).fit(feats["orig"].astype(float), y)
    p = h.predict_proba(feats["orig"].astype(float))
    assert p.shape == (600, 3) and np.allclose(p.sum(1), 1)
    assert h.member_logits(feats["orig"].astype(float)).shape == (3, 600, 3)


def test_pipeline_end_to_end_finds_gap():
    manifest, feats = _synthetic()
    manifest = assign_splits(manifest)
    res = run_audit(manifest, feats, coverage=0.8, n_members=3, n_boot=100,
                    dedup_threshold=None)
    t = res.scores_table.set_index("score")
    assert {"msp_ensemble", "random_tta", "rotation_asrs", "cf_unweighted", "cf_learned",
            "cf_learned+msp"} <= set(t.index)
    assert t.loc["msp_ensemble", "referral_gap_dark_minus_light"] > 0  # planted gap is found
    assert list(res.factor_table.columns) == ["underexpose", "blur"]
    assert res.factor_table.loc["dark"].mean() > res.factor_table.loc["light"].mean()
    assert len(res.test_scores) == (manifest["split"] == "test").sum()
    assert "FIX_consistency_heads" in set(t.index)
    assert set(res.extras["gap_after_matching_on_factor"]) == {"underexpose", "blur"}


def test_skin_mask_and_ita_from_image():
    skin = np.full((20, 20, 3), [224, 172, 140], dtype=np.uint8)
    blue = np.full((20, 20, 3), [30, 60, 200], dtype=np.uint8)
    assert skin_mask(skin).all() and not skin_mask(blue).any()
    assert np.isnan(ita_from_image(blue))
    assert ita_from_image(skin) > 10


# ---------- quality and dedup ----------

def test_quality_detects_blur_and_jpeg(img, tmp_path):
    sharp = image_quality(img)
    blurred = image_quality(cf.apply(img, "blur", 5))
    assert blurred["sharpness"] < sharp["sharpness"]
    p = tmp_path / "x.jpg"
    img.save(p, quality=20)
    assert image_quality(p)["jpeg_quality"] < 40
    assert np.isnan(sharp["jpeg_quality"])  # in-memory image has no JPEG tables


def test_matched_gap_removes_quality_confound():
    rng = np.random.default_rng(0)
    n = 4000
    groups = rng.choice(["light", "dark"], n)
    # dark photos are blurrier; referral depends ONLY on blur
    sharp = rng.normal(np.where(groups == "dark", -1.0, 1.0), 1.0)
    referred = rng.random(n) < 1 / (1 + np.exp(3 * sharp))
    q = pd.DataFrame({"sharpness": sharp, "brightness": rng.random(n)})
    raw, matched, used = matched_referral_gap(referred, groups, q, n_bins=8)
    assert raw > 0.2
    assert abs(matched) < raw / 2 and used > 0


def test_dedup_links_copies_but_not_same_source():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 32))
    X[1] = X[0] + 1e-3  # copy of 0, different source -> drop one
    X[3] = X[2] + 1e-3  # copy of 2, same source -> keep both
    src = np.array(["a", "b", "c", "c", "d", "e"], dtype=object)
    cl = duplicate_clusters(X, 0.99, src)
    assert cl[0] == cl[1] and cl[2] != cl[3]
    assert keep_mask(cl, src).tolist() == [True, False, True, True, True, True]
