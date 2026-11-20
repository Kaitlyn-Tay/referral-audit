# referral-audit

**Who gets referred, and why?** Auditing selective skin-image classifiers with capture-condition
counterfactuals.

A selective classifier answers only when it is confident and refers the rest to a doctor. Even
if it is accurate on the cases it answers, it can refer some patients much more often than others.
This repo measures that referral burden by skin tone and tests whether **factor-by-factor
counterfactual instability** (how much a prediction moves when only the lighting, white balance,
blur, compression or background changes) predicts and explains it, without needing skin-tone
labels at test time.

## Research questions

1. **Audit:** at a fixed overall coverage, is the referral rate higher for darker skin, and is
   accuracy on the answered cases still equal?
2. **Cause:** which capture condition (lighting, white balance, blur, compression, resolution,
   background) moves predictions most, and does that differ by skin tone?
3. **Predictor:** does a factor-wise instability score predict errors and group-level referral
   gaps better than softmax confidence, ensembles, random test-time augmentation and rotation
   sensitivity (ASRS, arXiv 2510.01683)?
4. **Fix (later):** does counterfactual-consistency training or an instability-aware referral rule
   shrink the gap at little cost in accuracy?

## Data (all open, no sign-up)

| Dataset | Images | Tone labels | Notes |
|---|---|---|---|
| [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) | ~16.5k clinical | Fitzpatrick 1-6 | Images hosted on atlas sites; some links dead. CC BY-NC-SA 3.0 |
| [SCIN](https://github.com/google-research-datasets/scin) | ~10k phone photos (5k cases) | Monk 1-10, Fitzpatrick | Public GCS bucket. Darker tones are rare (Monk 7-10 is about 4% of cases) |

Planned: PAD-UFES-20 (phone photos), Waterbirds (background shift sanity check), AZH wounds
(label-free case study using ITA tone measured from pixels, see `refaudit.groups.ita_degrees`).

## Quick start

```bash
pip install -e ".[torch,dev]"
pytest -q

# 1. data (writes data/<name>/manifest.csv with train / cal / test splits)
python scripts/download_data.py scin --root data/scin
python scripts/download_data.py fitzpatrick17k --root data/fitzpatrick17k

# 2. features for the original image and every counterfactual (GPU recommended)
python scripts/extract_features.py --manifest data/scin/manifest.csv --out features/scin_dinov2b.npz

# 3. audit: tables + figures in results/scin
python scripts/run_audit.py --manifest data/scin/manifest.csv \
    --features features/scin_dinov2b.npz --out results/scin --coverage 0.8
```

On Kaggle, open `notebooks/01_kaggle_quickstart.ipynb` (turn on GPU and Internet in Settings).

## How it works

```
image ──► counterfactuals (8 factors x 5 strengths, + random TTA, + rotations)
      ──► frozen backbone (DINOv2 by default) ──► stored features (.npz)
      ──► ensemble of logistic-regression heads (trained on original images only)
      ──► probabilities for every variant
      ──► confidence scores  (MSP, entropy, energy, ensemble agreement, random TTA, rotation)
          instability scores (per-factor KL; unweighted and learned combinations)
      ──► threshold on CAL split for a target coverage ──► referral audit on TEST split
```

Code map:

- `refaudit/counterfactuals.py` capture-condition changes (no change to the lesion itself)
- `refaudit/features.py` backbone feature extraction for all variants
- `refaudit/heads.py` head ensemble on frozen features
- `refaudit/selective.py` scores, coverage thresholds, AURC, per-group audit, bootstrap CIs
- `refaudit/instability.py` per-factor instability and combiners
- `refaudit/pipeline.py` end-to-end audit producing all tables
- `refaudit/groups.py` Fitzpatrick / Monk binning and ITA from pixels

## Outputs (`results/<name>/`)

- `scores.csv` per score: error AUROC, AURC, coverage, selective accuracy, dark minus light
  referral gap with 95% bootstrap CI
- `groups.csv` per score and group: coverage, referral rate, selective and full accuracy
- `factors.csv` mean instability per factor and group (the "cause" table)
- `tone_levels.csv` per tone level: referral rate vs mean instability (+ Spearman rho)
- `quality_by_group.csv` median photo quality per group
- `extras.json` near-duplicates removed; quality-matched gap; gap left after matching on each
  factor's instability (attribution); ITA-from-pixels gap estimate vs true-label gap (label-free
  check); real-vs-synthetic shift agreement on multi-photo SCIN cases
- the `FIX_consistency_heads` row in `scores.csv`: heads retrained on original + mild capture
  changes; compare its referral gap and accuracy with `msp_ensemble`
- `referral_by_group.png`, `instability_by_factor.png`

## Scope and safety

Research code only. Not a medical device and not for diagnosis.

## Licence

Code: MIT. Datasets keep their own licences; this repo never redistributes images.
