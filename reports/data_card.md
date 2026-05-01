# Data Card

## Dataset

Competition: [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)

The dataset contains EEG-derived information for classifying harmful brain activity in critically ill
patients. Kaggle provides metadata, raw EEG parquet files, and spectrogram parquet files through the
competition data page after the user accepts the competition rules.

## Labels

The six labels used by the competition are:

- seizure
- LPD
- GPD
- LRDA
- GRDA
- other

Training rows include expert vote columns, so models should treat labels as soft probability targets
when optimizing KL divergence.

## Repository Data Policy

This repository does not commit raw Kaggle data, parquet files, generated `.npy` feature tensors, or
model checkpoints. Those files can be large and are controlled by Kaggle competition access terms.

Tracked artifacts are limited to:

- public context tables
- small processed metadata generated from public summaries
- scripts that regenerate local metadata from `train.csv`
- documentation and tests

## Local Data Layout

Expected local layout after download:

```text
data/raw/
  train.csv
  test.csv
  sample_submission.csv
  train_eegs/
  test_eegs/
  train_spectrograms/
  test_spectrograms/
```

Generated local layout:

```text
data/processed/
  label_distribution.csv
  sample_manifest.csv
```

## Known Risks

- The hidden Kaggle test set is not public.
- Public validation can overfit if folds do not account for patient or spectrogram leakage.
- Medical EEG labels are uncertain and may reflect disagreement among expert reviewers.
- Competition performance does not imply clinical readiness.

