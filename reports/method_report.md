# Method Report

## Competition Objective

The HMS competition asks participants to classify harmful brain activity from EEG data. The output is
a six-way probability distribution over:

- seizure
- LPD, lateralized periodic discharges
- GPD, generalized periodic discharges
- LRDA, lateralized rhythmic delta activity
- GRDA, generalized rhythmic delta activity
- other

The solution archive uses soft vote targets rather than treating each row as a single hard class.
This is important because the annotations reflect expert disagreement and uncertainty.

## Model 1: EfficientNet Over Official and Generated Spectrograms

Model 1 is the strongest individual model family in the archived solution notes. It combines
competition-provided spectrograms with generated EEG spectrograms derived from raw signals. The
generated features use a bipolar montage idea common in EEG review workflows, converting channel
differences into time-frequency images.

Key rationale:

- Spectrograms make seizure-like and periodic patterns visible to 2D CNN backbones.
- Generated spectrograms add signal views not fully captured by the official features.
- Stage-wise training lets the model learn broad spectrogram structure before final specialization.

Original notebooks:

- `notebooks/original/model1/eeg_gen_spec.ipynb`
- `notebooks/original/model1/train_eff0_stage1.ipynb`
- `notebooks/original/model1/train_eff0_stage2.ipynb`
- `notebooks/original/model1/infer.ipynb`

## Model 2: Raw EEG ResNet1D-GRU

Model 2 uses raw EEG waveforms. It is designed to capture temporal waveform patterns that can be
blurred by spectrogram aggregation. The cleaned package implements a compact ResNet1D-GRU path for
smoke testing, while the original notebook preserves the full Kaggle implementation.

Key rationale:

- 1D convolutions capture local waveform motifs.
- GRU layers aggregate longer temporal dependencies.
- The model adds modality diversity to the final blend.

Original notebooks:

- `notebooks/original/model2/hms-resnet1d-gru-train.ipynb`
- `notebooks/original/model2/hms-resnet1d-gru-infer.ipynb`

## Model 3: Official Spectrogram EfficientNet

Model 3 is an auxiliary official-spectrogram-only model. It is less feature-rich than Model 1, but it
adds a more stable and simpler view of the same task.

Original notebooks:

- `notebooks/original/model3/hms-eff1-train-stage1.ipynb`
- `notebooks/original/model3/hms-eff1-train-stage2.ipynb`
- `notebooks/original/model3/hms-infer.ipynb`

## Ensemble

The repository exposes a weighted average blend in `configs/ensemble.yaml`. The default weights are
documented as a reproducible starting point, not as a claim that they exactly recover the hidden
leaderboard submission. Hidden Kaggle test labels are not public, so the only final public outcome
reported here is the certificate-backed medal result.

## Engineering Additions

The cleaned package adds:

- reusable EEG preprocessing and bipolar montage helpers
- NumPy/Pandas metadata and label summary utilities
- PyTorch model constructors for spectrogram and raw EEG paths
- KL-divergence utilities for soft-label evaluation
- Kaggle-style submission and weighted ensemble validation
- smoke tests that run without original raw competition files

## Medical AI Caveat

This is a competition solution archive. It is not a clinical decision system. Any medical deployment
would require external validation, calibration, privacy review, clinician-in-the-loop design, and
prospective safety analysis.

