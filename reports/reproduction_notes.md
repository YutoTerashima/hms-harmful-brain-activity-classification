# Reproduction Notes

## Environment

Recommended:

- Python 3.10+
- PyTorch with CUDA for full training
- `timm` for EfficientNet backbones
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`

Install:

```powershell
python -m pip install -e ".[dev,gpu]"
```

## Kaggle Data

1. Accept the competition rules on Kaggle.
2. Install Kaggle CLI:

```powershell
python -m pip install kaggle
```

3. Configure credentials with `kaggle.json` or environment variables.
4. Download:

```powershell
python scripts/download_data.py --competition hms-harmful-brain-activity-classification --data-dir data/raw
python scripts/prepare_metadata.py --data-dir data/raw --out-dir data/processed
```

## Smoke Validation

Smoke mode validates code paths without original competition data:

```powershell
pytest
python scripts/train_model1.py --config configs/model1_effnet_spectrogram.yaml --device cpu --smoke
python scripts/train_model2.py --config configs/model2_resnet1d_gru.yaml --device cpu --smoke
python scripts/train_model3.py --config configs/model3_effnet_official_spec.yaml --device cpu --smoke
python scripts/run_inference.py --device cpu --smoke
python scripts/blend_submissions.py --config configs/ensemble.yaml --smoke
```

## Full Training

Full training requires local Kaggle data and enough GPU memory:

```powershell
python scripts/train_model1.py --config configs/model1_effnet_spectrogram.yaml --device cuda
python scripts/train_model2.py --config configs/model2_resnet1d_gru.yaml --device cuda
python scripts/train_model3.py --config configs/model3_effnet_official_spec.yaml --device cuda
python scripts/blend_submissions.py --config configs/ensemble.yaml
```

The cleaned scripts are intentionally conservative wrappers. The original high-effort Kaggle
notebooks remain under `notebooks/original/` for exact archival reference.

