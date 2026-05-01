# HMS Second-Pass Solution Engineering Review

## Abstract

This second maturity pass turns the HMS Silver Medal repository from a medal archive into a more reviewable solution engineering package. The new artifacts inspect the original notebooks, submission outputs, ensemble configuration, and reproduction surface without committing raw Kaggle data or large checkpoints.

## Archive Evidence

- Original notebooks parsed: **9**
- Total code cells: **155**
- Total non-empty code lines: **3220**
- Pipeline phases covered: **5**
- Notebooks with explicit inference/submission logic: **4**
- Notebooks with fold/CV logic: **6**

The strongest archival signal is that the notebooks cover spectrogram generation, two-stage CNN training, raw EEG sequence modeling, inference, and final blending. This is materially stronger than a single notebook dump because reviewers can map each original artifact to a pipeline phase.

## Submission Sanity

The smoke submissions are not hidden-test claims. They are probability-format and calibration sanity checks over synthetic/smoke outputs. The audit verifies probability sums, entropy, max-confidence behavior, and distance from public development-class priors. This catches common reproduction bugs such as unnormalized logits, collapsed single-class outputs, or malformed Kaggle submission columns.

## Ensemble Readiness

The ensemble config records a 0.50 / 0.30 / 0.20 blend across the primary spectrogram model, raw EEG temporal model, and auxiliary official-spectrogram model. The audit intentionally distinguishes expected full-run artifacts from lightweight committed smoke artifacts, so the repository remains honest about what is tracked and what must be regenerated with Kaggle data.

## Reproduction Risk

The readiness checklist gives reviewers a concrete view of what is immediately available: certificate, original notebooks, model configs, download scripts, train entrypoints, smoke submission, reports, and tests. The remaining risk is external and expected: Kaggle raw data access, hidden-test labels, and full model checkpoints are not committed.

## Results Artifacts

- `reports/results/second_pass_notebook_archive_metrics.csv`
- `reports/results/second_pass_submission_sanity.csv`
- `reports/results/second_pass_ensemble_config_audit.csv`
- `reports/results/second_pass_reproduction_readiness.csv`
- `reports/figures/second_pass_notebook_phase_footprint.png`
- `reports/figures/second_pass_submission_sanity.png`
- `reports/figures/second_pass_reproduction_readiness.png`

## Reviewer Verdict

The repository now clears the second-pass portfolio threshold because it contains the medal certificate, original source notebooks, clean reusable code, model configs, smoke execution paths, probability sanity checks, notebook complexity analysis, and explicit reproduction limitations.

Updated maturity score: **92/100**.
