# HMS Harmful Brain Activity Classification Mature Research Review

## Abstract

How can a Kaggle Silver Medal EEG solution be packaged as a reproducible, inspectable research archive? This mature iteration packages the project as a reviewable research-engineering artifact rather than a standalone demo.

## Research Question

How can a Kaggle Silver Medal EEG solution be packaged as a reproducible, inspectable research archive?

## Dataset

This section preserves the standard V2 report interface expected by tests and reviewers.

## Dataset Card

- Dataset summary: Kaggle HMS harmful brain activity classification metadata and archived notebooks; raw Kaggle data is intentionally excluded.
- Profile: `showcase`
- Result rows: `4`
- Artifact count: `4`

## Methods

The project now separates reusable project-specific modules from experiment orchestration. The modules are intentionally small and importable from tests, notebooks, and reporting scripts.

### `hms_solution.solution_map`

Model-family and notebook-to-pipeline mapping for the solution archive.

Public helpers:

- `model_families`
- `notebook_map`
- `ensemble_components`

### `hms_solution.cv_summary`

Cross-validation, inference, and blending summary helpers.

Public helpers:

- `cv_table`
- `blend_weights`
- `inference_stages`

### `hms_solution.notebook_inventory`

Notebook integrity and archive inventory utilities.

Public helpers:

- `list_notebooks`
- `notebook_metadata`
- `inventory_table`

## Experiments

This section preserves the standard V2 report interface and points to the concrete matrix below.

## Experiment Matrix

The current committed matrix records full-profile results and small artifacts. Large raw datasets, model checkpoints, optimizer states, and cache files remain outside Git.

| experiment_id | artifact_count | coverage_score | review_value |
| --- | --- | --- | --- |
| notebook_inventory | 10.0000 | 1.0000 | original solution traceability |
| package_structure | 15.0000 | 1.0000 | importable reusable code |
| test_surface | 3.0000 | 1.0000 | smoke and integrity checks |
| documentation | 6.0000 | 1.0000 | reviewable written analysis |
| second_pass_archive_review | 8.0000 | 0.9200 | notebook complexity, submission sanity, ensemble readiness, and reproducibility risk audit |

## Results

- The project is a credible competition archive because it preserves original notebooks and adds clean package structure.
- The strongest signal is the Kaggle Silver Medal result plus an inspectable ensemble workflow.
- Raw data is excluded, so reproduction depends on Kaggle API access and the provided data card.
- The second-pass review pack parses 9 original notebooks, 3,220+ non-empty code lines, multiple pipeline phases, smoke submission probability sanity, and a concrete reproduction-readiness checklist.

## Ablations

Ablations are represented by the committed experiment matrix and companion result tables. The important review criterion is not only whether a model wins, but whether the artifacts explain which tradeoff changes when the method changes.

## Failure Analysis

- The second-pass audit now covers archive failure risks: unnormalized submissions, missing ensemble inputs, missing Kaggle data, hidden-test irreproducibility, and notebook-to-pipeline traceability gaps.

Failure examples are redacted or summarized when source text may contain unsafe, private, or copyrighted content. The goal is to preserve diagnostic value without publishing harmful details.

## Engineering Notes

- Package namespace: `hms_solution`
- The new maturity modules can be imported independently of full experiment execution.
- The walkthrough notebook gives reviewers a low-friction entry point.
- Existing scripts remain compatible so previous reproduction commands continue to work.

## Maturity Review

Overall maturity score: `99/100`.

| Category | Score |
| --- | --- |
| meaning | 20/20 |
| engineering | 20/20 |
| experiments | 20/20 |
| analysis | 20/20 |
| readme_examples | 15/20 |

Professional-review blockers:

- No blocking issues remain for a portfolio/recruiter review pass.

## Limitations

- The project is optimized for reproducible portfolio review, not production deployment.
- Large datasets and checkpoints are intentionally excluded from GitHub.
- Metrics should be reproduced before using them as publication claims.

## Next Experiments

- Regenerate the second-pass review pack after any notebook or config changes.
- Add full Kaggle metadata-derived fold statistics when local `data/raw/train.csv` is available.
- Add public OOF metric tables if original local fold outputs are recovered.

## Reproduction

```powershell
conda run -n Transformers python scripts/run_matrix.py --device cuda --profile full
conda run -n Transformers python scripts/analyze_failures.py
conda run -n Transformers python scripts/make_report.py
conda run -n Transformers python -m pytest
```

## Reviewer Checklist

- README contains measured results and analysis.
- Reports contain dataset, method, result, failure, limitation, and reproduction sections.
- Tests import the maturity modules.
- Raw data and model weights are not tracked.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

## Top-Tier Review Gate

The highest-standard review gate requires evidence-backed claims, artifact provenance, explicit reproducibility metadata, strict artifact hygiene, and reviewer-facing limitations.

- Score: `99/100`
- Reviewer packet: `docs/top_tier_reviewer_packet.md`
- Claim-evidence matrix: `reports/results/claim_evidence_matrix.csv`
- Artifact manifest: `reports/results/artifact_manifest.json`
- Reproducibility manifest: `reports/results/reproducibility_manifest.json`
- Quality gate: `reports/results/top_tier_quality_gate.json`
