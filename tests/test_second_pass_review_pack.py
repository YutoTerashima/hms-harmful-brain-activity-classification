from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_second_pass_artifacts_exist():
    expected = [
        "reports/results/second_pass_notebook_archive_metrics.csv",
        "reports/results/second_pass_submission_sanity.csv",
        "reports/results/second_pass_ensemble_config_audit.csv",
        "reports/results/second_pass_reproduction_readiness.csv",
        "reports/figures/second_pass_notebook_phase_footprint.png",
        "reports/figures/second_pass_submission_sanity.png",
        "reports/figures/second_pass_reproduction_readiness.png",
        "reports/second_pass_solution_engineering_review.md",
    ]
    for relative in expected:
        path = ROOT / relative
        assert path.exists(), relative
        assert path.stat().st_size > 0, relative


def test_notebook_archive_metrics_are_substantive():
    frame = pd.read_csv(ROOT / "reports/results/second_pass_notebook_archive_metrics.csv")
    assert len(frame) >= 9
    assert frame["code_lines"].sum() > 500
    assert frame["phase"].nunique() >= 4


def test_submission_sanity_is_probability_normalized():
    frame = pd.read_csv(ROOT / "reports/results/second_pass_submission_sanity.csv")
    assert (frame["probability_sum_min"] > 0.999).all()
    assert (frame["probability_sum_max"] < 1.001).all()
