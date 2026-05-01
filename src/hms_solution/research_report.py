from __future__ import annotations

"""Report metadata for the mature portfolio iteration."""

PROJECT_TITLE = 'HMS Harmful Brain Activity Classification'
RESEARCH_PROBLEM = 'How can a Kaggle Silver Medal EEG solution be packaged as a reproducible, inspectable research archive?'
DATASET_SUMMARY = 'Kaggle HMS harmful brain activity classification metadata and archived notebooks; raw Kaggle data is intentionally excluded.'
TAKEAWAYS = ['The project is a credible competition archive because it preserves original notebooks and adds clean package structure.', 'The strongest signal is the Kaggle Silver Medal result plus an inspectable ensemble workflow.', 'Raw data is excluded, so reproduction depends on Kaggle API access and the provided data card.']
NEXT_EXPERIMENTS = ['Add a rendered notebook map with model family diagrams.', 'Add synthetic smoke inference for each model family.', 'Document CV folds, blending, and known reproduction caveats in one solution report.']


def report_outline() -> list[str]:
    return [
        "Abstract",
        "Research question",
        "Dataset card",
        "Methods",
        "Experiment matrix",
        "Results",
        "Ablations",
        "Failure analysis",
        "Engineering notes",
        "Limitations",
        "Reproduction",
    ]


def maturity_claims() -> dict[str, object]:
    return {
        "title": PROJECT_TITLE,
        "problem": RESEARCH_PROBLEM,
        "dataset": DATASET_SUMMARY,
        "takeaways": TAKEAWAYS,
        "next_experiments": NEXT_EXPERIMENTS,
    }
