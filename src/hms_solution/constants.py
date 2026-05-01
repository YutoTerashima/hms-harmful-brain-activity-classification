"""Competition constants shared by scripts and tests."""

HMS_CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
TARGET_COLUMNS = [f"{name}_vote" for name in HMS_CLASSES]
PROBABILITY_COLUMNS = [f"{name}_probability" for name in HMS_CLASSES]

EEG_CHANNELS = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "F7",
    "T3",
    "T5",
    "O1",
    "Fz",
    "Cz",
    "Pz",
    "Fp2",
    "F4",
    "C4",
    "P4",
    "F8",
    "T4",
    "T6",
    "O2",
    "EKG",
]

BIPOLAR_PAIRS = [
    ("Fp1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("Fp2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
]

COMPETITION_SLUG = "hms-harmful-brain-activity-classification"

