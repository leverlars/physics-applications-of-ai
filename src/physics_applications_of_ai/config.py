"""Reusable run settings for the notebook workflows."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassifierSettings:
    """Training settings for one classifier."""

    max_iter: int
    l2_regularization: float
    learning_rate: float = 0.08


@dataclass(frozen=True)
class RunSettings:
    """Dataset, model, and output settings shared by the solution notebook."""

    random_state: int = 7
    test_size: float = 0.25
    output_dir: Path = Path("outputs")
    binary_n_per_class: int = 100_000
    multiclass_n_per_class: int = 75_000
    decorrelated_n_bins: int = 5
    decorrelated_max_per_bin_per_class: int = 4_000
    decorrelated_min_per_bin_per_class: int = 25
    binary_classifier: ClassifierSettings = ClassifierSettings(
        max_iter=180,
        l2_regularization=0.02,
    )
    multiclass_classifier: ClassifierSettings = ClassifierSettings(
        max_iter=220,
        l2_regularization=0.02,
    )
    decorrelated_classifier: ClassifierSettings = ClassifierSettings(
        max_iter=220,
        l2_regularization=0.08,
    )


FULL_RUN_SETTINGS = RunSettings()
QUICK_RUN_SETTINGS = RunSettings(
    binary_n_per_class=2_000,
    multiclass_n_per_class=2_000,
    decorrelated_n_bins=3,
    decorrelated_max_per_bin_per_class=300,
    decorrelated_min_per_bin_per_class=10,
    binary_classifier=ClassifierSettings(max_iter=30, l2_regularization=0.02),
    multiclass_classifier=ClassifierSettings(max_iter=35, l2_regularization=0.02),
    decorrelated_classifier=ClassifierSettings(max_iter=35, l2_regularization=0.08),
)
