"""Evaluation helpers for classification and momentum-decorrelation audits."""

import numpy
import pandas
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from physics_applications_of_ai.data import DISPLAY_LABELS, MOMENTUM_COLUMNS


def binary_metrics(
    y_true: numpy.ndarray,
    predictions: numpy.ndarray,
    probabilities: numpy.ndarray,
) -> dict[str, float]:
    """Return scalar metrics for a binary classifier."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "f1": f1_score(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
    }


def multiclass_metrics(
    y_true: numpy.ndarray,
    predictions: numpy.ndarray,
    probabilities: numpy.ndarray,
) -> dict[str, float]:
    """Return scalar metrics for a multiclass classifier."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro"),
        "weighted_f1": f1_score(y_true, predictions, average="weighted"),
        "macro_roc_auc_ovr": roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro",
        ),
    }


def probability_momentum_correlations(
    probabilities: numpy.ndarray,
    momenta: pandas.DataFrame,
    *,
    display_labels: list[str] = DISPLAY_LABELS,
    momentum_columns: list[str] = MOMENTUM_COLUMNS,
) -> pandas.DataFrame:
    """Measure linear dependence between each class probability and momentum variable."""
    rows = []
    for class_index, class_name in enumerate(display_labels):
        row = {"class": class_name}
        for momentum_column in momentum_columns:
            row[f"abs_corr_probability_{momentum_column}"] = abs(
                numpy.corrcoef(probabilities[:, class_index], momenta[momentum_column])[0, 1]
            )
        rows.append(row)
    return pandas.DataFrame(rows).set_index("class")


def prediction_momentum_eta_squared(
    predictions: numpy.ndarray,
    momenta: pandas.DataFrame,
    *,
    momentum_columns: list[str] = MOMENTUM_COLUMNS,
) -> pandas.DataFrame:
    """Correlation ratio eta^2: momentum variance explained by predicted class."""
    rows = []
    for momentum_column in momentum_columns:
        values = momenta[momentum_column].to_numpy()
        grand_mean = values.mean()
        total_sum_squares = ((values - grand_mean) ** 2).sum()
        between_sum_squares = 0.0
        for predicted_class in numpy.unique(predictions):
            group_values = values[predictions == predicted_class]
            between_sum_squares += len(group_values) * (group_values.mean() - grand_mean) ** 2
        rows.append(
            {
                "momentum_variable": momentum_column,
                "eta_squared_predicted_class": between_sum_squares / total_sum_squares,
            }
        )
    return pandas.DataFrame(rows).set_index("momentum_variable")
