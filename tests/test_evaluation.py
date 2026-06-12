import numpy
import pandas

from physics_applications_of_ai.evaluation import (
    binary_metrics,
    multiclass_metrics,
    prediction_momentum_eta_squared,
    probability_momentum_correlations,
)


def test_binary_metrics_returns_expected_keys():
    metrics = binary_metrics(
        numpy.array([0, 0, 1, 1]),
        numpy.array([0, 1, 1, 1]),
        numpy.array([0.1, 0.6, 0.8, 0.9]),
    )

    assert set(metrics) == {"accuracy", "f1", "roc_auc", "average_precision"}
    assert metrics["accuracy"] == 0.75


def test_multiclass_metrics_returns_expected_keys():
    y_true = numpy.array([0, 1, 2, 2])
    probabilities = numpy.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.2, 0.5, 0.3],
        ]
    )

    metrics = multiclass_metrics(y_true, probabilities.argmax(axis=1), probabilities)

    assert set(metrics) == {
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_roc_auc_ovr",
    }
    assert metrics["accuracy"] == 0.75


def test_momentum_audits_return_named_tables():
    momenta = pandas.DataFrame(
        {
            "pt": [1.0, 2.0, 3.0, 4.0],
            "eta": [0.1, 0.2, 0.3, 0.4],
            "phi": [0.2, 0.1, -0.1, -0.2],
            "mass": [10.0, 11.0, 12.0, 13.0],
        }
    )
    probabilities = numpy.array(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    predictions = probabilities.argmax(axis=1)

    correlations = probability_momentum_correlations(probabilities, momenta)
    eta_squared = prediction_momentum_eta_squared(predictions, momenta)

    assert correlations.index.tolist() == ["quark/gluon", "W/Z", "top"]
    assert "abs_corr_probability_pt" in correlations.columns
    assert eta_squared.index.tolist() == ["pt", "eta", "phi", "mass"]
    assert "eta_squared_predicted_class" in eta_squared.columns
