import numpy
import pandas

from physics_applications_of_ai.features import (
    clean_features,
    prepare_global_features,
    prepare_relative_constituent_features,
)


def test_clean_features_replaces_non_finite_values():
    features = pandas.DataFrame({"x": [1.0, numpy.inf, -numpy.inf, numpy.nan]})

    cleaned = clean_features(features)

    assert cleaned["x"].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_prepare_global_features_adds_ratios_and_logs():
    jets = pandas.DataFrame(
        {
            "pt": [99.0],
            "eta": [0.2],
            "phi": [0.4],
            "mass": [9.0],
            "tau1": [2.0],
            "tau2": [1.0],
            "tau3": [0.25],
            "d12": [4.0],
            "d23": [2.0],
            "ECF2": [8.0],
            "ECF3": [2.0],
        }
    )

    features = prepare_global_features(jets)

    assert "eta" not in features.columns
    assert "phi" not in features.columns
    assert features.loc[0, "tau21"] == numpy.float64(1.0 / (2.0 + 1e-8))
    assert features.loc[0, "d23_over_d12"] == numpy.float64(2.0 / (4.0 + 1e-8))
    assert features.loc[0, "log_pt"] == numpy.log1p(99.0)


def test_prepare_relative_constituent_features_wraps_phi_and_summarizes():
    constituents = numpy.array(
        [
            [
                [20.0, 1.1, -3.10, 0.5],
                [10.0, 0.8, 3.05, 0.2],
            ]
        ]
    )
    selected_jets = pandas.DataFrame({"pt": [100.0], "eta": [1.0], "phi": [3.10]})

    features = prepare_relative_constituent_features(
        constituents,
        selected_jets,
        include_constituent_mass=True,
        include_radius=True,
    )

    assert numpy.isclose(features.loc[0, "constituent_00_pt_fraction"], 0.2)
    assert numpy.isclose(features.loc[0, "constituent_01_delta_eta"], -0.2)
    assert abs(features.loc[0, "constituent_00_delta_phi"]) < 0.1
    assert features.loc[0, "constituent_00_mass"] == 0.5
    assert numpy.isclose(features.loc[0, "constituent_pt_fraction_sum"], 0.3)
    assert "constituent_radius_pt_weighted_mean" in features.columns
