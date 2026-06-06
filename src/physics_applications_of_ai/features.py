"""Feature engineering for jet-tagging models."""

import numpy
import pandas


def clean_features(features: pandas.DataFrame) -> pandas.DataFrame:
    """Replace non-finite values produced by ratios or logarithms."""
    return features.replace([numpy.inf, -numpy.inf], numpy.nan).fillna(0.0)


def prepare_global_features(jets: pandas.DataFrame) -> pandas.DataFrame:
    """Prepare global and substructure features for non-decorrelated classifiers."""
    eps = 1e-8
    features = jets.copy()
    features["tau21"] = features["tau2"] / (features["tau1"] + eps)
    features["tau32"] = features["tau3"] / (features["tau2"] + eps)
    features["d23_over_d12"] = features["d23"] / (features["d12"] + eps)
    features["ecf3_over_ecf2"] = features["ECF3"] / (features["ECF2"] + eps)

    for column in ["pt", "mass", "d12", "d23", "ECF2", "ECF3"]:
        features[f"log_{column}"] = numpy.log1p(features[column].clip(lower=0))

    selected_columns = [
        "pt",
        "mass",
        "tau1",
        "tau2",
        "tau3",
        "tau21",
        "tau32",
        "d12",
        "d23",
        "d23_over_d12",
        "ECF2",
        "ECF3",
        "ecf3_over_ecf2",
        "log_pt",
        "log_mass",
        "log_d12",
        "log_d23",
        "log_ECF2",
        "log_ECF3",
    ]
    return clean_features(features[selected_columns])


def prepare_decorrelated_substructure_features(jets: pandas.DataFrame) -> pandas.DataFrame:
    """Prepare mostly unitless substructure features while omitting four-momentum inputs."""
    eps = 1e-8
    features = pandas.DataFrame(index=jets.index)
    features["tau1"] = jets["tau1"]
    features["tau2"] = jets["tau2"]
    features["tau3"] = jets["tau3"]
    features["tau21"] = jets["tau2"] / (jets["tau1"] + eps)
    features["tau32"] = jets["tau3"] / (jets["tau2"] + eps)
    features["d23_over_d12"] = jets["d23"] / (jets["d12"] + eps)
    features["ecf3_over_ecf2"] = jets["ECF3"] / (jets["ECF2"] + eps)
    return clean_features(features)


def prepare_relative_constituent_features(
    constituents: numpy.ndarray,
    selected_jets: pandas.DataFrame,
    *,
    include_constituent_mass: bool = True,
    include_radius: bool = False,
) -> pandas.DataFrame:
    """Represent ordered constituents relative to their parent jet axis."""
    constituent_pt = constituents[:, :, 0]
    constituent_eta = constituents[:, :, 1]
    constituent_phi = constituents[:, :, 2]
    constituent_mass = constituents[:, :, 3]

    jet_pt = selected_jets["pt"].to_numpy()[:, numpy.newaxis]
    jet_eta = selected_jets["eta"].to_numpy()[:, numpy.newaxis]
    jet_phi = selected_jets["phi"].to_numpy()[:, numpy.newaxis]

    pt_fraction = constituent_pt / (jet_pt + 1e-8)
    delta_eta = constituent_eta - jet_eta
    delta_phi = (constituent_phi - jet_phi + numpy.pi) % (2 * numpy.pi) - numpy.pi
    radius = numpy.sqrt(delta_eta**2 + delta_phi**2)

    feature_columns = {}
    for constituent_index in range(constituents.shape[1]):
        prefix = f"constituent_{constituent_index:02d}"
        feature_columns[f"{prefix}_pt_fraction"] = pt_fraction[:, constituent_index]
        feature_columns[f"{prefix}_delta_eta"] = delta_eta[:, constituent_index]
        feature_columns[f"{prefix}_delta_phi"] = delta_phi[:, constituent_index]
        if include_constituent_mass:
            feature_columns[f"{prefix}_mass"] = constituent_mass[:, constituent_index]
        if include_radius:
            feature_columns[f"{prefix}_radius"] = radius[:, constituent_index]

    pt_fraction_sum = pt_fraction.sum(axis=1)
    feature_columns["constituent_pt_fraction_sum"] = pt_fraction_sum
    feature_columns["constituent_pt_fraction_std"] = pt_fraction.std(axis=1)

    if include_radius:
        radius_weighted_mean = (radius * pt_fraction).sum(axis=1) / (pt_fraction_sum + 1e-8)
        feature_columns["constituent_radius_pt_weighted_mean"] = radius_weighted_mean
        feature_columns["constituent_radius_pt_weighted_std"] = numpy.sqrt(
            (((radius - radius_weighted_mean[:, numpy.newaxis]) ** 2) * pt_fraction).sum(axis=1)
            / (pt_fraction_sum + 1e-8)
        )
    else:
        feature_columns["constituent_delta_eta_pt_weighted_mean"] = (
            delta_eta * pt_fraction
        ).sum(axis=1) / (pt_fraction_sum + 1e-8)
        feature_columns["constituent_delta_phi_pt_weighted_mean"] = (
            delta_phi * pt_fraction
        ).sum(axis=1) / (pt_fraction_sum + 1e-8)

    return clean_features(pandas.DataFrame(feature_columns))
