"""Dataset assembly helpers for the notebook workflows."""

import numpy
import pandas

from physics_applications_of_ai.data import (
    JET_KINDS,
    MOMENTUM_COLUMNS,
    load_constituents,
    load_global_properties,
)
from physics_applications_of_ai.features import (
    prepare_decorrelated_substructure_features,
    prepare_global_features,
    prepare_relative_constituent_features,
)
from physics_applications_of_ai.sampling import (
    random_class_indices,
    select_momentum_balanced_indices,
)


def make_binary_dataset(
    positive_jet_kind: str,
    *,
    n_per_class: int = 100_000,
    random_state: int = 7,
) -> tuple[pandas.DataFrame, numpy.ndarray]:
    """Create a balanced quark/gluon-vs-signal dataset."""
    rng = numpy.random.default_rng(random_state)
    background = load_global_properties("quark_gluon")
    signal = load_global_properties(positive_jet_kind)
    n_selected = min(n_per_class, len(background), len(signal))

    background_indices = random_class_indices(len(background), n_selected, rng)
    signal_indices = random_class_indices(len(signal), n_selected, rng)

    X = pandas.concat(
        [
            prepare_global_features(background.iloc[background_indices]),
            prepare_global_features(signal.iloc[signal_indices]),
        ],
        ignore_index=True,
    )
    y = numpy.concatenate(
        [numpy.zeros(n_selected, dtype=int), numpy.ones(n_selected, dtype=int)]
    )
    return X, y


def make_multiclass_dataset(
    *,
    n_per_class: int = 75_000,
    random_state: int = 7,
    jet_kinds: list[str] = JET_KINDS,
) -> tuple[pandas.DataFrame, numpy.ndarray]:
    """Build a balanced three-class dataset using global and constituent features."""
    rng = numpy.random.default_rng(random_state)
    feature_frames = []
    target_arrays = []

    for class_index, jet_kind in enumerate(jet_kinds):
        global_properties = load_global_properties(jet_kind)
        n_selected = min(n_per_class, len(global_properties))
        selected_indices = random_class_indices(
            len(global_properties), n_selected, rng, sort=True
        )
        selected_jets = global_properties.iloc[selected_indices].reset_index(drop=True)

        feature_frames.append(
            pandas.concat(
                [
                    prepare_global_features(selected_jets).reset_index(drop=True),
                    prepare_relative_constituent_features(
                        load_constituents(jet_kind, selected_indices),
                        selected_jets,
                        include_constituent_mass=True,
                        include_radius=False,
                    ),
                ],
                axis=1,
            )
        )
        target_arrays.append(numpy.full(n_selected, class_index, dtype=int))

    return pandas.concat(feature_frames, ignore_index=True), numpy.concatenate(target_arrays)


def make_decorrelated_dataset(
    *,
    n_bins: int = 5,
    max_per_bin_per_class: int = 4_000,
    min_per_bin_per_class: int = 25,
    random_state: int = 7,
    jet_kinds: list[str] = JET_KINDS,
) -> tuple[pandas.DataFrame, numpy.ndarray, pandas.DataFrame]:
    """Build a momentum-balanced dataset and separate momentum audit columns."""
    global_properties_by_class = {
        jet_kind: load_global_properties(jet_kind) for jet_kind in jet_kinds
    }
    selected_indices_by_class = select_momentum_balanced_indices(
        global_properties_by_class,
        jet_kinds=jet_kinds,
        n_bins=n_bins,
        max_per_bin_per_class=max_per_bin_per_class,
        min_per_bin_per_class=min_per_bin_per_class,
        random_state=random_state,
    )

    feature_frames = []
    target_arrays = []
    momentum_frames = []
    for class_index, jet_kind in enumerate(jet_kinds):
        selected_indices = selected_indices_by_class[jet_kind]
        selected_jets = global_properties_by_class[jet_kind].iloc[selected_indices].reset_index(drop=True)
        feature_frames.append(
            pandas.concat(
                [
                    prepare_decorrelated_substructure_features(selected_jets).reset_index(drop=True),
                    prepare_relative_constituent_features(
                        load_constituents(jet_kind, selected_indices),
                        selected_jets,
                        include_constituent_mass=False,
                        include_radius=True,
                    ),
                ],
                axis=1,
            )
        )
        target_arrays.append(numpy.full(len(selected_indices), class_index, dtype=int))
        momentum_frames.append(selected_jets[MOMENTUM_COLUMNS])

    return (
        pandas.concat(feature_frames, ignore_index=True),
        numpy.concatenate(target_arrays),
        pandas.concat(momentum_frames, ignore_index=True),
    )
