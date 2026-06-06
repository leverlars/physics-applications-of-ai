"""Sampling utilities for balanced jet-tagging datasets."""

import numpy
import pandas

from physics_applications_of_ai.data import JET_KINDS


def random_class_indices(
    n_available: int,
    n_selected: int,
    rng: numpy.random.Generator,
    *,
    sort: bool = False,
) -> numpy.ndarray:
    """Draw random row indices without replacement."""
    indices = rng.choice(n_available, min(n_selected, n_available), replace=False)
    if sort:
        return numpy.sort(indices)
    return indices


def select_momentum_balanced_indices(
    global_properties_by_class: dict[str, pandas.DataFrame],
    *,
    jet_kinds: list[str] = JET_KINDS,
    n_bins: int = 5,
    max_per_bin_per_class: int = 4_000,
    min_per_bin_per_class: int = 25,
    random_state: int = 7,
) -> dict[str, numpy.ndarray]:
    """Select equal class counts in shared pt/mass bins."""
    rng = numpy.random.default_rng(random_state)
    pooled_momenta = pandas.concat(
        [global_properties_by_class[jet_kind][["pt", "mass"]] for jet_kind in jet_kinds],
        ignore_index=True,
    )
    pt_edges = numpy.unique(
        numpy.quantile(pooled_momenta["pt"], numpy.linspace(0, 1, n_bins + 1))
    )
    mass_edges = numpy.unique(
        numpy.quantile(pooled_momenta["mass"], numpy.linspace(0, 1, n_bins + 1))
    )
    pt_edges[0] -= 1e-6
    pt_edges[-1] += 1e-6
    mass_edges[0] -= 1e-6
    mass_edges[-1] += 1e-6

    selected_indices_by_class = {jet_kind: [] for jet_kind in jet_kinds}
    for pt_bin in range(len(pt_edges) - 1):
        for mass_bin in range(len(mass_edges) - 1):
            candidate_indices_by_class = []
            for jet_kind in jet_kinds:
                global_properties = global_properties_by_class[jet_kind]
                in_bin = (
                    (global_properties["pt"] >= pt_edges[pt_bin])
                    & (global_properties["pt"] < pt_edges[pt_bin + 1])
                    & (global_properties["mass"] >= mass_edges[mass_bin])
                    & (global_properties["mass"] < mass_edges[mass_bin + 1])
                )
                candidate_indices_by_class.append(numpy.flatnonzero(in_bin.to_numpy()))

            n_selected = min(
                [len(indices) for indices in candidate_indices_by_class]
                + [max_per_bin_per_class]
            )
            if n_selected < min_per_bin_per_class:
                continue

            for jet_kind, candidate_indices in zip(jet_kinds, candidate_indices_by_class):
                selected_indices_by_class[jet_kind].append(
                    rng.choice(candidate_indices, n_selected, replace=False)
                )

    return {
        jet_kind: numpy.sort(numpy.concatenate(index_chunks))
        for jet_kind, index_chunks in selected_indices_by_class.items()
    }
