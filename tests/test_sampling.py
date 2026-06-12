import numpy
import pandas

from physics_applications_of_ai.sampling import (
    random_class_indices,
    select_momentum_balanced_indices,
)


def test_random_class_indices_can_sort_sample():
    rng = numpy.random.default_rng(7)

    indices = random_class_indices(20, 8, rng, sort=True)

    assert len(indices) == 8
    assert indices.tolist() == sorted(indices.tolist())
    assert len(set(indices.tolist())) == 8


def test_select_momentum_balanced_indices_keeps_equal_counts_per_class():
    jets_by_class = {
        "a": pandas.DataFrame({"pt": [10, 20, 30, 40], "mass": [5, 6, 7, 8]}),
        "b": pandas.DataFrame({"pt": [11, 21, 31, 41], "mass": [5, 6, 7, 8]}),
        "c": pandas.DataFrame({"pt": [12, 22, 32, 42], "mass": [5, 6, 7, 8]}),
    }

    selected = select_momentum_balanced_indices(
        jets_by_class,
        jet_kinds=["a", "b", "c"],
        n_bins=1,
        max_per_bin_per_class=3,
        min_per_bin_per_class=1,
        random_state=7,
    )

    assert set(selected) == {"a", "b", "c"}
    assert {len(indices) for indices in selected.values()} == {3}
    for indices in selected.values():
        assert numpy.all(indices[:-1] <= indices[1:])
