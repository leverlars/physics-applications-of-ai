# Project Status

## Current State

The three core assignment tasks are implemented in `solution.ipynb`, which uses reusable helpers from `src/physics_applications_of_ai/`:

- Task 1 trains binary classifiers for quark/gluon vs W/Z and quark/gluon vs top jets.
- Task 2 trains a balanced three-class classifier for quark/gluon, W/Z, and top jets.
- Task 3 trains a momentum-decorrelated three-class classifier and audits residual dependence on `pt`, `eta`, `phi`, and `mass`.

`instructions.ipynb` is restored as the original assignment handout. Keep it intact unless the handout itself changes upstream. The user-facing solution workflow is `solution.ipynb`, which handles orchestration, plotting, and reporting.

The local data and cache directories are ignored by git via `.gitignore`.

## Code Structure

- `src/physics_applications_of_ai/data.py`: HDF5 loading helpers and shared label constants.
- `src/physics_applications_of_ai/datasets.py`: task-ready dataset assembly for binary, multiclass, and decorrelated workflows.
- `src/physics_applications_of_ai/features.py`: global, substructure, and relative constituent feature preparation.
- `src/physics_applications_of_ai/sampling.py`: random class sampling and momentum-bin balancing.
- `src/physics_applications_of_ai/models.py`: classifier factory helpers.
- `src/physics_applications_of_ai/evaluation.py`: metric summaries and decorrelation audits.
- `instructions.ipynb`: original assignment handout.
- `solution.ipynb`: concise task-level orchestration, model training calls, plots, and displayed result tables.

## Data Reproducibility

The HDF5 data files and cached download archive are not tracked. `src/physics_applications_of_ai/data.py` downloads the official assignment archive on demand when the expected files are missing under `data/`, then caches the archive under `.cache/`. The old root-level `jettagging.zip` only contained a copy of the handout notebook and is not required.

## Known Tradeoffs

Task 3 intentionally trades classification performance for reduced dependence on global jet momentum. The current approach uses relative constituent features, removes direct four-momentum inputs, and balances samples across shared `(pt, mass)` bins.

The solution notebook still contains task orchestration and plotting code by design. This keeps it easy to present and tweak interactively, while the repeated mechanics live in `src/`.

## Recommended Next Work

- Add focused tests for feature preparation, balanced sampling, and metric calculation.
- Add a submission-oriented README section once the final folder layout is chosen.
- Consider adding saved plots or a small results summary for presentation preparation, while keeping large generated files out of git.
- If runtime becomes an issue, add smaller configurable sample sizes for quick smoke runs.

## Reproduction

Use `./startup.sh` from the repository root to synchronize the `uv` environment and start the solution notebook server.
