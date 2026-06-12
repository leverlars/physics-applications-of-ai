# Project Status

## Current State

This repository is in a hand-in-ready draft state for the **Physics Applications of AI** jet-tagging project.

The three core assignment tasks are implemented in `solution.ipynb`, which uses reusable helpers from `src/physics_applications_of_ai/`:

- Task 1 trains binary classifiers for quark/gluon vs W/Z and quark/gluon vs top jets.
- Task 2 trains a balanced three-class classifier for quark/gluon, W/Z, and top jets.
- Task 3 trains a momentum-decorrelated three-class classifier and audits residual dependence on `pt`, `eta`, `phi`, and `mass`.

`instructions.ipynb` is restored as the original assignment handout. Keep it intact unless the handout itself changes upstream. The user-facing solution workflow is `solution.ipynb`, which handles orchestration, plotting, reporting, and result artifact generation.

`README.md` has been rewritten as a submission-facing README with course context, project goals, method, qualitative result interpretation, reproduction instructions, checks, and submission notes.

## Recent Changes

The latest work session added project hygiene and hand-in polish:

- Added `src/physics_applications_of_ai/config.py` with `FULL_RUN_SETTINGS`, `QUICK_RUN_SETTINGS`, and centralized classifier/sample settings.
- Added `src/physics_applications_of_ai/artifacts.py` for saving metric tables and diagnostic plots.
- Updated `solution.ipynb` to use centralized settings and a `SMOKE_RUN` toggle.
- Updated `solution.ipynb` markdown with task-level commentary on expected results and the Task 3 performance/decorrelation tradeoff.
- Added unit tests under `tests/` for feature engineering, sampling, evaluation metrics, and artifact writing.
- Configured Ruff to exclude `instructions.ipynb`, because it is preserved as upstream assignment material.
- Added `outputs/` to `.gitignore`; generated result CSV/PNG artifacts should live there and remain untracked.

## Code Structure

- `src/physics_applications_of_ai/data.py`: HDF5 loading helpers and shared label constants.
- `src/physics_applications_of_ai/artifacts.py`: helpers for saving metric tables and figures.
- `src/physics_applications_of_ai/config.py`: centralized full-run and quick-run settings.
- `src/physics_applications_of_ai/datasets.py`: task-ready dataset assembly for binary, multiclass, and decorrelated workflows.
- `src/physics_applications_of_ai/features.py`: global, substructure, and relative constituent feature preparation.
- `src/physics_applications_of_ai/sampling.py`: random class sampling and momentum-bin balancing.
- `src/physics_applications_of_ai/models.py`: classifier factory helpers.
- `src/physics_applications_of_ai/evaluation.py`: metric summaries and decorrelation audits.
- `tests/`: focused unit tests for reusable package behavior.
- `instructions.ipynb`: original assignment handout.
- `solution.ipynb`: task-level orchestration, model training calls, plots, displayed result tables, and output saving.

## Data and Generated Files

The HDF5 data files and cached download archive are not tracked. `src/physics_applications_of_ai/data.py` downloads the official assignment archive on demand when the expected files are missing under `data/`, then caches the archive under `.cache/`.

Ignored local/generated paths include:

- `data/`
- `.cache/`
- `outputs/`
- `.venv/`

The notebook writes these files under `outputs/` when run:

- `task1_metrics.csv`
- `task1_binary_diagnostics.png`
- `task2_metrics.csv`
- `task2_confusion_matrix.png`
- `task3_metrics.csv`
- `task3_probability_momentum_correlations.csv`
- `task3_prediction_momentum_eta_squared.csv`
- `task3_decorrelation_diagnostics.png`

## Known Tradeoffs

Task 3 intentionally trades classification performance for reduced dependence on global jet momentum. The current approach uses relative constituent features, removes direct four-momentum inputs, and balances samples across shared `(pt, mass)` bins.

The solution notebook still contains task orchestration and plotting code by design. This keeps it easy to present and tweak interactively, while repeated mechanics live in `src/`.

The README and notebook currently describe results qualitatively. There are no checked-in executed notebook outputs or saved result artifacts, so avoid claiming exact metrics unless the notebook has been rerun and the resulting values are visible.

## Verification Status

Last verified in this workspace:

```bash
uv run ruff check .
uv run pytest
```

Both passed. `pytest` collected and passed 10 tests.

## Recommended Next Work

For the next Codex session, the most useful follow-up is to run `solution.ipynb` end-to-end, preferably first with `SMOKE_RUN = True` and then with `SMOKE_RUN = False` if runtime is acceptable.

After a full run:

- Review the generated `outputs/` metrics and plots.
- Optionally add a short numeric results table to the README or notebook summary using the actual generated values.
- Consider adding richer Task 3 before/after plots comparing the decorrelated model against the standard multiclass classifier.
- If submitting as a zip or repository snapshot, confirm whether generated `outputs/` should be included or left out according to course instructions.

## Reproduction

Use `./startup.sh` from the repository root to synchronize the `uv` environment and start the solution notebook server.

For a faster smoke run, set this in the notebook setup cell:

```python
SMOKE_RUN = True
```

For the full project run, leave:

```python
SMOKE_RUN = False
```
