# Physics Applications of AI: Jet Tagging

Course project for **Physics Applications of AI**. The goal is to classify ATLAS-like jets by their physical origin and to study how classifier performance changes when direct momentum dependence is reduced.

The implemented workflow is in `solution.ipynb`. The original assignment handout is preserved in `instructions.ipynb`, and reusable project code lives in `src/physics_applications_of_ai/`.

## Project Goals

The project addresses three related jet-tagging tasks:

1. Binary classification of quark/gluon jets against W/Z jets and against top jets.
2. Three-class classification of quark/gluon, W/Z, and top jets.
3. Momentum-decorrelated three-class classification, where the model should rely less directly on global jet momentum variables.

The main physics motivation is that high-performing taggers can learn correlations with jet kinematics such as `pt` and `mass`. Those correlations are useful for classification, but they can also bias downstream physics analyses. Task 3 therefore deliberately trades some classification power for weaker dependence on global momentum variables.

## Method

The solution uses gradient-boosted decision trees from scikit-learn on engineered jet features.

For Tasks 1 and 2, the feature set includes global jet and substructure information such as jet mass, N-subjettiness ratios, splitting scales, energy-correlation functions, and relative constituent features.

For Task 3, the feature set removes direct `pt`, `eta`, `phi`, and `mass` inputs. It instead uses mostly unitless substructure ratios and constituent coordinates expressed relative to the parent jet axis. The training sample is also balanced across shared `(pt, mass)` bins so that the classifier has less incentive to separate classes by obvious kinematic differences.

## Results and Interpretation

The notebook produces metric tables and diagnostic plots for each task.

Task 1 is expected to perform best because each model only separates quark/gluon jets from one signal class at a time. The ROC-AUC and average-precision scores are the most useful summary metrics because they evaluate ranking quality across classification thresholds, not only the default threshold.

Task 2 is harder because the classifier must separate all three jet origins simultaneously. Confusion between classes is more informative than raw accuracy alone: the normalized confusion matrix shows which jet types have overlapping feature patterns.

Task 3 should show the clearest performance-versus-decorrelation tradeoff. Its classification metrics are expected to be lower than the standard multiclass model, but the probability-momentum correlation table and eta-squared audit should show weaker residual dependence on `pt`, `eta`, `phi`, and `mass`. This is the desired behavior for a decorrelated tagger: it sacrifices some tagging power to reduce kinematic sculpting.

Generated outputs are written to `outputs/` when `solution.ipynb` is run:

- `task1_metrics.csv`
- `task1_binary_diagnostics.png`
- `task2_metrics.csv`
- `task2_confusion_matrix.png`
- `task3_metrics.csv`
- `task3_probability_momentum_correlations.csv`
- `task3_prediction_momentum_eta_squared.csv`
- `task3_decorrelation_diagnostics.png`

The `outputs/` directory is ignored by git so results can be regenerated without committing large or run-specific artifacts.

## Repository Layout

```text
.
├── instructions.ipynb                  # Original assignment handout
├── solution.ipynb                      # Main project solution notebook
├── src/physics_applications_of_ai/     # Reusable package code
│   ├── artifacts.py                    # CSV/figure saving helpers
│   ├── config.py                       # Full-run and smoke-run settings
│   ├── data.py                         # Data download and HDF5 loading
│   ├── datasets.py                     # Task dataset construction
│   ├── evaluation.py                   # Metrics and decorrelation audits
│   ├── features.py                     # Feature engineering
│   ├── models.py                       # Model factory
│   └── sampling.py                     # Balanced sampling utilities
├── tests/                              # Unit tests for reusable code
├── pyproject.toml                      # Project metadata and dependencies
└── uv.lock                             # Reproducible dependency lockfile
```

## Reproducing the Project

This repository uses [`uv`](https://docs.astral.sh/uv/) for Python environment and dependency management.

From the repository root:

```bash
uv sync
uv run jupyter notebook solution.ipynb
```

The data files are downloaded automatically on demand by `physics_applications_of_ai.data` if they are not already present under `data/`. The downloaded archive is cached under `.cache/`. Both directories are ignored by git.

For a faster end-to-end smoke run, set this toggle in the first code cell of `solution.ipynb`:

```python
SMOKE_RUN = True
```

For the full project run, leave it as:

```python
SMOKE_RUN = False
```

A convenience script is also provided:

```bash
./startup.sh
```

## Checks

Run the unit tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

The original `instructions.ipynb` handout is excluded from Ruff linting because it is preserved as upstream assignment material rather than maintained project code.

## Notes for Submission

The files to review for the submitted solution are `solution.ipynb`, `README.md`, and the reusable implementation under `src/physics_applications_of_ai/`. The notebook contains the task workflow and plots; the package code contains the data loading, feature construction, sampling, model setup, and evaluation logic used by the notebook.
