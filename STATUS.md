# Project Status

## Current State

The three core assignment tasks are implemented in `instructions.ipynb`:

- Task 1 trains binary classifiers for quark/gluon vs W/Z and quark/gluon vs top jets.
- Task 2 trains a balanced three-class classifier for quark/gluon, W/Z, and top jets.
- Task 3 trains a momentum-decorrelated three-class classifier and audits residual dependence on `pt`, `eta`, `phi`, and `mass`.

The local data and cache directories are ignored by git via `.gitignore`.

## Known Tradeoffs

Most of the project logic currently lives directly in notebook cells. This is acceptable for exploration, but it is not ideal for a final submission because the data loading, feature engineering, training, and evaluation code is duplicated across tasks and is harder to test or reuse.

Task 3 intentionally trades classification performance for reduced dependence on global jet momentum. The current approach uses relative constituent features, removes direct four-momentum inputs, and balances samples across shared `(pt, mass)` bins.

## Recommended Next Work

- Refactor repeated notebook code into importable modules under `src/physics_applications_of_ai/`.
- Keep `instructions.ipynb` as a lightweight analysis/reporting notebook that calls those modules.
- Add focused tests for feature preparation, balanced sampling, and metric calculation.
- Add a submission-oriented README section once the final folder layout is chosen.
- Consider adding saved plots or a small results summary for presentation preparation, while keeping large generated files out of git.

## Reproduction

Use `./startup.sh` from the repository root to synchronize the `uv` environment and start the notebook server.
