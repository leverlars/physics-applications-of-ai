# Physics Applications of AI: Jet Tagging

This repository contains the starter project setup for the jet-tagging assignment. The project uses
[`uv`](https://docs.astral.sh/uv/) to keep Python version selection, dependency locking, and command
execution reproducible from the repository root.

## Project setup

Run these commands from the top-level repository folder:

```bash
# Create a standalone Python project in this folder.
uv init --lib --no-workspace --name physics-applications-of-ai .

# Use a Python version compatible with the scientific stack used here.
uv python pin 3.12

# Add the runtime packages used by the instructions notebook and planned training code.
uv add torch numpy matplotlib h5py pandas tables notebook requests scikit-learn tqdm

# Add optional development tools for checks and tests.
uv add --dev pytest ruff

# Resolve and install the environment from pyproject.toml/uv.lock.
uv sync
```

The repository already includes the resulting top-level `pyproject.toml`, so a fresh checkout only needs:

```bash
uv sync
```

For convenience, the repository also includes a startup script that synchronizes the environment and opens the assignment notebook:

```bash
./startup.sh
```

## Useful commands

```bash
# Start the instructions notebook.
uv run jupyter notebook instructions.ipynb

# Run tests once project tests are added.
uv run pytest

# Run linting once project code is added.
uv run ruff check .
```

## Dependency notes

The main dependencies support the assignment workflow:

- `requests` downloads the cached jet-tagging archive.
- `h5py`, `pandas`, and `tables` load the HDF5 constituent and global-property files.
- `numpy` handles array preparation and feature transformations.
- `matplotlib` supports exploratory plots and result figures.
- `torch` supports neural-network training for binary, multiclass, and decorrelation tasks.
- `scikit-learn` supports metrics, splitting, baselines, and preprocessing.
- `notebook` runs `instructions.ipynb` interactively.
- `tqdm` provides progress bars for data preparation and training loops.
