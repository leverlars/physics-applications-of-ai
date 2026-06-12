"""Helpers for writing small, reproducible result artifacts."""

from pathlib import Path

import pandas
from matplotlib.figure import Figure


def save_table(table: pandas.DataFrame, output_dir: Path, filename: str) -> Path:
    """Save a dataframe as CSV and return the written path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    table.to_csv(output_path)
    return output_path


def save_figure(figure: Figure, output_dir: Path, filename: str, *, dpi: int = 160) -> Path:
    """Save a matplotlib figure and return the written path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path
