"""Data loading helpers for the jet-tagging assignment."""

from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy
import pandas
import requests

DATA_DIR_PATH = Path("data")
CACHE_DIR_PATH = Path(".cache")
DATA_ARCHIVE_PATH = CACHE_DIR_PATH / "jet_tagging.zip"
DATA_DOWNLOAD_URL = "https://drive.switch.ch/index.php/s/JpqDntLTwRKMgqW/download"
JET_KINDS = ["quark_gluon", "wz", "top"]
DISPLAY_LABELS = ["quark/gluon", "W/Z", "top"]
MOMENTUM_COLUMNS = ["pt", "eta", "phi", "mass"]


def ensure_data_available(
    data_dir: Path = DATA_DIR_PATH,
    archive_path: Path = DATA_ARCHIVE_PATH,
) -> None:
    """Download and extract the jet-tagging data if the HDF5 files are missing."""
    required_files = [
        data_dir / f"jet_properties_global_{jet_kind}.h5" for jet_kind in JET_KINDS
    ] + [
        data_dir / f"jet_constituents_{jet_kind}.h5" for jet_kind in JET_KINDS
    ]
    if all(path.exists() for path in required_files):
        return

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        response = requests.get(DATA_DOWNLOAD_URL, timeout=120)
        response.raise_for_status()
        archive_path.write_bytes(response.content)

    with ZipFile(archive_path) as archive:
        archive.extractall(data_dir)


def load_global_properties(
    jet_kind: str,
    data_dir: Path = DATA_DIR_PATH,
) -> pandas.DataFrame:
    """Load the global jet-level table for one jet origin."""
    ensure_data_available(data_dir=data_dir)
    return pandas.read_hdf(
        data_dir / f"jet_properties_global_{jet_kind}.h5",
        key="substructure",
    )


def load_constituents(
    jet_kind: str,
    indices: numpy.ndarray,
    data_dir: Path = DATA_DIR_PATH,
) -> numpy.ndarray:
    """Load selected constituent four-momenta for one jet origin."""
    ensure_data_available(data_dir=data_dir)
    with h5py.File(data_dir / f"jet_constituents_{jet_kind}.h5") as h5_file:
        return numpy.array(h5_file["constituents"][indices])
