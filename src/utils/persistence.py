"""Artifact persistence helpers for the Variant Disagreement Atlas.

All paths are resolved relative to the project root (the directory that
contains src/), so these functions work correctly regardless of which
directory a notebook is launched from.

Artifact inventory
------------------
  results/figures/          PNG + PDF, dpi=300, auto-versioned
  results/tables/           CSV or Parquet, auto-versioned
  data/processed/           DataFrames → parquet, arrays → .npy, other → pickle
  results/ARTIFACTS_LOG.csv append-only log of every saved artifact
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Project root = two levels up from this file (src/utils/persistence.py)
_ROOT = Path(__file__).resolve().parents[2]

_FIGURES_DIR   = _ROOT / "results" / "figures"
_TABLES_DIR    = _ROOT / "results" / "tables"
_PROCESSED_DIR = _ROOT / "data"    / "processed"
_LOG_PATH      = _ROOT / "results" / "ARTIFACTS_LOG.csv"

_LOG_COLUMNS = ["timestamp", "notebook", "name", "kind", "path", "description"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _versioned_stem(dest_dir: Path, name: str, ext: str) -> str:
    """Return name if the file doesn't exist, else name_v2, name_v3, …"""
    if not (dest_dir / f"{name}.{ext}").exists():
        return name
    v = 2
    while (dest_dir / f"{name}_v{v}.{ext}").exists():
        v += 1
    return f"{name}_v{v}"


def _mpl_figure(fig):
    """Extract a matplotlib Figure from fig or a seaborn ClusterGrid."""
    try:
        import seaborn as sns
        if isinstance(fig, sns.matrix.ClusterGrid):
            return fig.fig, fig   # (mpl_fig, original) — original used for .savefig
    except (ImportError, AttributeError):
        pass
    return fig, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_artifact(
    name: str,
    kind: str,
    path: str,
    description: str,
    notebook: str,
) -> None:
    """Append one row to results/ARTIFACTS_LOG.csv, creating it if needed."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "notebook":    notebook or "",
        "name":        name,
        "kind":        kind,
        "path":        str(path),
        "description": description or "",
    }])
    if not _LOG_PATH.exists():
        row.to_csv(_LOG_PATH, index=False)
    else:
        row.to_csv(_LOG_PATH, mode="a", header=False, index=False)


def save_figure(
    fig,
    name: str,
    subdir: Optional[str] = None,
    description: Optional[str] = None,
    notebook: Optional[str] = None,
) -> Path:
    """Save a figure as PNG (dpi=300) and PDF, auto-versioning the filename.

    Parameters
    ----------
    fig:
        A ``matplotlib.figure.Figure`` or a ``seaborn.matrix.ClusterGrid``.
    name:
        Base filename stem (no extension).
    subdir:
        Optional subdirectory under results/figures/.
    description:
        One-sentence description written to ARTIFACTS_LOG.csv.
    notebook:
        Notebook name (without .ipynb), written to ARTIFACTS_LOG.csv.

    Returns
    -------
    Path to the saved PNG file.
    """
    dest_dir = _FIGURES_DIR / subdir if subdir else _FIGURES_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    stem = _versioned_stem(dest_dir, name, "png")
    png_path = dest_dir / f"{stem}.png"
    pdf_path = dest_dir / f"{stem}.pdf"

    mpl_fig, cluster_grid = _mpl_figure(fig)

    if cluster_grid is not None:
        # seaborn ClusterGrid manages its own layout; use g.savefig for correctness
        cluster_grid.savefig(png_path, dpi=300)
        cluster_grid.savefig(pdf_path)
    else:
        mpl_fig.savefig(png_path, dpi=300, bbox_inches="tight")
        mpl_fig.savefig(pdf_path, bbox_inches="tight")

    log_artifact(stem, "figure", png_path, description or "", notebook or "")
    print(f"  [saved figure] {png_path.relative_to(_ROOT)}")
    return png_path


def save_table(
    df: pd.DataFrame,
    name: str,
    subdir: Optional[str] = None,
    description: Optional[str] = None,
    notebook: Optional[str] = None,
    format: str = "csv",
) -> Path:
    """Save a DataFrame to results/tables/, auto-versioning the filename.

    Parameters
    ----------
    df:
        DataFrame to save.
    name:
        Base filename stem (no extension).
    subdir:
        Optional subdirectory under results/tables/.
    description:
        One-sentence description written to ARTIFACTS_LOG.csv.
    notebook:
        Notebook name (without .ipynb).
    format:
        ``"csv"`` (default) or ``"parquet"``.

    Returns
    -------
    Path to the saved file.
    """
    dest_dir = _TABLES_DIR / subdir if subdir else _TABLES_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    ext = "parquet" if format == "parquet" else "csv"
    stem = _versioned_stem(dest_dir, name, ext)
    path = dest_dir / f"{stem}.{ext}"

    if ext == "csv":
        df.to_csv(path)
    else:
        df.to_parquet(path)

    log_artifact(stem, "table", path, description or "", notebook or "")
    print(f"  [saved table]  {path.relative_to(_ROOT)}")
    return path


def save_processed(
    obj,
    name: str,
    subdir: Optional[str] = None,
    description: Optional[str] = None,
    notebook: Optional[str] = None,
) -> Path:
    """Save an intermediate object to data/processed/, auto-versioning.

    Dispatch rules:
    - ``pd.DataFrame``  → Parquet
    - ``np.ndarray``    → .npy
    - anything else     → pickle

    Parameters
    ----------
    obj:
        Object to persist.
    name:
        Base filename stem (no extension).
    subdir:
        Optional subdirectory under data/processed/.
    description:
        One-sentence description written to ARTIFACTS_LOG.csv.
    notebook:
        Notebook name (without .ipynb).

    Returns
    -------
    Path to the saved file.
    """
    dest_dir = _PROCESSED_DIR / subdir if subdir else _PROCESSED_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(obj, pd.DataFrame):
        ext = "parquet"
        stem = _versioned_stem(dest_dir, name, ext)
        path = dest_dir / f"{stem}.{ext}"
        obj.to_parquet(path)
    elif isinstance(obj, np.ndarray):
        ext = "npy"
        stem = _versioned_stem(dest_dir, name, ext)
        path = dest_dir / f"{stem}.{ext}"
        np.save(path, obj)
    else:
        ext = "pkl"
        stem = _versioned_stem(dest_dir, name, ext)
        path = dest_dir / f"{stem}.{ext}"
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)

    log_artifact(stem, "processed", path, description or "", notebook or "")
    print(f"  [saved processed] {path.relative_to(_ROOT)}")
    return path
