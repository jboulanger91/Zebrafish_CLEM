"""Unified project root resolution.

All paths in the repository (input data, output, config) derive from a single
project root directory.  This module provides the canonical functions for
locating that root and its standard subdirectories.

Resolution order for the project root:

1. ``HBSF_ROOT`` environment variable (if set).
2. ``~/Desktop/hbsf/`` on macOS (if the directory exists).
3. ``~/hbsf/`` on any platform (if the directory exists).
4. Backward-compatible: derive from ``config/path_configuration.txt``.

Standard subdirectory layout::

    {HBSF_ROOT}/
      data/          # read-only scientific input data (or hbsf_input/)
      output/        # all generated output and intermediates (or hbsf_output/)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Subdirectory names, preferred first
_DATA_DIR_NAMES = ("data", "hbsf_input")
_OUTPUT_DIR_NAMES = ("hbsf_output", "output")


def get_project_root() -> Path:
    """Return the unified project root directory.

    Returns
    -------
        Path to the project root (e.g. ``~/Desktop/hbsf``).

    Raises
    ------
        FileNotFoundError: If no project root can be resolved.
    """
    # 1. Environment variable
    env = os.environ.get("HBSF_ROOT")
    if env:
        root = Path(env).expanduser()
        if root.is_dir():
            return root

    # 2. macOS Desktop convention
    if sys.platform == "darwin":
        candidate = Path.home() / "Desktop" / "hbsf"
        if candidate.is_dir():
            return candidate

    # 3. Home directory fallback
    candidate = Path.home() / "hbsf"
    if candidate.is_dir():
        return candidate

    # 4. Backward compat: derive from config file
    return _derive_root_from_config()


def _derive_root_from_config() -> Path:
    """Infer the project root from the legacy path_configuration.txt.

    If the configured data path is a child of a directory that also contains
    an output folder, use that parent as the project root.  Otherwise, fall
    back to the parent of the configured data path itself.
    """
    config_path = _REPO_ROOT / "config" / "path_configuration.txt"
    if not config_path.exists():
        msg = (
            "Cannot determine HBSF project root. Set the HBSF_ROOT "
            "environment variable or create ~/Desktop/hbsf/."
        )
        raise FileNotFoundError(msg)

    current_user = Path.home().name
    with config_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) == 2 and parts[0] == current_user:
                data_path = Path(parts[1]).expanduser()
                # If data_path's parent looks like a project root, use it
                parent = data_path.parent
                if any((parent / name).is_dir() for name in _OUTPUT_DIR_NAMES):
                    return parent
                return parent

    msg = (
        f"No path configured for user '{current_user}'. "
        "Set HBSF_ROOT or run: python scripts/setup_data_paths.py"
    )
    raise FileNotFoundError(msg)


def get_data_dir() -> Path:
    """Return the input data directory (``{HBSF_ROOT}/data/``).

    Checks for ``data/`` first, then ``hbsf_input/`` for backward
    compatibility.

    Returns
    -------
        Path to the data directory.
    """
    root = get_project_root()
    for name in _DATA_DIR_NAMES:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    # Default: create data/ if nothing exists yet
    default = root / _DATA_DIR_NAMES[0]
    default.mkdir(parents=True, exist_ok=True)
    return default


def get_output_root() -> Path:
    """Return the output directory root (``{HBSF_ROOT}/output/``).

    Checks for ``output/`` first, then ``hbsf_output/`` for backward
    compatibility.  Creates the directory if it doesn't exist.

    Returns
    -------
        Path to the output root directory.
    """
    root = get_project_root()
    for name in _OUTPUT_DIR_NAMES:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    # Default: create output/ if nothing exists yet
    default = root / _OUTPUT_DIR_NAMES[0]
    default.mkdir(parents=True, exist_ok=True)
    return default


# Default cell inventory filename (override with HBSF_CELL_INVENTORY env var)
_DEFAULT_CELL_INVENTORY = "metadata.xlsx"


def get_cell_inventory_xlsx() -> Path:
    """Return the path to the cell inventory xlsx.

    Resolution order:
    1. ``HBSF_CELL_INVENTORY`` environment variable (full path).
    2. ``{data_dir}/{_DEFAULT_CELL_INVENTORY}``.

    Returns
    -------
        Path to the cell inventory xlsx file.
    """
    env = os.environ.get("HBSF_CELL_INVENTORY")
    if env:
        return Path(env).expanduser()
    return get_data_dir() / _DEFAULT_CELL_INVENTORY


if __name__ == "__main__":
    try:
        root = get_project_root()
        print(f"Project root: {root}")
        print(f"Data dir:     {get_data_dir()}")
        print(f"Output root:  {get_output_root()}")
    except (FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}")
