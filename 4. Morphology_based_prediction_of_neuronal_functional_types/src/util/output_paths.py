"""Centralized output path configuration.

All scripts in the repository should use :func:`get_output_dir` to obtain
their output directory.  This ensures every generated file lands under a
single, predictable root, grouped by the module/script that created it.

The output root is determined by (in order of priority):
1. The ``HBSF_OUTPUT_ROOT`` environment variable, if set.
2. ``~/Desktop/hbsf_output/`` on macOS (where ``~/Desktop`` exists).
3. ``~/hbsf_output/`` on Linux, Windows, or any system without a Desktop.

Usage::

    from src.util.output_paths import get_output_dir

    out = get_output_dir("classifier_pipeline", "confusion_matrices")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_output_root() -> Path:
    """Detect the output root from unified project root, environment, or OS conventions."""
    # 1. Try unified project root
    try:
        from src.util.project_root import get_output_root

        return get_output_root()
    except (FileNotFoundError, ImportError):
        pass

    # 2. Legacy: HBSF_OUTPUT_ROOT environment variable
    env = os.environ.get("HBSF_OUTPUT_ROOT")
    if env:
        return Path(env)

    # 3. Platform defaults
    desktop = Path.home() / "Desktop"
    if sys.platform == "darwin" and desktop.is_dir():
        return desktop / "hbsf_output"
    return Path.home() / "hbsf_output"


OUTPUT_ROOT: Path = _resolve_output_root()


def get_output_dir(module: str, *subdirs: str) -> Path:
    """Return and create the output directory for a module.

    Args:
        module: Top-level subfolder name (e.g. ``"classifier_pipeline"``).
        *subdirs: Optional deeper subdirectories.

    Returns
    -------
        The resolved, existing directory path.
    """
    d = OUTPUT_ROOT / module
    for s in subdirs:
        d = d / s
    d.mkdir(parents=True, exist_ok=True)
    return d
