"""Utility functions for the hindbrain structure-function analysis."""

from .get_base_path import NotSetup, get_base_path

# Lazy re-exports -- avoids RuntimeWarning when running submodules with
# ``python -m src.util.project_root`` (the module would already be in
# sys.modules from this __init__ import before __main__ runs).


def get_project_root():  # noqa: D401
    """Canonical project root directory."""
    from .project_root import get_project_root as _f

    return _f()


def get_data_dir():  # noqa: D401
    """Input data directory under the project root."""
    from .project_root import get_data_dir as _f

    return _f()


def get_output_root():  # noqa: D401
    """Output directory root under the project root."""
    from .project_root import get_output_root as _f

    return _f()


def get_output_dir(module: str, *subdirs: str):
    """Return and create the output directory for a module."""
    from .output_paths import get_output_dir as _f

    return _f(module, *subdirs)


__all__ = [
    "get_base_path",
    "get_data_dir",
    "get_output_dir",
    "get_output_root",
    "get_project_root",
    "NotSetup",
]
