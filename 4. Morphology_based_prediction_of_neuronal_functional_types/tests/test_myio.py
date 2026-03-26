"""Unit tests for the src.myio I/O loader module.

Tests cover:
- Import smoke tests and API shape
- _parsing.py internal utilities
- Table loader signatures and error handling
- Mesh loader signatures
- Integration / pipeline function signatures

No actual data files are required -- tests exercise importability,
function signatures, synthetic data parsing, and temp-file I/O only.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure the repository root is on sys.path so ``src.*`` is importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "src")]:
    if _p not in sys.path:
        # Required for standalone execution (remove after pip install -e .)
        # _REPO_ROOT/"src": needed for bare myio.* / util.* imports
        sys.path.insert(0, _p)


# =========================================================================
# Group 1: Import and API shape (3 tests)
# =========================================================================


class TestImportAndAPIShape:
    """Verify that the myio module is importable and exposes expected symbols."""

    # Core symbols that do NOT depend on caveclient
    _CORE_SYMBOLS = [
        "load_clem_table",
        "load_em_table",
        "load_mesh",
        "load_pa_table",
    ]

    def test_all_public_symbols_importable(self):
        """All public symbols listed in __init__.py are importable."""
        mod = importlib.import_module("src.myio")
        for name in self._CORE_SYMBOLS:
            assert hasattr(mod, name), f"src.myio missing expected symbol: {name}"
            assert getattr(mod, name) is not None

    def test_key_functions_have_expected_signatures(self):
        """Key loader functions accept the parameters documented in their source."""
        from src.myio import load_pa_table

        # load_pa_table(xlsx_path, pa_data_dir)
        sig = inspect.signature(load_pa_table)
        assert "xlsx_path" in sig.parameters
        assert "pa_data_dir" in sig.parameters

    def test_all_list_matches_actual_exports(self):
        """__all__ contains the expected core symbols (non-CAVE subset)."""
        from src.myio import __all__ as myio_all

        # Every core symbol must be in __all__
        for name in self._CORE_SYMBOLS:
            assert name in myio_all, f"{name!r} missing from src.myio.__all__"



# =========================================================================
# Group 2: _parsing.py (2 tests)
# =========================================================================


class TestParsing:
    """Tests for myio._parsing.read_to_pandas_row."""

    def test_parses_key_value_format(self):
        """read_to_pandas_row correctly parses key=value lines."""
        from src.myio._parsing import read_to_pandas_row

        text = "name=cell_1\ncount=42\nvalues=[1, 2, 3]\nstatus=nan"
        df = read_to_pandas_row(text)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

        assert df["name"].iloc[0] == "cell_1"
        assert df["count"].iloc[0] == 42
        assert df["values"].iloc[0] == [1, 2, 3]
        assert pd.isna(df["status"].iloc[0])

    def test_handles_edge_cases(self):
        """read_to_pandas_row handles equals signs in values and quoted values."""
        from src.myio._parsing import read_to_pandas_row

        # Value containing an equals sign (split on first '=' only)
        text = 'formula=a=b+c\nquoted_val="hello world"'
        df = read_to_pandas_row(text)

        assert df["formula"].iloc[0] == "a=b+c"
        # Quotes are stripped
        assert df["quoted_val"].iloc[0] == "hello world"


# =========================================================================
# Group 3: Table loaders - signatures and error handling (5 tests)
# =========================================================================


class TestTableLoaders:
    """Tests for load_pa_table, load_clem_table, load_em_table."""

    def test_load_pa_table_callable_and_has_expected_params(self):
        """load_pa_table is callable and accepts xlsx_path and pa_data_dir."""
        from src.myio import load_pa_table

        assert callable(load_pa_table)
        sig = inspect.signature(load_pa_table)
        assert "xlsx_path" in sig.parameters
        assert "pa_data_dir" in sig.parameters

    def test_load_clem_table_callable_and_has_expected_params(self):
        """load_clem_table is callable and accepts xlsx_path."""
        from src.myio import load_clem_table

        assert callable(load_clem_table)
        sig = inspect.signature(load_clem_table)
        assert "xlsx_path" in sig.parameters
        assert "clem_data_dir" not in sig.parameters

    def test_load_em_table_callable_and_has_expected_params(self):
        """load_em_table is callable and accepts xlsx_path and em_data_dir."""
        from src.myio import load_em_table

        assert callable(load_em_table)
        sig = inspect.signature(load_em_table)
        assert "xlsx_path" in sig.parameters
        assert "em_data_dir" in sig.parameters

    def test_table_loaders_raise_on_nonexistent_path(self):
        """load_pa_table raises an error when given a nonexistent xlsx path."""
        from src.myio import load_pa_table

        bogus_xlsx = Path("/nonexistent/path/table.xlsx")
        bogus_dir = Path("/nonexistent/path/paGFP")
        with pytest.raises((FileNotFoundError, OSError)):
            load_pa_table(bogus_xlsx, bogus_dir)


# =========================================================================
# Group 4: Mesh loaders (1 test)
# =========================================================================


class TestMeshLoaders:
    """Tests for load_mesh."""

    def test_load_mesh_has_expected_params(self):
        """load_mesh accepts cell, swc."""
        from src.myio import load_mesh

        assert callable(load_mesh)
        sig = inspect.signature(load_mesh)
        param_names = list(sig.parameters.keys())
        assert "cell" in param_names
        assert "swc" in param_names


# =========================================================================
# Group 5: Integration (3 tests)
# =========================================================================


class TestIntegration:
    """Integration-level tests for the pipeline and module structure."""

    def test_load_cells_predictor_pipeline_has_expected_params(self):
        """load_cells_predictor_pipeline accepts all documented parameters."""
        from src.myio.load_cells2df import load_cells_predictor_pipeline

        assert callable(load_cells_predictor_pipeline)
        sig = inspect.signature(load_cells_predictor_pipeline)
        param_names = list(sig.parameters.keys())

        expected = [
            "modalities",
            "mirror",
            "path_to_data",
            "filter_incomplete_clem",
            "load_morphology",
        ]
        for name in expected:
            assert name in param_names, (
                f"load_cells_predictor_pipeline missing parameter: {name}"
            )

        # Check key defaults
        assert sig.parameters["mirror"].default is True
        assert sig.parameters["load_morphology"].default is True

    def test_loader_functions_have_return_annotations_or_documented_types(self):
        """All core loader functions have return type annotations."""
        from src.myio import (
            load_clem_table,
            load_em_table,
            load_pa_table,
        )

        functions_with_annotations = [
            load_pa_table,
            load_clem_table,
            load_em_table,
        ]

        for func in functions_with_annotations:
            sig = inspect.signature(func)
            assert sig.return_annotation is not inspect.Parameter.empty, (
                f"{func.__name__} is missing a return type annotation"
            )

    def test_module_docstring_exists(self):
        """The myio module has a docstring that describes key functionality."""
        import src.myio as myio_mod

        assert myio_mod.__doc__ is not None
        doc = myio_mod.__doc__
        assert len(doc) > 50, "Module docstring is too short"
        assert "load" in doc.lower() or "I/O" in doc or "cell" in doc.lower()
