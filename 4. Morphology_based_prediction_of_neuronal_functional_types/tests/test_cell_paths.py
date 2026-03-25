"""Tests for cell path resolver."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.util.cell_paths import MODALITY_ROOTS, get_cell_data_dir, get_cell_file_prefix


class TestModalityRoots:
    def test_pa_root(self):
        assert MODALITY_ROOTS["photoactivation"] == Path("paGFP")

    def test_clem_root(self):
        assert MODALITY_ROOTS["clem"] == Path("clem_zfish1")

    def test_em_root(self):
        assert MODALITY_ROOTS["em"] == Path("em_zfish1")


class TestGetCellFilePrefix:
    def test_pa(self):
        assert get_cell_file_prefix("photoactivation", "20210315.1") == "20210315.1"

    def test_clem(self):
        assert get_cell_file_prefix("clem", "cell_123") == "clem_zfish1_cell_123"

    def test_em(self):
        assert get_cell_file_prefix("EM", "456") == "em_fish1_456"

    def test_unknown_modality_raises(self):
        with pytest.raises(ValueError, match="Unknown modality"):
            get_cell_file_prefix("xray", "cell_1")


class TestGetCellDataDir:
    def test_pa(self, tmp_path):
        cell_dir = tmp_path / "paGFP" / "20210315.1"
        cell_dir.mkdir(parents=True)
        result = get_cell_data_dir(tmp_path, "photoactivation", "20210315.1")
        assert result == cell_dir

    def test_clem_functionally_imaged(self, tmp_path):
        cell_dir = (
            tmp_path / "clem_zfish1" / "new_batch_111224"
            / "functionally_imaged_111224" / "clem_zfish1_cell_123"
        )
        cell_dir.mkdir(parents=True)
        result = get_cell_data_dir(tmp_path, "clem", "cell_123")
        assert result == cell_dir

    def test_clem_non_functionally_imaged(self, tmp_path):
        cell_dir = (
            tmp_path / "clem_zfish1" / "new_batch_111224"
            / "non_functionally_imaged_111224" / "clem_zfish1_cell_456"
        )
        cell_dir.mkdir(parents=True)
        result = get_cell_data_dir(tmp_path, "clem", "cell_456")
        assert result == cell_dir

    def test_em(self, tmp_path):
        cell_dir = tmp_path / "em_zfish1" / "em_fish1_789"
        cell_dir.mkdir(parents=True)
        result = get_cell_data_dir(tmp_path, "EM", "789")
        assert result == cell_dir

    def test_not_found_returns_none(self, tmp_path):
        (tmp_path / "clem_zfish1").mkdir()
        result = get_cell_data_dir(tmp_path, "clem", "cell_nonexistent")
        assert result is None

    def test_clem_prefers_functionally_imaged(self, tmp_path):
        """When cell exists in both batch dirs, functionally_imaged wins."""
        for subdir in ("functionally_imaged_111224", "non_functionally_imaged_111224"):
            d = (
                tmp_path / "clem_zfish1" / "new_batch_111224"
                / subdir / "clem_zfish1_cell_both"
            )
            d.mkdir(parents=True)
        result = get_cell_data_dir(tmp_path, "clem", "cell_both")
        assert "functionally_imaged_111224" in str(result)

    def test_clem_all_cells(self, tmp_path):
        """CLEM cell found in all_cells directory."""
        cell_dir = tmp_path / "clem_zfish1" / "all_cells" / "clem_zfish1_cell_999"
        cell_dir.mkdir(parents=True)
        result = get_cell_data_dir(tmp_path, "clem", "cell_999")
        assert result == cell_dir
