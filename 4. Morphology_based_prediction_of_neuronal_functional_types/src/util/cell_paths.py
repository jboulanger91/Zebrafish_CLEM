"""Central cell path resolver.

Computes cell data directories and file prefixes from
(data_root, modality, cell_name). CLEM cells require a
directory probe because the batch subdirectory is not in the xlsx.
"""
from __future__ import annotations

from pathlib import Path

# Modality root directories (relative to data_root)
MODALITY_ROOTS = {
    "photoactivation": Path("paGFP"),
    "clem": Path("clem_zfish1"),
    "em": Path("em_zfish1"),
}

# CLEM subdirectories to probe, preferred order
_CLEM_BATCH_DIRS = (
    Path("clem_zfish1") / "new_batch_111224" / "functionally_imaged_111224",
    Path("clem_zfish1") / "new_batch_111224" / "non_functionally_imaged_111224",
    Path("clem_zfish1") / "all_cells",
)


def get_cell_file_prefix(modality: str, cell_name: str) -> str:
    """Return the on-disk file prefix for a cell.

    Parameters
    ----------
    modality : str
        One of 'photoactivation', 'clem', 'EM'.
    cell_name : str
        Cell name as stored in the DataFrame.
    """
    mod = modality.lower()
    if mod == "photoactivation":
        return cell_name
    if mod == "clem":
        if cell_name.startswith("clem_zfish1_"):
            return cell_name
        return f"clem_zfish1_{cell_name}"
    if mod == "em":
        return f"em_fish1_{cell_name}"
    raise ValueError(f"Unknown modality: {modality!r}")


def get_cell_data_dir(
    data_root: Path,
    modality: str,
    cell_name: str,
) -> Path | None:
    """Return the cell data directory on disk, or None if not found.

    Parameters
    ----------
    data_root : Path
        Base data directory (e.g. get_base_path()).
    modality : str
        One of 'photoactivation', 'clem', 'EM'.
    cell_name : str
        Cell name as stored in the DataFrame.
    """
    mod = modality.lower()
    if mod == "photoactivation":
        candidate = data_root / "paGFP" / cell_name
        return candidate if candidate.is_dir() else None
    if mod == "clem":
        # Axon segments already have the full dir name (clem_zfish1_axon_...)
        if cell_name.startswith("clem_zfish1_"):
            dir_name = cell_name
        else:
            dir_name = f"clem_zfish1_{cell_name}"
        for batch in _CLEM_BATCH_DIRS:
            candidate = data_root / batch / dir_name
            if candidate.is_dir():
                return candidate
        return None
    if mod == "em":
        candidate = data_root / "em_zfish1" / f"em_fish1_{cell_name}"
        return candidate if candidate.is_dir() else None
    raise ValueError(f"Unknown modality: {modality!r}")
