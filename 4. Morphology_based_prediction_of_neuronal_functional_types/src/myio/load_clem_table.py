"""Load CLEM cell metadata tables.

This module loads CLEM cell metadata from the cell inventory xlsx
(metadata.xlsx, CLEM sheet) and resolves per-cell data
directories from the CLEM data directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.util.cell_paths import get_cell_data_dir
from src.util.get_base_path import get_base_path


def load_clem_table(xlsx_path: Path | str) -> pd.DataFrame:
    """Load CLEM cell metadata from the cell inventory xlsx.

    Reads the CLEM sheet from metadata.xlsx, sets function/
    morphology columns directly from xlsx, computes reconstruction_complete
    boolean, and resolves metadata file paths.

    Parameters
    ----------
    xlsx_path : Path or str
        Path to metadata.xlsx.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with cell metadata and cell_data_dir column.
    """
    xlsx_path = Path(xlsx_path)

    clem_table = pd.read_excel(xlsx_path, sheet_name="CLEM")

    clem_table["imaging_modality"] = "clem"

    # Construct cell_name: cells use nucleus_id, axon segments use axon_id
    is_axon = clem_table.get("type", "") == "axon"
    clem_table["cell_name"] = "cell_" + clem_table["nucleus_id"].astype(str)
    if is_axon.any() and "axon_id" in clem_table.columns:
        clem_table.loc[is_axon, "cell_name"] = (
            "clem_zfish1_axon_" + clem_table.loc[is_axon, "axon_id"].astype(str)
        )

    # Filter original_function column: only recognized functional types are kept;
    # others (e.g. "myelinated") become NaN → "to_predict" downstream.
    from .load_cells2df import CELL_TYPE_CATEGORIES

    valid_functions = set(CELL_TYPE_CATEGORIES["function"])
    clem_table["original_function"] = clem_table["original_function"].where(
        clem_table["original_function"].isin(valid_functions)
    )

    # Mark whether reconstruction is complete (both axon and dendrites)
    clem_table["reconstruction_complete"] = (
        (clem_table["axon_reconstruction_status"] == "complete")
        & (clem_table["dendrite_reconstruction_status"] == "complete")
    )

    # Resolve cell_data_dir for each cell (probes batch subdirectories)
    data_root = get_base_path()
    clem_table["cell_data_dir"] = clem_table["cell_name"].apply(
        lambda name: get_cell_data_dir(data_root, "clem", name)
    )
    clem_table = clem_table.dropna(subset=["cell_data_dir"])

    return clem_table


if __name__ == "__main__":
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from src.util.project_root import get_cell_inventory_xlsx

    my_df = load_clem_table(get_cell_inventory_xlsx())
    print(f"Success! Loaded {len(my_df)} rows" if my_df is not None else "No data found")
