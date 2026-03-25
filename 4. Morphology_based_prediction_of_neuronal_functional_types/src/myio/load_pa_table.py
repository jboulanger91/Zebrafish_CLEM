"""Load photoactivation cell tables.

This module loads PA cell metadata from the cell inventory xlsx
(metadata.xlsx, PA sheet) and resolves per-cell data
directories from the paGFP data directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.util.cell_paths import get_cell_data_dir


def load_pa_table(xlsx_path: Path | str, pa_data_dir: Path | str) -> pd.DataFrame:
    """Load PA cell metadata from the cell inventory xlsx.

    Reads the PA sheet from metadata.xlsx, filters for
    valid cells, and resolves metadata file paths.

    Parameters
    ----------
    xlsx_path : Path or str
        Path to metadata.xlsx.
    pa_data_dir : Path or str
        Path to the paGFP data directory containing cell subdirectories.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with cell metadata and cell_data_dir column.
    """
    xlsx_path = Path(xlsx_path)
    pa_data_dir = Path(pa_data_dir)

    pa_table = pd.read_excel(xlsx_path, sheet_name="PA")

    pa_table = pa_table.dropna(subset=["original_function", "date_of_tracing"])
    pa_table["cell_name"] = pa_table["cell_name"].astype(str)

    pa_table["cell_data_dir"] = pa_table["cell_name"].apply(
        lambda name: get_cell_data_dir(pa_data_dir.parent, "photoactivation", name)
    )
    pa_table = pa_table.dropna(subset=["cell_data_dir"])

    return pa_table


if __name__ == "__main__":
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from src.util.get_base_path import get_base_path
    from src.util.project_root import get_cell_inventory_xlsx

    pa_table = load_pa_table(get_cell_inventory_xlsx(), get_base_path() / "paGFP")
    print(f"Success! Loaded {len(pa_table)} rows")
