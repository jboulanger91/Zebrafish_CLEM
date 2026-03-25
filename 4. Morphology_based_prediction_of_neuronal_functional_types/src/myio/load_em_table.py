"""Load EM cell metadata tables.

This module loads EM cell metadata from the cell inventory xlsx
(metadata.xlsx, EM sheet).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.util.cell_paths import get_cell_data_dir


def load_em_table(xlsx_path: Path | str, em_data_dir: Path | str) -> pd.DataFrame:
    """Load EM cell metadata from the cell inventory xlsx.

    Reads the EM sheet from metadata.xlsx and filters
    to cells with data on disk (has_data column).

    Parameters
    ----------
    xlsx_path : Path or str
        Path to metadata.xlsx.
    em_data_dir : Path or str
        Path to the em_zfish1 data directory (kept for signature compatibility).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with cell metadata.
    """
    xlsx_path = Path(xlsx_path)
    em_data_dir = Path(em_data_dir)

    em_table = pd.read_excel(xlsx_path, sheet_name="EM")

    # Use cell_id as the canonical cell name (matches existing convention:
    # the old directory-scanning loader renamed 'id' → 'cell_name')
    em_table["cell_name"] = em_table["cell_id"].astype(str)

    em_table["imaging_modality"] = "EM"

    # Filter to cells that have data on disk (excludes 11
    # cell_89189_postsyn_incomplete cells with no metadata/SWC)
    em_table = em_table[em_table["has_data"] == True]  # noqa: E712

    # Resolve cell_data_dir for each cell
    em_table["cell_data_dir"] = em_table["cell_name"].apply(
        lambda name: get_cell_data_dir(em_data_dir.parent, "EM", name)
    )

    return em_table


if __name__ == "__main__":
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from src.util.get_base_path import get_base_path
    from src.util.project_root import get_cell_inventory_xlsx

    my_df = load_em_table(get_cell_inventory_xlsx(), get_base_path() / "em_zfish1")
    print(f"Success! Loaded {len(my_df)} rows")
