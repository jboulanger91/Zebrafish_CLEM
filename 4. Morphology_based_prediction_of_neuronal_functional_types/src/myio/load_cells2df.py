"""Pipeline for loading multi-modality cell data into a unified DataFrame.

This module provides the main data loading pipeline for the hindbrain structure-function
project, loading and combining cell data from multiple imaging modalities:

- **PA (Photoactivation)**: Single-cell labeling using photoactivatable GFP
- **CLEM**: Correlative Light and Electron Microscopy
- **EM**: Electron Microscopy (connectomics data)

The pipeline handles:
1. Loading cell tables from each modality
2. Loading SWC skeletons
3. Mirroring cells to left hemisphere for consistent analysis
4. Extracting cell type labels (function, morphology, neurotransmitter)
5. Filtering incomplete reconstructions

Example:
-------
>>> from myio.load_cells2df import load_cells_predictor_pipeline
>>> all_cells = load_cells_predictor_pipeline(modalities=['pa', 'clem'])
>>> print(f"Loaded {len(all_cells)} cells")
"""

from __future__ import annotations

import sys
from pathlib import Path

import navis
import numpy as np
import pandas as pd

# Required for cross-package bare imports (util.*) in standalone execution
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Sibling imports use relative paths; cross-package use bare imports via sys.path
from util.get_base_path import get_base_path  # noqa: E402
from util.project_root import get_cell_inventory_xlsx  # noqa: E402

from .load_clem_table import load_clem_table  # noqa: E402
from .load_em_table import load_em_table  # noqa: E402
from .load_mesh import load_mesh  # noqa: E402
from .load_pa_table import load_pa_table  # noqa: E402

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Brain width in microns (for mirroring cells to left hemisphere)
BRAIN_WIDTH_UM = 495.56

# Valid cell type categories for label extraction
# Note: Old labels in metadata files are mapped to modern nomenclature:
#   - integrator -> motion_integrator (MI)
#   - dynamic_threshold -> motion_onset (MON)
#   - motor_command -> slow_motion_integrator (SMI)
CELL_TYPE_CATEGORIES = {
    "morphology": ["ipsilateral", "contralateral"],
    "neurotransmitter": ["inhibitory", "excitatory"],
    "function": [
        "motion_integrator",
        "motion_onset",
        "motion onset",
        "slow_motion_integrator",
        "slow motion integrator",
        # Legacy nomenclature (accepted for backward compatibility)
        "integrator",
        "dynamic threshold",
        "dynamic_threshold",
        "motor command",
        "motor_command",
        # Non-canonical labels
        "no response",
        "off-response",
        "motion responsive",
        "noisy, little modulation",
        "non-direction-selective, on response",
    ],
}

# Mapping from legacy nomenclature to modern nomenclature
# Based on: "Unbiased clustering revealed three main functional cell types.
# Based on their dynamics, we named these cells motion integrator (MI),
# motion onset (MON), and slow motion integrator (SMI) neurons."
FUNCTION_NAME_MAP = {
    "motion_integrator": "motion_integrator",
    "motion_onset": "motion_onset",
    "motion onset": "motion_onset",
    "motion_onset_neuron": "motion_onset",
    "slow_motion_integrator": "slow_motion_integrator",
    "slow motion integrator": "slow_motion_integrator",
    # Legacy nomenclature (accepted for backward compatibility)
    "integrator": "motion_integrator",
    "dynamic threshold": "motion_onset",
    "dynamic_threshold": "motion_onset",
    "motor command": "slow_motion_integrator",
    "motor_command": "slow_motion_integrator",
    # Directional variants (legacy ordering)
    "integrator_ipsilateral": "motion_integrator_ipsilateral",
    "integrator_contralateral": "motion_integrator_contralateral",
    "ipsilateral_integrator": "motion_integrator_ipsilateral",
    "contralateral_integrator": "motion_integrator_contralateral",
    "ipsilateral_motion_integrator": "motion_integrator_ipsilateral",
    "contralateral_motion_integrator": "motion_integrator_contralateral",
    # Short forms
    "MI": "motion_integrator",
    "MON": "motion_onset",
    "SMI": "slow_motion_integrator",
}

# Canonical functional types (modern nomenclature)
CANONICAL_FUNCTIONS = ["motion_integrator", "motion_onset", "slow_motion_integrator"]


def normalize_label(label: str) -> str:
    """Normalize a function label to modern nomenclature.

    Handles exact matches (e.g. 'integrator' -> 'motion_integrator') and
    prefix matches for suffixed variants (e.g. 'integrator_contralateral'
    -> 'motion_integrator_contralateral').

    Uses FUNCTION_NAME_MAP as the single source of truth.
    """
    if not isinstance(label, str):
        return label
    if label in FUNCTION_NAME_MAP:
        return FUNCTION_NAME_MAP[label]
    for old, new in FUNCTION_NAME_MAP.items():
        if label.startswith(old + "_"):
            return new + label[len(old):]
    return label



# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------


def _load_modality_tables(
    modalities: list[str],
    path_to_data: Path,
) -> pd.DataFrame:
    """Load and concatenate cell tables for specified imaging modalities.

    Parameters
    ----------
    modalities : list of str
        List of modality names to load. Valid options:
        'pa', 'clem', 'clem_predict', 'em'.
        Both 'clem' and 'clem_predict' load the full CLEM table;
        the training/prediction split is handled downstream by
        _filter_and_finalize via the used_for_training column.
    path_to_data : Path
        Base path to the CLEM paper data directory.

    Returns
    -------
    pd.DataFrame
        Combined cell table from all requested modalities.
    """
    tables = []

    if "pa" in modalities:
        pa_data_dir = path_to_data / "paGFP"
        tables.append(load_pa_table(get_cell_inventory_xlsx(), pa_data_dir))

    if "clem" in modalities or "clem_predict" in modalities:
        tables.append(load_clem_table(get_cell_inventory_xlsx()))

    if "em" in modalities:
        em_data_dir = path_to_data / "em_zfish1"
        if em_data_dir.exists():
            tables.append(load_em_table(get_cell_inventory_xlsx(), em_data_dir))

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _load_skeletons_for_cells(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Load SWC skeleton data for all cells.

    Parameters
    ----------
    df : pd.DataFrame
        Cell DataFrame with metadata.

    Returns
    -------
    pd.DataFrame
        DataFrame with loaded SWC data.
    """
    for i, cell in df.iterrows():
        df.loc[i, :] = load_mesh(cell, swc=True)
        if isinstance(df.loc[i, "swc"], float):
            print(f"{cell.cell_name} is not a TreeNeuron")
    return df


def _mirror_cells(df: pd.DataFrame, brain_width: float = BRAIN_WIDTH_UM) -> pd.DataFrame:
    """Mirror all cells in the right hemisphere to the left.

    Parameters
    ----------
    df : pd.DataFrame
        Cell DataFrame with SWC data.
    brain_width : float
        Brain width in microns.

    Returns
    -------
    pd.DataFrame
        DataFrame with mirrored cells.
    """
    for i, cell in df.iterrows():
        if "swc" in cell.index:
            swc = cell["swc"]
            if (
                swc is not None
                and not isinstance(swc, float)
                and swc.nodes.loc[0, "x"] > (brain_width / 2)
            ):
                    _mirror_cell_swc(df, i, brain_width)

    return df


def _mirror_cell_swc(df: pd.DataFrame, idx: int, brain_width: float) -> None:
    """Mirror SWC skeleton for a single cell."""
    swc = df.loc[idx, "swc"]
    df.loc[idx, "swc"].nodes.loc[:, ["x", "y", "z"]] = navis.transforms.mirror(
        np.array(swc.nodes.loc[:, ["x", "y", "z"]]), brain_width, "x"
    )
    if df.loc[idx, "swc"].connectors is not None:
        df.loc[idx, "swc"].connectors.loc[:, ["x", "y", "z"]] = navis.transforms.mirror(
            np.array(swc.connectors.loc[:, ["x", "y", "z"]]), brain_width, "x"
        )
    print(f"SWC of cell {df.loc[idx, 'cell_name']} mirrored")


def _filter_and_finalize(
    df: pd.DataFrame,
    filter_incomplete_clem: bool = True,
    filter_invalid_swc: bool = True,
    label_column: str = "kmeans_function",
) -> pd.DataFrame:
    """Filter cells, apply kmeans labels, and finalize the DataFrame.

    Steps:
    1. Drop empty rows and cells without valid SWC (if morphology loaded)
    2. Preserve raw xlsx ``function`` as ``original_function`` (normalized)
    3. Mark negative controls and ensure ``used_for_training`` exists
    4. Filter incomplete CLEM reconstructions (using ``original_function``)
    5. Set ``function``: training cells get their label from
       ``label_column``; neg_control cells keep ``original_function``;
       everything else (incl. EM) becomes ``to_predict``

    Columns produced:
    - ``original_function``: raw functional type from xlsx, normalized
    - ``function``: training label from xlsx ``label_column``

    Parameters
    ----------
    df : pd.DataFrame
        Cell DataFrame.
    filter_incomplete_clem : bool
        If True, filter out incomplete CLEM reconstructions that are canonical functions.
        Default True (enabled by default; set to False in kmeans scripts to use all data).
    filter_invalid_swc : bool
        If True, drop cells without valid SWC data. Default True.
        Set to False when loading metadata only (load_morphology=False).
    label_column : str
        Column in the xlsx to use as the functional label source.
        Default ``"kmeans_function"``.

    Returns
    -------
    pd.DataFrame
        Finalized DataFrame with ``original_function`` and ``function``.
    """
    df = df.dropna(how="all")

    # Report and drop cells without valid SWC (skip if morphology wasn't loaded)
    if filter_invalid_swc:
        valid_swc = df["swc"].apply(lambda x: isinstance(x, navis.TreeNeuron))
        print(f"{(~valid_swc).sum()} cells dropped because no swc")
        print(df.loc[~valid_swc, "cell_name"])
        df = df.loc[valid_swc]

    # Ensure required columns exist
    if "original_function" not in df.columns:
        df["original_function"] = np.nan
    if "reconstruction_complete" not in df.columns:
        df["reconstruction_complete"] = True

    # Normalize original_function (fill NaN, map old nomenclature)
    df.loc[df["original_function"].isna(), "original_function"] = "to_predict"
    df["original_function"] = df["original_function"].apply(
        lambda x: FUNCTION_NAME_MAP.get(str(x).replace(" ", "_"), str(x).replace(" ", "_"))
    )

    # Cells without a label in the chosen label column are negative controls
    # (exclude to_predict cells and canonical-function cells without training flag)
    df["is_neg_control"] = (
        df[label_column].isna()
        & (df["imaging_modality"] != "EM")
        & (df["original_function"] != "to_predict")
        & ~df["original_function"].isin(CANONICAL_FUNCTIONS)
    ) if label_column in df.columns else False
    # Ensure used_for_training exists for all cells (EM cells don't have it from xlsx)
    if "used_for_training" not in df.columns:
        df["used_for_training"] = False
    df["used_for_training"] = df["used_for_training"].fillna(False)

    # Drop canonical CLEM cells with incomplete reconstructions (unreliable features)
    if filter_incomplete_clem:
        exclude = (
            (df["imaging_modality"] == "clem")
            & ~df["reconstruction_complete"].fillna(True)
            & df["original_function"].isin(CANONICAL_FUNCTIONS)
        )
        if exclude.any():
            print(f"Filtering {exclude.sum()} incomplete CLEM reconstructions")
        df = df.loc[~exclude]

    # Set function from label_column (default: xlsx "kmeans_function").
    # - Training cells with valid label → label value (normalized)
    # - neg_control cells → keep original_function
    # - Everything else (incl. EM) → "to_predict"
    has_column = label_column in df.columns
    is_neg = df["is_neg_control"].fillna(False)
    df["function"] = "to_predict"  # default for all cells

    # neg_control cells keep their original function
    df.loc[is_neg, "function"] = df.loc[is_neg, "original_function"]

    # Training cells get their label from the chosen column
    if has_column:
        is_training = df["used_for_training"].fillna(False).astype(bool)
        has_label = df[label_column].notna()
        train_with_label = is_training & has_label & ~is_neg
        df.loc[train_with_label, "function"] = (
            df.loc[train_with_label, label_column]
            .astype(str)
            .map(lambda x: FUNCTION_NAME_MAP.get(x, x))
        )

    return df


def _apply_alt_neurotransmitter(df: pd.DataFrame) -> pd.DataFrame:
    """Override PA neurotransmitter with ``neurotransmitter_new`` from xlsx.

    The ``neurotransmitter_new`` column (derived from ``cells2show.xlsx``)
    was added to the PA sheet of ``metadata.xlsx``.  Where it
    has a value, it replaces the original ``neurotransmitter`` column.
    """
    if "neurotransmitter_new" not in df.columns:
        return df

    pa_mask = df["imaging_modality"] == "photoactivation"
    has_new = pa_mask & df["neurotransmitter_new"].notna()
    df.loc[has_new, "neurotransmitter"] = df.loc[has_new, "neurotransmitter_new"]
    return df


def _apply_morphology_gregor(df: pd.DataFrame) -> pd.DataFrame:
    """Override CLEM morphology with Gregor's manual annotations from xlsx.

    The ``morphology_gregor`` column in the CLEM sheet of
    ``metadata.xlsx`` contains expert morphology annotations
    for motion_onset cells.  Where it has a value, it replaces the
    original ``morphology`` column.
    """
    if "morphology_gregor" not in df.columns:
        return df

    has_gregor = df["morphology_gregor"].notna()
    df.loc[has_gregor, "morphology"] = df.loc[has_gregor, "morphology_gregor"]
    return df



# ------------------------------------------------------------------
# Main Pipeline Function
# ------------------------------------------------------------------


def load_cells_predictor_pipeline(
    modalities: list[str] | None = None,
    mirror: bool = True,
    path_to_data: Path | None = None,
    filter_incomplete_clem: bool = True,
    load_morphology: bool = True,
    label_column: str = "kmeans_function",
) -> pd.DataFrame:
    """Load and process cell data from multiple imaging modalities.

    This is the main pipeline function that:
    1. Loads cell metadata tables from specified modalities
    2. Loads SWC skeletons
    3. Optionally mirrors cells to left hemisphere
    4. Extracts cell type labels (function, morphology, neurotransmitter)
    5. Filters incomplete reconstructions

    Parameters
    ----------
    modalities : list of str
        Modalities to load. Options:
        - 'pa': Photoactivation cells
        - 'clem', 'clem_predict': CLEM cells (loads full CLEM table;
          training/prediction split handled downstream)
        - 'em': Electron microscopy
    mirror : bool
        Mirror cells in right hemisphere to left. Default True.
    path_to_data : Path, optional
        Base path to data directory. Uses get_base_path() if None.
    filter_incomplete_clem : bool
        Filter out incomplete CLEM reconstructions. Default True (enabled
        by default; set to False in kmeans scripts to use all data for clustering).
    load_morphology : bool
        Load SWC skeletons. Default True.
        Set to False for faster loading when only metadata is needed.
    label_column : str
        Column in the xlsx to use as the functional label source.
        Default ``"kmeans_function"``.  Use e.g. ``"function"`` for
        alternative label columns.

    Returns
    -------
    pd.DataFrame
        Combined cell data with columns:
        - cell_name: Unique cell identifier
        - imaging_modality: 'photoactivation', 'clem', or 'em'
        - swc: navis.TreeNeuron skeleton
        - function: Functional type (MI, MON, SMI, to_predict)
        - morphology: ipsilateral or contralateral
        - neurotransmitter: inhibitory or excitatory

    Examples
    --------
    Load all modalities:

    >>> all_cells = load_cells_predictor_pipeline()

    Load only photoactivation cells:

    >>> pa_cells = load_cells_predictor_pipeline(modalities=['pa'])

    Load CLEM cells for prediction:

    >>> clem_predict = load_cells_predictor_pipeline(
    ...     modalities=['clem_predict'],
    ... )
    """
    if modalities is None:
        modalities = ["pa", "clem", "em"]
    if path_to_data is None:
        path_to_data = get_base_path()

    all_cells = _load_modality_tables(modalities, path_to_data)
    if all_cells.empty:
        return all_cells

    # Initialize SWC column
    all_cells["swc"] = np.nan
    all_cells["swc"] = all_cells["swc"].astype(object)

    # Load SWC data (skip if load_morphology=False for faster metadata-only loading)
    if load_morphology:
        all_cells = _load_skeletons_for_cells(all_cells)

        # Mirror cells to left hemisphere (only if morphology was loaded)
        if mirror:
            all_cells = _mirror_cells(all_cells)
    # Drop empty rows
    all_cells = all_cells.dropna(how="all")
    all_cells = _filter_and_finalize(
        all_cells,
        filter_incomplete_clem,
        filter_invalid_swc=load_morphology,
        label_column=label_column,
    )

    # Override PA neurotransmitter with neurotransmitter_new from xlsx
    all_cells = _apply_alt_neurotransmitter(all_cells)
    # Override CLEM morphology with Gregor's expert annotations from xlsx
    all_cells = _apply_morphology_gregor(all_cells)

    return all_cells


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    all_cells = load_cells_predictor_pipeline(modalities=["clem_predict"])
    print(f"Loaded {len(all_cells)} cells")
