#!/usr/bin/env python3
"""
Helper functions for connectivity analysis in the zebrafish hindbrain (clem_zfish1).

This module provides shared utilities used by the connectivity analysis scripts,
including:

- Hemisphere classification from mapped meshes
- Loading and cleaning the metadata table
- Grouping nucleus/axon IDs by functional type (and hemisphere)
- Building directional connectivity matrices from NG-resolution synapse tables
- Plotting connectivity matrices (including inhibitory/excitatory overlays)

Functional type naming
----------------------
To match the terminology used in the manuscript, we use the following
functional labels in this module:

- ipsilateral_motion_integrator
- contralateral_motion_integrator
- motion_onset
- slow_motion_integrator

The underlying metadata table uses the functional classifier values:
- 'motion_integrator'
- 'motion_onset'
- 'slow_motion_integrator'
- 'myelinated'
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import navis
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Color dictionaries (updated functional names)
# -------------------------------------------------------------------------

COLOR_CELL_TYPE_DICT: Dict[str, Tuple[float, float, float, float]] = {
    "ipsilateral_motion_integrator":   (254 / 255, 179 / 255, 38 / 255, 0.7),  # Yellow-orange
    "contralateral_motion_integrator": (232 / 255, 77 / 255, 138 / 255, 0.7),  # Magenta-pink
    "motion_onset":                    (100 / 255, 197 / 255, 235 / 255, 0.7), # Light blue
    "slow_motion_integrator":          (127 / 255, 88 / 255, 175 / 255, 0.7),  # Purple
    "myelinated":                      (80 / 255, 220 / 255, 100 / 255, 0.7),  # Green
    "other_functional_types":                (220 / 255, 20 / 255, 60 / 255, 0.7),   # Crimson red

    # Axon-only categories
    "axon_rostral":                    (1.0, 1.0, 1.0, 0.7),                   # White
    "axon_caudal":                     (0.0, 0.0, 0.0, 0.7),                   # Black

    # Non-functionally imaged neurons
    "not_functionally_imaged":         (0.7, 0.7, 0.7, 0.7),                   # Medium gray
}

COLOR_CELL_TYPE_DICT_LR: Dict[str, Tuple[float, float, float, float]] = {
    "ipsilateral_motion_integrator_left": (254 / 255, 179 / 255, 38 / 255, 0.7),
    "ipsilateral_motion_integrator_right": (254 / 255, 179 / 255, 38 / 255, 0.7),
    "contralateral_motion_integrator_left": (232 / 255, 77 / 255, 138 / 255, 0.7),
    "contralateral_motion_integrator_right": (232 / 255, 77 / 255, 138 / 255, 0.7),
    "motion_onset_left": (100 / 255, 197 / 255, 235 / 255, 0.7),
    "motion_onset_right": (100 / 255, 197 / 255, 235 / 255, 0.7),
    "slow_motion_integrator_left": (127 / 255, 88 / 255, 175 / 255, 0.7),
    "slow_motion_integrator_right": (127 / 255, 88 / 255, 175 / 255, 0.7),
    "myelinated_left": (80 / 255, 220 / 255, 100 / 255, 0.7),
    "myelinated_right": (80 / 255, 220 / 255, 100 / 255, 0.7),
    "axon_rostral_left": (1.0, 1.0, 1.0, 0.7),
    "axon_rostral_right": (1.0, 1.0, 1.0, 0.7),
    "axon_caudal_left": (0.0, 0.0, 0.0, 0.7),
    "axon_caudal_right": (0.0, 0.0, 0.0, 0.7),
}


# -------------------------------------------------------------------------
# Hemisphere classification
# -------------------------------------------------------------------------

def determine_hemisphere(
    row: pd.Series,
    root_folder: str | Path,
    width_brain: float = 495.56,
    progress: dict | None = None,
) -> str | None:
    """
    Determine the hemisphere ('L' or 'R') of a neuron based on its mapped mesh.

    This function expects the standard clem_zfish1 directory structure, where
    each traced neuron has a folder of the form:

        clem_zfish1_cell_<nucleus_id>/mapped/clem_zfish1_cell_<nucleus_id>_mapped.obj

    or, for axons:

        clem_zfish1_axon_<axon_id>/mapped/clem_zfish1_axon_<axon_id>_axon_mapped.obj

    Parameters
    ----------
    row : pandas.Series
        One row from the metadata DataFrame. Must contain:
        - 'type' ('cell' or 'axon')
        - 'nucleus_id' (for cells)
        - 'axon_id' (for axons)
    root_folder : str or pathlib.Path
        Root folder containing all traced neuron subfolders.
    width_brain : float, optional
        Brain width in microns; used to define the midline. Meshes with mean
        x > width_brain/2 are assigned to the right hemisphere ('R'),
        otherwise left ('L').
    progress : dict, optional
        Mutable dict with keys 'processed_count' and 'total_rows'. If provided,
        it will be updated in-place and a progress message printed.

    Returns
    -------
    str or None
        'R' or 'L' if the hemisphere can be determined, otherwise None.
    """
    root_folder = Path(root_folder)

    try:
        if row["type"] == "cell":
            nid = row["nucleus_id"]
            mesh_path = (
                root_folder
                / f"clem_zfish1_cell_{nid}"
                / "mapped"
                / f"clem_zfish1_cell_{nid}_mapped.obj"
            )
        elif row["type"] == "axon":
            aid = row["axon_id"]
            mesh_path = (
                root_folder
                / f"clem_zfish1_axon_{aid}"
                / "mapped"
                / f"clem_zfish1_axon_{aid}_axon_mapped.obj"
            )
        else:
            result = None
            if progress is not None:
                progress["processed_count"] += 1
                print(
                    f"Processed {progress['processed_count']}/{progress['total_rows']} rows",
                    end="\r",
                )
            return result

        mesh = navis.read_mesh(mesh_path, units="um")
        result = "R" if np.mean(mesh._vertices[:, 0]) > (width_brain / 2) else "L"

    except Exception as e:  # noqa: BLE001
        print(f"Error processing row {row.name}: {e}")
        result = None

    if progress is not None:
        progress["processed_count"] += 1
        print(
            f"Processed {progress['processed_count']}/{progress['total_rows']} rows",
            end="\r",
        )

    return result


# -------------------------------------------------------------------------
# I/O + metadata helpers
# -------------------------------------------------------------------------

def load_and_clean_data(path: str | Path, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Load the metadata table and optionally drop duplicate axons.

    Parameters
    ----------
    path : str or Path
        Path to the CSV or Excel file.
    drop_duplicates : bool, default True
        If True, drop duplicated rows based on 'axon_id'.

    Returns
    -------
    pandas.DataFrame
        Loaded (and optionally de-duplicated) metadata table.
    """
    path = str(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if drop_duplicates and "axon_id" in df.columns:
        df = df.drop_duplicates(subset="axon_id")

    return df


def standardize_functional_naming(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that the 'functional classifier' column exists and is ready
    for downstream grouping.

    The clem_zfish1 metadata is assumed to already use the canonical labels:
        - 'motion_integrator'
        - 'motion_onset'
        - 'slow_motion_integrator'
        - 'myelinated'

    This function is kept as a hook for potential future normalization,
    but does not change any values in the current repository version.
    """
    if "functional classifier" not in df.columns:
        return df
    return df


def fetch_filtered_ids(
    df: pd.DataFrame,
    col_1: int,
    condition_1: str,
    col_2: int | None = None,
    condition_2: str | None = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Filter DataFrame by column index + condition, optionally with a second condition.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata DataFrame.
    col_1 : int
        Column index for the first condition.
    condition_1 : str
        Value to match in the first column.
    col_2 : int, optional
        Column index for the second condition.
    condition_2 : str, optional
        Value to match in the second column.

    Returns
    -------
    (pandas.Series, pandas.Series)
        Series of unique nucleus IDs (col 5) and functional IDs (col 1).
    """
    filtered = df[df.iloc[:, col_1] == condition_1]
    if col_2 is not None and condition_2 is not None:
        filtered = filtered[filtered.iloc[:, col_2] == condition_2]
    return filtered.iloc[:, 5].drop_duplicates(), filtered.iloc[:, 1].drop_duplicates()


# -------------------------------------------------------------------------
# Grouping IDs by functional type
# -------------------------------------------------------------------------

def create_nucleus_id_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Group nucleus/axon IDs by functional type.

    Groups correspond to:
    - axon_rostral
    - ipsilateral_motion_integrator
    - contralateral_motion_integrator
    - motion_onset
    - slow_motion_integrator
    - myelinated
    - axon_caudal

    The underlying DataFrame is assumed to use:
    - 'motion_integrator'      in 'functional classifier'
    - 'motion_onset'
    - 'slow_motion_integrator'
    - 'myelinated'
    plus 'projection classifier' ('ipsilateral'/'contralateral').

    Returns
    -------
    dict
        Mapping from functional group name to list of string IDs.
    """
    # Make sure functional classifier values are normalized
    df = standardize_functional_naming(df.copy())

    groups = {
        "axon_rostral": df.loc[
            (df["type"] == "axon")
            & (df["comment"] == "axon exits the volume rostrally"),
            "axon_id",
        ],
        "ipsilateral_motion_integrator": fetch_filtered_ids(
            df, 9, "motion_integrator", 11, "ipsilateral"
        )[0],
        "contralateral_motion_integrator": fetch_filtered_ids(
            df, 9, "motion_integrator", 11, "contralateral"
        )[0],
        "motion_onset": fetch_filtered_ids(df, 9, "motion_onset")[0],
        "slow_motion_integrator": fetch_filtered_ids(
            df, 9, "slow_motion_integrator"
        )[0],
        "myelinated": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "myelinated"),
            "nucleus_id",
        ],
        "axon_caudal": df.loc[
            (df["type"] == "axon")
            & (df["comment"] == "axon exits the volume caudally"),
            "axon_id",
        ],
    }

    return {k: [str(v) for v in values] for k, values in groups.items()}


def create_nucleus_id_groups_hemisphere(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Group nucleus/axon IDs by functional type *and* hemisphere.

    Returns groups such as:
    - ipsilateral_motion_integrator_left / right
    - contralateral_motion_integrator_left / right
    - motion_onset_left / right
    - slow_motion_integrator_left / right
    - myelinated_left / right
    - axon_rostral_left / right
    - axon_caudal_left / right

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table with 'hemisphere', 'functional classifier',
        'projection classifier', 'type', 'comment', etc.

    Returns
    -------
    dict
        Mapping from group name to list of string IDs.
    """
    # Ensure functional classifier names are in the new scheme
    df = standardize_functional_naming(df.copy())

    groups = {
        # Axon rostral
        "axon_rostral_left": df.loc[
            (df["type"] == "axon")
            & (df["comment"] == "axon exits the volume rostrally")
            & (df["hemisphere"] == "L"),
            "axon_id",
        ],
        "axon_rostral_right": df.loc[
            (df["type"] == "axon")
            & (df["comment"] == "axon exits the volume rostrally")
            & (df["hemisphere"] == "R"),
            "axon_id",
        ],
        # Motion integrators by projection + hemisphere
        "ipsilateral_motion_integrator_left": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "motion_integrator")
            & (df["projection classifier"] == "ipsilateral")
            & (df["hemisphere"] == "L"),
            "nucleus_id",
        ],
        "ipsilateral_motion_integrator_right": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "motion_integrator")
            & (df["projection classifier"] == "ipsilateral")
            & (df["hemisphere"] == "R"),
            "nucleus_id",
        ],
        "contralateral_motion_integrator_left": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "motion_integrator")
            & (df["projection classifier"] == "contralateral")
            & (df["hemisphere"] == "L"),
            "nucleus_id",
        ],
        "contralateral_motion_integrator_right": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "motion_integrator")
            & (df["projection classifier"] == "contralateral")
            & (df["hemisphere"] == "R"),
            "nucleus_id",
        ],
        # Motion onset
        "motion_onset_left": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "motion_onset")
            & (df["hemisphere"] == "L"),
            "nucleus_id",
        ],
        "motion_onset_right": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "motion_onset")
            & (df["hemisphere"] == "R"),
            "nucleus_id",
        ],
        # Slow motion integrator
        "slow_motion_integrator_left": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "slow_motion_integrator")
            & (df["hemisphere"] == "L"),
            "nucleus_id",
        ],
        "slow_motion_integrator_right": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "slow_motion_integrator")
            & (df["hemisphere"] == "R"),
            "nucleus_id",
        ],
        # Myelinated
        "myelinated_left": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "myelinated")
            & (df["hemisphere"] == "L"),
            "nucleus_id",
        ],
        "myelinated_right": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "myelinated")
            & (df["hemisphere"] == "R"),
            "nucleus_id",
        ],
        # Axon caudal
        "axon_caudal_left": df.loc[
            (df["type"] == "axon")
            & (df["comment"] == "axon exits the volume caudally")
            & (df["hemisphere"] == "L"),
            "axon_id",
        ],
        "axon_caudal_right": df.loc[
            (df["type"] == "axon")
            & (df["comment"] == "axon exits the volume caudally")
            & (df["hemisphere"] == "R"),
            "axon_id",
        ],
    }

    return {k: [str(v) for v in values] for k, values in groups.items()}


def generate_functional_types(nucleus_id_groups: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Invert a dict of {functional_type: [ids]} into {id: functional_type}.

    Parameters
    ----------
    nucleus_id_groups : dict
        Mapping from functional group name to list of IDs.

    Returns
    -------
    dict
        Mapping from individual ID to functional group name.
    """
    return {
        nucleus_id: functional_type
        for functional_type, ids in nucleus_id_groups.items()
        for nucleus_id in ids
    }


def filter_connectivity_matrix(
    matrix: pd.DataFrame,
    functional_types: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Filter connectivity matrix to only keep rows/cols that have any non-zero
    entries (either as source or target), and restrict functional_types to
    those IDs.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Original connectivity matrix.
    functional_types : dict
        Mapping of ID -> functional type.

    Returns
    -------
    (pandas.DataFrame, dict)
        Filtered matrix and corresponding functional_types mapping.
    """
    non_zero_indices = matrix.index[
        (matrix.sum(axis=1) != 0) | (matrix.sum(axis=0) != 0)
    ]
    filtered_matrix = matrix.loc[non_zero_indices, non_zero_indices]
    filtered_types = {
        id_: t for id_, t in functional_types.items() if id_ in filtered_matrix.index
    }
    return filtered_matrix, filtered_types


# -------------------------------------------------------------------------
# Connectivity extraction
# -------------------------------------------------------------------------

def read_synapse_table(path: Path) -> pd.DataFrame:
    """
    Read a synapse CSV written by the NG-res postprocessing step and
    standardize the partner column name to 'partner_cell_id'.
    """

    # CRITICAL: synapse files are whitespace-separated, not comma-separated
    df = pd.read_csv(path, sep=" ")

    # Identify partner column
    if "postsynaptic_ID" in df.columns:
        df["partner_cell_id"] = df["postsynaptic_ID"]
    elif "presynaptic_ID" in df.columns:
        df["partner_cell_id"] = df["presynaptic_ID"]
    else:
        raise ValueError(
            f"Could not find 'postsynaptic_ID' or 'presynaptic_ID' in {path}. "
            f"Found columns: {list(df.columns)}"
        )

    # Verify required columns
    for col in ["synapse_id", "validation_status"]:
        if col not in df.columns:
            raise ValueError(f"{col} is missing in {path}, columns found: {df.columns}")

    return df


def get_inputs_outputs_by_hemisphere_general(
    root_folder: str | Path,
    seed_cell_ids: Iterable[str],
    hemisphere_df: pd.DataFrame,
) -> dict:
    """
    Extract and categorize input/output neurons for a set of seed cell IDs,
    split by hemisphere (same-side vs different-side).

    This is a general function used to compute:
    - outputs: presynaptic targets of each seed (postsynaptic partners)
    - inputs: postsynaptic sources to each seed (presynaptic partners)

    It returns both:
    - cell-level tables (unique partners) and
    - synapse-level tables (all valid synapses).

    Parameters
    ----------
    root_folder : str or Path
        Root folder containing per-neuron NG synapse tables.
    seed_cell_ids : iterable of str
        List of nucleus or axon IDs used as seeds.
    hemisphere_df : pandas.DataFrame
        Metadata with columns including:
        - 'nucleus_id', 'axon_id', 'dendrite_id', 'hemisphere'
        - 'functional classifier', etc.

    Returns
    -------
    dict
        Nested dictionary with 'outputs', 'inputs', and 'counters' keys.
    """
    root_folder = Path(root_folder)

    hemisphere_df = hemisphere_df.copy()
    hemisphere_df["nucleus_id"] = hemisphere_df["nucleus_id"].astype(str)
    hemisphere_df.loc[hemisphere_df["nucleus_id"] == "0", "nucleus_id"] = (
        hemisphere_df["axon_id"].astype(str)
    )
    hemisphere_map = hemisphere_df.set_index("nucleus_id")["hemisphere"].to_dict()

    results: dict = {
        "outputs": {
            "cells": {"same_side": pd.DataFrame(), "different_side": pd.DataFrame()},
            "synapses": {"same_side": pd.DataFrame(), "different_side": pd.DataFrame()},
            "percentages": {"synapses": 0.0},
        },
        "inputs": {
            "cells": {"same_side": pd.DataFrame(), "different_side": pd.DataFrame()},
            "synapses": {"same_side": pd.DataFrame(), "different_side": pd.DataFrame()},
            "percentages": {"synapses": 0.0},
        },
        "counters": {"output_seed_counter": 0, "input_seed_counter": 0},
    }

    for seed_cell_id in seed_cell_ids:
        seed_cell_id_str = str(seed_cell_id)
        seed_hemi = hemisphere_map.get(seed_cell_id_str, None)
        if seed_hemi is None:
            print(f"Seed cell ID {seed_cell_id_str} has no hemisphere data. Skipping.")
            continue

        # -------------------------- OUTPUTS -------------------------------
        output_file_path = None
        # Try cell presynapses first
        cell_pattern = f"clem_zfish1_cell_{seed_cell_id_str}_presynapses.csv"
        for root, _, files in os.walk(root_folder):
            for filename in files:
                if filename == cell_pattern:
                    output_file_path = Path(root) / filename
                    break
            if output_file_path is not None:
                break

        # Fallback: axon presynapses
        if output_file_path is None or not output_file_path.exists():
            axon_pattern = f"clem_zfish1_axon_{seed_cell_id_str}_presynapses.csv"
            for root, _, files in os.walk(root_folder):
                for filename in files:
                    if filename == axon_pattern:
                        output_file_path = Path(root) / filename
                        break
                if output_file_path is not None:
                    break

        if output_file_path and output_file_path.exists():
            outputs_data = read_synapse_table(output_file_path)
            valid_outputs = outputs_data[
                outputs_data["validation_status"].astype(str).str.contains("valid", na=False)
            ]
            output_ids = valid_outputs["partner_cell_id"]

            traced_dendrites = output_ids[
                output_ids.isin(hemisphere_df["dendrite_id"])
            ]
            matched_outputs = (
                [
                    hemisphere_df[hemisphere_df["dendrite_id"] == dend].iloc[0]
                    for dend in traced_dendrites
                ]
                if not traced_dendrites.empty
                else []
            )

            output_connected_cells = pd.DataFrame(matched_outputs)
            if not output_connected_cells.empty:
                output_connected_cells_unique = output_connected_cells.drop_duplicates(
                    subset="axon_id"
                )

                # Percentages
                pct_syn = (
                    len(output_connected_cells) / len(valid_outputs)
                    if len(valid_outputs) > 0
                    else 0
                )
                results["outputs"]["percentages"]["synapses"] += pct_syn
                results["counters"]["output_seed_counter"] += 1

                if "hemisphere" in output_connected_cells_unique.columns:
                    same_outputs_cells = output_connected_cells_unique[
                        output_connected_cells_unique["hemisphere"] == seed_hemi
                    ]
                    diff_outputs_cells = output_connected_cells_unique[
                        output_connected_cells_unique["hemisphere"] != seed_hemi
                    ]
                    same_outputs_syn = output_connected_cells[
                        output_connected_cells["hemisphere"] == seed_hemi
                    ]
                    diff_outputs_syn = output_connected_cells[
                        output_connected_cells["hemisphere"] != seed_hemi
                    ]
                else:
                    same_outputs_cells = pd.DataFrame()
                    diff_outputs_cells = pd.DataFrame()
                    same_outputs_syn = pd.DataFrame()
                    diff_outputs_syn = pd.DataFrame()

                dfs = [
                    same_outputs_cells,
                    diff_outputs_cells,
                    same_outputs_syn,
                    diff_outputs_syn,
                ]
                for i, dff in enumerate(dfs):
                    if "functional classifier" in dff.columns:
                        dff = dff.copy()
                        dff["functional classifier"] = dff[
                            "functional classifier"
                        ].astype("object")
                        dff.loc[:, "functional classifier"] = dff[
                            "functional classifier"
                        ].fillna("not functionally imaged")
                        dfs[i] = dff

                (
                    same_outputs_cells,
                    diff_outputs_cells,
                    same_outputs_syn,
                    diff_outputs_syn,
                ) = dfs

                results["outputs"]["cells"]["same_side"] = pd.concat(
                    [results["outputs"]["cells"]["same_side"], same_outputs_cells],
                    ignore_index=True,
                )
                results["outputs"]["cells"]["different_side"] = pd.concat(
                    [
                        results["outputs"]["cells"]["different_side"],
                        diff_outputs_cells,
                    ],
                    ignore_index=True,
                )
                results["outputs"]["synapses"]["same_side"] = pd.concat(
                    [results["outputs"]["synapses"]["same_side"], same_outputs_syn],
                    ignore_index=True,
                )
                results["outputs"]["synapses"]["different_side"] = pd.concat(
                    [
                        results["outputs"]["synapses"]["different_side"],
                        diff_outputs_syn,
                    ],
                    ignore_index=True,
                )

        # --------------------------- INPUTS -------------------------------
        input_file_pattern = (
            f"clem_zfish1_cell_{seed_cell_id_str}_postsynapses.csv"
        )
        input_file_path = None
        for root, _, files in os.walk(root_folder):
            for filename in files:
                if filename == input_file_pattern:
                    input_file_path = Path(root) / filename
                    break
            if input_file_path is not None:
                break

        if input_file_path and input_file_path.exists():
            inputs_data = read_synapse_table(input_file_path)
            valid_inputs = inputs_data[
                inputs_data["validation_status"].astype(str).str.contains("valid", na=False)
            ]
            input_ids = valid_inputs["partner_cell_id"]

            traced_axons = input_ids[input_ids.isin(hemisphere_df["axon_id"])]
            matched_inputs = (
                [
                    hemisphere_df[hemisphere_df["axon_id"] == ax].iloc[0]
                    for ax in traced_axons
                ]
                if not traced_axons.empty
                else []
            )

            input_connected_cells = pd.DataFrame(matched_inputs)
            if not input_connected_cells.empty:
                input_connected_cells_unique = input_connected_cells.drop_duplicates(
                    subset="axon_id"
                )

                pct_syn = (
                    len(input_connected_cells) / len(valid_inputs)
                    if len(valid_inputs) > 0
                    else 0
                )
                results["inputs"]["percentages"]["synapses"] += pct_syn
                results["counters"]["input_seed_counter"] += 1

                if "hemisphere" in input_connected_cells_unique.columns:
                    same_inputs_cells = input_connected_cells_unique[
                        input_connected_cells_unique["hemisphere"] == seed_hemi
                    ]
                    diff_inputs_cells = input_connected_cells_unique[
                        input_connected_cells_unique["hemisphere"] != seed_hemi
                    ]
                    same_inputs_syn = input_connected_cells[
                        input_connected_cells["hemisphere"] == seed_hemi
                    ]
                    diff_inputs_syn = input_connected_cells[
                        input_connected_cells["hemisphere"] != seed_hemi
                    ]
                else:
                    same_inputs_cells = pd.DataFrame()
                    diff_inputs_cells = pd.DataFrame()
                    same_inputs_syn = pd.DataFrame()
                    diff_inputs_syn = pd.DataFrame()

                dfs = [
                    same_inputs_cells,
                    diff_inputs_cells,
                    same_inputs_syn,
                    diff_inputs_syn,
                ]
                for i, dff in enumerate(dfs):
                    if "functional classifier" in dff.columns:
                        dff = dff.copy()
                        dff["functional classifier"] = dff[
                            "functional classifier"
                        ].astype("object")
                        dff.loc[:, "functional classifier"] = dff[
                            "functional classifier"
                        ].fillna("not functionally imaged")
                        dfs[i] = dff

                (
                    same_inputs_cells,
                    diff_inputs_cells,
                    same_inputs_syn,
                    diff_inputs_syn,
                ) = dfs

                results["inputs"]["cells"]["same_side"] = pd.concat(
                    [results["inputs"]["cells"]["same_side"], same_inputs_cells],
                    ignore_index=True,
                )
                results["inputs"]["cells"]["different_side"] = pd.concat(
                    [
                        results["inputs"]["cells"]["different_side"],
                        diff_inputs_cells,
                    ],
                    ignore_index=True,
                )
                results["inputs"]["synapses"]["same_side"] = pd.concat(
                    [results["inputs"]["synapses"]["same_side"], same_inputs_syn],
                    ignore_index=True,
                )
                results["inputs"]["synapses"]["different_side"] = pd.concat(
                    [
                        results["inputs"]["synapses"]["different_side"],
                        diff_inputs_syn,
                    ],
                    ignore_index=True,
                )

    return results


def generate_directional_connectivity_matrix_general(
    root_folder: str | Path,
    seg_ids: Iterable[str],
    df_w_hemisphere: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a directional connectivity matrix for a set of functionally-labeled IDs.

    For each source ID, this function:
    - reads its presynapse and postsynapse tables (with headers),
    - uses hemisphere-aware connectivity (via get_inputs_outputs_by_hemisphere_general),
    - counts unique synapses between pairs of IDs in `seg_ids`,
      separately for inputs and outputs, and combines them into a single matrix.

    The matrix is:
        rows    = source (pre) IDs
        columns = target (post) IDs
    where entries represent the number of synapses (inputs+outputs) between them,
    with sign left untouched here; any inhibitory/excitatory flipping is done in
    the plotting function.
    """
    root_folder = Path(root_folder)
    seg_ids_str = [str(s) for s in seg_ids]
    connectivity_matrix = pd.DataFrame(0, index=seg_ids_str, columns=seg_ids_str)

    # globally tracked non-zero synapses so we don't double-count
    stored_nonzero_synapse_ids: set[int] = set()

    for source_id in seg_ids_str:
        # Hemisphere-aware grouping for this seed
        results = get_inputs_outputs_by_hemisphere_general(
            root_folder=root_folder,
            seed_cell_ids=[source_id],
            hemisphere_df=df_w_hemisphere,
        )

        # --------------------- OUTPUTS (source -> target) ----------------
        # Try cell presynapses first, then axon
        cell_name = f"clem_zfish1_cell_{source_id}"
        cell_file_name = f"{cell_name}_presynapses.csv"
        cell_output_file_path = root_folder / cell_name / cell_file_name

        if cell_output_file_path.exists():
            output_file_path = cell_output_file_path
        else:
            axon_name = f"clem_zfish1_axon_{source_id}"
            axon_file_name = f"{axon_name}_presynapses.csv"
            output_file_path = root_folder / axon_name / axon_file_name

        if not output_file_path.exists():
            # No presynaptic table for this seed
            continue

        # Use the header-aware helper
        outputs_data = read_synapse_table(output_file_path)
        valid_outputs = outputs_data[
            outputs_data["validation_status"].astype(str).str.contains("valid", na=False)
        ].copy()

        # Ensure we compare IDs as strings
        valid_outputs["partner_cell_id"] = valid_outputs["partner_cell_id"].astype(str)

        for direction in ["same_side", "different_side"]:
            outputs = results["outputs"]["synapses"][direction]
            if outputs.empty or "nucleus_id" not in outputs.columns:
                continue

            try:
                outputs = outputs.copy()
                # nucleus_id in metadata is numeric; we keep seg_ids_str as strings
                outputs["nucleus_id"] = outputs["nucleus_id"].astype(int)
                seg_ids_int = [int(s) for s in seg_ids_str]
                outputs = outputs[outputs["nucleus_id"].isin(seg_ids_int)]
            except ValueError as e:  # noqa: BLE001
                print(f"Error converting nucleus_id or seg_ids to int: {e}")
                continue

            for _, output_row in outputs.iterrows():
                target_dend = str(output_row["dendrite_id"])
                matching = valid_outputs[
                    valid_outputs["partner_cell_id"] == target_dend
                ]
                synapse_ids = matching["synapse_id"].tolist()

                zero_synapse_count = synapse_ids.count(0)
                nonzero_synapse_ids = [sid for sid in synapse_ids if sid != 0]
                new_nonzero = [
                    sid
                    for sid in nonzero_synapse_ids
                    if sid not in stored_nonzero_synapse_ids
                ]
                num_new = zero_synapse_count + len(new_nonzero)

                if num_new > 0:
                    target_func_id = str(output_row["nucleus_id"])
                    if (
                        source_id in connectivity_matrix.index
                        and target_func_id in connectivity_matrix.columns
                    ):
                        connectivity_matrix.loc[source_id, target_func_id] += num_new
                    stored_nonzero_synapse_ids.update(new_nonzero)

        # ---------------------- INPUTS (target -> source) ----------------
        input_file_name = f"{cell_name}_postsynapses.csv"
        input_file_path = root_folder / cell_name / input_file_name
        if not input_file_path.exists():
            continue

        inputs_data = read_synapse_table(input_file_path)
        valid_inputs = inputs_data[
            inputs_data["validation_status"].astype(str).str.contains("valid", na=False)
        ].copy()
        valid_inputs["partner_cell_id"] = valid_inputs["partner_cell_id"].astype(str)

        for direction in ["same_side", "different_side"]:
            inputs = results["inputs"]["synapses"][direction]
            if inputs.empty or "nucleus_id" not in inputs.columns:
                continue

            try:
                inputs = inputs.copy()
                inputs["nucleus_id"] = inputs["nucleus_id"].astype(int)
                seg_ids_int = [int(s) for s in seg_ids_str]
                inputs = inputs[inputs["nucleus_id"].isin(seg_ids_int)]
            except ValueError as e:  # noqa: BLE001
                print(f"Error converting nucleus_id or seg_ids to int: {e}")
                continue

            for _, input_row in inputs.iterrows():
                target_axon = str(input_row["axon_id"])
                matching = valid_inputs[
                    valid_inputs["partner_cell_id"] == target_axon
                ]
                synapse_ids = matching["synapse_id"].tolist()

                zero_synapse_count = synapse_ids.count(0)
                nonzero_synapse_ids = [sid for sid in synapse_ids if sid != 0]
                new_nonzero = [
                    sid
                    for sid in nonzero_synapse_ids
                    if sid not in stored_nonzero_synapse_ids
                ]
                num_new = zero_synapse_count + len(new_nonzero)

                if num_new > 0:
                    source_input_func_id = str(input_row["nucleus_id"])
                    if (
                        source_input_func_id in connectivity_matrix.index
                        and source_id in connectivity_matrix.columns
                    ):
                        connectivity_matrix.loc[
                            source_input_func_id, source_id
                        ] += num_new
                    stored_nonzero_synapse_ids.update(new_nonzero)

    return connectivity_matrix


# -------------------------------------------------------------------------
# Inhibitory / excitatory processing + plotting
# -------------------------------------------------------------------------

def process_matrix(
    matrix: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Modify the matrix rows based on the 'neurotransmitter classifier' in df.

    For each row index (ID), this function looks up the corresponding
    row in `df` using either 'nucleus_id' or 'axon_id' and:

    - if 'neurotransmitter classifier' == 'inhibitory', multiplies the row by -1
    - if 'excitatory' or missing/unknown, leaves it unchanged

    Rows that cannot be matched to either nucleus_id or axon_id are skipped.
    """
    df = df.copy()

    # Create string versions for safe matching
    if "nucleus_id" in df.columns:
        df["nucleus_id_str"] = df["nucleus_id"].astype(str)
    else:
        df["nucleus_id_str"] = ""

    if "axon_id" in df.columns:
        df["axon_id_str"] = df["axon_id"].astype(str)
    else:
        df["axon_id_str"] = ""

    for idx in matrix.index:
        # Try matching as nucleus_id first
        df_row = df.loc[df["nucleus_id_str"] == str(idx)]

        # If no match, try axon_id (for axon_rostral / axon_caudal rows)
        if df_row.empty:
            df_row = df.loc[df["axon_id_str"] == str(idx)]

        # If still no match, skip this row (do not modify)
        if df_row.empty:
            # Optional: print or log once if you want to see these
            # print(f"Warning: no metadata row for ID {idx}; leaving row unchanged.")
            continue

        classifier = df_row.iloc[0].get("neurotransmitter classifier", None)

        if classifier == "inhibitory":
            matrix.loc[idx] *= -1
        # If excitatory or None/NaN, we leave the row as-is

    return matrix

def plot_connectivity_matrix(
    matrix: pd.DataFrame,
    functional_types: Dict[str, str],
    output_path: str | Path,
    category_order: List[str],
    df: pd.DataFrame | None = None,
    title: str = "Directional Connectivity Matrix",
    display_type: str = "normal",
    plot_type: str = "raster",
    color_cell_type_dict: Dict[str, Tuple[float, float, float, float]] | None = None,
) -> None:
    """
    Plot a connectivity matrix, optionally encoding inhibitory / excitatory sign.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Connectivity matrix to plot (raw synapse counts; rows/cols are IDs).
    functional_types : dict
        Mapping from matrix index ID -> functional category name
        (e.g. 'motion_integrator', 'axon_rostral', 'axon_caudal_left', ...).
    output_path : str or Path
        Directory where the resulting PDF will be written.
    category_order : list of str
        Ordered list of functional categories for sorting indices and
        drawing separators.
    df : pandas.DataFrame, optional
        Metadata with at least 'nucleus_id' (or equivalent ID) and
        'neurotransmitter classifier'. Required if display_type == 'Inhibitory_Excitatory'.
    title : str, optional
        Plot title and base for the output filename.
    display_type : {'normal', 'Inhibitory_Excitatory'}, optional
        - 'normal'  : plot raw counts with a simple sequential colormap.
        - 'Inhibitory_Excitatory' :
              * use process_matrix(...) to map synapse sign/strength into
                discrete values in [-2, -1, 0, 1, 2],
              * plot with a 5-level inhibitory→excitatory colormap,
              * for axon categories (axon_rostral / axon_caudal ± _left/_right),
                override the color with light/dark gray depending on
                raw synapse count:
                    - 1 synapse  → light gray
                    - ≥ 2 synapses → dark gray
    plot_type : {'raster', 'scatter'}, optional
        - 'raster'  : use matshow to draw pixels.
        - 'scatter' : bubble-plot representation.
    color_cell_type_dict : dict, optional
        Mapping from functional category name -> RGBA color used for the
        outer row/column bars and the functional legend.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if color_cell_type_dict is None:
        color_cell_type_dict = COLOR_CELL_TYPE_DICT

    if display_type not in {"normal", "Inhibitory_Excitatory"}:
        raise ValueError("display_type must be 'normal' or 'Inhibitory_Excitatory'.")
    if plot_type not in {"raster", "scatter"}:
        raise ValueError("plot_type must be 'raster' or 'scatter'.")

    # Keep a copy of the raw integer counts for axon gray shading
    matrix_raw_counts = matrix.copy()

    # ------------------------------------------------------------------
    # 1) Possibly transform to inhibitory/excitatory representation
    # ------------------------------------------------------------------
    if display_type == "Inhibitory_Excitatory":
        if df is None:
            raise ValueError(
                "df is required for 'Inhibitory_Excitatory' display_type."
            )
        # process_matrix should return a matrix in [-2, 2] (signed)
        matrix = process_matrix(matrix.copy(), df)
    else:
        matrix = matrix.copy()

    # ------------------------------------------------------------------
    # 2) Filter to indices that have functional_types in category_order
    # ------------------------------------------------------------------
    functional_types = {
        k: v for k, v in functional_types.items()
        if v in category_order and k in matrix.index
    }

    filtered_indices = [
        idx
        for idx in matrix.index
        if functional_types.get(idx, "unknown") in category_order
    ]

    # Filter both transformed matrix and raw counts to the same subset
    filtered_matrix = matrix.loc[filtered_indices, filtered_indices]
    filtered_raw_counts = matrix_raw_counts.loc[filtered_indices, filtered_indices]

    # Sort indices according to category_order
    def sort_key(func_id: str) -> int:
        category = functional_types.get(func_id, "unknown")
        return category_order.index(category) if category in category_order else len(
            category_order
        )

    sorted_indices = sorted(filtered_indices, key=sort_key)

    # Final matrices used for plotting
    matrix_with_nan = filtered_matrix.loc[sorted_indices, sorted_indices]
    raw_counts_for_grey = filtered_raw_counts.loc[sorted_indices, sorted_indices]

    # ------------------------------------------------------------------
    # 3) Choose colormap / normalization
    # ------------------------------------------------------------------
    if display_type == "Inhibitory_Excitatory":
        # Clip to [-2,2] where:
        #   -2, -1 = inhibitory (weak/strong)
        #    0     = zero / unknown
        #    1,  2 = excitatory (weak/strong)
        matrix_with_nan = np.clip(matrix_with_nan, -2, 2)

        colors = [
            "#9B00AE",  # -2  strong inhibitory (dark magenta)
            "#FF4DFF",  # -1  weak inhibitory (light magenta)
            "#FFFFFF",  #  0  zero
            "#7CFF5A",  # +1  weak excitatory (light green)
            "#007A00",  # +2  strong excitatory (dark green)
        ]

        cmap = mcolors.ListedColormap(colors, name="Inhibitory-Excitatory")
        bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        ticks = [-2, -1, 0, 1, 2]
        cbar_label = "Synapse sign & strength (inhibitory → excitatory)"
    else:
        # Simple sequential colormap for raw counts (0–5+)
        cmap = mcolors.ListedColormap(
            ["white", "blue", "green", "yellow", "pink", "red"]
        )
        bounds = [0, 1, 2, 3, 4, 5, 6]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        ticks = [0, 1, 2, 3, 4, 5]
        cbar_label = "Number of synapses"

    # ------------------------------------------------------------------
    # 4) Plot matrix (raster or scatter)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))

    if plot_type == "raster":
        cax = ax.matshow(matrix_with_nan, cmap=cmap, norm=norm)
        artist_for_cbar = cax
    else:
        # Bubble / scatter representation
        x, y = np.meshgrid(
            range(len(matrix_with_nan.columns)),
            range(len(matrix_with_nan.index)),
        )
        x_flat, y_flat = x.flatten(), y.flatten()
        vals = matrix_with_nan.values.flatten()
        sizes = np.abs(vals) * 100.0  # bubble size ~ |value|
        scatter = ax.scatter(
            x_flat,
            y_flat,
            c=vals,
            s=sizes,
            cmap=cmap,
            norm=norm,
        )
        artist_for_cbar = scatter

    ax.set_aspect("equal")

    ax.set_xticks(range(len(matrix_with_nan.columns)))
    ax.set_yticks(range(len(matrix_with_nan.index)))
    ax.set_xticklabels(matrix_with_nan.columns, rotation=90, fontsize=5)
    ax.set_yticklabels(matrix_with_nan.index, fontsize=5)
    ax.set_xlabel("Pre-synaptic (axons)")
    ax.set_ylabel("Post-synaptic (dendrites)")
    ax.set_title(title, fontsize=12)

    # ------------------------------------------------------------------
    # 5) Functional-category color bars (outside matrix)
    # ------------------------------------------------------------------
    bar_width = 1.5
    bar_height = 1.5

    left_bar_x = -bar_width - 0.5
    top_bar_y = -bar_height - 0.5

    for i, functional_id in enumerate(matrix_with_nan.index):
        if functional_id not in functional_types:
            continue
        ftype = functional_types[functional_id]
        color = color_cell_type_dict.get(ftype, (0.8, 0.8, 0.8, 0.7))

        # Left bar (row indicator)
        ax.add_patch(
            patches.Rectangle(
                (left_bar_x, i - 0.5),
                bar_width,
                1,
                color=color,
                zorder=2,
            )
        )

        # Top bar (column indicator)
        ax.add_patch(
            patches.Rectangle(
                (i - 0.5, top_bar_y),
                1,
                bar_height,
                color=color,
                zorder=2,
            )
        )

    # ------------------------------------------------------------------
    # 6) Draw group boundaries between functional categories
    # ------------------------------------------------------------------
    group_boundaries: List[float] = []
    last_type = None
    for i, idx in enumerate(matrix_with_nan.index):
        current_type = functional_types.get(idx, "unknown")
        if current_type != last_type:
            group_boundaries.append(i - 0.5)
            last_type = current_type
    group_boundaries.append(len(matrix_with_nan.index) - 0.5)

    for boundary in group_boundaries:
        ax.axhline(boundary, color="black", linewidth=1.5, zorder=3)
        ax.axvline(boundary, color="black", linewidth=1.5, zorder=3)

    ax.set_xlim(-1.5, len(matrix_with_nan.columns) - 0.5)
    ax.set_ylim(len(matrix_with_nan.index) - 0.5, -1.5)

    # ------------------------------------------------------------------
    # 7) Overlay greys for axon rows in Inhibitory-Excitatory mode
    #    (works for BOTH raster and scatter)
    # ------------------------------------------------------------------
    if display_type == "Inhibitory_Excitatory":
        axon_categories = {
            "axon_rostral",
            "axon_caudal",
            "axon_rostral_left",
            "axon_rostral_right",
            "axon_caudal_left",
            "axon_caudal_right",
        }

        for i, row_id in enumerate(matrix_with_nan.index):
            row_cat = functional_types.get(row_id, "unknown")

            # Only apply greyscale override to axon categories
            if row_cat not in axon_categories:
                continue

            for j, col_id in enumerate(matrix_with_nan.columns):
                raw_val = raw_counts_for_grey.loc[row_id, col_id]

                if pd.isna(raw_val) or raw_val <= 0:
                    continue

                # Synapse-count → grey color
                if raw_val == 1:
                    grey = "#D0D0D0"   # light grey
                else:  # raw_val >= 2
                    grey = "#606060"   # dark grey

                if plot_type == "raster":
                    # Draw a full cell-sized rectangle over this position
                    rect = patches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        facecolor=grey,
                        edgecolor="none",
                        alpha=1.0,
                        zorder=4,
                    )
                    ax.add_patch(rect)
                else:
                    # Scatter mode: overlay a grey bubble exactly at (j, i)
                    ax.scatter(
                        [j],
                        [i],
                        s=200.0,          # size tuned to cover the cell
                        facecolor=grey,
                        edgecolor="none",
                        alpha=1.0,
                        zorder=4,
                    )

    # ------------------------------------------------------------------
    # 8) Colorbar (synapse strength / counts)
    # ------------------------------------------------------------------
    cbar = plt.colorbar(
        artist_for_cbar,
        ax=ax,
        boundaries=bounds,
        ticks=ticks,
        spacing="uniform",
        orientation="vertical",
        fraction=0.045,
        pad=0.02,
    )
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ------------------------------------------------------------------
    # 9) Functional-type legend (to the right of matrix+colorbar)
    # ------------------------------------------------------------------
    func_handles = []
    func_labels = []
    for cat in category_order:
        if cat in color_cell_type_dict:
            func_handles.append(
                patches.Patch(
                    color=color_cell_type_dict[cat],
                    label=cat.replace("_", " "),
                )
            )
            func_labels.append(cat.replace("_", " "))

    func_legend = ax.legend(
        handles=func_handles,
        labels=func_labels,
        loc="upper left",
        bbox_to_anchor=(1.25, 1.0),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        fontsize=7,
        title="Functional type",
        title_fontsize=8,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=0.2,
    )
    fig.add_artist(func_legend)

    # ------------------------------------------------------------------
    # 10) (Optional) Synapse-strength legend for IE mode
    # ------------------------------------------------------------------
    if display_type == "Inhibitory_Excitatory":
        strength_handles = [
            patches.Patch(color="#9B00AE", label="strong inhibitory (≤ -2)"),
            patches.Patch(color="#FF4DFF", label="weak inhibitory (-1)"),
            patches.Patch(color="#FFFFFF", label="zero (0)"),
            patches.Patch(color="#7CFF5A", label="weak excitatory (+1)"),
            patches.Patch(color="#007A00", label="strong excitatory (≥ +2)"),
        ]
        strength_legend = ax.legend(
            handles=strength_handles,
            loc="upper left",
            bbox_to_anchor=(1.25, 0.55),
            bbox_transform=ax.transAxes,
            borderaxespad=0.0,
            fontsize=7,
            title="Synapse sign/strength",
            title_fontsize=8,
            frameon=False,
            handlelength=1.5,
            handletextpad=0.4,
            labelspacing=0.2,
        )
        fig.add_artist(strength_legend)

        # Small note about greys for axons
        grey_handles = [
            patches.Patch(color="#D0D0D0", label="axon: 1 synapse"),
            patches.Patch(color="#606060", label="axon: ≥ 2 synapses"),
        ]
        grey_legend = ax.legend(
            handles=grey_handles,
            loc="upper left",
            bbox_to_anchor=(1.25, 0.20),
            bbox_transform=ax.transAxes,
            borderaxespad=0.0,
            fontsize=7,
            title="Axon inputs",
            title_fontsize=8,
            frameon=False,
            handlelength=1.5,
            handletextpad=0.4,
            labelspacing=0.2,
        )
        fig.add_artist(grey_legend)

    # ------------------------------------------------------------------
    # 11) Save
    # ------------------------------------------------------------------
    plt.tight_layout()
    sanitized_title = (
        title.lower().replace(" ", "_").replace(":", "").replace("/", "_")
    )
    output_pdf_path = output_path / f"{sanitized_title}.pdf"
    plt.savefig(output_pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()


# -------------------------------------------------------------------------
# Functions for differentiating LDA predicted vs. ground-truth cells
# -------------------------------------------------------------------------

def create_nucleus_id_groups_hemisphere_LDA(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Group nucleus/axon IDs by functional type, hemisphere, and LDA (native/predicted).

    This version expands each functional population into:
        <class>_<hemisphere>_<lda>
    e.g.:
        integrator_ipsilateral_left_native
        integrator_ipsilateral_left_predicted
    """
    groups = {}

    # -----------------------------
    # Helper to add grouped entries
    # -----------------------------
    def add_group(name, mask):
        groups[name] = df.loc[mask, "nucleus_id"].astype(str).tolist()

    # ============================= AXONS (no LDA split) =============================
    groups["axon_rostral_left"] = df.loc[
        (df.type == "axon") &
        (df.comment == "axon exits the volume rostrally") &
        (df.hemisphere == "L"),
        "axon_id"
    ].astype(str).tolist()

    groups["axon_rostral_right"] = df.loc[
        (df.type == "axon") &
        (df.comment == "axon exits the volume rostrally") &
        (df.hemisphere == "R"),
        "axon_id"
    ].astype(str).tolist()

    groups["axon_caudal_left"] = df.loc[
        (df.type == "axon") &
        (df.comment == "axon exits the volume caudally") &
        (df.hemisphere == "L"),
        "axon_id"
    ].astype(str).tolist()

    groups["axon_caudal_right"] = df.loc[
        (df.type == "axon") &
        (df.comment == "axon exits the volume caudally") &
        (df.hemisphere == "R"),
        "axon_id"
    ].astype(str).tolist()

    # ============================= CELLS (with LDA split) =============================
    for hemi in ["L", "R"]:
        hemi_str = "left" if hemi == "L" else "right"

        for lda_flag in ["native", "predicted"]:

            # Integrator ipsilateral
            add_group(
                f"motion_integrator_ipsilateral_{hemi_str}_{lda_flag}",
                (df.type == "cell") &
                (df["functional classifier"] == "motion_integrator") &
                (df["projection classifier"] == "ipsilateral") &
                (df.hemisphere == hemi) &
                (df.lda == lda_flag)
            )

            # Integrator contralateral
            add_group(
                f"motion_integrator_contralateral_{hemi_str}_{lda_flag}",
                (df.type == "cell") &
                (df["functional classifier"] == "motion_integrator") &
                (df["projection classifier"] == "contralateral") &
                (df.hemisphere == hemi) &
                (df.lda == lda_flag)
            )

            # Motion onset
            add_group(
                f"motion_onset_{hemi_str}_{lda_flag}",
                (df.type == "cell") &
                (df["functional classifier"] == "motion_onset") &
                (df.hemisphere == hemi) &
                (df.lda == lda_flag)
            )

            # Slow motion integrator
            add_group(
                f"slow_motion_integrator_{hemi_str}_{lda_flag}",
                (df.type == "cell") &
                (df["functional classifier"] == "slow_motion_integrator") &
                (df.hemisphere == hemi) &
                (df.lda == lda_flag)
            )

            # Myelinated (no LDA, but match format)
            add_group(
                f"myelinated_{hemi_str}",
                (df.type == "cell") &
                (df["functional classifier"] == "myelinated") &
                (df.hemisphere == hemi)
            )

    return groups

COLOR_CELL_TYPE_DICT_LR_LDA = {
# ===== Motion Integrators (Ipsi) =====
"motion_integrator_ipsilateral_left_native":  (254/255,179/255, 38/255,0.7),
"motion_integrator_ipsilateral_left_predicted": (254/255,220/255, 80/255,0.7),
"motion_integrator_ipsilateral_right_native": (254/255,179/255, 38/255,0.7),
"motion_integrator_ipsilateral_right_predicted": (254/255,220/255, 80/255,0.7),

# ===== Motion Integrators (Contra) =====
"motion_integrator_contralateral_left_native":  (232/255, 77/255,138/255,0.7),
"motion_integrator_contralateral_left_predicted": (255/255,105/255,180/255,0.7),
"motion_integrator_contralateral_right_native": (232/255, 77/255,138/255,0.7),
"motion_integrator_contralateral_right_predicted": (255/255,105/255,180/255,0.7),

# ===== Motion Onset =====
"motion_onset_left_native":  (100/255,197/255,235/255,0.7),
"motion_onset_left_predicted": (160/255,220/255,250/255,0.7),
"motion_onset_right_native": (100/255,197/255,235/255,0.7),
"motion_onset_right_predicted": (160/255,220/255,250/255,0.7),

# ===== Slow Motion Integrator =====
"slow_motion_integrator_left_native":  (127/255, 88/255,175/255,0.7),
"slow_motion_integrator_left_predicted": (180/255,130/255,210/255,0.7),
"slow_motion_integrator_right_native": (127/255, 88/255,175/255,0.7),
"slow_motion_integrator_right_predicted": (180/255,130/255,210/255,0.7),

# ===== Myelinated =====
"myelinated_left": (80/255,220/255,100/255,0.7),
"myelinated_right": (80/255,220/255,100/255,0.7),

# ===== Axons (no LDA split) =====
"axon_rostral_left": (1,1,1,0.7),
"axon_rostral_right": (1,1,1,0.7),
"axon_caudal_left": (0,0,0,0.7),
"axon_caudal_right": (0,0,0,0.7),
}

def plot_LDA_split_connectivity_matrix(
    df: pd.DataFrame,
    root_folder: Union[str, Path],
    output_folder: Union[str, Path],
    title: str,
    suffix: str = "",
):
    """
    Convenience wrapper that:
        1. Creates hemisphere + LDA split functional groups
        2. Creates ID → functional_type mapping
        3. Builds directional connectivity matrix
        4. Filters it
        5. Plots using Inhibitory–Excitatory scheme but with
           grey-only shading for axons.

    Produces a raster-mode IE matrix with all functional + predicted classes.
    """

    # 1. Build LDA-split groups
    nucleus_groups = create_nucleus_id_groups_hemisphere_LDA(df)
    func_types = generate_functional_types(nucleus_groups)

    # 2. Combine all IDs
    all_ids = np.concatenate([v for v in nucleus_groups.values()])

    # 3. Build matrix
    matrix = generate_directional_connectivity_matrix_general(
        root_folder=root_folder,
        seg_ids=all_ids,
        df_w_hemisphere=df,
    )

    # 4. Filter
    filtered_matrix, filtered_types = filter_connectivity_matrix(
        matrix, func_types
    )

    # 5. Build category order dynamically
    # Ordered by hemisphere left→right and functional block
    category_order = sorted(func_types.values(),
                            key=lambda x: (("_right" in x), x))

    # 6. Plot
    plot_connectivity_matrix(
        filtered_matrix,
        filtered_types,
        output_path=output_folder,
        category_order=category_order,
        df=df,
        title=f"{title}{suffix}",
        display_type="Inhibitory_Excitatory",
        plot_type="raster",
        color_cell_type_dict=COLOR_CELL_TYPE_DICT_LR_LDA,
    )