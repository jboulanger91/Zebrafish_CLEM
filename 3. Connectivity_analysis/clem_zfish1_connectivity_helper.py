#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

- ipsilateral_motion_integrator     (previously: integrator_ipsilateral)
- contralateral_motion_integrator   (previously: integrator_contralateral)
- motion_onset                      (previously: dynamic_threshold)
- slow_motion_integrator            (previously: motor_command)

The underlying metadata table still uses:
- 'integrator'       in the "functional classifier" column
- 'dynamic_threshold'
- 'motor_command'
- 'myelinated'

These are mapped to the new labels when constructing connectivity matrices
and color-coded summaries.
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
    "ipsilateral_motion_integrator": (254 / 255, 179 / 255, 38 / 255, 0.7),        # Yellow-orange
    "contralateral_motion_integrator": (232 / 255, 77 / 255, 138 / 255, 0.7),      # Magenta-pink
    "motion_onset": (100 / 255, 197 / 255, 235 / 255, 0.7),                        # Light blue
    "slow_motion_integrator": (127 / 255, 88 / 255, 175 / 255, 0.7),               # Purple
    "myelinated": (80 / 255, 220 / 255, 100 / 255, 0.7),                           # Green
    "axon_rostral": (105 / 255, 105 / 255, 105 / 255, 0.7),                        # Dim gray
    "axon_caudal": (192 / 255, 192 / 255, 192 / 255, 0.7),                         # Light gray
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
    "axon_rostral_left": (105 / 255, 105 / 255, 105 / 255, 0.7),
    "axon_rostral_right": (105 / 255, 105 / 255, 105 / 255, 0.7),
    "axon_caudal_left": (192 / 255, 192 / 255, 192 / 255, 0.7),
    "axon_caudal_right": (192 / 255, 192 / 255, 192 / 255, 0.7),
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
    Standardize naming in the 'functional classifier' column.

    This only normalizes spelling/spacing from the original annotations,
    e.g. 'dynamic threshold' -> 'dynamic_threshold'. Higher-level functional
    labels (e.g. 'motion_onset') are introduced when grouping IDs.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table.

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardized 'functional classifier' entries.
    """
    if "functional classifier" not in df.columns:
        return df

    replacements = {
        "dynamic threshold": "dynamic_threshold",
        "motor command": "motor_command",
    }
    df["functional classifier"] = df["functional classifier"].replace(replacements)
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
    - 'integrator'          in 'functional classifier'
    - 'dynamic_threshold'
    - 'motor_command'
    - 'myelinated'
    plus 'projection classifier' ('ipsilateral'/'contralateral').

    Returns
    -------
    dict
        Mapping from functional group name to list of string IDs.
    """
    groups = {
        "axon_rostral": df.loc[
            (df["type"] == "axon")
            & (df["comment"] == "axon exits the volume rostrally"),
            "axon_id",
        ],
        "ipsilateral_motion_integrator": fetch_filtered_ids(
            df, 9, "integrator", 11, "ipsilateral"
        )[0],
        "contralateral_motion_integrator": fetch_filtered_ids(
            df, 9, "integrator", 11, "contralateral"
        )[0],
        "motion_onset": fetch_filtered_ids(df, 9, "dynamic_threshold")[0],
        "slow_motion_integrator": fetch_filtered_ids(df, 9, "motor_command")[0],
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
        # Integrators by projection + hemisphere
        "ipsilateral_motion_integrator_left": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "integrator")
            & (df["projection classifier"] == "ipsilateral")
            & (df["hemisphere"] == "L"),
            "nucleus_id",
        ],
        "ipsilateral_motion_integrator_right": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "integrator")
            & (df["projection classifier"] == "ipsilateral")
            & (df["hemisphere"] == "R"),
            "nucleus_id",
        ],
        "contralateral_motion_integrator_left": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "integrator")
            & (df["projection classifier"] == "contralateral")
            & (df["hemisphere"] == "L"),
            "nucleus_id",
        ],
        "contralateral_motion_integrator_right": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "integrator")
            & (df["projection classifier"] == "contralateral")
            & (df["hemisphere"] == "R"),
            "nucleus_id",
        ],
        # Motion onset
        "motion_onset_left": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "dynamic_threshold")
            & (df["hemisphere"] == "L"),
            "nucleus_id",
        ],
        "motion_onset_right": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "dynamic_threshold")
            & (df["hemisphere"] == "R"),
            "nucleus_id",
        ],
        # Slow motion integrator
        "slow_motion_integrator_left": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "motor_command")
            & (df["hemisphere"] == "L"),
            "nucleus_id",
        ],
        "slow_motion_integrator_right": df.loc[
            (df["type"] == "cell")
            & (df["functional classifier"] == "motor_command")
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

    The file is expected to have:
    - either a 'postsynaptic ID' column (presynaptic file)
    - or a 'presynaptic_ID' column (postsynaptic file)
    plus 'synapse_id' and 'validation_status'.
    """
    df = pd.read_csv(path)

    if "postsynaptic ID" in df.columns:
        df["partner_cell_id"] = df["postsynaptic ID"]
    elif "presynaptic_ID" in df.columns:
        df["partner_cell_id"] = df["presynaptic_ID"]
    else:
        raise ValueError(
            f"Could not find 'postsynaptic ID' or 'presynaptic_ID' in {path}"
        )

    if "synapse_id" not in df.columns or "validation_status" not in df.columns:
        raise ValueError(
            f"'synapse_id' and/or 'validation_status' missing in {path}"
        )

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
                outputs_data["validation_status"].str.contains("valid", na=False)
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
                inputs_data["validation_status"].str.contains("valid", na=False)
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
    - reads its NG-res presynapse and postsynapse tables,
    - uses hemisphere-aware connectivity (via get_inputs_outputs_by_hemisphere_general),
    - counts unique synapses between pairs of IDs in `seg_ids`,
      separately for inputs and outputs, and combines them into a single matrix.

    The matrix is:
        rows    = source (pre) IDs
        columns = target (post) IDs
    where entries represent the number of synapses (inputs+outputs) between them,
    with sign left untouched here; any inhibitory/excitatory flipping is done in
    the plotting function.

    Parameters
    ----------
    root_folder : str or Path
        Root folder containing per-neuron synapse tables.
    seg_ids : iterable of str
        IDs used as both row and column labels of the matrix.
    df_w_hemisphere : pandas.DataFrame
        Metadata with hemisphere information.

    Returns
    -------
    pandas.DataFrame
        Square connectivity matrix indexed by seg_ids.
    """
    root_folder = Path(root_folder)
    seg_ids_str = [str(s) for s in seg_ids]
    connectivity_matrix = pd.DataFrame(0, index=seg_ids_str, columns=seg_ids_str)

    stored_nonzero_synapse_ids: set[int] = set()

    for source_id in seg_ids_str:
        results = get_inputs_outputs_by_hemisphere_general(
            root_folder=root_folder,
            seed_cell_ids=[source_id],
            hemisphere_df=df_w_hemisphere,
        )

        # --------------------- OUTPUTS (source -> target) ----------------
        # Try cell presynapses, then axon
        cell_name = f"clem_zfish1_cell_{source_id}"
        cell_file_name = f"{cell_name}_ng_res_presynapses.csv"
        cell_output_file_path = root_folder / cell_name / cell_file_name

        if cell_output_file_path.exists():
            output_file_path = cell_output_file_path
        else:
            axon_name = f"clem_zfish1_axon_{source_id}"
            axon_file_name = f"{axon_name}_ng_res_presynapses.csv"
            output_file_path = root_folder / axon_name / axon_file_name

        if not output_file_path.exists():
            continue

        outputs_data = pd.read_csv(
            output_file_path,
            comment="#",
            sep=" ",
            header=None,
            names=[
                "partner_cell_id",
                "x",
                "y",
                "z",
                "synapse_id",
                "size",
                "prediction_status",
                "validation_status",
                "date",
            ],
        )
        valid_outputs = outputs_data[
            outputs_data["validation_status"].str.contains("valid", na=False)
        ]

        for direction in ["same_side", "different_side"]:
            outputs = results["outputs"]["synapses"][direction]
            if outputs.empty or "nucleus_id" not in outputs.columns:
                continue

            try:
                outputs = outputs.copy()
                outputs["nucleus_id"] = outputs["nucleus_id"].astype(int)
                seg_ids_int = [int(s) for s in seg_ids_str]
                outputs = outputs[outputs["nucleus_id"].isin(seg_ids_int)]
            except ValueError as e:  # noqa: BLE001
                print(f"Error converting nucleus_id or seg_ids to int: {e}")
                continue

            for _, output_row in outputs.iterrows():
                target_dend = output_row["dendrite_id"]
                matching = valid_outputs[
                    valid_outputs["partner_cell_id"] == target_dend
                ]
                synapse_ids = matching["synapse_id"].tolist()

                zero_synapse_count = synapse_ids.count(0)
                nonzero_synapse_ids = [sid for sid in synapse_ids if sid != 0]
                new_nonzero = [
                    sid for sid in nonzero_synapse_ids if sid not in stored_nonzero_synapse_ids
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
        input_file_name = f"{cell_name}_ng_res_postsynapses.csv"
        input_file_path = root_folder / cell_name / input_file_name
        if not input_file_path.exists():
            continue

        inputs_data = pd.read_csv(
            input_file_path,
            comment="#",
            sep=" ",
            header=None,
            names=[
                "partner_cell_id",
                "x",
                "y",
                "z",
                "synapse_id",
                "size",
                "prediction_status",
                "validation_status",
                "date",
            ],
        )
        valid_inputs = inputs_data[
            inputs_data["validation_status"].str.contains("valid", na=False)
        ]

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
                target_axon = input_row["axon_id"]
                matching = valid_inputs[
                    valid_inputs["partner_cell_id"] == target_axon
                ]
                synapse_ids = matching["synapse_id"].tolist()

                zero_synapse_count = synapse_ids.count(0)
                nonzero_synapse_ids = [sid for sid in synapse_ids if sid != 0]
                new_nonzero = [
                    sid for sid in nonzero_synapse_ids if sid not in stored_nonzero_synapse_ids
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

    For each row index (functional ID), this function looks up the corresponding
    row in `df` (matching 'nucleus_id') and:

    - if 'neurotransmitter classifier' == 'inhibitory', multiplies the row by -1
    - if 'excitatory', leaves it unchanged

    Parameters
    ----------
    matrix : pandas.DataFrame
        Connectivity matrix (will be modified and returned).
    df : pandas.DataFrame
        Metadata with 'nucleus_id' and 'neurotransmitter classifier'.

    Returns
    -------
    pandas.DataFrame
        Modified matrix.
    """
    for idx in matrix.index:
        df_row = df.loc[df["nucleus_id"] == idx]
        if df_row.empty:
            raise ValueError(
                f"Index '{idx}' in the matrix does not have a matching 'nucleus_id' in df."
            )
        classifier = df_row.iloc[0]["neurotransmitter classifier"]
        if classifier == "inhibitory":
            matrix.loc[idx] *= -1

    return matrix


def plot_connectivity_matrix(
    matrix: pd.DataFrame,
    functional_types: Dict[str, str],
    output_path: str | Path,
    category_order: List[str],
    df: pd.DataFrame | None = None,
    title: str = "Directional Connectivity Matrix",
    display_type: str = "normal",
    plot_type: str = "heatmap",
    color_cell_type_dict: Dict[str, Tuple[float, float, float, float]] | None = None,
) -> None:
    """
    Plot a connectivity matrix with optional inhibitory/excitatory encoding.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Connectivity matrix to plot.
    functional_types : dict
        Mapping from matrix index ID -> functional category name.
    output_path : str or Path
        Directory where the resulting PDF will be written.
    category_order : list of str
        Ordered list of functional categories for sorting indices and
        drawing separators.
    df : pandas.DataFrame, optional
        Metadata DataFrame with 'nucleus_id' and 'neurotransmitter classifier'.
        Required if display_type == 'Inhibitory_Excitatory'.
    title : str, optional
        Plot title and base for the output filename.
    display_type : {'normal', 'Inhibitory_Excitatory'}, optional
        - 'normal' plots raw counts.
        - 'Inhibitory_Excitatory' flips inhibitory rows and uses a diverging
          colormap around zero.
    plot_type : {'heatmap', 'scatter'}, optional
        - 'heatmap' uses matshow.
        - 'scatter' uses a bubble-plot representation.
    color_cell_type_dict : dict, optional
        Mapping from functional category name -> RGBA color.

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
    if plot_type not in {"heatmap", "scatter"}:
        raise ValueError("plot_type must be 'heatmap' or 'scatter'.")

    if display_type == "Inhibitory_Excitatory":
        if df is None:
            raise ValueError(
                "df is required for 'Inhibitory_Excitatory' display_type."
            )
        matrix = process_matrix(matrix.copy(), df)
    else:
        matrix = matrix.copy()

    functional_types = {
        k: v for k, v in functional_types.items() if v in category_order and k in matrix.index
    }
    filtered_indices = [
        idx
        for idx in matrix.index
        if functional_types.get(idx, "unknown") in category_order
    ]
    filtered_matrix = matrix.loc[filtered_indices, filtered_indices]

    def sort_key(func_id: str) -> int:
        category = functional_types.get(func_id, "unknown")
        return category_order.index(category) if category in category_order else len(
            category_order
        )

    sorted_indices = sorted(filtered_indices, key=sort_key)
    matrix_with_nan = filtered_matrix.loc[sorted_indices, sorted_indices]

    if display_type == "Inhibitory_Excitatory":
        matrix_with_nan = np.clip(matrix_with_nan, -2, 2)

        colors = [
            "#9B00AE",  # strong inhibitory (dark magenta)
            "#FF4DFF",  # weak inhibitory (light magenta)
            "#FFFFFF",  # zero
            "#7CFF5A",  # weak excitatory (light green)
            "#007A00",  # strong excitatory (dark green)
        ]

        cmap = mcolors.ListedColormap(colors, name="Inhibitory-Excitatory")
        bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        ticks = [-2, -1, 0, 1, 2]
        cbar_label = "Synapse strength (inhibitory → excitatory)"
    else:
        cmap = mcolors.ListedColormap(
            ["white", "blue", "green", "yellow", "pink", "red"]
        )
        bounds = [0, 1, 2, 3, 4, 5, 6]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        ticks = [0, 1, 2, 3, 4, 5]
        cbar_label = "Number of synapses"

    fig, ax = plt.subplots(figsize=(10, 10))

    if plot_type == "heatmap":
        cax = ax.matshow(matrix_with_nan, cmap=cmap, norm=norm)
        artist_for_cbar = cax
    else:
        x, y = np.meshgrid(
            range(len(matrix_with_nan.columns)), range(len(matrix_with_nan.index))
        )
        x_flat, y_flat = x.flatten(), y.flatten()
        vals = matrix_with_nan.values.flatten()
        sizes = np.abs(vals) * 100.0
        scatter = ax.scatter(x_flat, y_flat, c=vals, s=sizes, cmap=cmap, norm=norm)
        artist_for_cbar = scatter

    ax.set_aspect("equal")

    cbar = plt.colorbar(
        artist_for_cbar,
        ax=ax,
        boundaries=bounds,
        ticks=ticks,
        spacing="uniform",
        orientation="horizontal",
        pad=0.1,
    )
    cbar.set_label(cbar_label)

    ax.set_xticks(range(len(matrix_with_nan.columns)))
    ax.set_yticks(range(len(matrix_with_nan.index)))
    ax.set_xticklabels(matrix_with_nan.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(matrix_with_nan.index, fontsize=8)
    ax.set_xlabel("Pre-synaptic (axons)")
    ax.set_ylabel("Post-synaptic (dendrites)")
    ax.set_title(title, fontsize=12)

    bar_width = 3
    bar_height = 3

    for i, functional_id in enumerate(matrix_with_nan.index):
        if functional_id not in functional_types:
            continue
        ftype = functional_types[functional_id]
        color = color_cell_type_dict.get(ftype, (0.8, 0.8, 0.8, 0.7))

        ax.add_patch(
            patches.Rectangle(
                (-bar_width, i - 0.5),
                bar_width,
                1,
                color=color,
                zorder=2,
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (i - 0.5, -bar_height),
                1,
                bar_height,
                color=color,
                zorder=2,
            )
        )

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

    plt.tight_layout()
    sanitized_title = (
        title.lower().replace(" ", "_").replace(":", "").replace("/", "_")
    )
    output_pdf_path = output_path / f"{sanitized_title}.pdf"
    plt.savefig(output_pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()