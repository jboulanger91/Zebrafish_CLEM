#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for hemisphere-aware connectome analysis and
two-layer network visualizations (zebrafish hindbrain).

Used by `regenerate_connectome_networks.py` to:
    - select seed neuron IDs from an LDA-annotated metadata table,
    - extract input/output partners split by hemisphere,
    - summarize synapse counts into probabilities,
    - draw compact two-layer network diagrams.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterable, Dict, Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from colorsys import rgb_to_hls, hls_to_rgb

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

# Functional cell-type colors (used for node fill colors)
COLOR_CELL_TYPE_DICT: Dict[str, tuple[float, float, float, float]] = {
    "integrator_ipsilateral": (254 / 255, 179 / 255, 38 / 255, 0.7),    # yellow-orange
    "integrator_contralateral": (232 / 255, 77 / 255, 138 / 255, 0.7),  # magenta-pink
    "dynamic_threshold": (100 / 255, 197 / 255, 235 / 255, 0.7),        # light blue
    "motor_command": (127 / 255, 88 / 255, 175 / 255, 0.7),             # purple
    "myelinated": (68 / 255, 252 / 255, 215 / 255, 1.0),                # teal
    "axon": (0.2, 0.2, 0.2, 0.7),                                       # dark gray
    "not functionally imaged": (0.5, 0.5, 0.5, 0.7),                    # gray
}


# ---------------------------------------------------------------------------
# Simple ID-selection helpers
# ---------------------------------------------------------------------------

def fetch_filtered_ids(
    df: pd.DataFrame,
    col_1_index: int,
    condition_1: Any,
    col_2_index: int | None = None,
    condition_2: Any | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Return unique nucleus IDs and functional IDs matching one or two conditions.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table.
    col_1_index : int
        Column index of the first condition (e.g. functional classifier).
    condition_1 : Any
        Value to match in column `col_1_index`.
    col_2_index : int, optional
        Column index of the second condition (e.g. projection classifier).
    condition_2 : Any, optional
        Value to match in column `col_2_index`.

    Returns
    -------
    nuclei_ids : pandas.Series
        Unique IDs from column 5 (nucleus_id).
    functional_ids : pandas.Series
        Unique IDs from column 1 (functional_id or similar).
    """
    filtered_rows = df.loc[df.iloc[:, col_1_index] == condition_1]

    if col_2_index is not None and condition_2 is not None:
        filtered_rows = filtered_rows.loc[
            filtered_rows.iloc[:, col_2_index] == condition_2
        ]

    nuclei_ids = filtered_rows.iloc[:, 5].drop_duplicates()
    functional_ids = filtered_rows.iloc[:, 1].drop_duplicates()
    return nuclei_ids, functional_ids


def fetch_filtered_ids_EI(
    df: pd.DataFrame,
    col_1_index: int,
    condition_1: Any,
    col_2_index: int | None = None,
    condition_2: Any | None = None,
    col_3_index: int | None = None,
    condition_3: Any | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Same as `fetch_filtered_ids`, but supports a third condition
    (used for integrator E/I + ipsi/contra splits).

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table.
    col_1_index, col_2_index, col_3_index : int
        Column indices used for filtering (e.g. functional, NT, projection).
    condition_1, condition_2, condition_3 : Any
        Values to match in the corresponding columns.

    Returns
    -------
    nuclei_ids : pandas.Series
        Unique IDs from column 5 (nucleus_id).
    functional_ids : pandas.Series
        Unique IDs from column 1 (functional_id or similar).
    """
    filtered_rows = df.loc[df.iloc[:, col_1_index] == condition_1]

    if col_2_index is not None and condition_2 is not None:
        filtered_rows = filtered_rows.loc[
            filtered_rows.iloc[:, col_2_index] == condition_2
        ]

    if col_3_index is not None and condition_3 is not None:
        filtered_rows = filtered_rows.loc[
            filtered_rows.iloc[:, col_3_index] == condition_3
        ]

    nuclei_ids = filtered_rows.iloc[:, 5].drop_duplicates()
    functional_ids = filtered_rows.iloc[:, 1].drop_duplicates()
    return nuclei_ids, functional_ids


# ---------------------------------------------------------------------------
# Hemisphere-aware connectivity extraction
# ---------------------------------------------------------------------------

def _fill_functional_classifier(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in 'functional classifier' with 'not functionally imaged'."""
    if "functional classifier" not in df.columns:
        return df

    df = df.copy()
    df["functional classifier"] = df["functional classifier"].astype("object")
    df["functional classifier"] = df["functional classifier"].fillna(
        "not functionally imaged"
    )
    return df


def get_inputs_outputs_by_hemisphere(
    root_folder: Path | str,
    seed_cell_ids: Iterable[int | str],
    hemisphere_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Extract and categorize input/output partners for given seed cells, split by hemisphere.

    For each seed nucleus ID:
        - locate its NG-res presynapse and postsynapse CSVs,
        - keep only synapses with 'valid' validation_status,
        - match partner IDs onto the metadata table (via dendrite_id / axon_id),
        - split partners into same-side vs different-side relative to the seed
          (using the 'hemisphere' column in `hemisphere_df`),
        - accumulate both per-cell and per-synapse tables.

    Parameters
    ----------
    root_folder : str or Path
        Root folder containing the per-neuron synapse CSV files.
    seed_cell_ids : iterable
        Nucleus IDs used as seeds.
    hemisphere_df : pandas.DataFrame
        Metadata table; must contain columns:
            'nucleus_id', 'axon_id', 'dendrite_id', 'hemisphere',
            'type', 'functional_id', 'functional classifier',
            'neurotransmitter classifier', 'projection classifier', 'comment'.

    Returns
    -------
    dict
        Nested dict with keys:
            results["outputs" or "inputs"]["cells" or "synapses"]["same_side"/"different_side"]
        plus simple counters and percentage sums.
    """
    root_folder = Path(root_folder)

    hemi_df = hemisphere_df.copy()
    hemi_df["nucleus_id"] = hemi_df["nucleus_id"].astype(str)
    hemisphere_map = hemi_df.set_index("nucleus_id")["hemisphere"].to_dict()

    results: Dict[str, Any] = {
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

    # Helper to find a single CSV for a given pattern anywhere under root_folder
    def _find_csv(pattern: str) -> Path | None:
        for root, _, files in os.walk(root_folder):
            for filename in fnmatch.filter(files, pattern):
                return Path(root) / filename
        return None

    for seed_cell_id in seed_cell_ids:
        seed_id_str = str(seed_cell_id)
        seed_hemi = hemisphere_map.get(seed_id_str, None)
        if seed_hemi is None:
            print(f"Seed cell ID {seed_cell_id} has no hemisphere data. Skipping.")
            continue

        # -------------------- OUTPUTS (presynapses) --------------------
        output_file_pattern = f"clem_zfish1_cell_{seed_id_str}_ng_res_presynapses.csv"
        output_file_path = _find_csv(output_file_pattern)

        if output_file_path is not None:
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
                outputs_data["validation_status"]
                .astype(str)
                .str.contains("valid", na=False)
            ]
            output_ids = valid_outputs["partner_cell_id"]

            traced_dendrites = output_ids[output_ids.isin(hemi_df["dendrite_id"])]
            matched_rows = (
                [
                    hemi_df[hemi_df["dendrite_id"] == dend].iloc[0]
                    for dend in traced_dendrites
                ]
                if not traced_dendrites.empty
                else []
            )
            output_connected = pd.DataFrame(matched_rows)

            if not output_connected.empty:
                output_connected_unique = output_connected.drop_duplicates(
                    subset="axon_id"
                )

                frac_syn = (
                    len(output_connected) / len(valid_outputs)
                    if len(valid_outputs) > 0
                    else 0.0
                )
                results["outputs"]["percentages"]["synapses"] += frac_syn
                results["counters"]["output_seed_counter"] += 1

                if "hemisphere" in output_connected_unique.columns:
                    same_cells = output_connected_unique[
                        output_connected_unique["hemisphere"] == seed_hemi
                    ]
                    diff_cells = output_connected_unique[
                        output_connected_unique["hemisphere"] != seed_hemi
                    ]

                    same_syn = output_connected[
                        output_connected["hemisphere"] == seed_hemi
                    ]
                    diff_syn = output_connected[
                        output_connected["hemisphere"] != seed_hemi
                    ]
                else:
                    same_cells = diff_cells = same_syn = diff_syn = pd.DataFrame()

                same_cells = _fill_functional_classifier(same_cells)
                diff_cells = _fill_functional_classifier(diff_cells)
                same_syn = _fill_functional_classifier(same_syn)
                diff_syn = _fill_functional_classifier(diff_syn)

                results["outputs"]["cells"]["same_side"] = pd.concat(
                    [results["outputs"]["cells"]["same_side"], same_cells],
                    ignore_index=True,
                )
                results["outputs"]["cells"]["different_side"] = pd.concat(
                    [results["outputs"]["cells"]["different_side"], diff_cells],
                    ignore_index=True,
                )
                results["outputs"]["synapses"]["same_side"] = pd.concat(
                    [results["outputs"]["synapses"]["same_side"], same_syn],
                    ignore_index=True,
                )
                results["outputs"]["synapses"]["different_side"] = pd.concat(
                    [results["outputs"]["synapses"]["different_side"], diff_syn],
                    ignore_index=True,
                )

        # -------------------- INPUTS (postsynapses) ---------------------
        input_file_pattern = f"clem_zfish1_cell_{seed_id_str}_ng_res_postsynapses.csv"
        input_file_path = _find_csv(input_file_pattern)

        if input_file_path is not None:
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
                inputs_data["validation_status"]
                .astype(str)
                .str.contains("valid", na=False)
            ]
            input_ids = valid_inputs["partner_cell_id"]

            traced_axons = input_ids[input_ids.isin(hemi_df["axon_id"])]
            matched_rows = (
                [
                    hemi_df[hemi_df["axon_id"] == ax].iloc[0]
                    for ax in traced_axons
                ]
                if not traced_axons.empty
                else []
            )
            input_connected = pd.DataFrame(matched_rows)

            if not input_connected.empty:
                input_connected_unique = input_connected.drop_duplicates(
                    subset="axon_id"
                )

                frac_syn = (
                    len(input_connected) / len(valid_inputs)
                    if len(valid_inputs) > 0
                    else 0.0
                )
                results["inputs"]["percentages"]["synapses"] += frac_syn
                results["counters"]["input_seed_counter"] += 1

                if "hemisphere" in input_connected_unique.columns:
                    same_cells = input_connected_unique[
                        input_connected_unique["hemisphere"] == seed_hemi
                    ]
                    diff_cells = input_connected_unique[
                        input_connected_unique["hemisphere"] != seed_hemi
                    ]

                    same_syn = input_connected[
                        input_connected["hemisphere"] == seed_hemi
                    ]
                    diff_syn = input_connected[
                        input_connected["hemisphere"] != seed_hemi
                    ]
                else:
                    same_cells = diff_cells = same_syn = diff_syn = pd.DataFrame()

                same_cells = _fill_functional_classifier(same_cells)
                diff_cells = _fill_functional_classifier(diff_cells)
                same_syn = _fill_functional_classifier(same_syn)
                diff_syn = _fill_functional_classifier(diff_syn)

                results["inputs"]["cells"]["same_side"] = pd.concat(
                    [results["inputs"]["cells"]["same_side"], same_cells],
                    ignore_index=True,
                )
                results["inputs"]["cells"]["different_side"] = pd.concat(
                    [results["inputs"]["cells"]["different_side"], diff_cells],
                    ignore_index=True,
                )
                results["inputs"]["synapses"]["same_side"] = pd.concat(
                    [results["inputs"]["synapses"]["same_side"], same_syn],
                    ignore_index=True,
                )
                results["inputs"]["synapses"]["different_side"] = pd.concat(
                    [results["inputs"]["synapses"]["different_side"], diff_syn],
                    ignore_index=True,
                )

    return results


# ---------------------------------------------------------------------------
# Counting + probabilities
# ---------------------------------------------------------------------------

def compute_count_probabilities_from_results(
    results: Dict[str, Any],
    functional_only: bool = False,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Turn the raw `get_inputs_outputs_by_hemisphere` results into
    compact count/probability tables.

    For each of:
        - conn_type in {'outputs', 'inputs'}
        - side in {'same_side', 'different_side'}
        - data_type in {'cells', 'synapses'}

    we build a table with columns:
        ['Functional Classifier',
         'Neurotransmitter Classifier',
         'Projection Classifier',
         'Axon Exit Direction',
         'Count', 'Probability'].

    Parameters
    ----------
    results : dict
        Output dict from `get_inputs_outputs_by_hemisphere`.
    functional_only : bool, optional
        If True, non-functionally imaged cells (functional_id == 'not functionally imaged'
        and functional classifier != 'myelinated') are excluded.

    Returns
    -------
    dict
        Nested dict: final_results[conn_type][side][data_type] -> DataFrame
    """

    def _process_category(df: pd.DataFrame, functional_only_flag: bool) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        rows: list[Dict[str, Any]] = []
        for _, row in df.iterrows():
            if row["type"] == "axon":
                # Axon-only partners (no functional classification)
                axon_exit = row.get("comment", None)
                if pd.notna(axon_exit):
                    rows.append(
                        {
                            "Functional Classifier": "axon",
                            "Neurotransmitter Classifier": None,
                            "Projection Classifier": None,
                            "Axon Exit Direction": axon_exit,
                        }
                    )
            elif row["type"] == "cell":
                if (
                    functional_only_flag
                    and row["functional_id"] == "not functionally imaged"
                    and row["functional classifier"] != "myelinated"
                ):
                    # Skip non-functional neurons when requested
                    continue
                rows.append(
                    {
                        "Functional Classifier": row["functional classifier"],
                        "Neurotransmitter Classifier": row["neurotransmitter classifier"],
                        "Projection Classifier": row["projection classifier"],
                        "Axon Exit Direction": None,
                    }
                )

        if not rows:
            return pd.DataFrame()

        conn_df = pd.DataFrame(rows).fillna("None")
        counts_df = conn_df.value_counts().reset_index(name="Count")
        counts_df["Probability"] = counts_df["Count"] / counts_df["Count"].sum()
        return counts_df

    final_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {
        conn_type: {
            side: {dtype: pd.DataFrame() for dtype in ["cells", "synapses"]}
            for side in ["same_side", "different_side"]
        }
        for conn_type in ["outputs", "inputs"]
    }

    for conn_type in ["outputs", "inputs"]:
        for side in ["same_side", "different_side"]:
            for dtype in ["cells", "synapses"]:
                df_cat = results.get(conn_type, {}).get(dtype, {}).get(side, pd.DataFrame())
                final_results[conn_type][side][dtype] = _process_category(
                    df_cat, functional_only
                )

    return final_results


# ---------------------------------------------------------------------------
# Two-layer network plotting
# ---------------------------------------------------------------------------

def _adjust_luminance(rgb: tuple[float, float, float, float], factor: float = 1.5) -> tuple[float, float, float]:
    """Return a luminance-scaled RGB triple (alpha is ignored)."""
    r, g, b = rgb[:3]
    h, l, s = rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * factor))
    return hls_to_rgb(h, l, s)


def draw_two_layer_neural_net(
    *,
    ax,
    left: float,
    right: float,
    bottom: float,
    top: float,
    data_df: pd.DataFrame,
    node_radius: float = 0.015,
    input_circle_color: str = "gray",
    input_cell_type: str = "excitatory",
    show_midline: bool = True,
    proportional_lines: bool = True,  # kept for API compatibility
    a: float = 5,
    b: float = 2,
    connection_type: str = "outputs",  # 'outputs' or 'inputs'
    add_legend: bool = True,
    # label controls
    label_mode: str = "count",         # 'count' | 'proportion' | 'probability' | 'none'
    label_as_percent: bool = True,
    label_decimals: int = 1,
    # cross-side totals (for normalization across both hemispheres)
    total_outputs: float | None = None,
    total_inputs: float | None = None,
) -> None:
    """
    Draw a two-layer network:

        layer 1: single "seed population" node
        layer 2: one node per row in `data_df` (connection category)

    Edge thickness ~ probability; node radius ~ probability; line style encodes
    neurotransmitter type (excitatory / inhibitory / unknown).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (created by caller).
    left, right, bottom, top : float
        Normalized coordinates for the drawing area (0–1).
    data_df : pandas.DataFrame
        Table produced by `compute_count_probabilities_from_results()['...']['synapses']`.
    node_radius : float, optional
        Base radius of output nodes.
    input_circle_color : str, optional
        Key into COLOR_CELL_TYPE_DICT for the seed node color.
    input_cell_type : {'excitatory', 'inhibitory', 'mixed'}, optional
        Used for drawing arrows/T-bars on output arrows.
    show_midline : bool, optional
        If True, draw a vertical midline (used in cross-side panels).
    proportional_lines : bool, optional
        Kept for compatibility; edges are always probability-scaled.
    a, b : float, optional
        Linear scaling for edge width: width = a * p + b.
    connection_type : {'outputs', 'inputs'}, optional
        Controls direction of arrows (seed→targets vs targets→seed).
    add_legend : bool, optional
        Whether to add a small legend for NT line styles.
    label_mode : str, optional
        How to label output nodes: 'count', 'proportion', 'probability', 'none'.
    label_as_percent : bool, optional
        If True, proportions/probabilities are formatted as percentages.
    label_decimals : int, optional
        Number of decimals for proportion/probability labels.
    total_outputs, total_inputs : float, optional
        Cross-side totals used to normalize proportions across panels.
        If None, the per-panel Count sum is used as denominator.
    """
    # Line style encodings for neurotransmitter classifier
    nt_outline = {
        "excitatory": "solid",
        "inhibitory": "dashed",
        "unknown": "dotted",
        "None": "dotted",
    }

    # Shrink drawing area slightly if midline is hidden
    if not show_midline:
        midpoint = (left + right) / 2
        if connection_type == "outputs":
            right = midpoint + (right - midpoint) * 0.8
        else:  # 'inputs'
            left = midpoint - (midpoint - left) * 0.8

    # ----- input (seed) node -----
    layer_sizes = [1, len(data_df)]
    v_spacing = (top - bottom) / float(max(layer_sizes) + 2)
    h_spacing = (right - left)  # single gap between the two layers

    input_y = bottom + (top - bottom) / 2.0
    input_x = left if connection_type == "outputs" else right
    input_center = (input_x, input_y)
    input_r = node_radius * 4.0

    seed_color = COLOR_CELL_TYPE_DICT.get(input_circle_color, (0.5, 0.5, 0.5, 0.7))
    input_circle = Circle(
        input_center,
        input_r,
        edgecolor=seed_color,
        facecolor=seed_color,
        lw=3,
        alpha=0.8,
    )
    ax.add_artist(input_circle)

    # ----- choose denominator for proportions -----
    panel_total = float(data_df["Count"].sum()) if "Count" in data_df.columns else 0.0
    if label_mode == "proportion":
        if connection_type == "outputs" and total_outputs is not None:
            denom = float(total_outputs)
        elif connection_type == "inputs" and total_inputs is not None:
            denom = float(total_inputs)
        else:
            denom = panel_total
    else:
        denom = panel_total

    # ----- merge special categories (not functionally imaged, dynamic_threshold) -----
    df = data_df.copy()

    def _merge_category(name: str, nt: str | None = None) -> pd.DataFrame:
        subset = df[df["Functional Classifier"] == name]
        if subset.empty:
            return df
        merged_prob = subset["Probability"].sum()
        merged_count = subset["Count"].sum()
        df_remaining = df[df["Functional Classifier"] != name]
        merged_row = {
            "Functional Classifier": name,
            "Projection Classifier": "None",
            "Neurotransmitter Classifier": nt if nt is not None else "unknown",
            "Probability": merged_prob,
            "Count": merged_count,
        }
        return pd.concat(
            [df_remaining, pd.DataFrame([merged_row])], ignore_index=True
        )

    df = _merge_category("not functionally imaged", nt="unknown")
    df = _merge_category("dynamic_threshold", nt="inhibitory")

    # ----- output nodes -----
    outputs: list[tuple[tuple[float, float], float]] = []
    layer2_top = bottom + (top - bottom) / 2.0 + v_spacing * (layer_sizes[1] - 1) / 2.0

    for idx, row in df.iterrows():
        y = layer2_top - idx * v_spacing
        x = right if connection_type == "outputs" else left
        center = (x, y)

        func = row["Functional Classifier"]
        proj = row["Projection Classifier"]

        if func in COLOR_CELL_TYPE_DICT:
            key = func
        elif func == "integrator":
            key = f"{func}_{proj}"
        else:
            key = "not functionally imaged"

        fill = COLOR_CELL_TYPE_DICT.get(key, COLOR_CELL_TYPE_DICT["not functionally imaged"])
        edge_rgb = _adjust_luminance(fill, factor=0.5)

        nt = row.get("Neurotransmitter Classifier", "unknown")
        nt = nt if isinstance(nt, str) else "unknown"
        style = nt_outline.get(nt, "solid")

        prob = float(row["Probability"])
        radius = node_radius * (1.0 + 4.0 * prob)

        circ = Circle(
            center,
            radius,
            edgecolor=edge_rgb,
            facecolor=fill,
            lw=3,
            alpha=0.8,
            linestyle=style,
        )
        ax.add_artist(circ)

        # ----- node labels -----
        label_text: str | None = None
        if label_mode != "none":
            if label_mode == "count":
                count_val = int(row["Count"]) if "Count" in row else 0
                label_text = f"{count_val}"
            elif label_mode == "proportion":
                count_val = float(row["Count"]) if "Count" in row else 0.0
                prop = (count_val / denom) if denom > 0 else 0.0
                if label_as_percent:
                    label_text = f"{prop * 100:.{label_decimals}f}%"
                else:
                    label_text = f"{prop:.{label_decimals}f}"
            elif label_mode == "probability":
                p = float(row.get("Probability", 0.0))
                if label_as_percent:
                    label_text = f"{p * 100:.{label_decimals}f}%"
                else:
                    label_text = f"{p:.{label_decimals}f}"

        if label_text is not None:
            offset = -radius * 1.5 if connection_type == "inputs" else radius * 1.5
            ax.text(
                center[0] + offset,
                center[1],
                label_text,
                fontsize=12,
                ha="left",
                va="center",
                color="black",
                fontname="Arial",
            )

        outputs.append((center, radius))

    # ----- connections -----
    for idx, (out_center, out_r) in enumerate(outputs):
        prob = float(df.iloc[idx]["Probability"]) if idx < len(df) else 0.0
        width = a * prob + b

        dx = out_center[0] - input_center[0]
        dy = out_center[1] - input_center[1]
        dist = np.hypot(dx, dy) if (dx or dy) else 1e-9

        src_x = input_center[0] + (input_r / dist) * dx
        src_y = input_center[1] + (input_r / dist) * dy
        dst_x = out_center[0] - (out_r / dist) * dx
        dst_y = out_center[1] - (out_r / dist) * dy

        if connection_type == "inputs":
            src_x, dst_x = dst_x, src_x
            src_y, dst_y = dst_y, src_y

        line = Line2D(
            [src_x, dst_x],
            [src_y, dst_y],
            c="black",
            lw=width,
            alpha=0.8,
        )
        ax.add_artist(line)

        # Arrow / T-bar endings for inputs
        if connection_type == "inputs":
            nt = df.iloc[idx]["Neurotransmitter Classifier"]
            if nt == "excitatory":
                arrow_start_x = dst_x - 0.1 * (dst_x - src_x)
                arrow_start_y = dst_y - 0.1 * (dst_y - src_y)
                ax.arrow(
                    arrow_start_x,
                    arrow_start_y,
                    (dst_x - arrow_start_x),
                    (dst_y - arrow_start_y),
                    head_width=width * 0.01,
                    head_length=width * 0.01,
                    fc="black",
                    ec="black",
                    length_includes_head=True,
                )
            elif nt == "inhibitory":
                t_len = width * 0.01
                t_dx = dy / dist
                t_dy = -dx / dist
                ax.plot(
                    [dst_x - t_len * t_dx, dst_x + t_len * t_dx],
                    [dst_y - t_len * t_dy, dst_y + t_len * t_dy],
                    c="black",
                    lw=2,
                )
        else:
            # Outputs: use `input_cell_type` for endings
            if input_cell_type == "excitatory":
                arrow_start_x = dst_x - 0.1 * (dst_x - src_x)
                arrow_start_y = dst_y - 0.1 * (dst_y - src_y)
                ax.arrow(
                    arrow_start_x,
                    arrow_start_y,
                    (dst_x - arrow_start_x),
                    (dst_y - arrow_start_y),
                    head_width=width * 0.01,
                    head_length=width * 0.01,
                    fc="black",
                    ec="black",
                    length_includes_head=True,
                )
            elif input_cell_type == "inhibitory":
                t_len = width * 0.01
                t_dx = -dy / dist
                t_dy = dx / dist
                ax.plot(
                    [dst_x - t_len * t_dx, dst_x + t_len * t_dx],
                    [dst_y - t_len * t_dy, dst_y + t_len * t_dy],
                    c="black",
                    lw=2,
                )
            # 'mixed' -> straight line only

    # Optional midline (used in cross-side panels)
    if show_midline:
        mid_x = (left + right) / 2.0
        ax.plot(
            [mid_x, mid_x],
            [bottom + 0.05, top - 0.05],
            color="lightgray",
            linestyle="--",
            linewidth=1.5,
        )

    # Legend for neurotransmitter line styles
    if add_legend:
        legend_elements = [
            Line2D([0], [0], color="black", lw=3, linestyle="solid", label="Excitatory"),
            Line2D([0], [0], color="black", lw=3, linestyle="dashed", label="Inhibitory"),
            Line2D([0], [0], color="black", lw=3, linestyle="dotted", label="Unknown"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=12,
            frameon=False,
            title="Neurotransmitter\nClassifier",
            title_fontsize=12,
            prop={"family": "Arial"},
        )

    ax.axis("off")
    ax.set_xlim(left - h_spacing, right + h_spacing)
    ax.set_ylim(bottom - v_spacing * 1.5, top + v_spacing)
    ax.set_aspect("equal", adjustable="datalim")