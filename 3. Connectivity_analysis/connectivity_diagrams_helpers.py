#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for two-layer network diagrams (hindbrain LDA connectomes).

This module is focused on the two-layer network plots used in
`make_connectome_diagrams.py`. It reuses generic utilities from
`matrix_helpers.py` and adds:

- LDA metadata loading
- Seed ID selection for functional populations (cMI, MON, MC, iMI±)
- Conversion of hemisphere-resolved connectivity into synapse-count
  probability tables
- A two-layer network drawing primitive

Functional naming conventions
-----------------------------
The LDA-based metadata may use older labels:

    'dynamic_threshold'
    'integrator'
    'motor_command'

For consistency with the rest of the project (and with `matrix_helpers.py`),
we standardize them to the canonical names:

    dynamic_threshold  -> motion_integrator
    integrator         -> motion_integrator
    motor_command      -> slow_motion_integrator

Other functional labels (e.g. 'motion_onset', 'myelinated') are left as-is.

Color mapping
-------------
Colors are taken from `matrix_helpers.COLOR_CELL_TYPE_DICT`, which uses keys:

    'ipsilateral_motion_integrator'
    'contralateral_motion_integrator'
    'motion_onset'
    'slow_motion_integrator'
    'myelinated'
    'axon_rostral'
    'axon_caudal'
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from colorsys import rgb_to_hls, hls_to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# -------------------------------------------------------------------------
# Import shared tools from the matrix helpers
# -------------------------------------------------------------------------

from matrix_helpers import (  # type: ignore[import]
    COLOR_CELL_TYPE_DICT,
    fetch_filtered_ids,
    get_inputs_outputs_by_hemisphere_general,
)


# -------------------------------------------------------------------------
# Simple I/O helpers
# -------------------------------------------------------------------------


def load_lda_metadata(path: Path | str) -> pd.DataFrame:
    """
    Load the LDA-annotated metadata table from CSV.

    Parameters
    ----------
    path : Path or str
        Path to the LDA metadata CSV.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(path)


# -------------------------------------------------------------------------
# Functional naming standardisation
# -------------------------------------------------------------------------


def standardize_functional_name(func: str | None) -> str | None:
    """
    Map LDA functional labels to canonical names used elsewhere.

    Mapping:
        dynamic_threshold  -> motion_integrator
        integrator         -> motion_integrator
        motor_command      -> slow_motion_integrator

    Any other value is returned unchanged.
    """
    if func is None or pd.isna(func):
        return func

    func = str(func)

    if func == "dynamic_threshold":
        return "motion_integrator"
    if func == "integrator":
        return "motion_integrator"
    if func == "motor_command":
        return "slow_motion_integrator"
    return func


# -------------------------------------------------------------------------
# Seed ID selection
# -------------------------------------------------------------------------


def fetch_filtered_ids_EI(
    df: pd.DataFrame,
    col_1_index: int,
    condition_1: str,
    col_2_index: int | None = None,
    condition_2: str | None = None,
    col_3_index: int | None = None,
    condition_3: str | None = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Filter DataFrame by up to three column index + condition pairs.

    All conditions are combined with logical AND.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table.
    col_1_index, condition_1 : int, str
        First column index and value to match.
    col_2_index, condition_2 : int, str, optional
        Second column index and value to match.
    col_3_index, condition_3 : int, str, optional
        Third column index and value to match.

    Returns
    -------
    (pandas.Series, pandas.Series)
        Series of unique nucleus IDs (column 5) and functional IDs (column 1).
    """
    filtered = df.loc[df.iloc[:, col_1_index] == condition_1]

    if col_2_index is not None and condition_2 is not None:
        filtered = filtered.loc[filtered.iloc[:, col_2_index] == condition_2]

    if col_3_index is not None and condition_3 is not None:
        filtered = filtered.loc[filtered.iloc[:, col_3_index] == condition_3]

    nuclei_ids = filtered.iloc[:, 5].drop_duplicates()
    functional_ids = filtered.iloc[:, 1].drop_duplicates()
    return nuclei_ids, functional_ids


from typing import Dict
import pandas as pd

def get_seed_id_sets(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Collect seed nucleus ID sets for each functional population.

    Assumes the 'functional classifier' column (col 9) uses the
    canonical labels:

        'motion_integrator'
        'motion_onset'
        'slow_motion_integrator'
        'myelinated'

    Uses `fetch_filtered_ids` / `fetch_filtered_ids_EI`, which operate
    on column indices rather than names:

        col 9  -> functional classifier
        col 10 -> neurotransmitter classifier
        col 11 -> projection classifier

    Populations
    -----------
    - cMI      : motion_integrator, contralateral
    - MON      : motion_onset
    - MC       : slow_motion_integrator
    - iMI_all  : motion_integrator, ipsilateral
    - iMI_plus : motion_integrator, ipsilateral, excitatory
    - iMI_minus: motion_integrator, ipsilateral, inhibitory
    """
    # MON: all motion_onset
    mon_ids_all_nuc, _ = fetch_filtered_ids(df, 9, "motion_onset")

    # cMI: contralateral motion_integrator
    cmi_ids_all_nuc, _ = fetch_filtered_ids(
        df, 9, "motion_integrator", 11, "contralateral"
    )

    # iMI (all ipsilateral motion_integrator)
    imi_ids_all_nuc, _ = fetch_filtered_ids(
        df, 9, "motion_integrator", 11, "ipsilateral"
    )

    # MC: all slow_motion_integrator
    mc_ids_all_nuc, _ = fetch_filtered_ids(df, 9, "slow_motion_integrator")

    # iMI+ : motion_integrator, ipsilateral, excitatory
    imi_ex_ids_all_nuc, _ = fetch_filtered_ids_EI(
        df, 9, "motion_integrator", 10, "excitatory", 11, "ipsilateral"
    )

    # iMI- : motion_integrator, ipsilateral, inhibitory
    imi_inh_ids_all_nuc, _ = fetch_filtered_ids_EI(
        df, 9, "motion_integrator", 10, "inhibitory", 11, "ipsilateral"
    )

    return {
        "cMI": cmi_ids_all_nuc,
        "MON": mon_ids_all_nuc,
        "MC": mc_ids_all_nuc,
        "iMI_plus": imi_ex_ids_all_nuc,
        "iMI_minus": imi_inh_ids_all_nuc,
        "iMI_all": imi_ids_all_nuc,
    }


# -------------------------------------------------------------------------
# Counts & probabilities
# -------------------------------------------------------------------------


def _process_category(df: pd.DataFrame, functional_only: bool) -> pd.DataFrame:
    """
    Compute counts and probabilities for a single category (cells or synapses).

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing individual connections (rows).
    functional_only : bool
        If True, exclude non-functional neurons except 'myelinated'.

    Returns
    -------
    pandas.DataFrame
        Table with columns:
            'Functional Classifier',
            'Neurotransmitter Classifier',
            'Projection Classifier',
            'Axon Exit Direction',
            'Count',
            'Probability'
    """
    if df.empty:
        return pd.DataFrame()

    connections = []

    for _, row in df.iterrows():
        row_type = row.get("type", None)

        if row_type == "axon":
            # Axon-only case: keep exit direction, no functional labels
            axon_exit_direction = row.get("comment", None)
            if pd.notna(axon_exit_direction):
                connections.append(
                    {
                        "Functional Classifier": "axon",
                        "Neurotransmitter Classifier": None,
                        "Projection Classifier": None,
                        "Axon Exit Direction": axon_exit_direction,
                    }
                )

        elif row_type == "cell":
            func_raw = row.get("functional classifier")
            func_std = standardize_functional_name(func_raw)

            if (
                functional_only
                and row.get("functional_id") == "not functionally imaged"
                and func_std != "myelinated"
            ):
                # Skip non-functional neurons when requested
                continue

            connections.append(
                {
                    "Functional Classifier": func_std,
                    "Neurotransmitter Classifier": row.get(
                        "neurotransmitter classifier"
                    ),
                    "Projection Classifier": row.get("projection classifier"),
                    "Axon Exit Direction": None,
                }
            )

    if not connections:
        return pd.DataFrame()

    connections_df = pd.DataFrame(connections).fillna("None")
    counts_df = connections_df.value_counts().reset_index(name="Count")
    counts_df["Probability"] = counts_df["Count"] / counts_df["Count"].sum()

    return counts_df


def compute_count_probabilities_from_results(
    results: dict,
    functional_only: bool = False,
) -> dict:
    """
    Convert hemisphere-resolved connectivity results into count/probability tables.

    Parameters
    ----------
    results : dict
        Output of `get_inputs_outputs_by_hemisphere(...)`.
    functional_only : bool, default False
        If True, drop non-functionally imaged neurons (except 'myelinated').

    Returns
    -------
    dict
        Nested dictionary:

            final_results[conn_type]["same_side" or "different_side"]["cells" or "synapses"]

        each entry being a DataFrame with columns described in `_process_category`.
    """
    final_results: dict = {
        conn_type: {
            side: {dtype: pd.DataFrame() for dtype in ("cells", "synapses")}
            for side in ("same_side", "different_side")
        }
        for conn_type in ("outputs", "inputs")
    }

    for conn_type in ("outputs", "inputs"):
        for side in ("same_side", "different_side"):
            for dtype in ("cells", "synapses"):
                df = results.get(conn_type, {}).get(dtype, {}).get(side, pd.DataFrame())
                final_results[conn_type][side][dtype] = _process_category(
                    df, functional_only
                )

    return final_results


# -------------------------------------------------------------------------
# 2-layer network drawing primitive
# -------------------------------------------------------------------------


def _adjust_luminance(
    rgba: Tuple[float, float, float, float],
    factor: float = 1.5,
) -> Tuple[float, float, float]:
    """Slightly darken/lighten an RGB triple for edge contrast."""
    r, g, b = rgba[:3]
    h, l, s = rgb_to_hls(r, g, b)
    l = min(1.0, max(0.0, l * factor))
    return hls_to_rgb(h, l, s)


def draw_two_layer_neural_net(
    ax,
    left: float,
    right: float,
    bottom: float,
    top: float,
    data_df: pd.DataFrame,
    node_radius: float = 0.015,
    input_circle_color: str = "ipsilateral_motion_integrator",
    input_cell_type: str = "excitatory",
    show_midline: bool = True,
    proportional_lines: bool = True,
    a: float = 5.0,
    b: float = 2.0,
    connection_type: str = "outputs",
    add_legend: bool = True,
    # Label behaviour
    label_mode: str = "count",        # 'count' | 'proportion' | 'probability' | 'none'
    label_as_percent: bool = True,
    label_decimals: int = 1,
    # Cross-side totals for normalization
    total_outputs: float | None = None,
    total_inputs: float | None = None,
) -> None:
    """
    Draw a 2-layer network:

        layer 1: one seed population node
        layer 2: connection categories from `data_df`

    `data_df` is expected to contain:

        'Functional Classifier'   (canonical names: motion_integrator,
                                   motion_onset, slow_motion_integrator, myelinated, axon, ...)
        'Neurotransmitter Classifier'
        'Projection Classifier'   ('ipsilateral', 'contralateral', or 'None')
        'Axon Exit Direction'
        'Probability'
        'Count'

    Colors come from `COLOR_CELL_TYPE_DICT` imported from matrix_helpers, with keys:
        'ipsilateral_motion_integrator', 'contralateral_motion_integrator',
        'motion_onset', 'slow_motion_integrator', 'myelinated',
        'axon_rostral', 'axon_caudal'.
    """
    if data_df.empty:
        ax.axis("off")
        return

    import matplotlib.pyplot as plt  # local import

    df = data_df.copy()

    # --- Merge "not functionally imaged" rows into a single category -------
    nfi = df[df["Functional Classifier"] == "not functionally imaged"]
    if not nfi.empty:
        merged_prob = nfi["Probability"].sum()
        merged_count = nfi["Count"].sum()
        df = df[df["Functional Classifier"] != "not functionally imaged"]
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "Functional Classifier": "not functionally imaged",
                            "Projection Classifier": "None",
                            "Neurotransmitter Classifier": "unknown",
                            "Axon Exit Direction": "None",
                            "Probability": merged_prob,
                            "Count": merged_count,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    # --- Optional horizontal shrink if no midline -------------------------
    if not show_midline:
        midpoint = (left + right) / 2.0
        if connection_type == "outputs":
            right = midpoint + (right - midpoint) * 0.8
        elif connection_type == "inputs":
            left = midpoint - (midpoint - left) * 0.8

    # --- Basic geometry ----------------------------------------------------
    layer_sizes = [1, len(df)]
    v_spacing = (top - bottom) / float(max(layer_sizes) + 2)
    h_spacing = (right - left) / float(1)

    # Input node (single seed population)
    input_center_y = bottom + (top - bottom) / 2.0
    input_center_x = left if connection_type == "outputs" else right
    input_center = (input_center_x, input_center_y)
    input_radius = node_radius * 4.0

    # Color for the seed node: use provided key directly in COLOR_CELL_TYPE_DICT
    seed_color = COLOR_CELL_TYPE_DICT.get(
        input_circle_color,
        (0.5, 0.5, 0.5, 0.7),
    )
    input_circle = Circle(
        input_center,
        input_radius,
        edgecolor=seed_color,
        facecolor=seed_color,
        lw=3,
        alpha=0.8,
    )
    ax.add_artist(input_circle)

    # --- Denominator for proportions --------------------------------------
    panel_total = float(df["Count"].sum()) if "Count" in df.columns else 0.0
    if label_mode == "proportion":
        if connection_type == "outputs" and total_outputs is not None:
            denom = float(total_outputs)
        elif connection_type == "inputs" and total_inputs is not None:
            denom = float(total_inputs)
        else:
            denom = panel_total
    else:
        denom = panel_total

    # --- Output nodes (one per row) ---------------------------------------
    node_positions: list[tuple[tuple[float, float], float]] = []
    output_top = bottom + (top - bottom) / 2.0 + v_spacing * (layer_sizes[1] - 1) / 2.0

    for idx, row in df.iterrows():
        node_y = output_top - idx * v_spacing
        node_x = right if connection_type == "outputs" else left
        node_center = (node_x, node_y)

        func_class = row["Functional Classifier"]
        proj_class = row["Projection Classifier"]
        ax_dir = row.get("Axon Exit Direction", "None")

        # Select color key based on functional + projection + axon direction
        if func_class == "motion_integrator":
            # projection-specific colors
            if proj_class == "ipsilateral":
                color_key = "ipsilateral_motion_integrator"
            elif proj_class == "contralateral":
                color_key = "contralateral_motion_integrator"
            else:
                # fall back if projection is missing
                color_key = "ipsilateral_motion_integrator"
        elif func_class == "slow_motion_integrator":
            color_key = "slow_motion_integrator"
        elif func_class == "motion_onset":
            color_key = "motion_onset"
        elif func_class == "myelinated":
            color_key = "myelinated"
        elif func_class == "axon":
            if isinstance(ax_dir, str) and "rostrally" in ax_dir:
                color_key = "axon_rostral"
            elif isinstance(ax_dir, str) and "caudally" in ax_dir:
                color_key = "axon_caudal"
            else:
                color_key = "axon_caudal"
        else:
            # not functionally imaged, etc.
            color_key = "axon_caudal"

        fill_color = COLOR_CELL_TYPE_DICT.get(color_key, (0.5, 0.5, 0.5, 0.7))
        outline_rgb = _adjust_luminance(fill_color, factor=0.5)

        nt_type = row.get("Neurotransmitter Classifier", "unknown")
        nt_type = str(nt_type) if nt_type is not None else "unknown"

        if nt_type == "excitatory":
            outline_style = "solid"
        elif nt_type == "inhibitory":
            outline_style = "dashed"
        else:
            outline_style = "dotted"

        prob = float(row["Probability"])
        radius = node_radius * (1.0 + prob * 4.0)

        circle = Circle(
            node_center,
            radius,
            edgecolor=outline_rgb,
            facecolor=fill_color,
            lw=3,
            alpha=0.8,
            linestyle=outline_style,
        )
        ax.add_artist(circle)

        # --- Label ---------------------------------------------------------
        label_text: str | None = None
        if label_mode != "none":
            if label_mode == "count":
                count_val = int(row.get("Count", 0))
                label_text = f"{count_val}"
            elif label_mode == "proportion":
                count_val = float(row.get("Count", 0.0))
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
                node_center[0] + offset,
                node_center[1],
                label_text,
                fontsize=12,
                ha="left",
                va="center",
                color="black",
                fontname="Arial",
            )

        node_positions.append((node_center, radius))

    # --- Connections from input → outputs ---------------------------------
    for idx, (out_center, out_radius) in enumerate(node_positions):
        prob = float(df.iloc[idx]["Probability"]) if idx < len(df) else 0.0
        width = a * prob + b if proportional_lines else b

        dx = out_center[0] - input_center[0]
        dy = out_center[1] - input_center[1]
        dist = np.hypot(dx, dy) if (dx or dy) else 1e-9

        src_x = input_center[0] + (input_radius / dist) * dx
        src_y = input_center[1] + (input_radius / dist) * dy
        dst_x = out_center[0] - (out_radius / dist) * dx
        dst_y = out_center[1] - (out_radius / dist) * dy

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

        # Arrow or T-bar (depending on direction / type)
        nt_type = df.iloc[idx]["Neurotransmitter Classifier"]
        if connection_type == "inputs":
            if nt_type == "excitatory":
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
            elif nt_type == "inhibitory":
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
            else:  # inhibitory population
                t_len = width * 0.01
                t_dx = -dy / dist
                t_dy = dx / dist
                ax.plot(
                    [dst_x - t_len * t_dx, dst_x + t_len * t_dx],
                    [dst_y - t_len * t_dy, dst_y + t_len * t_dy],
                    c="black",
                    lw=2,
                )

    # Midline between hemispheres (for cross-side panels)
    if show_midline:
        mid_x = (left + right) / 2.0
        top_y = top - 0.05
        bottom_y = bottom + 0.05
        ax.plot(
            [mid_x, mid_x],
            [bottom_y, top_y],
            color="lightgray",
            linestyle="--",
            linewidth=1.5,
        )

    # Legend for neurotransmitter outline styles (once per figure)
    if add_legend:
        legend_elements = [
            Line2D(
                [0],
                [0],
                color="black",
                lw=3,
                linestyle="solid",
                label="Excitatory",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                lw=3,
                linestyle="dashed",
                label="Inhibitory",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                lw=3,
                linestyle="dotted",
                label="Unknown",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=12,
            frameon=False,
            title="Neurotransmitter classifier",
            title_fontsize=12,
            prop={"family": "Arial"},
        )

    ax.axis("off")
    ax.set_xlim(left - h_spacing, right + h_spacing)
    ax.set_ylim(bottom - v_spacing * 1.5, top + v_spacing)
    ax.set_aspect("equal", adjustable="datalim")


# -------------------------------------------------------------------------
# High-level summariser used by the main script
# -------------------------------------------------------------------------


def summarize_connectome(
    root_folder: Path | str,
    seed_ids: Iterable[int | str],
    hemisphere_df: pd.DataFrame,
    outputs_total_mode: str = "both",
    functional_only: bool = False,
) -> dict:
    """
    Wrapper that:

        - builds hemisphere-resolved connectivity for a seed population,
        - converts to count/probability tables,
        - extracts synapse tables and global totals.

    Parameters
    ----------
    root_folder : Path or str
        Root folder with per-neuron synapse CSVs.
    seed_ids : iterable of int or str
        Nucleus IDs for the seed population.
    hemisphere_df : pandas.DataFrame
        LDA metadata with hemisphere + classifiers.
    outputs_total_mode : {'both', 'same_only'}
        If 'both', total_outputs = same + different side.
        If 'same_only', total_outputs = same-side only (used for iMI+ / iMI-).
    functional_only : bool, default False
        Passed through to `compute_count_probabilities_from_results`.

    Returns
    -------
    dict
        Keys:
            'probs'           : full nested dict of count/prob tables
            'same_in_syn'     : same-side input synapse table
            'diff_in_syn'     : cross-side input synapse table
            'same_out_syn'    : same-side output synapse table
            'diff_out_syn'    : cross-side output synapse table
            'total_inputs'    : total input synapse count (same+different)
            'total_outputs'   : total output synapse count (per `outputs_total_mode`)
    """
    results = get_inputs_outputs_by_hemisphere(
        root_folder=root_folder,
        seed_cell_ids=seed_ids,
        hemisphere_df=hemisphere_df,
    )

    probs = compute_count_probabilities_from_results(
        results,
        functional_only=functional_only,
    )

    same_out_syn = probs["outputs"]["same_side"]["synapses"]
    diff_out_syn = probs["outputs"]["different_side"]["synapses"]
    same_in_syn = probs["inputs"]["same_side"]["synapses"]
    diff_in_syn = probs["inputs"]["different_side"]["synapses"]

    if outputs_total_mode == "same_only":
        total_outputs = same_out_syn["Count"].sum()
    else:  # 'both'
        total_outputs = same_out_syn["Count"].sum() + diff_out_syn["Count"].sum()

    total_inputs = same_in_syn["Count"].sum() + diff_in_syn["Count"].sum()

    return {
        "probs": probs,
        "same_in_syn": same_in_syn,
        "diff_in_syn": diff_in_syn,
        "same_out_syn": same_out_syn,
        "diff_out_syn": diff_out_syn,
        "total_inputs": total_inputs,
        "total_outputs": total_outputs,
    }