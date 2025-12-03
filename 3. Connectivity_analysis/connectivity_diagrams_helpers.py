#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for two-layer network diagrams (hindbrain connectomes).

This module is focused on the two-layer network plots used in
`make_connectome_diagrams.py`. It reuses generic utilities from
`connectivity_matrices_helpers.py` and adds:

- LDA metadata loading
- Seed ID selection for functional populations (cMI, MON, SMI, iMI±)
- Conversion of hemisphere-resolved connectivity into synapse-count
  probability tables
- A two-layer network drawing primitive
- Export of detailed connectivity tables to TXT

Functional naming conventions
-----------------------------
We assume the metadata uses the canonical functional classifier labels:

    'motion_integrator'
    'motion_onset'
    'slow_motion_integrator'
    'myelinated'

All processing and plotting uses these labels directly.

Color mapping
-------------
Colors are taken from `connectivity_matrices_helpers.COLOR_CELL_TYPE_DICT`,
which uses keys:

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
from typing import Dict, Iterable, Tuple, List

import numpy as np
import pandas as pd
from colorsys import rgb_to_hls, hls_to_rgb
from matplotlib.patches import (
    Rectangle,
    Wedge,
    Patch,
    Circle,
    Polygon
)
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D

# -------------------------------------------------------------------------
# Import shared tools from the matrix helpers
# -------------------------------------------------------------------------

from connectivity_matrices_helpers import (  # type: ignore[import]
    COLOR_CELL_TYPE_DICT,
    fetch_filtered_ids,
    get_inputs_outputs_by_hemisphere_general,
)


# -------------------------------------------------------------------------
# Simple I/O helpers
# -------------------------------------------------------------------------


def load_csv_metadata(path: Path) -> pd.DataFrame:
    """
    Load the .csv metadata table. 

    The CSV is expected to contain (among others):
        - column 9 : 'functional classifier'
        - column 10: 'neurotransmitter classifier'
        - column 11: 'projection classifier'
        - column 5 : nucleus_id
        - column 1 : functional_id (or similar)

    Parameters
    ----------
    path : Path
        Path to the .csv metadata CSV.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(path)

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
    - SMI       : slow_motion_integrator
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

    # SMI: all slow_motion_integrator
    smi_ids_all_nuc, _ = fetch_filtered_ids(df, 9, "slow_motion_integrator")

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
        "SMI": smi_ids_all_nuc,
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

    connections: List[Dict[str, object]] = []

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
            func_cls = row.get("functional classifier")

            # Enforce canonical names only: we assume the CSV is already correct
            if (
                functional_only
                and row.get("functional_id") == "not functionally imaged"
                and func_cls != "myelinated"
            ):
                # Skip non-functional neurons when requested
                continue

            connections.append(
                {
                    "Functional Classifier": func_cls,
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
        Output of `get_inputs_outputs_by_hemisphere_general(...)`.
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


class HalfBlackWhiteHandler(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # left = black, right = white
        r_left = Rectangle(
            (xdescent, ydescent),
            width / 2.0,
            height,
            facecolor=(0.0, 0.0, 0.0, 1.0),
            edgecolor="black",
            transform=trans,
        )
        r_right = Rectangle(
            (xdescent + width / 2.0, ydescent),
            width / 2.0,
            height,
            facecolor=(1.0, 1.0, 1.0, 1.0),
            edgecolor="black",
            transform=trans,
        )
        return [r_left, r_right]


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

        'Functional Classifier'   (motion_integrator, motion_onset,
                                   slow_motion_integrator, myelinated, axon, ...)
        'Neurotransmitter Classifier'
        'Projection Classifier'   ('ipsilateral', 'contralateral', or 'None')
        'Axon Exit Direction'
        'Probability'
        'Count'

    Colors come from `COLOR_CELL_TYPE_DICT` with keys:
        'ipsilateral_motion_integrator', 'contralateral_motion_integrator',
        'motion_onset', 'slow_motion_integrator', 'myelinated',
        'not_functionally_imaged', 'axon_rostral', 'axon_caudal'.
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

    # --- Lump all *other* functional labels into "other_functional" ------
    valid_labels = {
        "motion_integrator",
        "motion_onset",
        "slow_motion_integrator",
        "myelinated",
        "not functionally imaged",
        "axon"
    }

    other_rows = df[~df["Functional Classifier"].isin(valid_labels)]
    if not other_rows.empty:
        other_count = other_rows["Count"].sum()
        other_prob = other_rows["Probability"].sum()

        # Keep only the valid labels, then append a single "other_functional" row
        df = df[df["Functional Classifier"].isin(valid_labels)]
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "Functional Classifier": "other_functional_types",
                            "Neurotransmitter Classifier": "unknown",
                            "Projection Classifier": "None",
                            "Axon Exit Direction": "None",
                            "Probability": other_prob,
                            "Count": other_count,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    # --- Merge ipsi/contra motion_onset into a single category ------------
    mo_rows = df[df["Functional Classifier"] == "motion_onset"]
    if not mo_rows.empty and len(mo_rows) > 1:
        mo_count = mo_rows["Count"].sum()
        mo_prob = mo_rows["Probability"].sum()
        df = df[df["Functional Classifier"] != "motion_onset"]
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "Functional Classifier": "motion_onset",
                            "Neurotransmitter Classifier": "inhibitory",
                            "Projection Classifier": "None",
                            "Axon Exit Direction": "None",
                            "Probability": mo_prob,
                            "Count": mo_count,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    # --- Merge "myelinated" neurons into a single category ---------------
    my_rows = df[df["Functional Classifier"] == "myelinated"]
    if not my_rows.empty and len(my_rows) > 1:
        my_count = my_rows["Count"].sum()
        my_prob = my_rows["Probability"].sum()
        df = df[df["Functional Classifier"] != "myelinated"]
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "Functional Classifier": "myelinated",
                            "Neurotransmitter Classifier": "unknown",
                            "Projection Classifier": "None",
                            "Axon Exit Direction": "None",
                            "Probability": my_prob,
                            "Count": my_count,
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

    seed_color = COLOR_CELL_TYPE_DICT.get(
        input_circle_color,
        (0.5, 0.5, 0.5, 0.7),
    )
    ax.add_patch(
        Circle(
            input_center,
            input_radius,
            edgecolor=seed_color,
            facecolor=seed_color,
            lw=3,
            alpha=0.8,
        )
    )

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
    node_positions: List[Tuple[Tuple[float, float], float]] = []
    output_top = bottom + (top - bottom) / 2.0 + v_spacing * (layer_sizes[1] - 1) / 2.0

    for idx, row in df.iterrows():
        node_y = output_top - idx * v_spacing
        node_x = right if connection_type == "outputs" else left
        node_center = (node_x, node_y)

        func_class = row["Functional Classifier"]
        proj_class = row["Projection Classifier"]
        ax_dir = row.get("Axon Exit Direction", "None")

        half_rostral_caudal = False

        if func_class == "motion_integrator":
            if proj_class == "ipsilateral":
                color_key = "ipsilateral_motion_integrator"
            elif proj_class == "contralateral":
                color_key = "contralateral_motion_integrator"
            else:
                color_key = "ipsilateral_motion_integrator"
        elif func_class == "slow_motion_integrator":
            color_key = "slow_motion_integrator"
        elif func_class == "motion_onset":
            color_key = "motion_onset"
        elif func_class == "myelinated":
            color_key = "myelinated"
        elif func_class == "other_functional_types":
            color_key = "other_functional_types"
        elif func_class == "axon":
            if isinstance(ax_dir, str) and ("rostrally" in ax_dir and "caudally" in ax_dir):
                color_key = None
                half_rostral_caudal = True
            elif isinstance(ax_dir, str) and "rostrally" in ax_dir:
                color_key = "axon_rostral"
            elif isinstance(ax_dir, str) and "caudally" in ax_dir:
                color_key = "axon_caudal"
            else:
                color_key = "axon_caudal"
        else:
            color_key = "not_functionally_imaged"

        if color_key is not None:
            fill_color = COLOR_CELL_TYPE_DICT.get(color_key, (0.5, 0.5, 0.5, 0.7))
        else:
            # fallback, not really used for half-circle case
            fill_color = (0.5, 0.5, 0.5, 0.7)

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

        if half_rostral_caudal:
            # --- Half white / half black circle (pure B/W), independent of COLOR_CELL_TYPE_DICT ---

            # 1) Full white circle as background
            ax.add_patch(
                Circle(
                    node_center,
                    radius,
                    facecolor="white",
                    edgecolor="none",
                    zorder=1,
                )
            )

            # 2) Left half (black) – 90° to 270° in data coordinates
            ax.add_patch(
                Wedge(
                    center=node_center,
                    r=radius,
                    theta1=90,
                    theta2=270,
                    facecolor="black",
                    edgecolor="none",
                    zorder=2,
                )
            )

            # 3) Outline in solid black, with the inhibitory/excitatory linestyle
            ax.add_patch(
                Circle(
                    node_center,
                    radius,
                    facecolor="none",
                    edgecolor="black",
                    lw=3,
                    linestyle=outline_style,
                    zorder=3,
                )
            )
        else:
            ax.add_patch(
                Circle(
                    node_center,
                    radius,
                    edgecolor=outline_rgb,
                    facecolor=fill_color,
                    lw=3,
                    alpha=0.8,
                    linestyle=outline_style,
                )
            )

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
            if connection_type == "inputs":
                offset = -radius * 3.5  # move labels farther left
            else:
                offset = radius * 1.5   # outputs stay the same
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

        ax.add_line(
            Line2D(
                [src_x, dst_x],
                [src_y, dst_y],
                c="black",
                lw=width,
                alpha=0.8,
            )
        )

        # Arrow or T-bar
        nt_type = df.iloc[idx]["Neurotransmitter Classifier"]
        if connection_type == "inputs":
            # arrow / T at the seed node
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
                # Inverted arrow: base on the circle outline, tip pointing away
                # dst_x, dst_y is the point on the circle (postsynaptic / seed)
                # src_x, src_y is the point toward the presynaptic node.

                # Unit vector from circle outward (seed -> presynaptic)
                ux = (src_x - dst_x) / dist
                uy = (src_y - dst_y) / dist

                # Perpendicular for the base width
                nx = -uy
                ny = ux

                # Arrow geometry (scaled by line width)
                arrow_len = width * 0.01   # length of arrow in data coords
                base_half = width * 0.01   # half-width of the base

                # Base centered exactly on the circle boundary
                base_left = (dst_x - base_half * nx, dst_y - base_half * ny)
                base_right = (dst_x + base_half * nx, dst_y + base_half * ny)
                tip = (dst_x + arrow_len * ux, dst_y + arrow_len * uy)

                arrow_poly = Polygon(
                    [base_left, base_right, tip],
                    closed=True,
                    facecolor="black",
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=5,
                )
                ax.add_patch(arrow_poly)

        else:
            # outputs: arrow/T at the target node
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
            else:  # inhibitory *outputs* (seed population is inhibitory)
                # Inverted arrow with base ON the circle outline, tip pointing outward
                px, py = dst_x, dst_y  # point on the target circle boundary

                # Unit vector from target-circle outward (opposite direction of input case)
                ux = (src_x - dst_x) / dist
                uy = (src_y - dst_y) / dist

                # Perpendicular unit vector for arrow base width
                nx = -uy
                ny = ux

                # Arrow geometry
                arrow_len = width * 0.01   # length of arrow
                base_half = width * 0.01   # half-width of triangle base

                # Base coordinates (touching circle)
                base_left = (px - base_half * nx, py - base_half * ny)
                base_right = (px + base_half * nx, py + base_half * ny)

                # Arrow tip (pointing outward)
                tip = (px + arrow_len * ux, py + arrow_len * uy)

                arrow_poly = Polygon(
                    [base_left, base_right, tip],
                    closed=True,
                    facecolor="black",
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=5,
                )
                ax.add_patch(arrow_poly)

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

    # ------------------------------------------------------------------
    # Legend (functional colors + neurotransmitter styles)
    # ------------------------------------------------------------------
    if add_legend:
        # Neurotransmitter legend
        nt_handles = [
            Line2D([0], [0], color="black", lw=3, linestyle="solid", label="Excitatory"),
            Line2D([0], [0], color="black", lw=3, linestyle="dashed", label="Inhibitory"),
            Line2D([0], [0], color="black", lw=3, linestyle="dotted", label="Mixed"),
        ]

        func_handles: List[Patch] = []
        for key in sorted(COLOR_CELL_TYPE_DICT.keys()):
            rgba = COLOR_CELL_TYPE_DICT[key]
            label = key.replace("_", " ")
            if key == input_circle_color:
                label += " (seed population)"
            func_handles.append(Patch(facecolor=rgba, edgecolor="black", label=label))

        # Special proxy for axons exiting both rostrally & caudally
        dual_handle = Patch(facecolor="none", edgecolor="none", label="axon exits rostrally & caudally")

        legend_handles = nt_handles + func_handles + [dual_handle]

        handler_map = {dual_handle: HalfBlackWhiteHandler()}

        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.20, 1.00),
            fontsize=9,
            frameon=False,
            title="Legend",
            title_fontsize=10,
            prop={"family": "Arial"},
            handler_map=handler_map,
        )

    ax.axis("off")
    ax.set_xlim(left - h_spacing, right + h_spacing)
    ax.set_ylim(bottom - v_spacing * 1.5, top + v_spacing)
    ax.set_aspect("equal", adjustable="datalim")

# -------------------------------------------------------------------------
# High-level summariser + TXT export used by the main script
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
        If 'both', total_outputs = same + different side (synapses).
        If 'same_only', total_outputs = ipsilateral only (used for iMI+ / iMI-).
    functional_only : bool, default False
        Passed through to `compute_count_probabilities_from_results`.

    Returns
    -------
    dict
        Keys:
            'probs'                 : full nested dict of count/prob tables
            'same_in_syn'           : ipsilateralside input synapse table
            'diff_in_syn'           : contralateral input synapse table
            'same_out_syn'          : ipsilateral output synapse table
            'diff_out_syn'          : contralateral output synapse table
            'total_inputs'          : total input synapse count (same+different)
            'total_outputs'         : total output synapse count (per outputs_total_mode)
            'total_inputs_cells'    : total input cell count
            'total_outputs_cells'   : total output cell count
            'panel_totals'          : per-side/per-unit totals for inputs/outputs
    """
    results = get_inputs_outputs_by_hemisphere_general(
        root_folder=root_folder,
        seed_cell_ids=seed_ids,
        hemisphere_df=hemisphere_df,
    )

    probs = compute_count_probabilities_from_results(
        results,
        functional_only=functional_only,
    )

    # Synapse-level tables
    same_out_syn = probs["outputs"]["same_side"]["synapses"]
    diff_out_syn = probs["outputs"]["different_side"]["synapses"]
    same_in_syn = probs["inputs"]["same_side"]["synapses"]
    diff_in_syn = probs["inputs"]["different_side"]["synapses"]

    # Cell-level tables
    same_out_cells = probs["outputs"]["same_side"]["cells"]
    diff_out_cells = probs["outputs"]["different_side"]["cells"]
    same_in_cells = probs["inputs"]["same_side"]["cells"]
    diff_in_cells = probs["inputs"]["different_side"]["cells"]

    # --- Synapse totals (match the logic used for line-width normalization) ---
    outputs_syn_same = float(same_out_syn["Count"].sum()) if not same_out_syn.empty else 0.0
    outputs_syn_diff = float(diff_out_syn["Count"].sum()) if not diff_out_syn.empty else 0.0
    inputs_syn_same = float(same_in_syn["Count"].sum()) if not same_in_syn.empty else 0.0
    inputs_syn_diff = float(diff_in_syn["Count"].sum()) if not diff_in_syn.empty else 0.0

    if outputs_total_mode == "same_only":
        # iMI+ / iMI− convention: normalize outputs by same-side counts only
        total_outputs = outputs_syn_same
    else:  # 'both'
        total_outputs = outputs_syn_same + outputs_syn_diff

    total_inputs = inputs_syn_same + inputs_syn_diff

    # --- Cell totals (for TXT summary & global proportions on cells) ----------
    outputs_cells_same = float(same_out_cells["Count"].sum()) if not same_out_cells.empty else 0.0
    outputs_cells_diff = float(diff_out_cells["Count"].sum()) if not diff_out_cells.empty else 0.0
    inputs_cells_same = float(same_in_cells["Count"].sum()) if not same_in_cells.empty else 0.0
    inputs_cells_diff = float(diff_in_cells["Count"].sum()) if not diff_in_cells.empty else 0.0

    if outputs_total_mode == "same_only":
        total_outputs_cells = outputs_cells_same
    else:
        total_outputs_cells = outputs_cells_same + outputs_cells_diff

    total_inputs_cells = inputs_cells_same + inputs_cells_diff

    panel_totals = {
        "inputs": {
            "synapses": {
                "same_side": inputs_syn_same,
                "different_side": inputs_syn_diff,
                "global": total_inputs,
            },
            "cells": {
                "same_side": inputs_cells_same,
                "different_side": inputs_cells_diff,
                "global": total_inputs_cells,
            },
        },
        "outputs": {
            "synapses": {
                "same_side": outputs_syn_same,
                "different_side": outputs_syn_diff,
                "global": total_outputs,
            },
            "cells": {
                "same_side": outputs_cells_same,
                "different_side": outputs_cells_diff,
                "global": total_outputs_cells,
            },
        },
    }

    return {
        "probs": probs,
        "same_in_syn": same_in_syn,
        "diff_in_syn": diff_in_syn,
        "same_out_syn": same_out_syn,
        "diff_out_syn": diff_out_syn,
        "total_inputs": total_inputs,
        "total_outputs": total_outputs,
        "total_inputs_cells": total_inputs_cells,
        "total_outputs_cells": total_outputs_cells,
        "panel_totals": panel_totals,
    }


def export_connectivity_tables_txt(
    population_name: str,
    summary: dict,
    output_folder: Path | str,
    suffix: str = "",
) -> Path:
    """
    Export a detailed TXT table of connectivity for a seed population.

    The exported table contains one row per category (cells/synapses) with:

        Connection Type (inputs/outputs)
        Side            (same_side/different_side)
        Unit Type       (cells/synapses)
        Functional Classifier
        Neurotransmitter Classifier
        Projection Classifier
        Axon Exit Direction
        Count
        Probability (panel-normalized)
        Panel_Total      (sum of Count within same Connection Type / Side / Unit)
        Global_Total     (total inputs/outputs for that Unit Type)
        Global_Proportion (Count / Global_Total)

    Parameters
    ----------
    population_name : str
        Short name for the seed population (e.g. 'cMI', 'MON').
    summary : dict
        Output of `summarize_connectome(...)`.
    output_folder : Path or str
        Folder where the TXT file will be written.
    suffix : str, optional
        Optional string appended to the output filename.

    Returns
    -------
    pathlib.Path
        Path to the written TXT file.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    probs = summary["probs"]
    records: List[pd.DataFrame] = []

    # Collect all panels into a single long table
    for conn_type in ("inputs", "outputs"):
        for side in ("same_side", "different_side"):
            for unit in ("cells", "synapses"):
                df = probs[conn_type][side][unit]
                if df.empty:
                    continue
                df_local = df.copy()
                df_local.insert(0, "Unit Type", unit)
                df_local.insert(0, "Side", side)
                df_local.insert(0, "Connection Type", conn_type)
                records.append(df_local)

    if not records:
        # still create an empty file with header
        columns = [
            "Connection Type",
            "Side",
            "Unit Type",
            "Functional Classifier",
            "Neurotransmitter Classifier",
            "Projection Classifier",
            "Axon Exit Direction",
            "Count",
            "Probability",
            "Panel_Total",
            "Global_Total",
            "Global_Proportion",
        ]
        full_table = pd.DataFrame(columns=columns)
    else:
        full_table = pd.concat(records, ignore_index=True)

        # --- Panel totals (same definition as used to compute Probability) ----
        full_table["Panel_Total"] = (
            full_table.groupby(["Connection Type", "Side", "Unit Type"])["Count"]
            .transform("sum")
            .astype(float)
        )

        # --- Global totals (match plotting normalization) --------------------
        total_in_syn = float(summary.get("total_inputs", 0.0))
        total_out_syn = float(summary.get("total_outputs", 0.0))
        total_in_cells = float(summary.get("total_inputs_cells", 0.0))
        total_out_cells = float(summary.get("total_outputs_cells", 0.0))

        def _global_total(row: pd.Series) -> float:
            if row["Unit Type"] == "synapses":
                return total_in_syn if row["Connection Type"] == "inputs" else total_out_syn
            else:  # cells
                return total_in_cells if row["Connection Type"] == "inputs" else total_out_cells

        full_table["Global_Total"] = full_table.apply(_global_total, axis=1).astype(float)

        # Avoid division by zero
        full_table["Global_Proportion"] = np.where(
            full_table["Global_Total"] > 0,
            full_table["Count"].astype(float) / full_table["Global_Total"],
            np.nan,
        )

        # --- Sort for clarity: inputs first, then outputs, synapses then cells ---
        full_table = full_table.sort_values(
            by=[
                "Connection Type",
                "Unit Type",
                "Side",
                "Functional Classifier",
                "Neurotransmitter Classifier",
            ],
            ignore_index=True,
        )

    clean_suffix = suffix.strip().lstrip("_")
    suffix_part = f"_{clean_suffix}" if clean_suffix else ""
    out_path = output_folder / f"{population_name}_connectivity_details{suffix_part}.txt"

    full_table.to_csv(out_path, sep="\t", index=False)
    return out_path