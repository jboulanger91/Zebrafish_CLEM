#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for two-layer network diagrams (hindbrain connectomes).

This module is focused on the two-layer network plots used in
`make_connectivity_diagrams.py`. It reuses generic utilities from
`connectivity_matrices_helpers.py` and adds:

- Metadata loading
- Seed ID selection for functional populations (cMI, MON, SMI, iMI±, iMI_all)
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import (
    Rectangle,
    Wedge,
    Circle,
    Patch,
    Polygon
)
from matplotlib.legend_handler import HandlerPatch, HandlerBase
from matplotlib.lines import Line2D
from typing import Iterable


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
    a: float = 110,
    b: float = 0.1,
    connection_type: str = "outputs",
    # Label behaviour
    label_mode: str = "count",        # 'count' | 'proportion' | 'probability' | 'none'
    label_as_percent: bool = True,
    label_decimals: int = 1,
    # Cross-side totals for normalization
    total_outputs: float | None = None,
    total_inputs: float | None = None,
) -> None:
    """
    Draw a schematic 2-layer connectivity diagram for a single seed population.

    Overview
    --------
    The diagram shows:

    - **Layer 1**: one large seed node (the seed population).
    - **Layer 2**: one node per connectivity *category* (rows of `data_df` after
      pre-processing).
    - **Edges**: straight lines between the seed node and each category node,
      with line width and terminal symbol encoding connection strength and sign.

    This function **does not** draw any legends; legends are handled separately
    (e.g. by `draw_full_connectivity_legend`).

    Expected columns in `data_df`
    -----------------------------
    Each row corresponds to a single connectivity category and must contain:

    - `'Functional Classifier'`:
        e.g. ``motion_integrator``, ``motion_onset``, ``slow_motion_integrator``,
        ``myelinated``, ``axon``, ``not functionally imaged``, etc.
    - `'Neurotransmitter Classifier'`:
        ``excitatory``, ``inhibitory``, or any other value (treated as mixed/unknown).
    - `'Projection Classifier'`:
        ``'ipsilateral'``, ``'contralateral'``, or ``'None'``.
    - `'Axon Exit Direction'`:
        free text; substrings ``'rostrally'`` / ``'caudally'`` are used to infer
        axon exit direction.
    - `'Probability'`:
        per-panel probability of that category (used as a fallback if needed).
    - `'Count'`:
        number of synapses in that category.

    Pre-processing of categories
    ----------------------------
    Before plotting, `data_df` is simplified as follows:

    1. **Not functionally imaged**  
       All rows with ``Functional Classifier == 'not functionally imaged'`` are
       merged into a single row with summed ``Count`` / ``Probability``, and
       dummy classifiers:
           - Projection = ``'None'``
           - Neurotransmitter = ``'mixed'``.

    2. **Other functional labels**  
       Any functional label *not* in:

           {``motion_integrator``, ``motion_onset``, ``slow_motion_integrator``,
            ``myelinated``, ``not functionally imaged``, ``axon``}

       is merged into a single category ``'other_functional_types'`` with
       neurotransmitter = ``'mixed'`` and projection = ``'None'``.

    3. **Motion onset**  
       All ``motion_onset`` rows (ipsi + contra) are merged into a single
       ``motion_onset`` category with:
           - Neurotransmitter = ``'inhibitory'``
           - Projection = ``'None'``.

    4. **Myelinated**  
       Multiple ``myelinated`` rows are merged into a single category with
       summed ``Count`` / ``Probability``.

    Node geometry and appearance
    ----------------------------
    - **Placement**:
        - Category nodes are placed in a vertical stack between ``bottom`` and
          ``top``.
        - The seed node is centered vertically and placed at:
            - ``x = left``  if ``connection_type='outputs'``
            - ``x = right`` if ``connection_type='inputs'``.

    - **Seed node**:
        - Radius: ``input_radius = 4 * node_radius``.
        - Color: fetched from ``COLOR_CELL_TYPE_DICT[input_circle_color]``.

    - **Category node fill color** (via `COLOR_CELL_TYPE_DICT`):

        - ``motion_integrator``:
            - ipsilateral   → ``'ipsilateral_motion_integrator'``
            - contralateral → ``'contralateral_motion_integrator'``
        - ``slow_motion_integrator`` → ``'slow_motion_integrator'``
        - ``motion_onset``           → ``'motion_onset'``
        - ``myelinated``             → ``'myelinated'``
        - ``other_functional_types`` → ``'other_functional_types'``
        - ``axon``:
            - exits rostrally only  → ``'axon_rostral'``
            - exits caudally only   → ``'axon_caudal'``
            - exits both rostrally and caudally → special half-white/half-black
              circle (independent of `COLOR_CELL_TYPE_DICT`).
        - anything else             → ``'not_functionally_imaged'``.

    - **Outline style (cell outline encoding)**:

        - solid   → ``Neurotransmitter Classifier == 'excitatory'``
        - dashed  → ``'inhibitory'``
        - dotted  → all other values (mixed / unknown)

      The outline color is a darkened version of the fill color
      (via ``_adjust_luminance``). For the special half-white/half-black axon
      case, the outline is a black circle.

    - **Node size**:

        A *normalized proportion* is computed for each row:

            prop_norm = Count / denom   (if denom > 0)
                        or Probability  (if denom == 0)

        where ``denom`` depends on context (see below). The category radius is:

            radius = node_radius * (1 + 4 * prop_norm)

        so higher-fraction categories are drawn larger.

        The seed node radius is fixed at ``4 * node_radius`` and is not
        scaled by `prop_norm`.

    Normalization and denominators
    ------------------------------
    A single denominator `denom` is used per panel for *both* node radii
    and line widths (and for labels when `label_mode='proportion'`):

    - If `label_mode == 'proportion'`:
        - outputs panel and `total_outputs` is not None:
            ``denom = total_outputs`` (cross-panel normalization)
        - inputs panel and `total_inputs` is not None:
            ``denom = total_inputs``
        - otherwise:
            ``denom = sum(Count)`` within this panel.

    - For other label modes (`'count'`, `'probability'`, `'none'`), the same
      logic is used so that radii and line widths are still normalized
      consistently across panels.

    Edge geometry, line width, and connector type
    ---------------------------------------------
    For each category node, a straight line is drawn between the seed node and
    the category node.

    - **Direction**:
        - ``connection_type='outputs'``:
            line runs from seed → target node.
        - ``connection_type='inputs'``:
            line runs from target → seed (arguments are swapped so the line
            still points toward the postsynaptic neuron).

    - **Line width (connection strength)**:

        The same normalized proportion `prop_norm` used for node radius is
        used for line width:

            linewidth = a * prop_norm + b     if proportional_lines
                         b                    otherwise

        Default values are `a = 110`, `b = 0.1`. This mapping is mirrored in
        the external legend (e.g. via `draw_full_connectivity_legend`).

    - **Connector glyphs at the line tip**:

        *Input panels* (``connection_type='inputs'``) use the partner’s
        neurotransmitter (`Neurotransmitter Classifier`) to choose the glyph:

        - ``'excitatory'``:
            small arrowhead pointing toward the postsynaptic seed node.
        - ``'inhibitory'``:
            inverted arrow (triangle) whose base lies on the circle outline and
            whose tip points away from the circle (outwards).
        - anything else (mixed/unknown):
            small filled black circle at the boundary.

        *Output panels* (``connection_type='outputs'``) instead use the
        **seed population type** (`input_cell_type`) to choose the glyph:

        - ``input_cell_type == 'excitatory'``:
            arrowhead at the target node (seed is excitatory).
        - ``input_cell_type == 'inhibitory'``:
            inverted arrow at the target node (seed is inhibitory).
        - ``input_cell_type == 'mixed'``:
            small filled black circle at the boundary.

    Node labels
    -----------
    Controlled by `label_mode`:

    - ``'none'``:
        no labels are drawn.
    - ``'count'``:
        label is the integer `Count`.
    - ``'proportion'``:
        label is `prop_norm`, optionally formatted as a percentage.
    - ``'probability'``:
        label is taken from the `Probability` column.

    If `label_as_percent=True`, labels are formatted as percentages with
    `label_decimals` decimal places. Labels are offset to the left or right
    of the node depending on `connection_type`.

    Midline
    -------
    If `show_midline=True`, a dashed vertical line is drawn halfway between
    `left` and `right` to mark the anatomical midline (used for cross-side
    panels).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the diagram.
    left, right, bottom, top : float
        Logical bounds for positioning nodes in this axes.
    data_df : pandas.DataFrame
        Connectivity summary with the columns described above.
    node_radius : float
        Base radius for category nodes (seed node radius = 4× this value).
    input_circle_color : str
        Key in `COLOR_CELL_TYPE_DICT` specifying the seed node color.
    input_cell_type : {'excitatory', 'inhibitory', 'mixed'}
        High-level type of the seed population, used to choose connector
        glyphs in output panels.
    show_midline : bool
        Whether to draw a dashed midline.
    proportional_lines : bool
        Whether line width scales with normalized synapse fraction.
    a, b : float
        Parameters for line-width scaling: ``lw = a * prop_norm + b``.
    connection_type : {'inputs', 'outputs'}
        Whether this panel shows inputs to the seed or outputs from it.
    label_mode : {'count', 'proportion', 'probability', 'none'}
        What each category node label displays.
    label_as_percent : bool
        If True and label_mode in {'proportion', 'probability'}, format labels
        as percentages.
    label_decimals : int
        Number of decimal places for proportion/probability labels.
    total_outputs, total_inputs : float or None
        Optional cross-panel denominators for output and input panels,
        respectively. If provided, they override within-panel totals when
        computing `prop_norm`.
    """
    if data_df.empty:
        ax.axis("off")
        return

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
                            "Neurotransmitter Classifier": "mixed",
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
        "axon",
    }

    other_rows = df[~df["Functional Classifier"].isin(valid_labels)]
    if not other_rows.empty:
        other_count = other_rows["Count"].sum()
        other_prob = other_rows["Probability"].sum()

        df = df[df["Functional Classifier"].isin(valid_labels)]
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "Functional Classifier": "other_functional_types",
                            "Neurotransmitter Classifier": "mixed",
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
            lw=1,
            alpha=0.8,
        )
    )

    # --- Denominator for proportions --------------------------------------
    panel_total = float(df["Count"].sum()) if "Count" in df.columns else 0.0
    if label_mode == "proportion":
        if connection_type == "outputs" and total_outputs is not None:
            # cross-panel normalization for outputs
            denom = float(total_outputs)
        elif connection_type == "inputs" and total_inputs is not None:
            # cross-panel normalization for inputs
            denom = float(total_inputs)
        else:
            # fall back to within-panel normalization
            denom = panel_total
    else:
        # even if labels are not "proportion", we still use this denom
        # to normalize radii and line widths so the visual encoding
        # is comparable across panels
        if connection_type == "outputs" and total_outputs is not None:
            denom = float(total_outputs)
        elif connection_type == "inputs" and total_inputs is not None:
            denom = float(total_inputs)
        else:
            denom = panel_total

    # --- Output nodes (one per row) ---------------------------------------
    node_positions: List[Tuple[Tuple[float, float], float]] = []
    row_props: List[float] = []   # normalized proportions for each row
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

        # ----- unified normalized proportion for this row ------------------
        count_val = float(row.get("Count", 0.0))
        if denom > 0:
            prop_norm = count_val / denom
        else:
            # fall back to the Probability column if denom is zero
            prop_norm = float(row.get("Probability", 0.0))

        # use the same prop_norm for radius, line width, and (optionally) label
        radius = node_radius * (1.0 + prop_norm * 4.0)

        if half_rostral_caudal:
            # Half white / half black circle (pure B/W), independent of COLOR_CELL_TYPE_DICT
            ax.add_patch(
                Circle(
                    node_center,
                    radius,
                    facecolor="white",
                    edgecolor="none",
                    zorder=1,
                )
            )
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
            ax.add_patch(
                Circle(
                    node_center,
                    radius,
                    facecolor="none",
                    edgecolor="black",
                    lw=1,
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
                    lw=1,
                    alpha=0.8,
                    linestyle=outline_style,
                )
            )

        # --- Label ---------------------------------------------------------
        label_text: str | None = None
        if label_mode != "none":
            if label_mode == "count":
                label_text = f"{int(count_val)}"
            elif label_mode == "proportion":
                prop = prop_norm
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
                offset = -radius * 4.5  # move labels farther left
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
        row_props.append(prop_norm)

    # --- Connections from input → outputs ---------------------------------
    for idx, (out_center, out_radius) in enumerate(node_positions):
        # use the same normalized proportion for line width
        prop_for_width = row_props[idx] if idx < len(row_props) else 0.0
        width = a * prop_for_width + b if proportional_lines else b

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

        # Circle, arrow or inverted arrow at the target node
        nt_type = df.iloc[idx]["Neurotransmitter Classifier"]
        if connection_type == "inputs":

            # oinputs
            if nt_type == "mixed":
                # Mixed: small black circle at boundary
                ax.add_patch(
                    Circle(
                        (dst_x, dst_y),
                        radius=width * 0.005,
                        facecolor="black",
                        edgecolor="black",
                        zorder=5,
                    )
                )

            elif nt_type == "excitatory":
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
                ux = (src_x - dst_x) / dist
                uy = (src_y - dst_y) / dist
                nx = -uy
                ny = ux
                arrow_len = width * 0.01
                base_half = width * 0.01

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
            # outputs
            if input_cell_type == "mixed":
                # Mixed: small black circle at boundary
                ax.add_patch(
                    Circle(
                        (dst_x, dst_y),
                        radius=width * 0.005,
                        facecolor="black",
                        edgecolor="black",
                        zorder=5,
                    )
                )
            elif input_cell_type == "excitatory":
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
            else:  # inhibitory *outputs*
                ux = (src_x - dst_x) / dist
                uy = (src_y - dst_y) / dist
                nx = -uy
                ny = ux
                arrow_len = width * 0.01
                base_half = width * 0.01

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

    ax.axis("off")
    ax.set_xlim(left - h_spacing, right + h_spacing)
    ax.set_ylim(bottom - v_spacing * 1.5, top + v_spacing)
    ax.set_aspect("equal", adjustable="datalim")


def draw_full_connectivity_legend(
    legend_ax,
    *,
    input_circle_color: str,
    a_for_width: float = 12.0,
    b_for_width: float = 1.0,
) -> None:
    """
    Draw the full connectivity legend into a dedicated Axes.

    The legend is split into three stacked boxes:

    1) Cell outlines
       - Solid stroke   → excitatory cell
       - Dashed stroke  → inhibitory cell
       - Dotted stroke  → mixed / unknown cell

    2) Functional types (node fill colors)
       - One colored square for each entry in COLOR_CELL_TYPE_DICT
       - The seed population is annotated as '(seed population)'
       - A special half–white / half–black disk for 'axon exits rostrally & caudally'
         is rendered via HalfBlackWhiteHandler on a dummy Patch.

    3) Fraction of synapses and connector types
       - Three horizontal example strokes for 1%, 10%, 50% of synapses:
           lw = a_for_width * p + b_for_width
         (same mapping as used in the network diagrams)
       - Four connector types drawn at the *tip* of the stroke:
           • Excitation: arrowhead at the right end (→)
           • Inhibition: inverted arrow at the right end (base at tip, point inward)
           • Mixed: small filled circle at the right end
           • Unknown: plain black line (no marker)
    """
    legend_ax.axis("off")

    # -------------------------------------------------------------
    # 1) Cell outlines
    # -------------------------------------------------------------
    outline_handles = [
        Line2D([0, 1], [0, 0], color="black", lw=3, linestyle="solid",
               label="Excitatory outline"),
        Line2D([0, 1], [0, 0], color="black", lw=3, linestyle="dashed",
               label="Inhibitory outline"),
        Line2D([0, 1], [0, 0], color="black", lw=3, linestyle="dotted",
               label="Mixed / unknown outline"),
    ]

    ax_outline = legend_ax.inset_axes([0.0, 0.62, 1.0, 0.35])
    ax_outline.axis("off")
    ax_outline.legend(
        handles=outline_handles,
        loc="upper left",
        fontsize=9,
        frameon=True,
        title="Cell outlines",
        title_fontsize=10,
        prop={"family": "Arial"},
        borderpad=0.6,
    )

    # -------------------------------------------------------------
    # 2) Functional types (node fill colors)
    # -------------------------------------------------------------
    from connectivity_diagrams_helpers import HalfBlackWhiteHandler  # avoid circular import at top if needed

    func_handles: list[Patch] = []
    for key in sorted(COLOR_CELL_TYPE_DICT.keys()):
        rgba = COLOR_CELL_TYPE_DICT[key]
        label = key.replace("_", " ")
        if key == input_circle_color:
            label += " (seed population)"
        func_handles.append(Patch(facecolor=rgba, edgecolor="black", label=label))

    dual_handle = Patch(
        facecolor="none",
        edgecolor="black",
        label="axon exits rostrally & caudally",
    )
    handler_map_func = {dual_handle: HalfBlackWhiteHandler()}

    ax_func = legend_ax.inset_axes([0.0, 0.30, 1.0, 0.50])
    ax_func.axis("off")
    ax_func.legend(
        handles=func_handles + [dual_handle],
        handler_map=handler_map_func,
        loc="upper left",
        fontsize=9,
        frameon=True,
        title="Functional types",
        title_fontsize=10,
        prop={"family": "Arial"},
        borderpad=0.6,
    )

    # -------------------------------------------------------------
    # 3) Fraction of synapses & connector types
    # -------------------------------------------------------------
    # Line thickness vs fraction: same mapping as in draw_two_layer_neural_net
    example_fracs = [0.01, 0.10, 0.50]
    frac_handles = []
    for p in example_fracs:
        lw_ex = a_for_width * p + b_for_width
        frac_handles.append(
            Line2D([0, 1], [0, 0], color="black", lw=lw_ex, label=f"{p:g}")
        )

    # Use a mid example fraction for connector thickness, purely visual
    lw_conn = a_for_width * 0.10 + b_for_width

    # Dummy handles for connector types (actual shapes via custom handlers)
    conn_exc = Line2D([], [], label="Excitation")
    conn_inh = Line2D([], [], label="Inhibition")
    conn_mix = Line2D([], [], label="Mixed")
    conn_unknown = Line2D([], [], label="Unknown")

    ax_frac = legend_ax.inset_axes([0.0, 0.05, 1.0, 0.40])
    ax_frac.axis("off")

    class _ExcConnHandler(HandlerBase):
        """Line with an arrowhead at the right tip (excitatory)."""

        def create_artists(
            self, legend, orig_handle, x0, y0, width, height, fontsize, trans
        ):
            y_mid = y0 + 0.5 * height
            x_start = x0
            x_end = x0 + width

            # Shaft
            line = Line2D(
                [x_start, x_end],
                [y_mid, y_mid],
                color="black",
                linewidth=lw_conn,
                transform=trans,
            )

            # Arrowhead (triangle) at the right tip, pointing outward (→)
            head_len = 0.18 * width
            head_half = 0.30 * height
            p1 = (x_end, y_mid)
            p2 = (x_end - head_len, y_mid + head_half)
            p3 = (x_end - head_len, y_mid - head_half)
            head = Polygon(
                [p1, p2, p3],
                closed=True,
                facecolor="black",
                edgecolor="black",
                transform=trans,
            )
            return [line, head]

    class _InhConnHandler(HandlerBase):
        """
        Inhibitory connector.

        Horizontal line from left→right, with an *inverted* arrow at the
        RIGHT end: the flat base is at the very right tip, and the point of
        the arrow is shifted inward (to the left).
        """

        def create_artists(
            self, legend, orig_handle, x0, y0, width, height, fontsize, trans
        ):
            y_mid = y0 + 0.5 * height
            x_start = x0
            x_end = x0 + width

            # Shaft: left → right
            line = Line2D(
                [x_start, x_end],
                [y_mid, y_mid],
                color="black",
                linewidth=lw_conn,
                transform=trans,
            )

            # Inverted arrow at the RIGHT end
            head_len = 0.18 * width
            head_half = 0.30 * height

            base_top = (x_end, y_mid + head_half)
            base_bottom = (x_end, y_mid - head_half)
            tip = (x_end - head_len, y_mid)  # tip pointing inward

            head = Polygon(
                [base_top, base_bottom, tip],
                closed=True,
                facecolor="black",
                edgecolor="black",
                transform=trans,
            )

            return [line, head]

    class _MixedConnHandler(HandlerBase):
        """Line with a small filled circle at the right tip (mixed)."""

        def create_artists(
            self, legend, orig_handle, x0, y0, width, height, fontsize, trans
        ):
            y_mid = y0 + 0.5 * height
            x_start = x0
            x_end = x0 + width

            line = Line2D(
                [x_start, x_end],
                [y_mid, y_mid],
                color="black",
                linewidth=lw_conn,
                transform=trans,
            )

            marker = Line2D(
                [x_end],
                [y_mid],
                color="black",
                marker="o",
                markersize=6,
                transform=trans,
            )
            return [line, marker]

    class _UnknownConnHandler(HandlerBase):
        """Plain black line, no marker (unknown)."""

        def create_artists(
            self, legend, orig_handle, x0, y0, width, height, fontsize, trans
        ):
            y_mid = y0 + 0.5 * height
            x_start = x0
            x_end = x0 + width

            line = Line2D(
                [x_start, x_end],
                [y_mid, y_mid],
                color="black",
                linewidth=lw_conn,
                transform=trans,
            )
            return [line]

    handler_map_conn = {
        conn_exc: _ExcConnHandler(),
        conn_inh: _InhConnHandler(),
        conn_mix: _MixedConnHandler(),
        conn_unknown: _UnknownConnHandler(),
    }

    ax_frac.legend(
        handles=frac_handles + [conn_exc, conn_inh, conn_mix, conn_unknown],
        handler_map=handler_map_conn,
        loc="upper left",
        fontsize=9,
        frameon=True,
        title="Fraction of synapses\n& connectors",
        title_fontsize=10,
        prop={"family": "Arial"},
        borderpad=0.6,
    )


def plot_population_networks(
    *,
    population_name: str,
    seed_ids: Iterable[int] | Iterable[str],
    metadata_df: pd.DataFrame,
    root_folder: Path,
    output_folder: Path,
    plot_suffix: str,
    input_circle_color: str,
    input_cell_type: str,
    outputs_total_mode: str = "both",
) -> None:
    """
    Generate a 4-panel two-layer network figure for a given seed population
    and export a detailed text connectivity table.

    Layout
    ------
    The main figure consists of a 2×2 grid of network panels plus a separate
    legend column on the right:

        [0,0] Ipsilateral input synapses
        [0,1] Contralateral input synapses
        [1,0] Ipsilateral output synapses
        [1,1] Contralateral output synapses

    Each panel is a two-layer network:

        • Layer 1 (input): a single seed-population node, colored according
          to `input_circle_color`.

        • Layer 2 (output): one node per connection category, where a
          category is defined by a unique combination of:

              - Functional Classifier
              - Neurotransmitter Classifier
              - Projection Classifier
              - Axon Exit Direction

        • Edges: the line thickness encodes the fraction of synapses
          assigned to that category in the corresponding panel. Fractions
          are computed by `summarize_connectome` and mapped to a line
          width via:

              line_width = a * p + b

          with `a = 12` and `b = 1` in the current implementation
          (passed to `draw_two_layer_neural_net`).

        • Connector glyphs: a symbol at the *tip* of each line encodes
          the synaptic sign, using the same conventions as the legend
          drawn by `draw_full_connectivity_legend`:

              - Excitation : arrow pointing outward from the postsynaptic node
              - Inhibition : inverted arrow / T-like symbol pointing inward
              - Mixed      : small filled black circle at the tip
              - Unknown    : straight line, no marker

    All four panels share identical x/y limits so that circle sizes and line
    widths are visually comparable across panels and populations, independent
    of how many categories appear in each panel.

    Parameters
    ----------
    population_name : str
        Short label used for console messages and filenames
        (e.g. 'cMI', 'MON', 'SMI', 'iMI+', 'iMI-', 'iMI_all').
    seed_ids : iterable of int or str
        Nucleus IDs used as seed neurons when building the connectome.
    metadata_df : pandas.DataFrame
        Metadata for all reconstructed neurons/axons. Must contain hemisphere
        information and the functional / neurotransmitter / projection
        classifiers consumed by `summarize_connectome`. No LDA-specific
        columns are assumed at this stage.
    root_folder : Path
        Root folder containing traced-neuron subfolders with synapse CSVs.
    output_folder : Path
        Folder where the resulting PDF network diagrams and TXT tables
        will be saved.
    plot_suffix : str
        Suffix included in the output filenames (e.g. 'ic_all_lda_110725').
    input_circle_color : str
        Functional color key passed to `draw_two_layer_neural_net` to
        color the seed node, e.g.:
            'ipsilateral_motion_integrator',
            'contralateral_motion_integrator',
            'motion_onset',
            'slow_motion_integrator',
            'myelinated'.
    input_cell_type : str
        High-level description of the seed population, used by
        `draw_two_layer_neural_net` to select the connector type for
        *output* synapses. Typical values:
            'inhibitory', 'excitatory', 'mixed'.
    outputs_total_mode : {'both', 'same_only'}, optional
        Controls how the total number of output synapses is computed
        for cross-panel normalization of output fractions:
            'both'      : ipsilateral + contralateral outputs (default).
            'same_only' : ipsilateral outputs only (used for iMI populations).
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build and summarize the connectome for this seed population
    # ------------------------------------------------------------------
    print(f"\n=== Population: {population_name} ===")
    summary = summarize_connectome(
        root_folder=root_folder,
        seed_ids=seed_ids,
        hemisphere_df=metadata_df,
        outputs_total_mode=outputs_total_mode,
        functional_only=False,
    )

    same_out_syn = summary["same_out_syn"]
    diff_out_syn = summary["diff_out_syn"]
    same_in_syn = summary["same_in_syn"]
    diff_in_syn = summary["diff_in_syn"]
    total_outputs = summary["total_outputs"]
    total_inputs = summary["total_inputs"]

    print(f"Total output synapses: {total_outputs}")
    print(f"Total input synapses:  {total_inputs}")

    # ------------------------------------------------------------------
    # 2. Build the 2×2 figure + legend column
    # ------------------------------------------------------------------
    # We make a 2×3 grid: 2 rows of network panels, 3rd column only for legends.
    fig = plt.figure(figsize=(18, 12))

    gs = gridspec.GridSpec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.7],  # last column reserved for legend
        wspace=0.30,
        hspace=0.35,
    )

    ax00 = fig.add_subplot(gs[0, 0])  # ipsilateral inputs
    ax01 = fig.add_subplot(gs[0, 1])  # contralateral inputs
    ax10 = fig.add_subplot(gs[1, 0])  # ipsilateral outputs
    ax11 = fig.add_subplot(gs[1, 1])  # contralateral outputs

    # Legend axis (spans both rows); we draw no data here, only legends.
    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.axis("off")

    axes = [ax00, ax01, ax10, ax11]

    titles = [
        "Ipsilateral input synapses",
        "Contralateral input synapses",
        "Ipsilateral output synapses",
        "Contralateral output synapses",
    ]

    dataframes = [
        same_in_syn,
        diff_in_syn,
        same_out_syn,
        diff_out_syn,
    ]

    connection_types = [
        "inputs",
        "inputs",
        "outputs",
        "outputs",
    ]

    # Midline only makes sense for the cross-side (contralateral) panels:
    # → only the right-hand panels (index 1 and 3) get midlines.
    show_midlines = [False, True, False, True]

    # ------------------------------------------------------------------
    # 2a. Draw the four network panels
    # ------------------------------------------------------------------
    for ax, title, df_syn, ctype, show_midline_flag in zip(
        axes, titles, dataframes, connection_types, show_midlines
    ):
        kwargs: dict = {}
        if ctype == "outputs":
            kwargs["total_outputs"] = total_outputs
        else:
            kwargs["total_inputs"] = total_inputs

        draw_two_layer_neural_net(
            ax=ax,
            left=0.1,
            right=0.6,
            bottom=0.5,
            top=1.1,
            data_df=df_syn,
            node_radius=0.02,
            input_circle_color=input_circle_color,
            input_cell_type=input_cell_type,
            show_midline=show_midline_flag,
            proportional_lines=True,
            a=12,
            b=1,
            connection_type=ctype,
            label_mode="proportion",
            label_as_percent=True,
            label_decimals=1,
            **kwargs,
        )
        ax.set_title(title, fontsize=14)

    # ------------------------------------------------------------------
    # 2b. Enforce common data limits across panels
    # ------------------------------------------------------------------
    # Explicit limits so that node radii and line widths are visually
    # comparable across panels and populations.
    for ax in axes:
        ax.set_xlim(-0.4, 1.1)  # tuned to give a good layout
        ax.set_ylim(0.3, 1.3)

    # ------------------------------------------------------------------
    # 3. Draw the legends into the dedicated legend axis
    # ------------------------------------------------------------------
    draw_full_connectivity_legend(
        legend_ax,
        input_circle_color=input_circle_color,
        a_for_width=12.0,  # must match the 'a' passed to draw_two_layer_neural_net
        b_for_width=1.0,   # must match the 'b' passed to draw_two_layer_neural_net
    )

    # ------------------------------------------------------------------
    # 4. Global title
    # ------------------------------------------------------------------
    fig.suptitle(
        f"{population_name}: connectivity diagrams",
        fontsize=16,
        y=0.98,
    )

    # ------------------------------------------------------------------
    # 5. Save figure
    # ------------------------------------------------------------------
    basename_pdf = f"{plot_suffix}.pdf"
    out_path_pdf = output_folder / basename_pdf
    plt.savefig(out_path_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()
    print(f"Saved figure: {out_path_pdf}")

    # ------------------------------------------------------------------
    # 6. Export detailed TXT connectivity table
    # ------------------------------------------------------------------
    txt_path = export_connectivity_tables_txt(
        population_name=population_name,
        summary=summary,
        output_folder=output_folder,
        suffix=plot_suffix,
    )
    print(f"Saved connectivity table: {txt_path}")

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