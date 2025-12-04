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
    The diagram has:

    - **Layer 1**: one large seed node (the population of interest).
    - **Layer 2**: one node per connection *category* derived from `data_df`.
    - **Edges**: straight lines from the seed node to each category node, with
      width and terminal symbol (arrow / inverted arrow / dot) encoding
      connection strength and sign.

    Expected columns in `data_df`
    -----------------------------
    Each row represents one connectivity category and must contain:

    - ``'Functional Classifier'``: e.g. ``motion_integrator``,
      ``motion_onset``, ``slow_motion_integrator``, ``myelinated``,
      ``axon``, ``not functionally imaged``, etc.
    - ``'Neurotransmitter Classifier'``: ``excitatory``, ``inhibitory``,
      or anything else (treated as mixed / unknown).
    - ``'Projection Classifier'``: ``ipsilateral``, ``contralateral``, or ``'None'``.
    - ``'Axon Exit Direction'``: free text; we search for the substrings
      ``'rostrally'`` and/or ``'caudally'`` to determine axon exit direction.
    - ``'Probability'``: panel-wise probability of that category.
    - ``'Count'``: number of synapses in that category.

    Pre-processing of categories
    ----------------------------
    Before plotting, the function simplifies `data_df`:

    1. **“Not functionally imaged”** rows are merged into a single row with
       summed ``Count`` / ``Probability`` and dummy classifiers
       (projection = ``'None'``, neurotransmitter = ``'unknown'``).

    2. **Other functional labels** (anything not in
       {``motion_integrator``, ``motion_onset``, ``slow_motion_integrator``,
       ``myelinated``, ``not functionally imaged``, ``axon``}) are merged into a
       single category ``other_functional_types`` (also with ``unknown`` /
       ``None`` classifiers).

    3. **Motion onset**: ipsilateral and contralateral motion-onset rows are
       merged into a single ``motion_onset`` category, with its neurotransmitter
       set to ``inhibitory`` and projection to ``None``.

    4. **Myelinated** rows are merged into a single ``myelinated`` category.

    Node appearance
    ---------------
    - **Position**: category nodes are spaced vertically between ``bottom`` and
      ``top``; the seed node is centered vertically at the left (for
      ``connection_type='outputs'``) or right (for ``'inputs'``).

    - **Fill color**:
        - motion_integrator → ipsi/contra-specific keys
          ``ipsilateral_motion_integrator`` or ``contralateral_motion_integrator``.
        - slow_motion_integrator → ``slow_motion_integrator`` color.
        - motion_onset           → ``motion_onset`` color.
        - myelinated             → ``myelinated`` color.
        - other_functional_types → ``other_functional_types`` color.
        - axon:
            * exits rostrally only  → ``axon_rostral`` color.
            * exits caudally only   → ``axon_caudal`` color.
            * exits rostrally **and** caudally → special half black / half white
              circle (independent of `COLOR_CELL_TYPE_DICT`).
        - anything else            → ``not_functionally_imaged`` color.

      Seed node color is taken from `COLOR_CELL_TYPE_DICT[input_circle_color]`.

    - **Outline style (cell outline legend)**:
        - solid   → ``Neurotransmitter Classifier == 'excitatory'``
        - dashed  → ``'inhibitory'``
        - dotted  → all other values (mixed / unknown)

      Outline color is a darkened version of the fill color
      (via ``_adjust_luminance``).

    - **Node size**:
        ``radius = node_radius * (1 + 4 * Probability)``, so more probable
        classes are drawn larger.  The seed node radius is 4× `node_radius`.

    - **Labels**:
        Controlled by ``label_mode``:

        - ``'count'``       → raw ``Count``.
        - ``'proportion'``  → ``Count / denom`` where
            * denom = ``total_outputs`` for output panels (if given),
            * denom = ``total_inputs`` for input panels (if given),
            * otherwise denom = total count in this panel.
        - ``'probability'`` → uses the ``Probability`` column.

        If ``label_as_percent=True``, labels are formatted as percentages
        with ``label_decimals`` decimal places.

    Edges, line width & connector type
    ----------------------------------
    For each category node, a straight line is drawn between the seed node
    and the category node.

    - **Direction**:
        - For ``connection_type='outputs'``: seed → target.
        - For ``'inputs'``: target → seed (the arguments are swapped so the
          line still points toward the postsynaptic cell).

    - **Line width (strength of connectivity)**:

        If ``proportional_lines=True``:

            ``linewidth = a * Probability + b``

        so that high-probability categories are drawn with thicker links.
        If ``proportional_lines=False``, all links have width ``b``.

        In the legend, three example line widths (e.g. p = 0.05, 0.1, 0.5)
        are shown to illustrate this mapping (“Fraction of synapses”).

    - **Connector type (connector legend)**:

        For **input panels** (``connection_type='inputs'``), the connector
        depends on the partner’s neurotransmitter:

        - excitatory → arrowhead pointing toward the postsynaptic seed.
        - inhibitory → inverted arrow (small triangle) whose base lies on the
          circle outline and whose tip points away from the circle.
        - mixed / unknown → no extra symbol (just a line).

        For **output panels** (``connection_type='outputs'``), the connector
        depends on the *seed* population type:

        - ``input_cell_type='excitatory'`` → all arrows.
        - ``'inhibitory'`` → all inverted arrows at the target node.
        - ``'mixed'`` → black circle at the line endpoint (no arrow).

    Midline
    -------
    If ``show_midline=True``, a dashed vertical line is drawn at the center
    between ``left`` and ``right`` to indicate the midline (used for
    cross-hemisphere panels).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the diagram.
    left, right, bottom, top : float
        Logical bounds in axis coordinates for placing nodes.
    data_df : pandas.DataFrame
        Connectivity summary with the columns described above.
    node_radius : float
        Base radius for category nodes (seed node is 4× this radius).
    input_circle_color : str
        Key into `COLOR_CELL_TYPE_DICT` for the seed node.
    input_cell_type : {'excitatory', 'inhibitory', 'mixed'}
        Type of seed population, used for output connector style.
    show_midline : bool
        Whether to draw a dashed midline.
    proportional_lines : bool
        Whether edge line width scales with probability.
    a, b : float
        Parameters for line-width scaling: ``lw = a * Probability + b``.
    connection_type : {'inputs', 'outputs'}
        Whether this panel shows inputs to the seed or outputs from it.
    add_legend : bool
        If True, draws the three legends described above.
    label_mode, label_as_percent, label_decimals :
        Control content and formatting of node labels.
    total_outputs, total_inputs : float or None
        Optional global totals for proportion labels.
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
            lw=3,
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