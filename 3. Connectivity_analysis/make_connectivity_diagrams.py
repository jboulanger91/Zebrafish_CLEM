#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate two-layer network diagrams for zebrafish hindbrain connectomes.

This script:
    1. Loads a .csv annotated metadata table (one row per reconstructed neuron/axon).
    2. Selects seed populations (cMI, MON, MC, iMI+, iMI-).
    3. For each population:
        - extracts input/output connections with hemisphere info,
        - computes synapse-count probabilities,
        - plots four two-layer networks:
              (same-side inputs, different-side inputs,
               same-side outputs, different-side outputs),
          with line thickness proportional to connection strength.
    4. Saves each figure as a PDF in the chosen output folder.

Helpers are split across two modules:

Matrix helpers (matrix_helpers.py)
----------------------------------
    - COLOR_CELL_TYPE_DICT
    - fetch_filtered_ids
    - get_inputs_outputs_by_hemisphere_general
    - (plus matrix-related utilities not used here)

Connectivity-diagram helpers (connectivity_helpers.py)
------------------------------------------------------
    - compute_count_probabilities_from_results
    - draw_two_layer_neural_net
    - fetch_filtered_ids_EI

Typical usage
-------------
python3 make_connectome_diagrams.py \
    --lda-csv /path/to/all_cells_with_hemisphere_lda.csv \
    --root-folder /path/to/traced_axons_neurons \
    --output-folder /path/to/network_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd

# --- imports from the MATRIX helper (ID selection + hemisphere-aware IO) -----
from connectivity_matrices_helpers import (
    fetch_filtered_ids,
    get_inputs_outputs_by_hemisphere_general,
)

# --- imports from the CONNECTIVITY-DIAGRAM helper ----------------------------
from connectivity_diagrams_helpers import (
    compute_count_probabilities_from_results,
    draw_two_layer_neural_net,
    fetch_filtered_ids_EI,
)


# ----------------------------------------------------------------------
# ID selection helpers
# ----------------------------------------------------------------------


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


def get_seed_id_sets(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Collect seed nucleus ID sets for each functional population.

    This version assumes the *updated* functional classifier column
    (column index 9) uses the new labels:

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
    imi_all_ids_all_nuc, _ = fetch_filtered_ids(
        df, 9, "motion_integrator", 11, "ipsilateral"
    )

    # MC: all slow_motion_integrator
    mc_ids_all_nuc, _ = fetch_filtered_ids(df, 9, "slow_motion_integrator")

    # iMI+ : motion_integrator, ipsilateral, excitatory
    imi_plus_ids_all_nuc, _ = fetch_filtered_ids_EI(
        df, 9, "motion_integrator", 10, "excitatory", 11, "ipsilateral"
    )

    # iMI- : motion_integrator, ipsilateral, inhibitory
    imi_minus_ids_all_nuc, _ = fetch_filtered_ids_EI(
        df, 9, "motion_integrator", 10, "inhibitory", 11, "ipsilateral"
    )

    return {
        "cMI": cmi_ids_all_nuc,
        "MON": mon_ids_all_nuc,
        "MC": mc_ids_all_nuc,
        "iMI_plus": imi_plus_ids_all_nuc,
        "iMI_minus": imi_minus_ids_all_nuc,
        "iMI_all": imi_all_ids_all_nuc,
    }


# ----------------------------------------------------------------------
# Core plotting routine
# ----------------------------------------------------------------------


def plot_population_networks(
    *,
    population_name: str,
    seed_ids: Iterable[int] | Iterable[str],
    lda_df: pd.DataFrame,
    root_folder: Path,
    output_folder: Path,
    plot_suffix: str,
    input_circle_color: str,
    input_cell_type: str,
    outputs_total_mode: str = "both",
) -> None:
    """
    Generate a 4-panel two-layer network figure for a given seed population.

    Panels (2x2):
        [0,0] same-side inputs
        [0,1] different-side inputs
        [1,0] same-side outputs
        [1,1] different-side outputs

    Parameters
    ----------
    population_name : str
        Short label used only for console messages.
    seed_ids : iterable of int or str
        Nucleus IDs used as seed neurons when building the connectome.
    lda_df : pandas.DataFrame
        LDA-annotated metadata (contains hemisphere, classifiers, etc.).
    root_folder : Path
        Folder containing NG-resolution synapse tables for each neuron.
    output_folder : Path
        Folder where the resulting PDF will be saved.
    plot_suffix : str
        Suffix included in the output filename (e.g. 'ic_all_lda_110725').
    input_circle_color : str
        Functional color key passed to `draw_two_layer_neural_net`
        (e.g. 'contralateral_motion_integrator', 'motion_onset').
    input_cell_type : str
        High-level description of the seed population, used by the helper
        plotting function (e.g. 'inhibitory', 'excitatory', 'mixed').
    outputs_total_mode : {'both', 'same_only'}, optional
        Controls how the total number of output synapses is computed
        for normalization:
            - 'both'      : same-side + different-side outputs
            - 'same_only' : same-side outputs only (used for iMI+ / iMI-)
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # --- 1. Build connectome for this seed population ------------------
    print(f"\n=== Population: {population_name} ===")
    connectome = get_inputs_outputs_by_hemisphere_general(
        root_folder=root_folder,
        seed_cell_ids=seed_ids,
        hemisphere_df=lda_df,
    )

    # --- 2. Compute synapse-count probabilities (helper returns nested dict)
    probs = compute_count_probabilities_from_results(
        connectome, functional_only=False
    )

    same_out_cells = probs["outputs"]["same_side"]["cells"]
    same_out_syn = probs["outputs"]["same_side"]["synapses"]
    same_in_cells = probs["inputs"]["same_side"]["cells"]
    same_in_syn = probs["inputs"]["same_side"]["synapses"]

    diff_out_cells = probs["outputs"]["different_side"]["cells"]
    diff_out_syn = probs["outputs"]["different_side"]["synapses"]
    diff_in_cells = probs["inputs"]["different_side"]["cells"]
    diff_in_syn = probs["inputs"]["different_side"]["synapses"]

    # --- 3. Totals used for proportional line widths -------------------
    if outputs_total_mode == "same_only":
        # iMI+ / iMI- convention: normalize outputs by same-side counts
        total_outputs = same_out_syn["Count"].sum()
    else:  # 'both'
        total_outputs = same_out_syn["Count"].sum() + diff_out_syn["Count"].sum()

    total_inputs = same_in_syn["Count"].sum() + diff_in_syn["Count"].sum()

    print(f"Total output synapses: {total_outputs}")
    print(f"Total input synapses:  {total_inputs}")

    # --- 4. Build the 2x2 figure --------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    titles = [
        "Same-side input synapses",
        "Cross-side input synapses",
        "Same-side output synapses",
        "Cross-side output synapses",
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

    # Midline only makes sense for cross-side panels
    show_midlines = [False, True, False, True]

    for i, (ax, title, df_syn, ctype, show_midline) in enumerate(
        zip(axes.flatten(), titles, dataframes, connection_types, show_midlines)
    ):
        # Normalization key passed to `draw_two_layer_neural_net`
        kwargs = {}
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
            show_midline=show_midline,
            proportional_lines=True,
            a=6,
            b=2,
            connection_type=ctype,
            add_legend=(i == 0),          # show legend once (top-left panel)
            label_mode="proportion",      # line labels show proportions
            label_as_percent=True,        # ...as percent
            label_decimals=1,
            **kwargs,
        )
        ax.set_title(title, fontsize=14)

    plt.tight_layout()

    # --- 5. Save figure ------------------------------------------------
    basename = f"neural_network_visualization_with_lda_{plot_suffix}.pdf"
    out_path = output_folder / basename
    plt.savefig(out_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()
    print(f"Saved figure: {out_path}")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate two-layer network diagrams for hindbrain connectomes."
    )
    parser.add_argument(
        "--lda-csv",
        type=Path,
        required=True,
        help="CSV with hemisphere + LDA annotations for all reconstructed neurons.",
    )
    parser.add_argument(
        "--root-folder",
        type=Path,
        required=True,
        help="Root folder containing traced neuron subfolders with synapse CSVs.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Folder where PDF network diagrams will be written.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional global suffix appended to all figure names (e.g. '110725').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lda_df = load_csv_metadata(args.lda_csv)
    seed_sets = get_seed_id_sets(lda_df)

    # Normalise CLI suffix so you can pass '110725' or '_110725'
    global_suffix = args.suffix.strip().lstrip("_")
    suffix_part = f"_{global_suffix}" if global_suffix else ""

    out_folder = args.output_folder

    # 1) cMI (contralateral motion integrator, mostly inhibitory)
    plot_population_networks(
        population_name="cMI",
        seed_ids=seed_sets["cMI"],
        lda_df=lda_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"cmi_all_lda{suffix_part}",
        input_circle_color="contralateral_motion_integrator",
        input_cell_type="inhibitory",
        outputs_total_mode="both",
    )

    # 2) MON (motion onset, mostly inhibitory)
    plot_population_networks(
        population_name="MON",
        seed_ids=seed_sets["MON"],
        lda_df=lda_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"mon_all_lda{suffix_part}",
        input_circle_color="motion_onset",
        input_cell_type="inhibitory",
        outputs_total_mode="both",
    )

    # 3) MC (slow motion integrator / motor command, mixed)
    plot_population_networks(
        population_name="MC",
        seed_ids=seed_sets["MC"],
        lda_df=lda_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"mc_all_lda{suffix_part}",
        input_circle_color="slow_motion_integrator",
        input_cell_type="mixed",
        outputs_total_mode="both",
    )

    # 4) iMI+ (ipsilateral motion integrator, excitatory, same-side outputs for norm)
    plot_population_networks(
        population_name="iMI+",
        seed_ids=seed_sets["iMI_plus"],
        lda_df=lda_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"imi_plus_all_lda{suffix_part}",
        input_circle_color="ipsilateral_motion_integrator",
        input_cell_type="excitatory",
        outputs_total_mode="same_only",
    )

    # 5) iMI- (ipsilateral motion integrator, inhibitory, same-side outputs for norm)
    plot_population_networks(
        population_name="iMI-",
        seed_ids=seed_sets["iMI_minus"],
        lda_df=lda_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"imi_minus_all_lda{suffix_part}",
        input_circle_color="ipsilateral_motion_integrator",
        input_cell_type="inhibitory",
        outputs_total_mode="same_only",
    )


if __name__ == "__main__":
    main()