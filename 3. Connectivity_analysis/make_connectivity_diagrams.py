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
python3 make_connectivity_diagrams.py \
    --lda-csv ".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/all_reconstructed_neurons.csv" \
    --root-folder ".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/traced_axons_neurons" \
    --output-folder connectivity_diagrams \
    --suffix gt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Patch,
)
import pandas as pd


# --- imports from the CONNECTIVITY-DIAGRAM helper ----------------------------
from connectivity_diagrams_helpers import (
    load_csv_metadata,
    get_seed_id_sets,
    summarize_connectome,
    draw_two_layer_neural_net,
    export_connectivity_tables_txt,
    HalfBlackWhiteHandler
)

from connectivity_matrices_helpers import (  
    COLOR_CELL_TYPE_DICT,
)

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
    Generate a 4-panel two-layer network figure for a given seed population
    and export a detailed TXT connectivity table.

    Panels (2x2):
        [0,0] same-side inputs
        [0,1] different-side inputs
        [1,0] same-side outputs
        [1,1] different-side outputs

    Parameters
    ----------
    population_name : str
        Short label used for console messages and filenames (e.g. 'cMI', 'MON').
    seed_ids : iterable of int or str
        Nucleus IDs used as seed neurons when building the connectome.
    lda_df : pandas.DataFrame
        LDA-annotated metadata (contains hemisphere, classifiers, etc.).
    root_folder : Path
        Folder containing NG-resolution synapse tables for each neuron.
    output_folder : Path
        Folder where the resulting PDF and TXT will be saved.
    plot_suffix : str
        Suffix included in the output filename (e.g. 'ic_all_lda_110725').
    input_circle_color : str
        Functional color key passed to `draw_two_layer_neural_net`
        (e.g. 'ipsilateral_motion_integrator',
              'contralateral_motion_integrator',
              'motion_onset',
              'slow_motion_integrator',
              'myelinated').
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

    # --- 1. Build and summarize connectome for this seed population ----
    print(f"\n=== Population: {population_name} ===")
    summary = summarize_connectome(
        root_folder=root_folder,
        seed_ids=seed_ids,
        hemisphere_df=lda_df,
        outputs_total_mode=outputs_total_mode,
        functional_only=False,
    )

    probs = summary["probs"]
    same_out_syn = summary["same_out_syn"]
    diff_out_syn = summary["diff_out_syn"]
    same_in_syn = summary["same_in_syn"]
    diff_in_syn = summary["diff_in_syn"]
    total_outputs = summary["total_outputs"]
    total_inputs = summary["total_inputs"]

    print(f"Total output synapses: {total_outputs}")
    print(f"Total input synapses:  {total_inputs}")

    # --- 2. Build the 2x2 figure + legend column --------------------------
    # We make a 2x3 grid: 2 rows of panels, 3rd column is only for legends.
    fig = plt.figure(figsize=(16, 10))

    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[1.0, 1.0, 0.7],   # last column reserved for legend
        wspace=0.30,
        hspace=0.35,
    )

    ax00 = fig.add_subplot(gs[0, 0])   # ipsi inputs
    ax01 = fig.add_subplot(gs[0, 1])   # contra inputs
    ax10 = fig.add_subplot(gs[1, 0])   # ipsi outputs
    ax11 = fig.add_subplot(gs[1, 1])   # contra outputs

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

    # Midline only makes sense for the cross-side (contralateral) panels
    # → only the right-hand panels (index 1 and 3) get midlines.
    show_midlines = [False, True, False, True]

    for i, (ax, title, df_syn, ctype, show_midline_flag) in enumerate(
        zip(axes, titles, dataframes, connection_types, show_midlines)
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

    # --- 2b. Force all four network panels to share the same data limits ---
    # This makes node radii and line widths *visually* comparable across subplots,
    # independent of how many nodes each panel has or whether it has a midline.
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for ax in axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xmins.append(x0)
        xmaxs.append(x1)
        ymins.append(y0)
        ymaxs.append(y1)

    common_xlim = (min(xmins), max(xmaxs))
    common_ylim = (min(ymins), max(ymaxs))

    # --- Optional: zoom all panels by a fixed factor around the center ----
    # factor < 1 → zoom in (features look bigger)
    # factor > 1 → zoom out (more whitespace, features look smaller)
    zoom_factor = 0.7   # 0.7 ≈ “30% larger” appearance

    x_center = 0.5 * (common_xlim[0] + common_xlim[1])
    y_center = 0.5 * (common_ylim[0] + common_ylim[1])

    x_half = 0.5 * (common_xlim[1] - common_xlim[0]) * zoom_factor
    y_half = 0.5 * (common_ylim[1] - common_ylim[0]) * zoom_factor

    zoomed_xlim = (x_center - x_half, x_center + x_half)
    zoomed_ylim = (y_center - y_half, y_center + y_half)

    for ax in axes:
        ax.set_xlim(zoomed_xlim)
        ax.set_ylim(zoomed_ylim)

    # --- 3. Draw the legends INTO legend_ax -------------------------------
    # Three stacked legend boxes: cell outlines, functional types, and
    # fraction-of-synapses + connector types.

    # 3.1 Cell outlines
    outline_handles = [
        Line2D([0, 1], [0, 0], color="black", lw=3, linestyle="solid",
               label="Excitatory outline"),
        Line2D([0, 1], [0, 0], color="black", lw=3, linestyle="dashed",
               label="Inhibitory outline"),
        Line2D([0, 1], [0, 0], color="black", lw=3, linestyle="dotted",
               label="Mixed / unknown outline"),
    ]
    # Slightly more compact vertical spacing between legend blocks
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

    # 3.2 Functional types (node fill colors)
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

    # 3.3 Fraction of synapses & connector types
    example_fracs = [0.01, 0.10, 0.50]
    frac_handles = []
    for p in example_fracs:
        # Use same mapping as lines in the network: lw = a * p + b with a=12, b=1
        lw_ex = 12 * p + 1
        frac_handles.append(
            Line2D([0, 1], [0, 0], color="black", lw=lw_ex, label=f"{p:g}")
        )

    # Use a mid example fraction (0.10) for connector thickness
    lw_conn = 12 * 0.10 + 1
    conn_exc = Line2D(
        [0, 1], [0, 0],
        color="black",
        lw=lw_conn,
        marker=">",
        markersize=7,
        markevery=(1,),
        label="Excitation",
    )
    conn_inh = Line2D(
        [0, 1], [0, 0],
        color="black",
        lw=lw_conn,
        marker="<",
        markersize=7,
        markevery=(1,),
        label="Inhibition",
    )
    conn_mix = Line2D(
        [0, 1], [0, 0],
        color="black",
        lw=lw_conn,
        marker="o",
        markersize=6,
        markevery=(1,),
        label="Mixed",
    )

    ax_frac = legend_ax.inset_axes([0.0, 0.05, 1.0, 0.40])
    ax_frac.axis("off")
    ax_frac.legend(
        handles=frac_handles + [conn_exc, conn_inh, conn_mix],
        loc="upper left",
        fontsize=9,
        frameon=True,
        title="Fraction of synapses\n& connectors",
        title_fontsize=10,
        prop={"family": "Arial"},
        borderpad=0.6,
    )

    # --- 4. Global title & layout -----------------------------------------
    fig.suptitle(
        f"{population_name}: two-layer connectivity diagrams",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout(rect=[0.0, 0.0, 0.98, 0.60])  # leave a bit of room for suptitle

    # --- 5. Save figure ----------------------------------------------------
    basename_pdf = f"{plot_suffix}.pdf"
    out_path_pdf = output_folder / basename_pdf
    plt.savefig(out_path_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()
    print(f"Saved figure: {out_path_pdf}")

    # --- 6. Export detailed TXT connectivity table ------------------------
    txt_path = export_connectivity_tables_txt(
        population_name=population_name,
        summary=summary,
        output_folder=output_folder,
        suffix=plot_suffix,
    )
    print(f"Saved connectivity table: {txt_path}")

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

    csv_df = load_csv_metadata(args.lda_csv)
    seed_sets = get_seed_id_sets(csv_df)

    global_suffix = args.suffix.strip().lstrip("_")
    suffix_part = f"_{global_suffix}" if global_suffix else ""

    out_folder = args.output_folder

    # 1) cMI (contralateral motion integrator, mostly inhibitory)
    plot_population_networks(
        population_name="cMI",
        seed_ids=seed_sets["cMI"],
        lda_df=csv_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"cMI_all{suffix_part}",
        input_circle_color="contralateral_motion_integrator",
        input_cell_type="inhibitory",
        outputs_total_mode="both",
    )

    # 2) MON (motion onset, mostly inhibitory)
    plot_population_networks(
        population_name="MON",
        seed_ids=seed_sets["MON"],
        lda_df=csv_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"MON_all{suffix_part}",
        input_circle_color="motion_onset",
        input_cell_type="inhibitory",
        outputs_total_mode="both",
    )

    # 3) SMI (slow motion integrator / motor command, mixed)
    plot_population_networks(
        population_name="SMI",
        seed_ids=seed_sets["SMI"],
        lda_df=csv_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"SMI_all{suffix_part}",
        input_circle_color="slow_motion_integrator",
        input_cell_type="mixed",
        outputs_total_mode="both",
    )

    # 4) iMI+ (ipsilateral motion integrator, excitatory, same-side outputs for norm)
    plot_population_networks(
        population_name="iMI+",
        seed_ids=seed_sets["iMI_plus"],
        lda_df=csv_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"iMI_plus{suffix_part}",
        input_circle_color="ipsilateral_motion_integrator",
        input_cell_type="excitatory",
        outputs_total_mode="same_only",
    )

    # 5) iMI- (ipsilateral motion integrator, inhibitory, same-side outputs for norm)
    plot_population_networks(
        population_name="iMI-",
        seed_ids=seed_sets["iMI_minus"],
        lda_df=csv_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"iMI_minus{suffix_part}",
        input_circle_color="ipsilateral_motion_integrator",
        input_cell_type="inhibitory",
        outputs_total_mode="same_only",
    )

    # 6) iMI (ipsilateral motion integrator, same-side outputs for norm)
    plot_population_networks(
        population_name="iMI_all",
        seed_ids=seed_sets["iMI_all"],
        lda_df=csv_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"iMI{suffix_part}",
        input_circle_color="ipsilateral_motion_integrator",
        input_cell_type="mixed",
        outputs_total_mode="same_only",
    )

if __name__ == "__main__":
    main()