#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate two-layer network diagrams for zebrafish hindbrain connectomes.

Overview
--------
This script rebuilds compact, two-layer connectivity diagrams for several
functionally defined seed populations in the zebrafish hindbrain
(cMI, MON, SMI, iMI+, iMI-, iMI_all). For each population it:

    1. Loads a .csv metadata table with one row per reconstructed neuron/axon
       (e.g. hemisphere, functional classifier, neurotransmitter classifier,
       projection type, etc.).

    2. Uses that table to define seed populations (via `get_seed_id_sets`).

    3. For each seed population:
        - extracts its input and output synapses split by hemisphere
          (same-side vs. cross-side),
        - computes synapse-count probabilities for each combination of
          functional / neurotransmitter / projection class
          (`summarize_connectome`, called inside `plot_population_networks`),
        - visualizes the connectivity as four two-layer diagrams:

              [0,0] Ipsilateral input synapses
              [0,1] Contralateral input synapses
              [1,0] Ipsilateral output synapses
              [1,1] Contralateral output synapses

          Each diagram consists of:
              - a single seed population node,
              - multiple target-category nodes,
              - connection lines whose thickness scales with the fraction
                of synapses in that category,
              - connector glyphs at the tips of the lines encoding
                excitatory / inhibitory / mixed / unknown.

        - embeds a dedicated legend column that explains:
              (i)   cell outline styles (E / I / mixed),
              (ii)  functional node colors,
              (iii) line thickness vs. fraction of synapses and the
                    different connector glyph types.

    4. Writes each multi-panel figure as a PDF in the chosen output folder.

    5. Exports a detailed, text-based connectivity table summarizing the same
       statistics for downstream analysis (`export_connectivity_tables_txt`).

Helpers
-------
All heavy lifting is delegated to `connectivity_diagrams_helpers.py`, which
provides:

    - load_csv_metadata
    - get_seed_id_sets
    - plot_population_networks

Typical usage
-------------
python3 make_connectivity_diagrams.py \
    --metadata-csv ".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/all_reconstructed_neurons.csv" \
    --root-folder ".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/traced_axons_neurons" \
    --output-folder connectivity_diagrams \
    --suffix gt
"""

from __future__ import annotations

import argparse
from pathlib import Path

# --- imports from the connectivity-diagram helpers ---------------------------
from connectivity_diagrams_helpers import (
    load_csv_metadata,
    get_seed_id_sets,
    plot_population_networks,
)

# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate two-layer network diagrams for hindbrain connectomes."
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help=(
            "CSV with metadata for all reconstructed neurons/axons "
            "(hemisphere, functional / neurotransmitter / projection classifiers, etc.)."
        ),
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

    # 1) Load metadata and define seed sets
    metadata_df = load_csv_metadata(args.metadata_csv)
    seed_sets = get_seed_id_sets(metadata_df)

    # Optional global suffix for filenames
    global_suffix = args.suffix.strip().lstrip("_")
    suffix_part = f"_{global_suffix}" if global_suffix else ""

    out_folder = args.output_folder

    # 2) Run the same plotting pipeline for each seed population  ----------
    # Each call to `plot_population_networks`:
    #   - builds the connectome summary for that seed set,
    #   - generates a 4-panel figure + legend,
    #   - saves the PDF + connectivity TXT.
    #
    # The `input_circle_color` determines the seed-node color in the diagrams,
    # and `input_cell_type` controls the connector glyph used for outputs
    # (excitatory / inhibitory / mixed).

    # 1) cMI (contralateral motion integrator, mostly inhibitory)
    plot_population_networks(
        population_name="cMI",
        seed_ids=seed_sets["cMI"],
        metadata_df=metadata_df,
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
        metadata_df=metadata_df,
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
        metadata_df=metadata_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"SMI_all{suffix_part}",
        input_circle_color="slow_motion_integrator",
        input_cell_type="mixed",
        outputs_total_mode="both",
    )

    # 4) iMI+ (ipsilateral motion integrator, excitatory,
    #          same-side outputs used for normalization)
    plot_population_networks(
        population_name="iMI+",
        seed_ids=seed_sets["iMI_plus"],
        metadata_df=metadata_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"iMI_plus{suffix_part}",
        input_circle_color="ipsilateral_motion_integrator",
        input_cell_type="excitatory",
        outputs_total_mode="same_only",
    )

    # 5) iMI- (ipsilateral motion integrator, inhibitory,
    #          same-side outputs used for normalization)
    plot_population_networks(
        population_name="iMI-",
        seed_ids=seed_sets["iMI_minus"],
        metadata_df=metadata_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"iMI_minus{suffix_part}",
        input_circle_color="ipsilateral_motion_integrator",
        input_cell_type="inhibitory",
        outputs_total_mode="same_only",
    )

    # 6) iMI_all (ipsilateral motion integrator, pooled;
    #             same-side outputs used for normalization)
    plot_population_networks(
        population_name="iMI_all",
        seed_ids=seed_sets["iMI_all"],
        metadata_df=metadata_df,
        root_folder=args.root_folder,
        output_folder=out_folder,
        plot_suffix=f"iMI{suffix_part}",
        input_circle_color="ipsilateral_motion_integrator",
        input_cell_type="mixed",
        outputs_total_mode="same_only",
    )


if __name__ == "__main__":
    main()