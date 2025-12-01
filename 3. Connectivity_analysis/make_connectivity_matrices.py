#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make connectivity matrices for zebrafish hindbrain neurons (clem_zfish1).

This script builds directional connectivity matrices from synapse
tables and a metadata CSV. It produces:

1. A single pooled matrix across hemispheres, grouping neurons into:
   - axon_rostral
   - ipsilateral_motion_integrator
   - contralateral_motion_integrator
   - motion_onset
   - slow_motion_integrator
   - myelinated
   - axon_caudal

2. A left/right-split matrix, with the same functional classes but separated
   by hemisphere (e.g., ipsilateral_motion_integrator_left/right, etc.), and
   optionally rendered in an inhibitory/excitatory “signed” representation.

The underlying metadata table is assumed to contain:
- 'type' ('cell' or 'axon')
- 'functional classifier' (e.g., 'motion_integrator',
  'motion_onset', 'slow_motion_integrator', 'myelinated')
- 'projection classifier' ('ipsilateral'/'contralateral')
- 'neurotransmitter classifier' ('inhibitory'/'excitatory')
- 'comment' (e.g., 'axon exits the volume caudally/rostrally')
- 'nucleus_id', 'axon_id', 'dendrite_id'
- optionally 'hemisphere'

If 'hemisphere' is missing or entirely NaN, it will be computed from mapped
mesh files using `determine_hemisphere` from the helper module.

Typical usage
-------------
python3 make_connectivity_matrices.py \
    --metadata-csv "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/all_reconstructed_neurons.csv" \
    --root-folder "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/traced_axons_neurons" \
    --output-folder /connectivity_matrices \
    --plot-type scatter \
    --suffix ground_truth_scatter

All paths should be adapted to your local setup.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from clem_zfish1_connectivity_helper import (
    COLOR_CELL_TYPE_DICT,
    COLOR_CELL_TYPE_DICT_LR,
    create_nucleus_id_groups,
    create_nucleus_id_groups_hemisphere,
    determine_hemisphere,
    filter_connectivity_matrix,
    generate_directional_connectivity_matrix_general,
    generate_functional_types,
    load_and_clean_data,
    plot_connectivity_matrix,
    standardize_functional_naming,
)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging format and level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the connectivity matrix script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:
        - metadata_csv
        - root_folder
        - output_folder
        - recompute_hemisphere
        - plot_type
        - suffix
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate directional connectivity matrices for clem_zfish1 neurons "
            "from NG-resolution synapse tables and metadata."
        )
    )

    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="CSV metadata file (e.g., all_cells_111224_with_hemisphere.csv).",
    )
    parser.add_argument(
        "--root-folder",
        type=Path,
        required=True,
        help=(
            "Root folder containing traced neuron subfolders with NG synapse "
            "tables (e.g., traced_neurons/all_cells_111224/)."
        ),
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Folder where connectivity matrix plots (PDFs) will be written.",
    )
    parser.add_argument(
        "--recompute-hemisphere",
        action="store_true",
        help=(
            "Force re-computation of hemisphere labels from meshes, even if "
            "the metadata already contains a 'hemisphere' column."
        ),
    )
    parser.add_argument(
        "--plot-type",
        choices=["heatmap", "scatter"],
        default="heatmap",
        help="Plot style for both matrices ('heatmap' or 'scatter').",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help=(
            "Optional suffix appended to output PDF filenames "
            "(e.g., 'lda_022825'). Do not include the file extension."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point for building connectivity matrices from clem_zfish1 metadata.

    This function:
    1. Loads the metadata table.
    2. Standardizes functional naming.
    3. Ensures a 'hemisphere' column (computes it if needed).
    4. Creates functional groups (pooled and L/R split).
    5. Builds connectivity matrices using NG synapse tables.
    6. Plots and saves the resulting matrices as PDFs.

    The plot type (heatmap vs scatter) and an optional filename suffix can
    be configured via command-line arguments.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    args = parse_args()
    metadata_csv: Path = args.metadata_csv
    root_folder: Path = args.root_folder
    output_folder: Path = args.output_folder
    recompute_hemisphere: bool = args.recompute_hemisphere
    plot_type: str = args.plot_type
    suffix: str = args.suffix.strip()

    if suffix:
        suffix_str = f"_{suffix}"
    else:
        suffix_str = ""

    logger.info("Loading metadata from %s", metadata_csv)
    all_cells = load_and_clean_data(metadata_csv)
    all_cells = standardize_functional_naming(all_cells)

    # ------------------------------------------------------------------
    # Ensure hemisphere information
    # ------------------------------------------------------------------
    need_hemi = (
        recompute_hemisphere
        or ("hemisphere" not in all_cells.columns)
        or all_cells["hemisphere"].isna().all()
    )
    if need_hemi:
        logger.info(
            "Computing 'hemisphere' column from mapped meshes in %s ...", root_folder
        )
        progress = {"processed_count": 0, "total_rows": len(all_cells)}
        all_cells["hemisphere"] = all_cells.apply(
            determine_hemisphere,
            axis=1,
            root_folder=root_folder,
            progress=progress,
        )
        logger.info("Hemisphere classification completed.")
    else:
        logger.info("Using existing 'hemisphere' column from metadata.")

    # ------------------------------------------------------------------
    # 1. Pooled connectivity matrix (no L/R split)
    # ------------------------------------------------------------------
    logger.info("Building pooled functional groups (no L/R split)...")
    nucleus_id_groups = create_nucleus_id_groups(all_cells)
    functional_types = generate_functional_types(nucleus_id_groups)

    all_ids_nuc = np.concatenate(list(nucleus_id_groups.values()))
    logger.info("Total pooled IDs used in matrix: %d", len(all_ids_nuc))

    logger.info("Computing pooled directional connectivity matrix...")
    connectivity_matrix = generate_directional_connectivity_matrix_general(
        root_folder=root_folder,
        seg_ids=all_ids_nuc,
        df_w_hemisphere=all_cells,
    )

    logger.info("Filtering pooled connectivity matrix to non-zero rows/cols...")
    filtered_matrix, filtered_types = filter_connectivity_matrix(
        connectivity_matrix,
        functional_types,
    )

    category_order_pooled = [
        "axon_rostral",
        "ipsilateral_motion_integrator",
        "contralateral_motion_integrator",
        "motion_onset",
        "slow_motion_integrator",
        "myelinated",
        "axon_caudal",
    ]

    pooled_title = f"pooled_connectivity_matrix{suffix_str}"
    logger.info("Plotting pooled connectivity matrix (%s)...", plot_type)
    plot_connectivity_matrix(
        filtered_matrix,
        filtered_types,
        output_path=output_folder,
        category_order=category_order_pooled,
        df=all_cells,
        title=pooled_title,
        display_type="normal",
        plot_type=plot_type,
        color_cell_type_dict=COLOR_CELL_TYPE_DICT,
    )

    # Simple counts for console
    logger.info("Counts per pooled functional category (non-zero entries only):")
    counts = {}
    for fid, ftype in filtered_types.items():
        counts[ftype] = counts.get(ftype, 0) + 1
    for category, count in counts.items():
        logger.info("  %s: %d", category, count)

    # ------------------------------------------------------------------
    # 2. Left/right split connectivity matrix
    # ------------------------------------------------------------------
    logger.info("Building L/R split functional groups...")
    nucleus_id_groups_lr = create_nucleus_id_groups_hemisphere(all_cells)
    functional_types_lr = generate_functional_types(nucleus_id_groups_lr)

    all_ids_nuc_lr = np.concatenate(list(nucleus_id_groups_lr.values()))
    logger.info("Total L/R IDs used in matrix: %d", len(all_ids_nuc_lr))

    logger.info("Computing L/R directional connectivity matrix...")
    connectivity_matrix_lr = generate_directional_connectivity_matrix_general(
        root_folder=root_folder,
        seg_ids=all_ids_nuc_lr,
        df_w_hemisphere=all_cells,
    )

    logger.info("Filtering L/R connectivity matrix to non-zero rows/cols...")
    filtered_matrix_lr, filtered_types_lr = filter_connectivity_matrix(
        connectivity_matrix_lr,
        functional_types_lr,
    )

    category_order_lr = [
        "axon_rostral_left",
        "ipsilateral_motion_integrator_left",
        "contralateral_motion_integrator_left",
        "motion_onset_left",
        "slow_motion_integrator_left",
        "myelinated_left",
        "axon_caudal_left",
        "axon_rostral_right",
        "ipsilateral_motion_integrator_right",
        "contralateral_motion_integrator_right",
        "motion_onset_right",
        "slow_motion_integrator_right",
        "myelinated_right",
        "axon_caudal_right",
    ]

    lr_title = f"lr_split_connectivity_matrix_inhibitory_excitatory{suffix_str}"
    logger.info(
        "Plotting L/R connectivity matrix in inhibitory/excitatory mode (%s)...",
        plot_type,
    )
    plot_connectivity_matrix(
        filtered_matrix_lr,
        filtered_types_lr,
        output_path=output_folder,
        category_order=category_order_lr,
        df=all_cells,
        title=lr_title,
        display_type="Inhibitory_Excitatory",
        plot_type=plot_type,
        color_cell_type_dict=COLOR_CELL_TYPE_DICT_LR,
    )

    logger.info("Connectivity matrix generation finished successfully.")


if __name__ == "__main__":
    main()