#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neuroglancer segment and synapse retrieval pipeline for the zebrafish connectome (clem_zfish1).

This module retrieves neuronal segments and synapses from the Lichtman/Engert zebrafish 
hindbrain connectome (clem_zfish1) via Neuroglancer/CloudVolume and CAVEclient. It builds 
per-neuron metadata, downloads mesh components, merges automatic and manually annotated 
synapses, and optionally extracts functional imaging dynamics for downstream analyses.

Before running this script, CAVE credentials must be configured using the accompanying 
``CAVE_setup.ipynb`` notebook. That notebook establishes authentication required by 
``CAVEclient`` to access the relevant datastack.

Main functionality
------------------
- **Segment and synapse retrieval:**  
  Retrieve neuronal segments and synapse tables using CloudVolume and CAVEclient, including
  identification of problematic axons and dendrites.

- **Metadata generation:**  
  Create per-neuron metadata files describing structural, functional, and reconstruction 
  attributes, including nucleus, soma, axon, and dendrite IDs and Neuroglancer links.

- **Mesh export:**  
  Download and save soma, nucleus, axon, dendrite, and whole-neuron meshes (OBJ files) in
  organized per-neuron directories.

- **Synapse export:**  
  Query pre- and postsynaptic synapses, merge automatic and manual annotations, apply
  size-based filtering, and export NG-resolution synapse tables.

- **Optional functional dynamics:**  
  For functionally imaged neurons, extract ΔF/F activity from the HDF5 dataset, save it in
  per-neuron HDF5 files, and optionally generate diagnostic activity plots.

Dependencies
------------
- Python 3.10+
- navis
- cloudvolume
- numpy
- pandas
- h5py
- matplotlib
- scipy
- caveclient

Environment setup
-----------------
Create the recommended environment using:

    conda env create --file pull_from_neuroglancer.yaml

Command-line usage
------------------
After configuring CAVE credentials using ``CAVE_setup.ipynb``, run the full pipeline as:

    python3 clem_zfish1_neuroglancer_pipeline.py \
        --excel-file /path/to/rgc_axons_output_020525.csv \
        --root-path /path/to/traced_axons_neurons/ \
        --manual-synapses-path /path/to/manual_synapses/ \
        --hdf5-path /path/to/all_cells.h5 \
        --size-cutoff 44

Required inputs include:
- a CSV file listing neurons and segment IDs,
- an output directory where per-neuron folders will be created,
- a directory containing manual synapse annotations,
- an HDF5 file containing functional imaging data (if available).

This script forms part of the zebrafish hindbrain functional connectomics analysis 
pipeline and is released under the MIT License.
"""

__author__ = "Jonathan Boulanger-Weill"
__version__ = "1.0"
__date__ = "2025-11-25"


import argparse
import datetime
from datetime import date
import logging
import os
from pathlib import Path

import cloudvolume as cv
import h5py
import matplotlib.pyplot as plt
import navis
import numpy as np
import pandas as pd
from caveclient import CAVEclient
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings("ignore")

from clem_zfish1_neuroglancer_helper import (
    init_helpers,
    check_problematic_synapses,
    generate_metadata_files,
)


# ------------------------------------------------------------------------
# Dataset-level constants (these are fine to keep hard-coded)
# ------------------------------------------------------------------------

CLOUD_VOLUME_URL = (
    "graphene://https://data.proofreading.zetta.ai/segmentation/api/v1/"
    "lichtman_zebrafish_hindbrain_001"
)
DATASTACK_NAME = "lichtman_zebrafish_hindbrain"
SERVER_ADDRESS = "https://proofreading.zetta.ai"

# These will be set from CLI arguments in main()
EXCEL_FILE_PATH: Path
MANUAL_SYNAPSES_PATH: Path
ROOT_PATH: Path
HDF5_PATH: Path
SIZE_CUT_OFF: int

# Globals that will be initialized in main()
vol: cv.CloudVolume
client: CAVEclient
df: pd.DataFrame
num_cells: int
synapse_table: str


# ------------------------------------------------------------------------
# CLI + logging setup
# ------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for the script."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve segments and synapses from clem_zfish1, and generate "
            "per-neuron metadata, meshes, and synapse tables."
        )
    )
    parser.add_argument(
        "--excel-file",
        type=Path,
        required=True,
        help="CSV file listing neurons/axons and their segment IDs, see provided example",
    )
    parser.add_argument(
        "--root-path",
        type=Path,
        required=True,
        help="Output directory where per-neuron folders will be created.",
    )
    parser.add_argument(
        "--manual-synapses-path",
        type=Path,
        required=True,
        help="Directory containing manual synapse CSV files.",
    )
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        required=True,
        help="HDF5 file containing functional imaging data (all_cells.h5).",
    )
    parser.add_argument(
        "--size-cutoff",
        type=int,
        default=44,
        help="Minimum synapse size (in voxels) considered valid.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the clem_zfish1 Neuroglancer pipeline."""
    global EXCEL_FILE_PATH, MANUAL_SYNAPSES_PATH, ROOT_PATH, HDF5_PATH, SIZE_CUT_OFF
    global vol, client, df, num_cells, synapse_table

    setup_logging()
    logger = logging.getLogger(__name__)

    args = parse_args()

    # Set paths and parameters from CLI
    EXCEL_FILE_PATH = args.excel_file
    MANUAL_SYNAPSES_PATH = args.manual_synapses_path
    ROOT_PATH = args.root_path
    HDF5_PATH = args.hdf5_path
    SIZE_CUT_OFF = args.size_cutoff

    logger.info("Patching CloudVolume for navis.")
    navis.patch_cloudvolume()

    logger.info("Initializing CloudVolume and CAVEclient...")
    vol = cv.CloudVolume(CLOUD_VOLUME_URL, use_https=True, progress=False)
    client = CAVEclient(datastack_name=DATASTACK_NAME, server_address=SERVER_ADDRESS)

    logger.info("Loading cell table from %s", EXCEL_FILE_PATH)
    df = pd.read_csv(EXCEL_FILE_PATH, dtype=str)
    num_cells = len(df)
    logger.info("Loaded %d entries from cell table.", num_cells)

    synapse_table = client.info.get_datastack_info()["synapse_table"]
    logger.info("Using synapse table: %s", synapse_table)

    # ------------------------------------------------------------------
    # 1. Check problematic axons and dendrites
    # ------------------------------------------------------------------
    logger.info("Checking for problematic axons and dendrites...")
    problematic_axons, problematic_dendrites = check_problematic_synapses(df, synapse_table)
    logger.info(
        "Found %d problematic axons and %d problematic dendrites.",
        len(problematic_axons),
        len(problematic_dendrites),
    )

    # ------------------------------------------------------------------
    # 2. Generate metadata, meshes, synapse files, and functional data
    # ------------------------------------------------------------------
    logger.info("Generating metadata files and retrieving segments...")
    generate_metadata_files(df, ROOT_PATH)

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()