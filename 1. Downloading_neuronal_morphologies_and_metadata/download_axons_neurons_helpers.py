"""
Helper functions for the clem_zfish1 Neuroglancer segment and synapse pipeline.

This module contains the core functionality used by
`clem_zfish1_neuroglancer_pipeline.py`, including:

- checking for problematic axons and dendrites,
- generating per-neuron metadata files,
- downloading mesh components,
- querying and exporting synapse tables,
- and extracting functional imaging dynamics.

The main script is responsible for:
- configuring logging,
- parsing command-line arguments,
- initializing CloudVolume and CAVEclient,
- and calling these helpers.
"""

from __future__ import annotations

import datetime
from datetime import date
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

import cloudvolume as cv
import h5py
import navis
import numpy as np
import pandas as pd
from caveclient import CAVEclient
from scipy.signal import savgol_filter


# ------------------------------------------------------------------------
# Globals initialized from the main pipeline script via init_helpers()
# ------------------------------------------------------------------------

vol: cv.CloudVolume | None = None
client: CAVEclient | None = None
MANUAL_SYNAPSES_PATH: Path | None = None
ROOT_PATH: Path | None = None
HDF5_PATH: Path | None = None
SIZE_CUT_OFF: int = 44


def init_helpers(
    volume: cv.CloudVolume,
    cave_client: CAVEclient,
    manual_synapses_path: Path,
    root_path: Path,
    hdf5_path: Path,
    size_cutoff: int = 44,
) -> None:
    """
    Initialize module-level state for helper functions.

    This function must be called once from the main pipeline script after
    CloudVolume and CAVEclient have been created and paths have been resolved.

    Parameters
    ----------
    volume : cloudvolume.CloudVolume
        Initialized CloudVolume instance for the clem_zfish1 segmentation.
    cave_client : CAVEclient
        Initialized CAVEclient for the zebrafish hindbrain datastack.
    manual_synapses_path : Path
        Directory containing manual synapse annotation files.
    root_path : Path
        Root directory where per-neuron folders and metadata will be written.
    hdf5_path : Path
        HDF5 file containing functional imaging data (all_cells.h5).
    size_cutoff : int, optional
        Minimum synapse size (in voxels) considered "valid" for predicted synapses.
    """
    global vol, client, MANUAL_SYNAPSES_PATH, ROOT_PATH, HDF5_PATH, SIZE_CUT_OFF

    vol = volume
    client = cave_client
    MANUAL_SYNAPSES_PATH = manual_synapses_path
    ROOT_PATH = root_path
    HDF5_PATH = hdf5_path
    SIZE_CUT_OFF = size_cutoff


# ------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------

def format_synapse(
    segment_id: Any,
    position: list[float],
    synapse_id: Any,
    size: Any,
    source: str,
    validation: str,
    date_str: str,
) -> str:
    """
    Format a synapse entry as a comma-separated string.

    Parameters
    ----------
    segment_id : any
        Partner segment (root) ID.
    position : list of float
        [x, y, z] position in NG space.
    synapse_id : any
        Synapse ID (may be None for manual synapses).
    size : any
        Synapse size (may be None for manual synapses).
    source : {"predicted", "manual"}
        Indicates whether the synapse is predicted or manually annotated.
    validation : str
        Validation status (e.g. "valid", "below cut-off").
    date_str : str
        Date string in YYYY-MM-DD format.

    Returns
    -------
    str
        Formatted synapse string.
    """
    segment_str = str(segment_id)
    position_str = ",".join(map(str, position))
    id_str = "" if synapse_id is None else str(synapse_id)
    size_str = "" if size is None else str(size)
    return f"{segment_str},{position_str},{id_str},{size_str},{source},{validation},{date_str}"


def convert_to_int_safe(value: Any, default: int = 0) -> int:
    """Safely convert a value to int, returning `default` on failure."""
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------------
# Problematic synapse detection
# ------------------------------------------------------------------------

def check_problematic_segments(
    df: pd.DataFrame,
    synapse_table: str,
) -> Tuple[List[str], List[str]]:
    """
    Check which axon and dendrite segment IDs are NOT latest in the chunkedgraph.

    A segment is considered "problematic" here if
    `client.chunkedgraph.is_latest_roots(..., timestamp=mat_time)` returns False
    or if the check raises an error (e.g. invalid / deleted root).

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing neuron/axon entries and segment IDs.
        Assumes:
        - axon IDs     are in column 7
        - dendrite IDs are in column 8
    synapse_table : str
        Name of the synapse table in the materialized database.
        (Kept for API compatibility; not used directly in this function.)

    Returns
    -------
    problematic_axons : list of str
        Axon IDs that are not latest or could not be checked.
    problematic_dendrites : list of str
        Dendrite IDs that are not latest or could not be checked.
    """
    if client is None:
        raise RuntimeError(
            "Helper module not initialized: `client` is None. "
            "Call `init_helpers(...)` from the main script first."
        )

    logger = logging.getLogger(__name__)

    # Timestamp used for "latest" checks
    mat_time = client.materialize.get_timestamp()

    problematic_axons: List[str] = []
    problematic_dendrites: List[str] = []

    num_cells = len(df)
    logger.info(
        "Checking %d rows for outdated axon/dendrite root IDs (synapse table: %s)...",
        num_cells,
        synapse_table,
    )

    for idx in range(num_cells):
        print(f"Checking row {idx+1}/{num_cells}", end="\r")

        # ------------------------
        # AXON CHECK
        # ------------------------
        axon_id_raw = df.iloc[idx, 7]
        axon_id_str = str(axon_id_raw)

        if axon_id_str not in ("0", "nan", "NaN", "None") and axon_id_str.strip() != "":
            try:
                axon_root = int(axon_id_str)
                status = client.chunkedgraph.is_latest_roots(
                    [axon_root], timestamp=mat_time
                )
                is_latest = bool(status[0])

                if not is_latest:
                    problematic_axons.append(axon_id_str)
                    logger.warning(
                        "Outdated / non-latest axon root ID detected: %s", axon_id_str
                    )

            except Exception as e:
                problematic_axons.append(axon_id_str)
                logger.error(
                    "Error checking latest-root status for axon ID %s: %s",
                    axon_id_str,
                    e,
                )

        # ------------------------
        # DENDRITE CHECK
        # ------------------------
        dend_id_raw = df.iloc[idx, 8]
        dend_id_str = str(dend_id_raw)

        # Skip if "0" (axon-only row) or empty
        if dend_id_str not in ("0", "nan", "NaN", "None") and dend_id_str.strip() != "":
            try:
                dend_root = int(dend_id_str)
                status = client.chunkedgraph.is_latest_roots(
                    [dend_root], timestamp=mat_time
                )
                is_latest = bool(status[0])

                if not is_latest:
                    problematic_dendrites.append(dend_id_str)
                    logger.warning(
                        "Outdated / non-latest dendrite root ID detected: %s",
                        dend_id_str,
                    )

            except Exception as e:
                problematic_dendrites.append(dend_id_str)
                logger.error(
                    "Error checking latest-root status for dendrite ID %s: %s",
                    dend_id_str,
                    e,
                )

    logger.info(
        "Finished checking latest-root status. "
        "Problematic axons: %d, problematic dendrites: %d",
        len(problematic_axons),
        len(problematic_dendrites),
    )

    return problematic_axons, problematic_dendrites


# ------------------------------------------------------------------------
# Metadata generation and mesh export
# ------------------------------------------------------------------------

def generate_metadata_files(df: pd.DataFrame, root_path: Path) -> None:
    """
    Generate per-neuron metadata files and retrieve associated mesh segments.

    For each row in the input DataFrame, this function:
    - constructs a segment ID (cell or axon),
    - writes a metadata text file describing structural/functional attributes,
    - downloads and saves mesh components (soma, nucleus, axon, dendrites),
    - writes synapse files (predicted + manual, pre and post),
    - and processes functional data when available.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing neuron/axon entries and metadata.
    root_path : Path
        Root directory where per-neuron folders will be created.
    """
    num_cells = len(df)
    for idx in range(num_cells):
        element_type = str(df.iloc[idx, 0])
        print(
            f"Processing {element_type} {idx + 1} / {num_cells}, "
            f"ID: {df.iloc[idx, 5]}, functional ID: {df.iloc[idx, 1]}"
        )

        # Common fields
        functional_id = str(df.iloc[idx, 1])
        connectivity = "na" if pd.isnull(df.iloc[idx, 2]) else str(df.iloc[idx, 2])
        comment = "na" if pd.isnull(df.iloc[idx, 3]) else str(df.iloc[idx, 3])
        reconstruction_status = str(df.iloc[idx, 4])
        neuroglancer_link = str(df.iloc[idx, 12])
        imaging_modality = str(df.iloc[idx, 13])
        date_of_tracing = str(df.iloc[idx, 14])
        tracer_names = str(df.iloc[idx, 15])

        if element_type == "cell":
            nucleus_id = str(df.iloc[idx, 5])
            soma_id = str(df.iloc[idx, 6])
            axon_id = str(df.iloc[idx, 7])
            dendrites_id = str(df.iloc[idx, 8])
            segment_id = f"clem_zfish1_cell_{nucleus_id}"

            functional_classifier = str(df.iloc[idx, 9])
            neurotransmitter_classifier = str(df.iloc[idx, 10])
            projection_classifier = str(df.iloc[idx, 11])

            if axon_id == "0":
                axon_id = "na"
            if dendrites_id == "0":
                dendrites_id = "na"

            lines = [
                'type = "cell"',
                f"cell_name = {nucleus_id}",
                f"nucleus_id = {nucleus_id}",
                f"soma_id = {soma_id}",
                f"axon_id = {axon_id}",
                f"dendrites_id = {dendrites_id}",
                f'functional_id = "{functional_id}"',
                "cell_type_labels = "
                f'["{functional_classifier}", "{neurotransmitter_classifier}", "{projection_classifier}"]',
                f'imaging_modality = "{imaging_modality}"',
                f"date_of_tracing =  {date_of_tracing}",
                f'tracer_names = "{tracer_names}"',
                f'neuroglancer_link = "{neuroglancer_link}"',
                f'connectivity = "{connectivity}"',
                f'reconstruction_status = "{reconstruction_status}"',
                f'comment = "{comment}"',
            ]
        else:
            nucleus_id = "na"
            soma_id = "na"
            axon_id = str(df.iloc[idx, 7])
            dendrites_id = "na"
            segment_id = f"clem_zfish1_axon_{axon_id}"

            functional_classifier = "na"
            neurotransmitter_classifier = "na"
            projection_classifier = "na"

            lines = [
                'type = "axon"',
                'cell_name = "na"',
                'nucleus_id = "na"',
                'soma_id = "na"',
                f"axon_id = {axon_id}",
                'dendrites_id = "na"',
                f'functional_id = "{functional_id}"',
                'cell_type_labels = ["na", "na", "na"]',
                f'imaging_modality = "{imaging_modality}"',
                f"date_of_tracing =  {date_of_tracing}",
                f'tracer_names = "{tracer_names}"',
                f'neuroglancer_link = "{neuroglancer_link}"',
                f'connectivity = "{connectivity}"',
                f'reconstruction_status = "{reconstruction_status}"',
                f'comment = "{comment}"',
            ]

        # Create directory for the element
        element_path = os.path.join(root_path, segment_id)
        if not os.path.exists(element_path):
            os.makedirs(element_path)

        # Write metadata to a text file
        path_text_file = os.path.join(element_path, f"{segment_id}_metadata.txt")
        with open(path_text_file, "w") as f:
            for line in lines:
                f.write(line + "\n")

        upload_segments(element_type, df, idx, segment_id)
        process_functional_data(df, idx, segment_id)


def upload_segments(
    element_type: str,
    df: pd.DataFrame,
    idx: int,
    segment_id: str,
) -> None:
    """
    Upload mesh segments for a given cell or axon and write synapse files.

    Parameters
    ----------
    element_type : {"cell", "axon"}
        Indicates whether this row corresponds to a full cell or just an axon.
    df : pandas.DataFrame
        Input table with neuron/axon info.
    idx : int
        Row index in the DataFrame corresponding to the element.
    segment_id : str
        Segment identifier used for folder and file naming.
    """
    if element_type == "cell":
        upload_cell_segments(df, idx, segment_id)
    else:
        upload_axon_segments(df, idx, segment_id)


def upload_cell_segments(df: pd.DataFrame, idx: int, segment_id: str) -> None:
    """Download and save neuron parts for a cell and write its synapse file."""
    nucleus_id = str(df.iloc[idx, 5])
    soma_id = str(df.iloc[idx, 6])
    axon_id = str(df.iloc[idx, 7])
    dendrites_id = str(df.iloc[idx, 8])

    save_mesh(segment_id, soma_id, nucleus_id, axon_id, dendrites_id)

    synapse_file_path = ROOT_PATH / segment_id / f"{segment_id}_synapses.txt"
    today = date.today().strftime("%Y-%m-%d")
    write_synapse_file(synapse_file_path, axon_id, dendrites_id, segment_id, today)


def upload_axon_segments(df: pd.DataFrame, idx: int, segment_id: str) -> None:
    """Download and save axon mesh and write its synapse file."""
    axon_id = str(df.iloc[idx, 7])

    save_mesh(segment_id, soma_id=None, nucleus_id=None, axon_id=axon_id, dendrites_id=None)

    synapse_file_path = ROOT_PATH / segment_id / f"{segment_id}_synapses.txt"
    today = date.today().strftime("%Y-%m-%d")
    write_synapse_file(synapse_file_path, axon_id, "0", segment_id, today)


def save_mesh(
    segment_id: str,
    soma_id: str | None = None,
    nucleus_id: str | None = None,
    axon_id: str | None = None,
    dendrites_id: str | None = None,
) -> None:
    """
    Download and save neuron mesh components (soma, nucleus, axon, dendrites, full neuron).

    Parameters
    ----------
    segment_id : str
        Base identifier for the neuron/axon (used to name files and folders).
    soma_id : str or None
        Segment ID of soma.
    nucleus_id : str or None
        Segment ID of nucleus.
    axon_id : str or None
        Segment ID of axon.
    dendrites_id : str or None
        Segment ID of dendritic arbor.
    """
    if vol is None or ROOT_PATH is None:
        raise RuntimeError("Helper module not initialized: `vol` or `ROOT_PATH` is None. "
                           "Call `init_helpers(...)` from the main script first.")

    segment_path = os.path.join(ROOT_PATH, segment_id)
    os.makedirs(segment_path, exist_ok=True)

    neuron_parts: list[str] = []

    def is_valid_id(mesh_id: str | None) -> bool:
        return mesh_id is not None and mesh_id != "0" and mesh_id.lower() != "na"

    # Soma + nucleus
    if is_valid_id(soma_id) and is_valid_id(nucleus_id):
        soma_parts = vol.mesh.get([soma_id, nucleus_id], as_navis=True)
        soma_path = os.path.join(segment_path, f"{segment_id}_soma.obj")
        soma_nuc = navis.combine_neurons(soma_parts)
        navis.write_mesh(soma_nuc, soma_path, filetype="obj")
        neuron_parts.extend([soma_id, nucleus_id])

    # Axon
    if is_valid_id(axon_id):
        axon = vol.mesh.get([axon_id], as_navis=True)
        axon_path = os.path.join(segment_path, f"{segment_id}_axon.obj")
        navis.write_mesh(axon, axon_path, filetype="obj")
        neuron_parts.append(axon_id)

    # Dendrites
    if is_valid_id(dendrites_id):
        dendrites = vol.mesh.get([dendrites_id], as_navis=True)
        dendrites_path = os.path.join(segment_path, f"{segment_id}_dendrite.obj")
        navis.write_mesh(dendrites, dendrites_path, filetype="obj")
        neuron_parts.append(dendrites_id)

    # Full neuron
    if len(neuron_parts) > 1:
        neuron_parts_data = vol.mesh.get(neuron_parts, as_navis=True)
        neuron_path = os.path.join(segment_path, f"{segment_id}.obj")
        neuron = navis.combine_neurons(neuron_parts_data)
        navis.write_mesh(neuron, neuron_path, filetype="obj")
        # Optional: plot3d for QC (currently disabled)
        # neuron.plot3d()


# ------------------------------------------------------------------------
# Synapse export
# ------------------------------------------------------------------------

def write_synapse_file(
    synapse_file_path: Path,
    axon_id: str,
    dendrites_id: str,
    segment_id: str,
    run_date: str,
) -> None:
    """
    Query synapses for a given neuron/axon and write pre/post synapse tables.

    This function:
    - queries predicted pre- and postsynaptic synapses for the given axon and dendrite IDs,
    - merges them with manual synapses (if available),
    - writes an intermediate combined text file,
    - and finally exports NG-resolution presynaptic and postsynaptic CSV tables.

    Parameters
    ----------
    synapse_file_path : Path
        Path to the intermediate synapse text file to be written.
    axon_id : str
        Axon root ID used for presynaptic synapses (pre_pt_root_id).
    dendrites_id : str
        Dendrite root ID used for postsynaptic synapses (post_pt_root_id).
    segment_id : str
        Segment identifier (e.g. "clem_zfish1_cell_<id>" or "clem_zfish1_axon_<id>").
    run_date : str
        Date string (YYYY-MM-DD) associated with the predicted synapses.
    """
    if client is None or MANUAL_SYNAPSES_PATH is None:
        raise RuntimeError("Helper module not initialized: `client` or `MANUAL_SYNAPSES_PATH` is None. "
                           "Call `init_helpers(...)` from the main script first.")

    synapse_table = client.info.get_datastack_info()["synapse_table"]

    # Initialize output_synapses and input_synapses with empty dataframes
    output_synapses = pd.DataFrame()
    input_synapses = pd.DataFrame()

    # Fetch output synapses if axon_id is not "0"
    if axon_id != "0":
        output_synapses = client.materialize.live_query(
            synapse_table,
            datetime.datetime.now(datetime.timezone.utc),
            filter_equal_dict={"pre_pt_root_id": int(axon_id)},
        )

    # Fetch input synapses if dendrites_id is not "0"
    if dendrites_id != "0":
        input_synapses = client.materialize.live_query(
            synapse_table,
            datetime.datetime.now(datetime.timezone.utc),
            filter_equal_dict={"post_pt_root_id": int(dendrites_id)},
        )

    size_cut_off = SIZE_CUT_OFF

    # Output synapses
    if not output_synapses.empty:
        output_segment = output_synapses.post_pt_root_id
        output_position = output_synapses.ctr_pt_position.apply(
            lambda x: [2 * x[0], 2 * x[1], x[2]]
        )
        output_synapse_id = output_synapses.id
        output_size = output_synapses.iloc[:, 4]
        output_prediction_list = ["predicted"] * len(output_synapses)
        output_validation_list = [
            "valid" if value > size_cut_off else "below cut-off"
            for value in output_synapses.iloc[:, 4]
        ]
    else:
        output_segment = []
        output_position = []
        output_synapse_id = []
        output_size = []
        output_prediction_list = []
        output_validation_list = []

    # Input synapses
    if not input_synapses.empty:
        input_segment = input_synapses.pre_pt_root_id
        input_position = input_synapses.ctr_pt_position.apply(
            lambda x: [2 * x[0], 2 * x[1], x[2]]
        )
        input_synapse_id = input_synapses.id
        input_size = input_synapses.iloc[:, 4]
        input_prediction_list = ["predicted"] * len(input_synapses)
        input_validation_list = [
            "valid" if value > size_cut_off else "below cut-off"
            for value in input_synapses.iloc[:, 4]
        ]
    else:
        input_segment = []
        input_position = []
        input_synapse_id = []
        input_size = []
        input_prediction_list = []
        input_validation_list = []

    # Manually annotated synapses
    manual_synapses: Dict[str, Dict[str, list]] = {"pre": {}, "post": {}}

    for syn_type in ["pre", "post"]:
        manual_file_xlsx = MANUAL_SYNAPSES_PATH / f"{segment_id}_{syn_type}synapses_manual.xlsx"
        manual_file_csv = MANUAL_SYNAPSES_PATH / f"{segment_id}_{syn_type}synapses_manual.csv"

        if manual_file_xlsx.exists():
            manual_synapses_df = pd.read_excel(manual_file_xlsx)
        elif manual_file_csv.exists():
            manual_synapses_df = pd.read_csv(manual_file_csv)
        else:
            manual_synapses[syn_type] = {
                "segments": [],
                "positions": [],
                "ids": [],
                "sizes": [],
                "prediction_list": [],
                "validation_list": [],
                "date_list": [],
            }
            continue

        segments = manual_synapses_df["segment_id"].tolist()
        positions = manual_synapses_df[["position_x", "position_y", "position_z"]].values.tolist()
        date_list = manual_synapses_df["date"].astype(str).tolist()

        manual_synapses[syn_type] = {
            "segments": segments,
            "positions": positions,
            "ids": [None] * len(segments),
            "sizes": [None] * len(segments),
            "prediction_list": ["manual"] * len(segments),
            "validation_list": ["valid"] * len(segments),
            "date_list": date_list,
        }

    # Write formatted synapses to file
    date_list_output = [run_date for _ in range(len(output_segment))]
    date_list_input = [run_date for _ in range(len(input_segment))]

    with open(synapse_file_path, "w") as file:
        file.write("(presynaptic: [")
        for segment, position, syn_id, size, manual, validation, date_val in zip(
            output_segment,
            output_position,
            output_synapse_id,
            output_size,
            output_prediction_list,
            output_validation_list,
            date_list_output,
        ):
            file.write(
                "'" + format_synapse(segment, position, syn_id, size, manual, validation, date_val) + "', "
            )

        for segment, position, syn_id, size, manual, validation, date_val in zip(
            manual_synapses["pre"]["segments"],
            manual_synapses["pre"]["positions"],
            manual_synapses["pre"]["ids"],
            manual_synapses["pre"]["sizes"],
            manual_synapses["pre"]["prediction_list"],
            manual_synapses["pre"]["validation_list"],
            manual_synapses["pre"]["date_list"],
        ):
            file.write(
                "'" + format_synapse(segment, position, syn_id, size, manual, validation, date_val) + "', "
            )

        file.write("],postsynaptic: [")

        for segment, position, syn_id, size, manual, validation, date_val in zip(
            input_segment,
            input_position,
            input_synapse_id,
            input_size,
            input_prediction_list,
            input_validation_list,
            date_list_input,
        ):
            file.write(
                "'" + format_synapse(segment, position, syn_id, size, manual, validation, date_val) + "', "
            )

        for segment, position, syn_id, size, manual, validation, date_val in zip(
            manual_synapses["post"]["segments"],
            manual_synapses["post"]["positions"],
            manual_synapses["post"]["ids"],
            manual_synapses["post"]["sizes"],
            manual_synapses["post"]["prediction_list"],
            manual_synapses["post"]["validation_list"],
            manual_synapses["post"]["date_list"],
        ):
            file.write(
                "'" + format_synapse(segment, position, syn_id, size, manual, validation, date_val) + "', "
            )

        file.write("])")

    # Split the file into pre and post synapses, then write CSVs
    with open(synapse_file_path, "r") as fp:
        split_data = fp.read().split(",postsynaptic")

    presynaptic_data_str = split_data[0].replace("'", "")
    postsynaptic_data_str = split_data[1].replace("'", "")

    def convert_to_int_safe_from_float(value):
        try:
            if value is None:
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None

    for i, data_str in enumerate([presynaptic_data_str, postsynaptic_data_str]):
        start_index = data_str.find("[")
        end_index = data_str.find("]")
        synaptic_data = data_str[start_index + 1 : end_index]
        synaptic_list = synaptic_data.split(", ")

        table_data = []
        for entry in synaptic_list:
            if entry not in ("[]", ""):
                values = entry.split(",")
                partner_cell_id = values[0].strip("'")
                date_val = values[8].strip("'")

                table_data.append(
                    {
                        "partner_cell_id": partner_cell_id,
                        "x": int(values[1]),
                        "y": int(values[2]),
                        "z": int(values[3]),
                        "synapse_id": convert_to_int_safe(values[4]),
                        # NOTE: use float-safe conversion here instead of int-only
                        "size": convert_to_int_safe_from_float(values[5]),
                        "prediction_status": values[6],
                        "validation_status": values[7],
                        "date": date_val,
                    }
                )
        df_out = pd.DataFrame(table_data)

        # ---- NEW: set column headers depending on file type ----
        if i == 0:
            # presynaptic file: partner is postsynaptic
            columns = [
                "postsynaptic_ID",   # consistent naming, no space
                "x_(8_nm)",
                "y_(8_nm)",
                "z_(30_nm)",
                "synapse_id",
                "size",
                "prediction_status",
                "validation_status",
                "date",
            ]
        else:
            # postsynaptic file: partner is presynaptic
            columns = [
                "presynaptic_ID",
                "x_(8_nm)",
                "y_(8_nm)",
                "z_(30_nm)",
                "synapse_id",
                "size",
                "prediction_status",
                "validation_status",
                "date",
            ]

        if df_out.empty:
            # Create an empty table with the correct columns
            df_out = pd.DataFrame(columns=columns)
        else:
            # Just rename the existing columns
            df_out.columns = columns
        # --------------------------------------------------------

        synapse_file_path = Path(synapse_file_path)
        new_stem = synapse_file_path.stem.replace("_synapses", "")
        new_synapse_file_path = synapse_file_path.with_name(new_stem + synapse_file_path.suffix)

        # ---- NEW: remove 'ng_res' from the filenames ----
        suffix = "_presynapses.csv" if i == 0 else "_postsynapses.csv"
        # -------------------------------------------------
        output_file = new_synapse_file_path.with_name(new_synapse_file_path.stem + suffix)

        df_out.to_csv(
            output_file,
            index=False,
            sep=" ",     # keep space-separated
            header=True, # now write the header line
            float_format="%.8f",
        )

# ------------------------------------------------------------------------
# Functional imaging dynamics
# ------------------------------------------------------------------------

def process_functional_data(
    df: pd.DataFrame,
    idx: int,
    segment_id: str,
    make_plots: bool = True,
) -> None:
    """
    Extract and save functional imaging dynamics for functionally imaged neurons.

    For neurons with a `functional_id` different from "not functionally imaged",
    this function:

    - loads the corresponding ΔF/F activity from the global HDF5 dataset
      (specified via `init_helpers`),
    - writes a per-neuron HDF5 file containing single-trial and averaged traces
      for leftward and rightward random-dot and sine-wave motion stimuli,
    - optionally generates a PDF with trial and mean traces for random-dot stimuli.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing neuron/axon entries and metadata. Column 1 is assumed
        to contain the `functional_id`.
    idx : int
        Row index corresponding to the neuron.
    segment_id : str
        Segment identifier used to name the per-neuron folder and files
        (e.g. "clem_zfish1_cell_<id>" or "clem_zfish1_axon_<id>").
    make_plots : bool, optional
        If True, generate a PDF with trial and mean traces for left and right
        random-dot stimuli. By default, plots are created.
    """
    if HDF5_PATH is None or ROOT_PATH is None:
        raise RuntimeError(
            "Helper module not initialized: `HDF5_PATH` or `ROOT_PATH` is None. "
            "Call `init_helpers(...)` from the main script first."
        )

    functional_id = str(df.iloc[idx, 1])
    if functional_id == "not functionally imaged":
        # Nothing to do for non-imaged neurons
        return

    neuron_group_name = f"neuron_{functional_id}"

    # ------------------------------------------------------------------
    # Load data from the global functional HDF5
    # ------------------------------------------------------------------
    with h5py.File(HDF5_PATH, "r") as hdf_file:
        if neuron_group_name not in hdf_file:
            raise KeyError(
                f"Group '{neuron_group_name}' not found in HDF5 file {HDF5_PATH}."
            )
        neuron_group = hdf_file[neuron_group_name]

        # Trial-resolved ΔF/F data
        left_dots = neuron_group["dff_trials_left_dots"][()]
        right_dots = neuron_group["dff_trials_right_dots"][()]
        left_sine = neuron_group["dff_trials_left_sine"][()]
        right_sine = neuron_group["dff_trials_right_sine"][()]

    # Compute mean traces across trials (time along axis=1)
    mean_left_dots = np.nanmean(left_dots, axis=0)
    mean_right_dots = np.nanmean(right_dots, axis=0)
    mean_left_sine = np.nanmean(left_sine, axis=0)
    mean_right_sine = np.nanmean(right_sine, axis=0)

    # ------------------------------------------------------------------
    # Create per-neuron HDF5 file with dynamics
    # ------------------------------------------------------------------
    dest_path = ROOT_PATH / segment_id / f"{segment_id}_dynamics.hdf5"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(dest_path, "w") as f:
        grp = f.create_group("dF_F")

        # Store trial-resolved data
        grp.create_dataset("dff_trials_left_dots", data=left_dots)
        grp.create_dataset("dff_trials_right_dots", data=right_dots)
        grp.create_dataset("dff_trials_left_sine", data=left_sine)
        grp.create_dataset("dff_trials_right_sine", data=right_sine)

        # Store mean traces
        grp.create_dataset("mean_left_dots", data=mean_left_dots)
        grp.create_dataset("mean_right_dots", data=mean_right_dots)
        grp.create_dataset("mean_left_sine", data=mean_left_sine)
        grp.create_dataset("mean_right_sine", data=mean_right_sine)

    # ------------------------------------------------------------------
    # Optional plotting (Random Dot stimuli only)
    # ------------------------------------------------------------------
    if make_plots:
        import matplotlib.pyplot as plt  # local import to keep dependency optional

        # Time axis (frames -> seconds)
        dt = 0.5  # adjust if your frame rate is different
        n_timepoints = left_dots.shape[1]
        time_axis = np.arange(n_timepoints) * dt

        fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Leftward motion (dots)
        for trial in left_dots:
            ax[0].plot(time_axis, trial, color="orange", alpha=0.4, linewidth=1)
        ax[0].plot(time_axis, mean_left_dots, color="darkorange", linewidth=2)
        ax[0].axvspan(20, 60, color="gray", alpha=0.2, label="Stimulus ON")
        ax[0].set_title("Leftward Motion (Dots)")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("ΔF/F (%)")
        ax[0].legend(loc="upper left", frameon=False, fontsize=12)

        # Rightward motion (dots)
        for trial in right_dots:
            ax[1].plot(time_axis, trial, color="blue", alpha=0.2, linewidth=1)
        ax[1].plot(time_axis, mean_right_dots, color="darkblue", linewidth=2)
        ax[1].axvspan(20, 60, color="gray", alpha=0.2)
        ax[1].set_title("Rightward Motion (Dots)")
        ax[1].set_xlabel("Time (s)")

        # Square aspect ratio
        for axis in ax:
            axis.set_box_aspect(1)

        plt.suptitle(f"Neuron {functional_id} – Random Dot Stimuli")
        plt.tight_layout()

        pdf_path = ROOT_PATH / segment_id / f"{segment_id}_dynamics.pdf"
        plt.savefig(pdf_path)
        plt.close(fig)