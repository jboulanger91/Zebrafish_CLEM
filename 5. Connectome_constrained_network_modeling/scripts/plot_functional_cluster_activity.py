#!/usr/bin/env python3
"""
plot_functional_cluster_activity.py

Display and export activity traces for functionally classified neurons.

For each cell type (new naming convention):
  - ipsilateral_motion_integrator
  - contralateral_motion_integrator
  - slow_motion_integrator
  - motion_onset

This script:
  - Loads neuron IDs from an Excel metadata file
  - Loads average activity (left/right) from an HDF5 file
  - Chooses the preferred direction (largest response)
  - Smooths and normalizes to 0–100%
  - Plots all traces and the mean trace
  - Saves:
        * <cell_type>_activity_traces.pdf
        * <cell_type>_activity_traces.csv
"""

import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# -------------------------------------------------------------------
# USER CONFIGURATION
# -------------------------------------------------------------------

PATH_ALL_CELLS = (
    "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
    "Jonathan Boulanger-Weill/Projects/Zebrafish_CLEM/"
    "1. Downloading_neuronal_morphologies_and_metadata/all_reconstructed_neurons.csv"
)

HDF5_FILE_PATH = (
    "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/"
    "hb_connectome/hindbrain_structure_function/clem_zfish1/function/"
    "all_cells_091024.h5"
)

PATH_OUTPUT = (
    "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
    "Jonathan Boulanger-Weill/Projects/Zebrafish_CLEM/"
    "5. Connectome_constrained_network_modeling"
)

os.makedirs(PATH_OUTPUT, exist_ok=True)

# Time step (seconds/sample)
DT = 0.5

# Window for determining preferred direction
PEAK_START = 40
PEAK_END   = 120

# Stimulus shading (seconds)
STIM_START = 20
STIM_END   = 60

# Force Arial font if available
plt.rcParams["font.family"] = "Arial"

# Color dictionary — NEW naming convention
COLOR_CELL_TYPE_DICT = {
    "ipsilateral_motion_integrator":   (254/255, 179/255, 38/255, 0.7),   # yellow-orange
    "contralateral_motion_integrator": (232/255, 77/255, 138/255, 0.7),   # magenta
    "slow_motion_integrator":          (127/255, 88/255, 175/255, 0.7),   # purple
    "motion_onset":                    (100/255, 197/255, 235/255, 0.7),  # light blue
}

# -------------------------------------------------------------------
# Extract neuron IDs per functional type
# -------------------------------------------------------------------

def extract_cell_ids(path_excel: str) -> dict:
    """
    Extract neuron IDs using the NEW naming convention.
    """
    df = pd.read_csv(path_excel)

    func_col = df.columns[9]     # functional classifier
    proj_col = df.columns[11]    # projection classifier
    id_col   = df.columns[1]     # neuron ID

    ids = {
        "ipsilateral_motion_integrator": df[
            (df[func_col] == "motion_integrator") &
            (df[proj_col].astype(str).str.contains("ipsilateral"))
        ][id_col],

        "contralateral_motion_integrator": df[
            (df[func_col] == "motion_integrator") &
            (df[proj_col].astype(str).str.contains("contralateral"))
        ][id_col],

        "motion_onset": df[df[func_col] == "motion_onset"][id_col],

        "slow_motion_integrator": df[df[func_col] == "slow_motion_integrator"][id_col],
    }

    return ids


# -------------------------------------------------------------------
# Core function: plot + CSV export
# -------------------------------------------------------------------

def plot_and_export_activity_pref_null(
    cell_ids,
    hdf5_path: str,
    cell_type: str,
    out_dir: str,
    color_dict,
) -> None:
    """
    For a given cell type:
      - For each neuron: load left/right average activity
      - Decide preferred vs null direction based on peak response
      - Normalize both traces to the preferred peak (0–100%)
      - Save separate plots and CSVs for preferred and null responses.

    CSVs contain ONLY the mean trace over all neurons for that direction.
    """
    cell_ids = list(cell_ids)
    preferred_traces = []
    null_traces = []

    with h5py.File(hdf5_path, "r") as hdf:
        for nid in cell_ids:
            key = f"neuron_{int(nid)}"
            if key not in hdf:
                continue

            grp = hdf[key]
            if ("average_activity_left" not in grp) or ("average_activity_right" not in grp):
                continue

            left = np.asarray(grp["average_activity_left"])
            right = np.asarray(grp["average_activity_right"])

            # Ensure traces are long enough for smoothing/window
            if len(left) < PEAK_END or len(right) < PEAK_END:
                continue

            # Smooth with Savitzky-Golay
            win_len = min(21, len(left) - (len(left) + 1) % 2)
            if win_len < 5:
                continue

            left_smooth  = savgol_filter(left,  win_len, 3)
            right_smooth = savgol_filter(right, win_len, 3)

            # Peak in analysis window
            win_slice = slice(PEAK_START, PEAK_END)
            left_peak  = np.nanmax(left_smooth[win_slice])
            right_peak = np.nanmax(right_smooth[win_slice])

            # Determine preferred / null
            if left_peak >= right_peak:
                pref_raw = left_smooth
                null_raw = right_smooth
            else:
                pref_raw = right_smooth
                null_raw = left_smooth

            pref_peak = np.nanmax(pref_raw)
            if not np.isfinite(pref_peak) or pref_peak == 0:
                continue

            # Normalize both to preferred peak (0–100%)
            pref_norm = (pref_raw / pref_peak) * 100.0
            null_norm = (null_raw / pref_peak) * 100.0

            preferred_traces.append(pref_norm)
            null_traces.append(null_norm)

    if not preferred_traces:
        print(f"[WARN] No usable traces for {cell_type}")
        return

    # Time axis
    T = len(preferred_traces[0])
    time = np.arange(T) * DT

    color = color_dict[cell_type]

    # Plot preferred stack
    _plot_traces_stack(
        time=time,
        traces=preferred_traces,
        cell_type=cell_type,
        out_dir=out_dir,
        color=color,
        suffix="preferred",
    )

    # Plot null stack
    _plot_traces_stack(
        time=time,
        traces=null_traces,
        cell_type=cell_type,
        out_dir=out_dir,
        color=color,
        suffix="null",
    )

    # -------------------------------------------------------
    # Save CSVs with MEAN traces only
    # -------------------------------------------------------
    mean_pref = np.nanmean(np.vstack(preferred_traces), axis=0)
    mean_null = np.nanmean(np.vstack(null_traces), axis=0)

    # Preferred
    df_pref = pd.DataFrame({
        "time_s": time,
        "mean_preferred_norm": mean_pref,
    })
    csv_pref = os.path.join(out_dir, f"{cell_type}_preferred_activity_traces.csv")
    df_pref.to_csv(csv_pref, index=False)
    print(f"[INFO] Saved CSV (preferred, mean only): {csv_pref}")

    # Null
    df_null = pd.DataFrame({
        "time_s": time,
        "mean_null_norm": mean_null,
    })
    csv_null = os.path.join(out_dir, f"{cell_type}_null_activity_traces.csv")
    df_null.to_csv(csv_null, index=False)
    print(f"[INFO] Saved CSV (null, mean only): {csv_null}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    ids_dict = extract_cell_ids(PATH_ALL_CELLS)

    for cell_type, ids in ids_dict.items():
        print(f"\n=== Processing {cell_type} (n={len(ids)}) ===")
        plot_and_export_activity_pref_null(
            cell_ids=ids,
            hdf5_path=HDF5_FILE_PATH,
            cell_type=cell_type,
            out_dir=PATH_OUTPUT,
            color_dict=COLOR_CELL_TYPE_DICT,
        )


if __name__ == "__main__":
    main()