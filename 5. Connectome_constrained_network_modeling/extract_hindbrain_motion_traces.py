#!/usr/bin/env python3
"""
extract_hindbrain_motion_traces.py

Purpose
-------
Extract hindbrain neurons belonging to motion-related functional clusters
(slow motion integrator, motion onset, motion integrator), compute each
neuron's preferred-direction *raw* trace (no normalization), and export:

  - One CSV per cluster type containing ONLY the mean preferred trace:
        time_s, mean_preferred_raw

Also prints how many functional (cluster) types were exported.
"""

import os
import re
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams["font.family"] = "Arial"

# -------------------------------------------------------------------
# User configuration
# -------------------------------------------------------------------

BRAIN_REG_CSV_PATH = (
    "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/"
    "hindbrain_structure_function/clem_zfish1/function/inputs/regions_021725.csv"
)
REGION_COL_NAME = "mapzebrain"

HDF5_PATH = (
    "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/"
    "hb_connectome/hindbrain_structure_function/clem_zfish1/function/"
    "all_cells_091024.h5"
)

OUTPUT_DIR = (
    "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
    "Jonathan Boulanger-Weill/Projects/Zebrafish_CLEM/"
    "5. Connectome_constrained_network_modeling"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cluster mapping
CLUSTER_TO_CELLTYPE: Dict[int, str] = {
    0: "slow_motion_integrator",
    1: "motion_onset",
    2: "motion_integrator",
}
CELLTYPE_TO_TITLE: Dict[str, str] = {
    "slow_motion_integrator": "Slow motion integrator",
    "motion_onset": "Motion onset",
    "motion_integrator": "Motion integrator",
}
VALID_CLUSTERS = [0, 1, 2]  # keep order for plotting

# Optional colors (your palette)
CELLTYPE_TO_COLOR: Dict[str, tuple] = {
    "motion_integrator": (232 / 255, 77 / 255, 138 / 255),
    "motion_onset": (100 / 255, 197 / 255, 235 / 255),
    "slow_motion_integrator": (127 / 255, 88 / 255, 175 / 255),
}

# Activity processing params
DT = 0.5        # seconds per sample (edit if needed)
PEAK_START = 40 # inclusive index
PEAK_END = 120  # exclusive index

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def find_hindbrain_neuron_ids(
    brain_reg_df: pd.DataFrame,
    region_col: str = REGION_COL_NAME,
) -> List[int]:
    """Find hindbrain neuron IDs (1-based) excluding mixed cerebellum/hindbrain label."""
    brain_reg_df = brain_reg_df.copy()
    brain_reg_df[region_col] = brain_reg_df[region_col].astype(str)

    mixed = brain_reg_df[region_col].str.contains(
        re.escape("cerebellum,rhombencephalon_(hindbrain)"), na=False
    )
    mixed_idx = set(brain_reg_df[mixed].index)

    hindbrain = brain_reg_df[region_col].str.contains(
        re.escape("rhombencephalon_(hindbrain)"), na=False
    ) & ~brain_reg_df.index.isin(mixed_idx)

    hind_idx = brain_reg_df[hindbrain].index
    return [i + 1 for i in hind_idx]  # 0-based row -> 1-based neuron_id


def _safe_savgol(x: np.ndarray, win: int = 21, poly: int = 3):
    """Savgol with safe window sizing; returns None if too short."""
    x = np.asarray(x, dtype=float).ravel()
    if len(x) < 5:
        return None
    win_len = min(win, len(x))
    if win_len % 2 == 0:
        win_len -= 1
    if win_len < 5:
        return None
    poly = min(poly, win_len - 2)
    if poly < 2:
        return None
    return savgol_filter(x, win_len, poly)


def get_hindbrain_ids_by_celltype(
    hindbrain_ids: List[int],
    hdf5_path: str,
) -> Dict[str, List[int]]:
    """Intersect hindbrain IDs with HDF5 and keep only clusters 0/1/2."""
    out: Dict[str, List[int]] = {CLUSTER_TO_CELLTYPE[c]: [] for c in VALID_CLUSTERS}
    with h5py.File(hdf5_path, "r") as h5:
        for nid in hindbrain_ids:
            key = f"neuron_{int(nid)}"
            if key not in h5:
                continue
            grp = h5[key]
            if "cluster_id" not in grp:
                continue
            cl = int(grp["cluster_id"][()])
            if cl in CLUSTER_TO_CELLTYPE:
                out[CLUSTER_TO_CELLTYPE[cl]].append(int(nid))
    return out


def extract_preferred_raw_traces(
    cell_ids: List[int],
    hdf5_path: str,
):
    """
    For each neuron:
      - Load average_activity_left/right
      - Smooth both traces
      - Choose preferred by peak in [PEAK_START:PEAK_END]
      - Return preferred RAW (smoothed) trace

    Returns:
      traces: (N, T) array
      time:   (T,) seconds
    """
    traces = []

    with h5py.File(hdf5_path, "r") as h5:
        for nid in cell_ids:
            key = f"neuron_{int(nid)}"
            if key not in h5:
                continue
            grp = h5[key]
            if ("average_activity_left" not in grp) or ("average_activity_right" not in grp):
                continue

            left = np.asarray(grp["average_activity_left"])
            right = np.asarray(grp["average_activity_right"])

            if len(left) < PEAK_END or len(right) < PEAK_END:
                continue

            left_s = _safe_savgol(left)
            right_s = _safe_savgol(right)
            if left_s is None or right_s is None:
                continue

            win = slice(PEAK_START, PEAK_END)
            left_peak = np.nanmax(left_s[win])
            right_peak = np.nanmax(right_s[win])

            pref = left_s if left_peak >= right_peak else right_s
            if np.all(~np.isfinite(pref)):
                continue

            traces.append(pref.astype(float))

    if not traces:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    # enforce consistent length across neurons
    T = min(len(t) for t in traces)
    traces = np.vstack([t[:T] for t in traces])
    time = np.arange(T) * DT
    return traces, time


def export_mean_trace_csv(
    traces: np.ndarray,
    time: np.ndarray,
    out_csv: str,
) -> None:
    """Save ONLY the mean preferred raw trace."""
    mean_tr = np.nanmean(traces, axis=0)
    df = pd.DataFrame({"time_s": time, "mean_preferred_raw": mean_tr})
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved mean trace CSV: {out_csv}")


def plot_three_clusters_side_by_side(
    traces_by_celltype: Dict[str, np.ndarray],
    time_by_celltype: Dict[str, np.ndarray],
    out_pdf: str,
) -> None:
    """
    Save a single PDF with 3 side-by-side SQUARE subplots.
    Each subplot: all traces (colored, faint) + mean (black, bold).
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

    for ax, cl in zip(axes, VALID_CLUSTERS):
        cell_type = CLUSTER_TO_CELLTYPE[cl]
        title = CELLTYPE_TO_TITLE.get(cell_type, cell_type)

        traces = traces_by_celltype.get(cell_type)
        time = time_by_celltype.get(cell_type)

        ax.set_box_aspect(1)  # ⬅️ square subplot

        if traces is None or traces.size == 0:
            ax.set_title(f"{title}\n(n=0)")
            ax.set_xlabel("Time (s)")
            continue

        color = CELLTYPE_TO_COLOR.get(cell_type, (0.5, 0.5, 0.5))

        # Individual traces
        for i in range(traces.shape[0]):
            ax.plot(time, traces[i, :], color=color, alpha=0.2, lw=0.2)

        # Mean trace — BLACK
        mean_tr = np.nanmean(traces, axis=0)
        ax.plot(time, mean_tr, color="black", lw=1)

        ax.set_title(f"{title}\n(n={traces.shape[0]})")
        ax.set_xlabel("Time (s)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Preferred raw activity (a.u.)")

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved 3-panel PDF: {out_pdf}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    brain_reg_df = pd.read_csv(BRAIN_REG_CSV_PATH)
    hindbrain_ids = find_hindbrain_neuron_ids(brain_reg_df, REGION_COL_NAME)
    print(f"[INFO] Hindbrain IDs from CSV: {len(hindbrain_ids)}")

    celltype_to_ids = get_hindbrain_ids_by_celltype(hindbrain_ids, HDF5_PATH)

    exported_types = 0
    out_base = os.path.join(OUTPUT_DIR, "hindbrain_motion_traces")
    os.makedirs(out_base, exist_ok=True)

    traces_by_celltype: Dict[str, np.ndarray] = {}
    time_by_celltype: Dict[str, np.ndarray] = {}

    for cl in VALID_CLUSTERS:
        cell_type = CLUSTER_TO_CELLTYPE[cl]
        ids = celltype_to_ids.get(cell_type, [])

        if not ids:
            print(f"[WARN] {cell_type}: 0 neurons -> skipping")
            traces_by_celltype[cell_type] = np.zeros((0, 0))
            time_by_celltype[cell_type] = np.zeros((0,))
            continue

        traces, time = extract_preferred_raw_traces(ids, HDF5_PATH)
        traces_by_celltype[cell_type] = traces
        time_by_celltype[cell_type] = time

        if traces.size == 0:
            print(f"[WARN] {cell_type}: no usable traces -> skipping")
            continue

        out_csv = os.path.join(out_base, f"{cell_type}_hindbrain_mean_preferred_raw.csv")
        export_mean_trace_csv(traces, time, out_csv)
        exported_types += 1
        print(f"[INFO] {cell_type}: exported mean (n={traces.shape[0]} neurons)")

    # 3-panel figure (single file)
    out_pdf = os.path.join(out_base, "hindbrain_preferred_raw_traces_3panel.pdf")
    plot_three_clusters_side_by_side(traces_by_celltype, time_by_celltype, out_pdf)

    print(f"[DONE] Exported {exported_types} hindbrain functional type(s) (clusters) to CSV.")


if __name__ == "__main__":
    main()