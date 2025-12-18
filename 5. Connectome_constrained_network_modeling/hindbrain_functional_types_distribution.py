#!/usr/bin/env python3
"""
hindbrain_functional_types_distribution.py

Identify neurons located in the rhombencephalon (hindbrain), extract their
3D positions from an HDF5 file, and:

1. Plot the distribution of their x, y, and z coordinates as stacked
   histograms for clusters 0, 1, 2, using their functional names:

      - Cluster 0: Slow motion integrator
      - Cluster 1: Motion onset
      - Cluster 2: Motion integrator

2. Interpolate the z-axis density (using the 12 µm sampling) and estimate
   the total number of each functional type in the hindbrain by integrating
   the interpolated density.

x and y coordinates are in micrometers.
z is converted to micrometers assuming 12 µm sampling between planes.

Usage:
    python hindbrain_functional_types_distribution.py
"""

import os
import re
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simpson

# Use Arial everywhere (falls back silently if not installed)
plt.rcParams["font.family"] = "Arial"

# -------------------------------------------------------------------
# User configuration
# -------------------------------------------------------------------

# Path to the brain-region annotation table
BRAIN_REG_CSV_PATH = (
    "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/"
    "hindbrain_structure_function/clem_zfish1/function/inputs/regions_021725.csv"
)

# Column name containing MapZebrain region strings
REGION_COL_NAME = "mapzebrain"

# HDF5 file with neuron data
HDF5_PATH = (
    "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/"
    "hb_connectome/hindbrain_structure_function/clem_zfish1/function/"
    "all_cells_091024.h5"
)

# Output directory for IDs and figures
OUTPUT_DIR = (
    "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
    "Jonathan Boulanger-Weill/Projects/Zebrafish_CLEM/"
    "5. Connectome_constrained_network_modeling"
)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors: local motion palette for this script
COLOR_CELL_TYPE_DICT: Dict[str, Tuple[float, float, float, float]] = {
    "motion_integrator":       (232 / 255, 77 / 255, 138 / 255, 0.7),  # Magenta
    "motion_onset":            (100 / 255, 197 / 255, 235 / 255, 0.7), # Light blue
    "slow_motion_integrator":  (127 / 255, 88 / 255, 175 / 255, 0.7),  # Purple
    "not_clustered":           (0.6, 0.6, 0.6, 0.7),                   # Gray
}

# Map cluster IDs to color and functional name
CLUSTER_TO_COLOR: Dict[int, Tuple[float, float, float, float]] = {
    0: COLOR_CELL_TYPE_DICT["slow_motion_integrator"],
    1: COLOR_CELL_TYPE_DICT["motion_onset"],
    2: COLOR_CELL_TYPE_DICT["motion_integrator"],
}

CLUSTER_TO_NAME: Dict[int, str] = {
    0: "Slow motion integrator",
    1: "Motion onset",
    2: "Motion integrator",
}

# Only these clusters are visualized and used for estimation
VALID_CLUSTERS = {0, 1, 2}

# Z-step in micrometers
Z_STEP_UM = 12.0


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def find_hindbrain_neuron_ids(brain_reg_df: pd.DataFrame,
                              region_col: str = REGION_COL_NAME
                              ) -> List[int]:
    """
    Return a list of neuron IDs (1-based indices) that belong to
    'rhombencephalon_(hindbrain)', excluding those that carry the mixed label
    'cerebellum,rhombencephalon_(hindbrain)'.
    """
    brain_reg_df = brain_reg_df.copy()
    brain_reg_df[region_col] = brain_reg_df[region_col].astype(str)

    # Indices of mixed label
    cereb_rhomb_mask = brain_reg_df[region_col].str.contains(
        re.escape("cerebellum,rhombencephalon_(hindbrain)"), na=False
    )
    cereb_rhomb_indices = brain_reg_df[cereb_rhomb_mask].index

    # Hindbrain-only
    hindbrain_mask = (
        brain_reg_df[region_col].str.contains(
            re.escape("rhombencephalon_(hindbrain)"), na=False
        )
        & ~brain_reg_df.index.isin(cereb_rhomb_indices)
    )

    hindbrain_indices = brain_reg_df[hindbrain_mask].index

    # Convert 0-based row index -> 1-based neuron ID
    neuron_ids = [idx + 1 for idx in hindbrain_indices]
    return neuron_ids


def extract_positions_for_neurons(h5_path: str,
                                  neuron_ids: List[int]):
    """
    Given a list of neuron IDs, open the HDF5 file and extract:

        - positions_x, positions_y, positions_z (one point per neuron)
        - cluster_ids

    Only clusters in VALID_CLUSTERS (0,1,2) are kept for visualization.
    'neuron_positions' can be either a length-3 vector [x, y, z] or
    an (N, 3) array; we then use the mean position per neuron.
    """
    positions_x = []
    positions_y = []
    positions_z = []
    clusters = []

    with h5py.File(h5_path, "r") as hdf_file:
        for neuron_id in neuron_ids:
            neuron_name = f"neuron_{neuron_id}"
            if neuron_name not in hdf_file:
                continue

            neuron_group = hdf_file[neuron_name]

            if "cluster_id" not in neuron_group or "neuron_positions" not in neuron_group:
                continue

            cluster = int(neuron_group["cluster_id"][()])

            # Only keep clusters 0, 1, 2 for visualization & estimation
            if cluster not in VALID_CLUSTERS:
                continue

            pos_arr = np.asarray(neuron_group["neuron_positions"][:])

            # Handle 1D (x, y, z) or (N, 3)
            if pos_arr.ndim == 1 and pos_arr.size == 3:
                x, y, z = pos_arr
            elif pos_arr.ndim == 2 and pos_arr.shape[1] == 3:
                x, y, z = pos_arr.mean(axis=0)
            else:
                continue  # unexpected shape

            positions_x.append(float(x))
            positions_y.append(float(y))
            positions_z.append(float(z))
            clusters.append(cluster)

    return np.array(positions_x), np.array(positions_y), np.array(positions_z), np.array(clusters)


def plot_xyz_distributions(x_um: np.ndarray,
                           y_um: np.ndarray,
                           z_um: np.ndarray,
                           clusters: np.ndarray,
                           output_path: str) -> None:
    """
    Plot stacked histograms of x, y, and z coordinates for hindbrain neurons.
    For each axis, bins are stacked by functional type (cluster 0,1,2).
    """

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    axes_labels = [
        "x position (µm)",
        "y position (µm)",
        f"z position (µm, {Z_STEP_UM:.0f} µm steps)",
    ]
    data_arrays = [x_um, y_um, z_um]

    n_bins = 25  # adjust as needed

    for ax, axis_label, data in zip(axes, axes_labels, data_arrays):
        # Define common bins for this axis
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        bins = np.linspace(data_min, data_max, n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0]

        bottom = np.zeros_like(bin_centers)

        # Stacked bars by cluster/functional type
        for cluster_id in sorted(np.unique(clusters)):
            mask = clusters == cluster_id
            if not np.any(mask):
                continue

            counts, _ = np.histogram(data[mask], bins=bins)
            color = CLUSTER_TO_COLOR.get(cluster_id, COLOR_CELL_TYPE_DICT["not_clustered"])
            label = CLUSTER_TO_NAME.get(cluster_id, "Not clustered")

            ax.bar(
                bin_centers,
                counts,
                width=bin_width,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=0.3,
                label=label,
            )
            bottom = bottom + counts

        ax.set_xlabel(axis_label, fontsize=8)
        ax.set_ylabel("Neuron count", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def estimate_cell_counts_from_z(z_positions_um: np.ndarray,
                                clusters: np.ndarray,
                                output_pdf: str,
                                soma_diameter_um: float = 5.0) -> Dict[str, Dict[str, float]]:
    """
    Interpolate along z using histogrammed counts and estimate total number
    of each functional type in the hindbrain.

    We assume:
        - Sampling step along z = Z_STEP_UM (e.g. 12 µm)
        - Effective sampling thickness per plane ~ soma_diameter_um (e.g. 5 µm)

    So the fraction of the volume that is actually sampled is:
        coverage_fraction = soma_diameter_um / Z_STEP_UM

    And the estimated total number of cells is:
        N_full ≈ N_observed / coverage_fraction = N_observed * (Z_STEP_UM / soma_diameter_um)

    We apply this scaling to the z-density, so the integral of the
    corrected density gives the estimated full count.
    """

    dz = Z_STEP_UM  # 12 µm

    # Bin edges (common for all clusters)
    zmin = float(np.nanmin(z_positions_um))
    zmax = float(np.nanmax(z_positions_um))
    bins = np.arange(zmin, zmax + dz, dz)

    # Coverage scaling
    coverage_fraction = soma_diameter_um / dz
    if coverage_fraction <= 0:
        raise ValueError("soma_diameter_um must be > 0.")
    scale_factor = dz / soma_diameter_um  # = 1 / coverage_fraction

    results: Dict[str, Dict[str, float]] = {}

    fig, ax = plt.subplots(figsize=(6, 4))

    for cluster_id in sorted(np.unique(clusters)):
        mask = clusters == cluster_id
        if not np.any(mask):
            continue

        z_vals = z_positions_um[mask]
        n_observed = float(len(z_vals))

        # Histogram for this cluster
        counts, edges = np.histogram(z_vals, bins=bins)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # Observed density: cells per µm within sampled planes
        density_observed = counts / dz

        # Correct density to account for unsampled gaps along z
        density_corrected = density_observed * scale_factor

        # Interpolate corrected density along z
        interpolator = interp1d(
            bin_centers,
            density_corrected,
            kind="linear",
            fill_value="extrapolate"
        )

        # High-resolution z-grid for integration
        z_hr = np.linspace(bin_centers.min(), bin_centers.max(), 5000)
        density_hr = interpolator(z_hr)

        # Integrate corrected density to get estimated full count
        estimated_full = simpson(density_hr, z_hr)

        func_name = CLUSTER_TO_NAME.get(cluster_id, "Not clustered")
        results[func_name] = {
            "observed": n_observed,
            "estimated_full": estimated_full,
        }

        # Plot corrected density curve
        color = CLUSTER_TO_COLOR.get(cluster_id, COLOR_CELL_TYPE_DICT["not_clustered"])
        ax.plot(z_hr, density_hr, label=func_name, color=color)

    ax.set_xlabel("z position (µm)", fontsize=10)
    ax.set_ylabel("Estimated cell density (cells / µm)", fontsize=10)
    ax.set_title(
        f"Interpolated hindbrain z-density\n"
        f"(corrected for {soma_diameter_um:.1f} µm soma, {Z_STEP_UM:.0f} µm z-step)",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    return results

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    # ---------------------------------------------------------------
    # Load brain region annotations
    # ---------------------------------------------------------------
    if not os.path.exists(BRAIN_REG_CSV_PATH):
        raise FileNotFoundError(f"Brain region CSV not found:\n{BRAIN_REG_CSV_PATH}")

    brain_reg_df = pd.read_csv(BRAIN_REG_CSV_PATH)

    if REGION_COL_NAME not in brain_reg_df.columns:
        raise KeyError(f"Expected column '{REGION_COL_NAME}' not found in CSV.")

    # ---------------------------------------------------------------
    # Identify hindbrain neuron IDs
    # ---------------------------------------------------------------
    hindbrain_ids = find_hindbrain_neuron_ids(brain_reg_df, region_col=REGION_COL_NAME)

    print(f"Found {len(hindbrain_ids)} neurons in rhombencephalon_(hindbrain) (before cluster filtering).")

    # Save IDs to a text file for later use
    ids_path = os.path.join(OUTPUT_DIR, "hindbrain_neuron_ids.txt")
    with open(ids_path, "w") as f:
        for nid in hindbrain_ids:
            f.write(f"{nid}\n")
    print(f"Hindbrain neuron IDs saved to: {ids_path}")

    # ---------------------------------------------------------------
    # Extract positions from HDF5 (only clusters 0,1,2)
    # ---------------------------------------------------------------
    if not os.path.exists(HDF5_PATH):
        raise FileNotFoundError(f"HDF5 file not found:\n{HDF5_PATH}")

    x, y, z, clusters = extract_positions_for_neurons(HDF5_PATH, hindbrain_ids)

    print(f"Extracted positions for {len(x)} neurons with clusters in {sorted(VALID_CLUSTERS)}.")

    # Convert to micrometers
    x_um = x  # already in µm
    y_um = y  # already in µm
    z_um = z * Z_STEP_UM  # convert slices to µm

    # Per-functional-type counts (sanity check)
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    print("\nFunctional type counts in hindbrain subset (clusters 0,1,2):")
    for cid, cnt in zip(unique_clusters, cluster_counts):
        name = CLUSTER_TO_NAME.get(cid, "Not clustered")
        print(f"  {name} (cluster {cid}): {cnt}")

    # ---------------------------------------------------------------
    # Plot x, y, z stacked distributions
    # ---------------------------------------------------------------
    fig_path = os.path.join(OUTPUT_DIR, "hindbrain_xyz_distributions.pdf")
    plot_xyz_distributions(x_um, y_um, z_um, clusters, fig_path)
    print(f"XYZ stacked distribution figure saved to: {fig_path}")

    # ---------------------------------------------------------------
    # Interpolate along z and estimate total cell counts
    # ---------------------------------------------------------------
    density_fig_path = os.path.join(OUTPUT_DIR, "hindbrain_z_interpolated_density.pdf")
    results = estimate_cell_counts_from_z(z_um, clusters, density_fig_path, soma_diameter_um=5.0)

    print("\nObserved vs estimated total neurons in hindbrain (corrected for z sampling):")
    print(f"{'Functional type':30s} {'observed':>10s} {'estimated_full':>15s}")
    for func_name, vals in results.items():
        print(f"{func_name:30s} {vals['observed']:10.0f} {vals['estimated_full']:15.1f}")

    print(f"\nInterpolated density figure saved to: {density_fig_path}")


if __name__ == "__main__":
    main()