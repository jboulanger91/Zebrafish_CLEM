#!/usr/bin/env python3
"""
count_motion_cell_types.py

Count motion-related cell types in the CLEM dataset, distinguishing
between native and LDA-predicted labels based on the 'lda' column.

We specifically count:
    - motion_integrator
        - separated by:
            - ipsilateral vs contralateral (projection classifier)
            - excitatory vs inhibitory (neurotransmitter classifier)
        - counts computed separately for native vs predicted
          and visualized as:
              * bar 1: native
              * bar 2: native + predicted (total)
    - slow_motion_integrator (native vs predicted)
    - motion_onset (native vs predicted)

We generate simple bar plots using the usual color scheme and save them as PDFs.

Usage:
    python count_motion_cell_types.py
"""

import os
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Use Arial everywhere (falls back silently if not installed)
plt.rcParams["font.family"] = "Arial"

# -------------------------------------------------------------------
# User configuration
# -------------------------------------------------------------------

CSV_PATH = (
    "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
    "Jonathan Boulanger-Weill/Projects/Zebrafish_CLEM/"
    "4. Morphology_based_prediction_of_neuronal_functional_types/"
    "all_reconstructed_neurons_with_LDA_predictions.csv"
)

# Correct color dictionary (as specified)
COLOR_CELL_TYPE_DICT: Dict[str, Tuple[float, float, float, float]] = {
    "ipsilateral_motion_integrator":   (254 / 255, 179 / 255, 38 / 255, 0.7),   # Yellow-orange
    "contralateral_motion_integrator": (232 / 255, 77 / 255, 138 / 255, 0.7),   # Magenta-pink
    "motion_onset":                    (100 / 255, 197 / 255, 235 / 255, 0.7),  # Light blue
    "slow_motion_integrator":          (127 / 255, 88 / 255, 175 / 255, 0.7),   # Purple
    "myelinated":                      (80 / 255, 220 / 255, 100 / 255, 0.7),   # Green
    "other_functional_types":          (220 / 255, 20 / 255, 60 / 255, 0.7),    # Crimson red
}

# Map motion-related plotting categories to colors (using ONLY new keys)
MOTION_COLOR_MAP: Dict[str, Tuple[float, float, float, float]] = {
    "motion_integrator_ipsi":   COLOR_CELL_TYPE_DICT["ipsilateral_motion_integrator"],
    "motion_integrator_contra": COLOR_CELL_TYPE_DICT["contralateral_motion_integrator"],
    "slow_motion_integrator":   COLOR_CELL_TYPE_DICT["slow_motion_integrator"],
    "motion_onset":             COLOR_CELL_TYPE_DICT["motion_onset"],
}


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def normalize_label_series(s: pd.Series) -> pd.Series:
    """
    Normalize textual labels for robust matching:
    - convert to string
    - lowercase
    - strip whitespace
    - replace spaces with underscores
    """
    return (
        s.astype(str)
         .str.lower()
         .str.strip()
         .str.replace(" ", "_", regex=False)
    )


def normalize_projection_series(s: pd.Series) -> pd.Series:
    """
    Normalize projection labels to 'ipsilateral' or 'contralateral'
    when possible, keeping original (normalized) text otherwise.
    """
    s_norm = normalize_label_series(s)

    def _map_proj(val: str) -> str:
        if "ipsi" in val:
            return "ipsilateral"
        if "contra" in val:
            return "contralateral"
        return val

    return s_norm.apply(_map_proj)


def normalize_neurotransmitter_series(s: pd.Series) -> pd.Series:
    """
    Normalize neurotransmitter labels to 'excitatory' or 'inhibitory'
    when possible, keeping original (normalized) text otherwise.
    """
    s_norm = normalize_label_series(s)

    def _map_nt(val: str) -> str:
        if "excit" in val:
            return "excitatory"
        if "inhib" in val:
            return "inhibitory"
        return val

    return s_norm.apply(_map_nt)


# -------------------------------------------------------------------
# Counting functions
# -------------------------------------------------------------------

def count_motion_integrator_subtypes_by_lda(
    df: pd.DataFrame,
    func_col_norm: str,
    proj_col_norm: str,
    nt_col_norm: str,
    lda_col_norm: str,
    lda_value: str,
) -> Dict[str, int]:
    """
    Count motion_integrator cells, restricted to rows where lda == lda_value,
    split by:
        - ipsilateral / contralateral
        - excitatory / inhibitory

    lda_value should be 'native' or 'predicted' (normalized).

    Returns a dict with keys:
        'ipsi_excit', 'ipsi_inhib', 'contra_excit', 'contra_inhib'
    """
    base_mask = (
        (df[func_col_norm] == "motion_integrator")
        & (df[lda_col_norm] == lda_value)
    )

    categories = [
        ("ipsi_excit",  "ipsilateral",   "excitatory"),
        ("ipsi_inhib",  "ipsilateral",   "inhibitory"),
        ("contra_excit","contralateral", "excitatory"),
        ("contra_inhib","contralateral", "inhibitory"),
    ]

    counts: Dict[str, int] = {}
    for key, proj_val, nt_val in categories:
        mask = (
            base_mask
            & (df[proj_col_norm] == proj_val)
            & (df[nt_col_norm] == nt_val)
        )
        counts[key] = int(mask.sum())

    return counts


def count_simple_class_by_lda(
    df: pd.DataFrame,
    func_col_norm: str,
    lda_col_norm: str,
    class_name: str,
    lda_value: str,
) -> int:
    """
    Count rows where functional classifier == class_name and lda == lda_value.
    """
    mask = (
        (df[func_col_norm] == class_name)
        & (df[lda_col_norm] == lda_value)
    )
    return int(mask.sum())


# -------------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------------

def _add_bar_labels(ax, bars):
    """
    Add integer value labels on top of bars.
    """
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_native_vs_total_subtypes(
    native_counts: Dict[str, int],
    total_counts: Dict[str, int],
    output_path: str = "motion_integrator_native_vs_total.pdf",
) -> None:
    """
    Plot native and (native + predicted) side by side for motion_integrator subtypes.
    """

    subtypes = ["ipsi_excit", "ipsi_inhib", "contra_excit", "contra_inhib"]
    x_labels = ["ipsi\nexc", "ipsi\ninh", "contra\nexc", "contra\ninh"]

    native_vals = [native_counts[k] for k in subtypes]
    total_vals  = [total_counts[k] for k in subtypes]

    x = np.arange(len(subtypes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))

    # Colors
    native_colors = []
    total_colors = []
    for key in subtypes:
        if key.startswith("ipsi"):
            c = COLOR_CELL_TYPE_DICT["ipsilateral_motion_integrator"]
        else:
            c = COLOR_CELL_TYPE_DICT["contralateral_motion_integrator"]
        native_colors.append(c)
        total_colors.append(c)

    bars1 = ax.bar(
        x - width/2, native_vals, width,
        label="Native", color=native_colors
    )
    bars2 = ax.bar(
        x + width/2, total_vals, width,
        label="Total (native + predicted)", color=total_colors, alpha=0.5
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Number of cells")
    ax.set_title("motion_integrator subtypes: native vs total")

    _add_bar_labels(ax, bars1)
    _add_bar_labels(ax, bars2)

    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_native_vs_total_classes(
    native_counts: Dict[str, int],
    total_counts: Dict[str, int],
    output_path: str = "motion_classes_native_vs_total.pdf",
) -> None:
    """
    Plot native and (native + predicted) side by side for major motion classes.
    """

    classes = ["motion_integrator", "slow_motion_integrator", "motion_onset"]
    x_labels = ["MI", "slow MI", "onset"]

    native_vals = [native_counts[c] for c in classes]
    total_vals  = [total_counts[c] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = [
        COLOR_CELL_TYPE_DICT["ipsilateral_motion_integrator"],
        COLOR_CELL_TYPE_DICT["slow_motion_integrator"],
        COLOR_CELL_TYPE_DICT["motion_onset"],
    ]

    bars1 = ax.bar(
        x - width/2, native_vals, width,
        color=colors, label="Native"
    )
    bars2 = ax.bar(
        x + width/2, total_vals, width,
        color=colors, alpha=0.5,
        label="Total (native + predicted)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Number of cells")
    ax.set_title("Motion-related classes: native vs total")

    _add_bar_labels(ax, bars1)
    _add_bar_labels(ax, bars2)

    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def main() -> None:
    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found:\n{CSV_PATH}")

    # sep=None lets pandas infer comma vs tab etc.
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")

    # -----------------------------------------------------------------
    # Basic column checks
    # -----------------------------------------------------------------
    required_cols = [
        "functional classifier",
        "neurotransmitter classifier",
        "projection classifier",
        "lda",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in CSV.")

    # -----------------------------------------------------------------
    # Normalize columns for robust matching
    # -----------------------------------------------------------------
    df["func_norm"] = normalize_label_series(df["functional classifier"])
    df["proj_norm"] = normalize_projection_series(df["projection classifier"])
    df["nt_norm"]   = normalize_neurotransmitter_series(df["neurotransmitter classifier"])
    df["lda_norm"]  = normalize_label_series(df["lda"])

    # Keep only rows where lda is 'native' or 'predicted'
    lda_values_of_interest = ["native", "predicted"]
    df = df[df["lda_norm"].isin(lda_values_of_interest)].copy()

    # -----------------------------------------------------------------
    # Count motion_integrator subtypes (native vs predicted)
    # -----------------------------------------------------------------
    native_motion_sub = count_motion_integrator_subtypes_by_lda(
        df,
        func_col_norm="func_norm",
        proj_col_norm="proj_norm",
        nt_col_norm="nt_norm",
        lda_col_norm="lda_norm",
        lda_value="native",
    )
    pred_motion_sub = count_motion_integrator_subtypes_by_lda(
        df,
        func_col_norm="func_norm",
        proj_col_norm="proj_norm",
        nt_col_norm="nt_norm",
        lda_col_norm="lda_norm",
        lda_value="predicted",
    )

    # Total (native + predicted) for subtypes
    total_motion_sub = {
        key: native_motion_sub.get(key, 0) + pred_motion_sub.get(key, 0)
        for key in set(native_motion_sub.keys()) | set(pred_motion_sub.keys())
    }

    # -----------------------------------------------------------------
    # Count motion_integrator, slow_motion_integrator, motion_onset
    # separately for native vs predicted
    # -----------------------------------------------------------------
    motion_classes = ["motion_integrator", "slow_motion_integrator", "motion_onset"]

    native_simple_counts = {
        cls: count_simple_class_by_lda(
            df,
            func_col_norm="func_norm",
            lda_col_norm="lda_norm",
            class_name=cls,
            lda_value="native",
        )
        for cls in motion_classes
    }

    pred_simple_counts = {
        cls: count_simple_class_by_lda(
            df,
            func_col_norm="func_norm",
            lda_col_norm="lda_norm",
            class_name=cls,
            lda_value="predicted",
        )
        for cls in motion_classes
    }

    total_simple_counts = {
        cls: native_simple_counts.get(cls, 0) + pred_simple_counts.get(cls, 0)
        for cls in motion_classes
    }

# -----------------------------------------------------------------
    # Generate simplified final table
    # -----------------------------------------------------------------

    table_rows = [
        ("imi+", native_motion_sub["ipsi_excit"],   total_motion_sub["ipsi_excit"]),
        ("imi-", native_motion_sub["ipsi_inhib"],   total_motion_sub["ipsi_inhib"]),
        ("cmi+", native_motion_sub["contra_excit"], total_motion_sub["contra_excit"]),
        ("cmi-", native_motion_sub["contra_inhib"], total_motion_sub["contra_inhib"]),
        ("smi",  native_simple_counts["slow_motion_integrator"], total_simple_counts["slow_motion_integrator"]),
        ("mon",  native_simple_counts["motion_onset"],           total_simple_counts["motion_onset"]),
    ]

    # Compute totals
    total_native = sum(row[1] for row in table_rows)
    total_total  = sum(row[2] for row in table_rows)

    # Add TOTAL row
    table_rows.append(("total", total_native, total_total))

    # Print table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'type':<6}{'native':>10}{'native+pred':>15}")
    for label, val_n, val_t in table_rows:
        print(f"{label:<6}{val_n:>10}{val_t:>15}")

    # -----------------------------------------------------------------
    # Plots: native vs total (PDF)
    # -----------------------------------------------------------------
    plot_native_vs_total_subtypes(
        native_motion_sub,
        total_motion_sub,
        output_path="motion_integrator_native_vs_total.pdf",
    )

    plot_native_vs_total_classes(
        native_simple_counts,
        total_simple_counts,
        output_path="motion_classes_native_vs_total.pdf",
    )

    print("\nSaved plots:")
    print("  - motion_integrator_native_vs_total.pdf")
    print("  - motion_classes_native_vs_total.pdf")


if __name__ == "__main__":
    main()