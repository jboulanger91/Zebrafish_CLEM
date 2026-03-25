"""Classification visualization functions for confusion matrices and predictions.

This module provides publication-quality visualizations for analyzing machine
learning classification results on hindbrain neuron data. Includes:
    - Single confusion matrices with customizable formatting
    - Multi-panel confusion matrix grids for comparing modality combinations
    - Prediction summary visualizations with confidence distributions

Design Philosophy:
    - Stateless functions for easy reuse
    - Consistent styling across all plots
    - Both raster (PNG) and vector (PDF) output
    - Integration with src.viz.colors for unified color schemes

Usage:
    >>> from src.viz.classification_plots import plot_confusion_matrix
    >>> import matplotlib.pyplot as plt
    >>>
    >>> fig, ax = plt.subplots()
    >>> plot_confusion_matrix(
    ...     cm, ["MON", "cMI", "iMI", "SMI"], ax=ax,
    ...     title="Train: CLEM, Test: PA"
    ... )
    >>> plt.savefig("cm.png")

Functions:
    plot_confusion_matrix: Single confusion matrix with diagonal highlighting
    plot_confusion_matrix_grid: Grid of matrices for modality comparisons
    plot_prediction_summary: Multi-panel prediction analysis visualization

Author: Florian Kämpf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    ax: plt.Axes | None = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    cmap: str = "Blues",
    spines_red: bool = False,
    fontsize: int = 12,
) -> plt.Axes:
    """Plot a single confusion matrix with diagonal highlighting.

    Creates a heatmap visualization of a confusion matrix with:
        - Color-coded cells showing classification results
        - Green rectangles highlighting correctly classified instances (diagonal)
        - Optional normalization by true labels (rows)
        - Customizable color scheme and text size

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes).
        Element [i, j] represents samples with true label i predicted as label j.
    class_names : list[str]
        Names of classes for axis labels (e.g., ["MON", "cMI", "iMI", "SMI"]).
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure with size (8, 6).
    title : str, optional
        Plot title. Default: "Confusion Matrix"
    normalize : bool, optional
        If True, normalize confusion matrix by row (true labels) to show
        percentages instead of counts. Default: False
    cmap : str, optional
        Matplotlib colormap name for heatmap. Default: "Blues"
    spines_red : bool, optional
        If True, highlight plot borders in red (useful for emphasizing specific
        comparisons in grid layouts). Default: False
    fontsize : int, optional
        Base font size for annotations. Labels will be fontsize+2, title
        will be fontsize+4. Default: 12

    Returns
    -------
    plt.Axes
        The axes object containing the plotted confusion matrix.

    Notes
    -----
    - Diagonal elements (correct predictions) are highlighted with green borders
    - Color intensity represents count (or proportion if normalized)
    - Output is suitable for publication-quality figures

    Examples
    --------
    Plot a single confusion matrix:

    >>> cm = np.array([[45, 3, 2, 0], [5, 38, 4, 3], [1, 2, 42, 5], [0, 1, 3, 46]])
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> plot_confusion_matrix(
    ...     cm,
    ...     ["MON", "cMI", "iMI", "SMI"],
    ...     ax=ax,
    ...     title="Train: CLEM, Test: PA",
    ...     normalize=True
    ... )
    >>> plt.show()

    Use in a custom layout:

    >>> fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    >>> plot_confusion_matrix(cm1, classes, ax=axes[0], title="Model A")
    >>> plot_confusion_matrix(cm2, classes, ax=axes[1], title="Model B")
    >>> plt.tight_layout()
    >>> plt.savefig("comparison.png", dpi=300)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize if requested
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    # Plot heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"fontsize": fontsize},
    )

    # Set labels
    ax.set_xlabel("Predicted Label", fontsize=fontsize + 2)
    ax.set_ylabel("True Label", fontsize=fontsize + 2)
    ax.set_title(title, fontsize=fontsize + 4, pad=10)

    # Highlight diagonal (correct predictions)
    for i in range(len(class_names)):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor="green", lw=2))

    # Optionally highlight borders
    if spines_red:
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(3)

    return ax


def plot_confusion_matrix_grid(
    confusion_matrices: dict[str, np.ndarray],
    class_names: list[str],
    save_path: Path,
    suptitle: str = "Confusion Matrices",
    modalities: list[str] | None = None,
    normalize: bool = False,
    highlight_combinations: list[tuple[str, str]] | None = None,
) -> None:
    """Plot grid of confusion matrices for comparing modality combinations.

    Creates a comprehensive grid showing classifier performance across all
    combinations of training and testing modalities. Useful for:
        - Cross-modal generalization analysis (train on CLEM, test on PA)
        - Within-modality performance (train and test on same modality)
        - Identifying modality-specific biases

    Grid layout example (3x3 for ['all', 'clem', 'pa']):

                    Test: All    Test: CLEM   Test: PA
        Train: All    [CM]         [CM]         [CM]
        Train: CLEM   [CM]         [CM]         [CM]
        Train: PA     [CM]         [CM]         [CM]

    Parameters
    ----------
    confusion_matrices : dict[str, np.ndarray]
        Dictionary mapping '{train}_{test}' to confusion matrix array.
        Keys should match pattern from modalities list (e.g., "all_clem", "pa_pa").
    class_names : list[str]
        Names of classes for axis labels.
    save_path : Path
        Path where output figure will be saved. Creates both PNG and PDF versions.
        Parent directories will be created if they don't exist.
    suptitle : str, optional
        Overall title for the entire grid. Default: "Confusion Matrices"
    modalities : list[str], optional
        List of modality names in order for grid axes.
        Default: ["all", "clem", "pa"]
    normalize : bool, optional
        If True, normalize each confusion matrix by row. Default: False
    highlight_combinations : list[tuple[str, str]], optional
        List of (train, test) tuples to highlight with red borders.
        Default: [("all", "clem"), ("clem", "clem"), ("pa", "pa")]

    Returns
    -------
    None
        Saves figure to disk at save_path and save_path.with_suffix('.pdf').

    Notes
    -----
    - Missing combinations will print a warning and leave subplot blank
    - Figure size automatically scales with grid dimensions (20x20 for 3x3)
    - Console output includes progress and output paths
    - Both PNG (300 dpi) and PDF versions are saved

    Examples
    --------
    Standard 3x3 grid for modality analysis:

    >>> cms = {
    ...     "all_all": cm_all_all,
    ...     "all_clem": cm_all_clem,
    ...     "all_pa": cm_all_pa,
    ...     "clem_all": cm_clem_all,
    ...     "clem_clem": cm_clem_clem,
    ...     "clem_pa": cm_clem_pa,
    ...     "pa_all": cm_pa_all,
    ...     "pa_clem": cm_pa_clem,
    ...     "pa_pa": cm_pa_pa,
    ... }
    >>> plot_confusion_matrix_grid(
    ...     cms,
    ...     ["MON", "cMI", "iMI", "SMI"],
    ...     Path("confusion_matrices/modality_comparison.png"),
    ...     suptitle="Cross-Modal Classification Performance",
    ...     highlight_combinations=[("all", "clem"), ("clem", "clem")],
    ... )

    Custom modalities (e.g., CLEM vs EM):

    >>> plot_confusion_matrix_grid(
    ...     clem_em_cms,
    ...     class_names,
    ...     Path("clem_em_comparison.png"),
    ...     modalities=["clem", "em"],
    ...     normalize=True,
    ... )
    """
    # Handle mutable default
    if modalities is None:
        modalities = ["all", "clem", "pa"]

    print("\n📊 Plotting confusion matrix grid...")
    print(f"   Grid size: {len(modalities)}x{len(modalities)}")
    print(f"   Output path: {save_path}")

    # Create highlight set
    if highlight_combinations is None:
        highlight_combinations = [("all", "clem"), ("clem", "clem"), ("pa", "pa")]
    highlight_set = set(highlight_combinations)

    # Create figure
    fig, axes = plt.subplots(len(modalities), len(modalities), figsize=(20, 20))

    # Plot each combination
    for i, train_mod in enumerate(modalities):
        for j, test_mod in enumerate(modalities):
            key = f"{train_mod}_{test_mod}"

            if key not in confusion_matrices:
                print(f"   Warning: Missing CM for {key}")
                continue

            cm = confusion_matrices[key]
            title = f"Train: {train_mod.upper()}\nTest: {test_mod.upper()}"
            spines_red = (train_mod, test_mod) in highlight_set

            plot_confusion_matrix(
                cm,
                class_names,
                ax=axes[i, j],
                title=title,
                normalize=normalize,
                spines_red=spines_red,
            )

    # Set overall title
    fig.suptitle(suptitle, fontsize=24, y=0.995)

    # Adjust layout
    plt.tight_layout()

    # Save figures
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")

    print("✅ Confusion matrix grid saved\n")
    plt.close()


def plot_prediction_summary(
    predictions: pd.DataFrame,
    save_path: Path,
    title: str = "Prediction Summary",
) -> None:
    """Create comprehensive multi-panel visualization of prediction results.

    Generates a publication-quality figure with three panels:
        1. Prediction counts (bar chart comparing scaled vs unscaled)
        2. Confidence distribution (histogram of maximum probabilities)
        3. Probability heatmap (all classes vs all cells)

    Useful for:
        - Quality control of predictions
        - Identifying low-confidence predictions that need review
        - Comparing effects of probability scaling/calibration
        - Visualizing class balance in predictions
        - Spotting systematic biases in probability assignments

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame containing prediction results with columns:
            - 'prediction': Unscaled predicted class labels
            - 'prediction_scaled': Scaled/calibrated predicted class labels
            - '*_proba': Probability columns for each class (e.g., 'MON_proba')
            - '*_proba_scaled': Optional scaled probability columns
    save_path : Path
        Path where output figure will be saved (PNG format, 300 dpi).
        Parent directories will be created if they don't exist.
    title : str, optional
        Overall title for the figure. Default: "Prediction Summary"

    Returns
    -------
    None
        Saves figure to disk at save_path.

    Notes
    -----
    Panel Descriptions:
        - Top-left: Side-by-side bar chart showing class distribution before
          and after probability scaling. Useful for detecting calibration shifts.
        - Top-right: Histogram of maximum predicted probabilities across all
          samples. Mean line indicates average confidence. Low mean suggests
          classifier uncertainty.
        - Bottom: Heatmap showing probability assigned to each class (rows)
          for each cell (columns). Vertical stripes indicate cells with
          ambiguous predictions.

    The function automatically:
        - Filters probability columns (excludes scaled versions for main heatmap)
        - Calculates maximum probability per sample for confidence analysis
        - Creates directory structure if needed
        - Handles missing columns gracefully

    Examples
    --------
    Basic usage with prediction DataFrame:

    >>> pred_df = pd.DataFrame({
    ...     'prediction': ['MON', 'cMI', 'iMI', 'SMI', 'MON'],
    ...     'prediction_scaled': ['MON', 'cMI', 'iMI', 'iMI', 'MON'],
    ...     'MON_proba': [0.85, 0.12, 0.05, 0.03, 0.92],
    ...     'cMI_proba': [0.10, 0.78, 0.15, 0.08, 0.05],
    ...     'iMI_proba': [0.03, 0.08, 0.75, 0.18, 0.02],
    ...     'SMI_proba': [0.02, 0.02, 0.05, 0.71, 0.01],
    ... })
    >>> plot_prediction_summary(
    ...     pred_df,
    ...     Path("results/prediction_summary.png"),
    ...     title="EM Cell Type Predictions"
    ... )

    After model calibration:

    >>> plot_prediction_summary(
    ...     calibrated_predictions,
    ...     Path("results/calibrated_summary.png"),
    ...     title="Calibrated Predictions (Platt Scaling)"
    ... )
    """
    print("\n📊 Creating prediction summary visualization...")
    print(f"   Output: {save_path}")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Prediction counts comparison (scaled vs unscaled)
    ax1 = fig.add_subplot(gs[0, 0])
    unscaled_counts = predictions["prediction"].value_counts()
    scaled_counts = predictions["prediction_scaled"].value_counts()

    x = np.arange(len(unscaled_counts))
    width = 0.35

    ax1.bar(x - width / 2, unscaled_counts.values, width, label="Unscaled", alpha=0.8)
    ax1.bar(x + width / 2, scaled_counts.values, width, label="Scaled", alpha=0.8)

    ax1.set_xlabel("Cell Type", fontsize=14)
    ax1.set_ylabel("Count", fontsize=14)
    ax1.set_title("Prediction Counts", fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(unscaled_counts.index, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 2. Maximum probability distribution (confidence)
    ax2 = fig.add_subplot(gs[0, 1])
    proba_cols = [
        col for col in predictions.columns if col.endswith("_proba") and "scaled" not in col
    ]
    max_proba = predictions[proba_cols].max(axis=1)

    ax2.hist(max_proba, bins=30, edgecolor="black", alpha=0.7)
    ax2.axvline(
        max_proba.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {max_proba.mean():.2f}",
    )
    ax2.set_xlabel("Maximum Probability", fontsize=14)
    ax2.set_ylabel("Count", fontsize=14)
    ax2.set_title("Prediction Confidence Distribution", fontsize=16)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # 3. Probability heatmap (all classes vs all cells)
    ax3 = fig.add_subplot(gs[1, :])
    proba_matrix = predictions[proba_cols].values.T

    im = ax3.imshow(proba_matrix, aspect="auto", cmap="viridis")
    ax3.set_yticks(range(len(proba_cols)))
    ax3.set_yticklabels([col.replace("_proba", "") for col in proba_cols])
    ax3.set_xlabel("Cell Index", fontsize=14)
    ax3.set_ylabel("Cell Type", fontsize=14)
    ax3.set_title("Prediction Probabilities Heatmap", fontsize=16)

    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label("Probability", fontsize=12)

    fig.suptitle(title, fontsize=20, y=0.995)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print("✅ Prediction summary saved\n")
    plt.close()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "plot_confusion_matrix",
    "plot_confusion_matrix_grid",
    "plot_prediction_summary",
]
