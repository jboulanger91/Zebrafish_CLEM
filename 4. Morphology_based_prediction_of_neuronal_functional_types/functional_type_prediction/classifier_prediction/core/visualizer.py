"""Visualization Module for Cell Type Prediction Results.

Provides publication-quality visualizations for:
    - **Confusion Matrices**: Heatmap plots showing classification performance
      across different train/test modality combinations (PA, CLEM, EM).

Design Philosophy (Zen of Python):
    - Simple is better than complex: Each class has a focused responsibility
    - Beautiful is better than ugly: Consistent styling and color schemes
    - Sparse is better than dense: Clean layouts with informative annotations

Usage:
    All plotter classes use static methods for stateless operation.
    Output is saved to specified paths in both PNG (raster) and PDF (vector).

Example:
    >>> from visualizer import ConfusionMatrixPlotter
    >>>
    >>> # Plot confusion matrix
    >>> fig, ax = plt.subplots()
    >>> ConfusionMatrixPlotter.plot_single_cm(
    ...     cm, ["MON", "cMI", "iMI", "SMI"], ax=ax, title="Train: CLEM, Test: PA"
    ... )
    >>> plt.savefig("cm.png")

Classes:
    ConfusionMatrixPlotter: Single and grid confusion matrix visualizations.

Author: Florian Kämpf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


class ConfusionMatrixPlotter:
    """Handles confusion matrix visualization.

    Provides clean interface for plotting confusion matrices
    with customizable formatting and annotations.

    Methods
    -------
    plot_single_cm(cm, class_names, ax, title, normalize)
        Plot a single confusion matrix
    plot_cm_grid(cms, class_names, save_path, suptitle)
        Plot 3x3 grid of confusion matrices for modality combinations
    """

    @staticmethod
    def plot_single_cm(
        cm: np.ndarray,
        class_names: list[str],
        ax: plt.Axes | None = None,
        title: str = "Confusion Matrix",
        normalize: bool = False,
        cmap: str = "Blues",
        spines_red: bool = False,
        fontsize: int = 12,
    ) -> plt.Axes:
        """Plot a single confusion matrix.

        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix (n_classes, n_classes)
        class_names : List[str]
            Names of classes for axis labels
        ax : plt.Axes, optional
            Matplotlib axes to plot on. If None, creates new figure.
        title : str, optional
            Plot title. Default: 'Confusion Matrix'
        normalize : bool, optional
            If True, normalize by true labels (rows). Default: False
        cmap : str, optional
            Colormap name. Default: 'Blues'
        spines_red : bool, optional
            If True, highlight borders in red. Default: False
        fontsize : int, optional
            Font size for annotations. Default: 12

        Returns
        -------
        plt.Axes
            The axes with the plotted confusion matrix

        Examples
        --------
        >>> fig, ax = plt.subplots()
        >>> ConfusionMatrixPlotter.plot_single_cm(
        ...     cm, ["MON", "cMI", "iMI", "SMI"], ax=ax, title="Train=CLEM, Test=PA"
        ... )
        >>> plt.show()
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

    @staticmethod
    def plot_cm_grid(
        confusion_matrices: dict[str, np.ndarray],
        class_names: list[str],
        save_path: Path,
        suptitle: str = "Confusion Matrices",
        modalities: list[str] = None,
        normalize: bool = False,
        highlight_combinations: list[tuple[str, str]] | None = None,
    ) -> None:
        """Plot 3x3 grid of confusion matrices for modality combinations.

        Creates a grid showing train→test performance for all
        combinations of training and testing modalities.

        Parameters
        ----------
        confusion_matrices : Dict[str, np.ndarray]
            Dictionary mapping '{train}_{test}' to confusion matrix
        class_names : List[str]
            Names of classes
        save_path : Path
            Path where figure will be saved
        suptitle : str, optional
            Overall title for the grid. Default: 'Confusion Matrices'
        modalities : List[str], optional
            List of modalities. Default: ['all', 'clem', 'pa']
        normalize : bool, optional
            If True, normalize each CM. Default: False
        highlight_combinations : List[Tuple[str, str]], optional
            List of (train, test) combinations to highlight in red

        Notes
        -----
        The grid layout is:
                    Test: All    Test: CLEM   Test: PA
        Train: All    [CM]         [CM]         [CM]
        Train: CLEM   [CM]         [CM]         [CM]
        Train: PA     [CM]         [CM]         [CM]

        Saves both PNG and PDF versions.

        Examples
        --------
        >>> cms = {
        ...     "all_all": cm1,
        ...     "all_clem": cm2,
        ...     "all_pa": cm3,
        ...     "clem_all": cm4,
        ...     "clem_clem": cm5,
        ...     "clem_pa": cm6,
        ...     "pa_all": cm7,
        ...     "pa_clem": cm8,
        ...     "pa_pa": cm9,
        ... }
        >>> ConfusionMatrixPlotter.plot_cm_grid(
        ...     cms,
        ...     ["MON", "cMI", "iMI", "SMI"],
        ...     Path("confusion_matrices/grid.png"),
        ...     highlight_combinations=[("all", "clem"), ("clem", "clem")],
        ... )
        """
        # Handle mutable default
        if modalities is None:
            modalities = ["all", "clem", "pa"]
        print("\n Plotting confusion matrix grid...")
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

                ConfusionMatrixPlotter.plot_single_cm(
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

        print(" Confusion matrix grid saved\n")
        plt.close()

