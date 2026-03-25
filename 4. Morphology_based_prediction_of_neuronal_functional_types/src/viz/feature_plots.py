"""Feature analysis visualization for machine learning models.

Provides publication-quality visualizations for feature selection and
importance analysis:
    - **RFE Curves**: Performance vs. number of features for Recursive
      Feature Elimination
    - **Feature Importance**: Bar charts showing most influential features
    - **SelectKBest Comparison**: Side-by-side comparison of different
      feature selection scoring functions

Design Philosophy:
    - Modular functions that can be used independently
    - Consistent styling through integration with src.viz.colors
    - Support for both PNG (raster) and PDF (vector) output
    - Clear annotations with maximum values highlighted

Usage:
    >>> from src.viz.feature_plots import plot_rfe_curve, plot_feature_importance
    >>>
    >>> # Plot RFE performance curve
    >>> plot_rfe_curve(
    ...     n_features=range(1, 11),
    ...     scores=[75.2, 82.1, 88.5, 90.2, 89.8, 88.1, 87.5, 86.9, 86.2, 85.8],
    ...     output_path="results/rfe_curve.png",
    ...     title="Feature Selection Performance"
    ... )
    >>>
    >>> # Plot feature importance
    >>> plot_feature_importance(
    ...     feature_names=["length", "width", "volume", "surface_area"],
    ...     importances=np.array([0.45, 0.30, 0.15, 0.10]),
    ...     output_path="results/feature_importance.png",
    ...     top_n=10
    ... )

Functions:
    plot_rfe_curve: Plot Recursive Feature Elimination performance curve
    plot_feature_importance: Plot feature importance bar chart
    plot_selectkbest_comparison: Compare SelectKBest scoring functions

Author: Florian Kämpf
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_rfe_curve(
    n_features: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    output_path: str | Path,
    title: str = "RFE Feature Selection",
    xlabel: str = "Number of Features",
    ylabel: str = "Accuracy (%)",
    estimator_name: str | None = None,
    color: str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """Plot Recursive Feature Elimination performance curve.

    Creates a line plot showing classification performance as a function
    of the number of features used. Highlights the maximum performance
    with a red star and annotation.

    Parameters
    ----------
    n_features : Sequence[int] or np.ndarray
        Number of features at each evaluation point
    scores : Sequence[float] or np.ndarray
        Performance scores (e.g., accuracy percentages) corresponding
        to each n_features value
    output_path : str or Path
        Path where figure will be saved (both PNG and PDF)
    title : str, optional
        Plot title. Default: "RFE Feature Selection"
    xlabel : str, optional
        X-axis label. Default: "Number of Features"
    ylabel : str, optional
        Y-axis label. Default: "Accuracy (%)"
    estimator_name : str, optional
        Name of the estimator (shown in legend). If None, no legend.
    color : str, optional
        Line color. If None, uses matplotlib default color cycle.
    figsize : Tuple[int, int], optional
        Figure size (width, height). Default: (12, 8)

    Returns
    -------
    None
        Saves figure to output_path as PNG and PDF

    Examples
    --------
    Single estimator:
    >>> plot_rfe_curve(
    ...     n_features=range(1, 11),
    ...     scores=[75.2, 82.1, 88.5, 90.2, 89.8, 88.1, 87.5, 86.9, 86.2, 85.8],
    ...     output_path="rfe_curve.png",
    ...     title="LDA Feature Selection",
    ...     estimator_name="LDA"
    ... )

    Multiple estimators (call multiple times on same figure):
    >>> fig, ax = plt.subplots(figsize=(12, 8))
    >>> for name, scores in results.items():
    ...     plot_rfe_curve(
    ...         range(1, len(scores)+1), scores, "rfe_multi.png",
    ...         estimator_name=name
    ...     )

    Notes
    -----
    - Automatically highlights the maximum score with a red star
    - Saves both PNG (300 dpi) and PDF versions
    - Creates parent directories if they don't exist
    """
    print("\n📈 Plotting RFE curve...")
    print(f"   Output: {output_path}")

    # Convert to Path object
    output_path = Path(output_path)

    # Convert inputs to arrays for consistent handling
    n_features = np.asarray(n_features)
    scores = np.asarray(scores)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot curve
    plot_kwargs = {"marker": "o", "linewidth": 2}
    if color is not None:
        plot_kwargs["color"] = color
    if estimator_name is not None:
        plot_kwargs["label"] = estimator_name

    ax.plot(n_features, scores, **plot_kwargs)

    # Mark maximum
    max_idx = np.argmax(scores)
    max_score = scores[max_idx]
    max_n_features = n_features[max_idx]

    ax.plot(max_n_features, max_score, "r*", markersize=15)
    ax.annotate(
        f"{max_score:.1f}%",
        (max_n_features, max_score),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    # Configure axes
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, pad=15)
    if estimator_name is not None:
        ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    print("✅ RFE curve saved\n")
    plt.close()


def plot_rfe_comparison(
    results: dict[str, Sequence[float] | np.ndarray],
    output_path: str | Path,
    title: str = "RFE Feature Selection Comparison",
    xlabel: str = "Number of Features",
    ylabel: str = "Accuracy (%)",
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """Plot RFE curves for multiple estimators on the same figure.

    Convenience function for comparing multiple estimators. Each estimator
    gets a different color from the default color cycle.

    Parameters
    ----------
    results : Dict[str, Sequence[float] or np.ndarray]
        Dictionary mapping estimator name to scores
    output_path : str or Path
        Path where figure will be saved
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    figsize : Tuple[int, int], optional
        Figure size. Default: (12, 8)

    Returns
    -------
    None
        Saves figure to output_path

    Examples
    --------
    >>> results = {
    ...     "LDA": [75.2, 82.1, 88.5, 90.2, 89.8],
    ...     "AdaBoost": [71.3, 78.9, 85.1, 87.3, 86.9],
    ...     "RandomForest": [73.5, 80.2, 86.3, 88.1, 87.5],
    ... }
    >>> plot_rfe_comparison(results, "rfe_comparison.png")
    """
    print(f"\n📈 Plotting RFE comparison ({len(results)} estimators)...")
    print(f"   Output: {output_path}")

    # Convert to Path object
    output_path = Path(output_path)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each estimator
    for estimator_name, scores in results.items():
        scores = np.asarray(scores)
        n_features = np.arange(1, len(scores) + 1)

        ax.plot(n_features, scores, marker="o", label=estimator_name, linewidth=2)

        # Mark maximum
        max_idx = np.argmax(scores)
        max_score = scores[max_idx]
        ax.plot(max_idx + 1, max_score, "r*", markersize=15)
        ax.annotate(
            f"{max_score:.1f}%",
            (max_idx + 1, max_score),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
        )

    # Configure axes
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, pad=15)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    print("✅ RFE comparison saved\n")
    plt.close()


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: np.ndarray | Sequence[float],
    output_path: str | Path,
    title: str = "Feature Importance",
    top_n: int = 20,
    color: str | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Plot feature importance as horizontal bar chart.

    Shows the most important features ranked by their importance scores.
    Useful for understanding which features drive model predictions.

    Parameters
    ----------
    feature_names : Sequence[str]
        Names of all features
    importances : np.ndarray or Sequence[float]
        Importance score for each feature (same length as feature_names)
    output_path : str or Path
        Path where figure will be saved (both PNG and PDF)
    title : str, optional
        Plot title. Default: "Feature Importance"
    top_n : int, optional
        Number of top features to display. Default: 20
    color : str, optional
        Bar color. If None, uses matplotlib default.
    figsize : Tuple[int, int], optional
        Figure size (width, height). Default: (10, 8)

    Returns
    -------
    None
        Saves figure to output_path as PNG and PDF

    Examples
    --------
    >>> feature_names = ["length", "width", "volume", "surface_area", "density"]
    >>> importances = np.array([0.45, 0.30, 0.15, 0.10, 0.05])
    >>> plot_feature_importance(
    ...     feature_names, importances, "feature_importance.png", top_n=5
    ... )

    From sklearn RandomForest:
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> rf = RandomForestClassifier()
    >>> rf.fit(X_train, y_train)
    >>> plot_feature_importance(
    ...     feature_names=X_train.columns,
    ...     importances=rf.feature_importances_,
    ...     output_path="rf_importance.png",
    ...     title="Random Forest Feature Importance"
    ... )

    Notes
    -----
    - Automatically sorts features by importance (descending)
    - Shows only top_n features to avoid cluttered plots
    - Saves both PNG (300 dpi) and PDF versions
    """
    print("\n📊 Plotting feature importance...")
    print(f"   Output: {output_path}")

    # Convert to Path object
    output_path = Path(output_path)

    # Convert to arrays
    feature_names = np.asarray(feature_names)
    importances = np.asarray(importances)

    # Validate inputs
    if len(feature_names) != len(importances):
        raise ValueError(
            f"Length mismatch: {len(feature_names)} features but "
            f"{len(importances)} importance scores"
        )

    # Sort by importance and take top N
    sorted_idx = np.argsort(importances)[-top_n:]
    top_features = feature_names[sorted_idx]
    top_importances = importances[sorted_idx]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot horizontal bars
    bar_kwargs = {}
    if color is not None:
        bar_kwargs["color"] = color

    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importances, **bar_kwargs)

    # Configure axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Importance Score", fontsize=14)
    ax.set_title(title, fontsize=16, pad=15)
    ax.grid(axis="x", alpha=0.3)

    # Save
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    print("✅ Feature importance plot saved\n")
    plt.close()


def plot_selectkbest_comparison(
    results_dict: dict[str, dict[str, Sequence[float]]],
    output_path: str | Path,
    title: str = "SelectKBest Comparison",
    figsize: tuple[int, int] = (16, 6),
) -> None:
    """Compare different SelectKBest scoring functions side-by-side.

    Creates a two-panel figure showing overall accuracy and per-class
    accuracy for different feature selection scoring functions (e.g.,
    f_classif, mutual_info_classif, chi2).

    Parameters
    ----------
    results_dict : Dict[str, Dict[str, Sequence[float]]]
        Nested dictionary structure:
        {
            "evaluator_name": {
                "overall": [acc1, acc2, ...],
                "per_class": [acc1, acc2, ...]
            }
        }
    output_path : str or Path
        Path where figure will be saved (both PNG and PDF)
    title : str, optional
        Overall plot title. Default: "SelectKBest Comparison"
    figsize : Tuple[int, int], optional
        Figure size (width, height). Default: (16, 6)

    Returns
    -------
    None
        Saves figure to output_path as PNG and PDF

    Examples
    --------
    >>> results = {
    ...     "f_classif": {
    ...         "overall": [0.75, 0.82, 0.88, 0.90],
    ...         "per_class": [0.70, 0.78, 0.85, 0.87]
    ...     },
    ...     "mutual_info": {
    ...         "overall": [0.73, 0.80, 0.86, 0.88],
    ...         "per_class": [0.68, 0.76, 0.83, 0.85]
    ...     }
    ... }
    >>> plot_selectkbest_comparison(results, "selectkbest_comparison.png")

    Alternative format (separate dicts):
    >>> overall = {"f_classif": [0.75, 0.82], "mutual_info": [0.73, 0.80]}
    >>> per_class = {"f_classif": [0.70, 0.78], "mutual_info": [0.68, 0.76]}
    >>> results = {
    ...     name: {"overall": overall[name], "per_class": per_class[name]}
    ...     for name in overall
    ... }
    >>> plot_selectkbest_comparison(results, "selectkbest.png")

    Notes
    -----
    - Left panel shows overall classification accuracy
    - Right panel shows per-class accuracy (averaged across classes)
    - Each evaluator gets a different color from the color cycle
    - Saves both PNG (300 dpi) and PDF versions
    """
    print("\n📊 Plotting SelectKBest comparison...")
    print(f"   Output: {output_path}")

    # Convert to Path object
    output_path = Path(output_path)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot overall accuracy
    for evaluator, metrics in results_dict.items():
        scores = np.asarray(metrics["overall"])
        n_features = np.arange(1, len(scores) + 1)
        ax1.plot(n_features, scores, marker="o", label=evaluator, linewidth=2)

    ax1.set_xlabel("Number of Features", fontsize=14)
    ax1.set_ylabel("Overall Accuracy", fontsize=14)
    ax1.set_title("Overall Accuracy", fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot per-class accuracy
    for evaluator, metrics in results_dict.items():
        scores = np.asarray(metrics["per_class"])
        n_features = np.arange(1, len(scores) + 1)
        ax2.plot(n_features, scores, marker="o", label=evaluator, linewidth=2)

    ax2.set_xlabel("Number of Features", fontsize=14)
    ax2.set_ylabel("Per-Class Accuracy", fontsize=14)
    ax2.set_title("Per-Class Accuracy", fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(title, fontsize=18, y=1.02)

    # Save
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    print("✅ SelectKBest comparison saved\n")
    plt.close()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "plot_rfe_curve",
    "plot_rfe_comparison",
    "plot_feature_importance",
    "plot_selectkbest_comparison",
]
