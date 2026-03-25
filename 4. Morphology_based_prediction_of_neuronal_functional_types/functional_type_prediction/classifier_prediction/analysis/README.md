# analysis

Research and evaluation scripts for analyzing classifier pipeline outputs.

## Overview

This package contains standalone analysis scripts used for model evaluation, feature importance analysis, and prediction comparison. These scripts are designed for research exploration and are not part of the production pipeline. Each script loads data through `ClassPredictor`, runs a specific analysis, and saves plots and summaries to dedicated output folders.

## Files

| File | Description | Output folder |
|------|-------------|---------------|
| `feature_importance.py` | Computes permutation importance for RFE-selected features. Shuffles each feature K=50 times and measures the drop in weighted F1 score. Produces a mean permuted accuracy plot, a sorted importance bar chart, and a CSV. | `classifier_pipeline/feature_importance/` |
| `find_best_feature_selector.py` | Evaluates RFE with multiple classifier estimators to find the best feature subset and count. Generates per-estimator RFE curve plots. | `classifier_pipeline/find_feature_selector/` |
| `find_optimal_proba_cutoff.py` | Sweeps probability cutoff thresholds (0.01–0.99) to find the optimal confidence threshold. For each cutoff, runs cross-validation and records accuracy and retained cell count. Produces a dual-axis accuracy vs. coverage plot. | `classifier_pipeline/proba_cutoff/` |
| `calculate_published_metrics.py` | Reproduces metrics matching the published classification results. Runs LDA confusion matrices for three feature types (`pv`, `ps`, `ff`) using Leave-Pair-Out cross-validation. | `classifier_pipeline/published_metrics/` |

## Dependencies

- **`core/`** -- `ClassPredictor` (also available as `class_predictor` alias) for data loading and pipeline operations
- **`src/util/`** -- `get_base_path()` for data path resolution, `get_output_dir()` for output paths
- External: scikit-learn (LDA, AdaBoost), pandas, numpy, matplotlib, tqdm

## Output Location

All scripts save to subdirectories of `~/Desktop/hbsf_output/classifier_pipeline/` via `get_output_dir()`. No scripts use interactive display (`plt.show()`); all use the non-interactive `Agg` backend.

## How to Run

Each script is run standalone from the command line:

```bash
python analysis/feature_importance.py
python analysis/find_best_feature_selector.py
python analysis/find_optimal_proba_cutoff.py
python analysis/calculate_published_metrics.py
```

All scripts use `if __name__ == "__main__"` guards and require that the data path (resolved via `get_base_path()`) points to a valid data directory containing the HDF5 feature files and cell metadata.
