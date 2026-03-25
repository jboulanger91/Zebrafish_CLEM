"""Analysis Scripts.

This package contains scripts for analyzing predictions, evaluating models,
and exploring feature importance.

Scripts
-------
feature_importance : Analyze feature importance for classification
find_best_feature_selector : Optimize feature selection method
find_optimal_proba_cutoff : Optimize probability cutoff for predictions
calculate_published_metrics : Calculate metrics matching published results

Usage
-----
These scripts are typically run from the command line:
    python analysis/feature_importance.py
    python analysis/find_best_feature_selector.py
    python analysis/find_optimal_proba_cutoff.py
    python analysis/calculate_published_metrics.py

Notes
-----
Analysis scripts are for research and evaluation, not production use.
"""

__all__ = [
    'feature_importance',
    'find_best_feature_selector',
    'find_optimal_proba_cutoff',
    'calculate_published_metrics',
]
