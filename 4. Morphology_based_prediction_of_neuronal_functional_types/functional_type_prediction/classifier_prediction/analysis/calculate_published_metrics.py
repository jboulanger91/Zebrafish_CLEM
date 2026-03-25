"""Calculate metrics matching published classification results.

Loads pre-computed features from HDF5, runs the published metrics
(confusion matrices with LDA Leave-Pair-Out cross-validation),
and saves plots without displaying them.

Usage:
    python functional_type_prediction/classifier_prediction/analysis/calculate_published_metrics.py
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no plt.show()

import sys
from pathlib import Path

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

_SCRIPT_DIR = Path(__file__).resolve().parent
_CLASSIFIER_DIR = _SCRIPT_DIR.parent
_SRC = _CLASSIFIER_DIR.parent.parent / "src"
for _p in [str(_CLASSIFIER_DIR), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)  # Required for standalone execution

from src.util.get_base_path import get_base_path  # noqa: E402

from core.class_predictor import class_predictor  # noqa: E402

if __name__ == "__main__":
    _data_path = get_base_path()
    predictor = class_predictor(_data_path)
    predictor.load_cells_df(
        modalities=['pa', 'clem241211', 'em', 'clem_predict241211'],
    )

    # Load pre-computed features (skip metric recalculation)
    predictor.load_cells_features('test', drop_neurotransmitter=False)
    predictor.calculate_published_metrics()

    # LDA confusion matrices (Leave-Pair-Out)
    clf_pv = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_ps = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_ff = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    predictor.confusion_matrices(clf_pv, method='lpo', feature_type='pv', output_subdir='published_metrics')
    predictor.confusion_matrices(clf_ps, method='lpo', feature_type='ps', output_subdir='published_metrics')
    predictor.confusion_matrices(clf_ff, method='lpo', feature_type='ff', output_subdir='published_metrics')

    print("\nDone. Plots saved (no interactive display).")
