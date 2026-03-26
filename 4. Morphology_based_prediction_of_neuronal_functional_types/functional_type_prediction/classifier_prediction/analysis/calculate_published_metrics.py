"""Calculate metrics matching published classification results.

Loads pre-computed features from HDF5, runs the published metrics
(confusion matrices with LDA Leave-One-Out cross-validation),
and saves plots without displaying them.

Usage:
    python functional_type_prediction/classifier_prediction/analysis/calculate_published_metrics.py
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

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


def run_published_metrics(
    data_path=None,
    features_file="test",
    modalities=None,
):
    """Run published metrics (confusion matrices for pv, ps, ff features).

    Parameters
    ----------
    data_path : Path, optional
        Base data directory. Uses get_base_path() if None.
    features_file : str
        HDF5 features file name (without _features.hdf5 suffix).
    modalities : list of str, optional
        Imaging modalities to load. Default: pa, clem, em, clem_predict.
    """
    if data_path is None:
        data_path = get_base_path()
    if modalities is None:
        modalities = ["pa", "clem", "em", "clem_predict"]

    predictor = class_predictor(data_path)
    predictor.load_cells_df(modalities=modalities)
    predictor.load_cells_features(features_file, drop_neurotransmitter=False)
    predictor.calculate_published_metrics()

    # LDA confusion matrices (Leave-One-Out) for three feature types
    for feature_type in ("pv", "ps", "ff"):
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        predictor.confusion_matrices(
            clf, method="lpo", feature_type=feature_type,
            output_subdir="published_metrics",
        )

    print("\nDone. Plots saved (no interactive display).")


if __name__ == "__main__":
    run_published_metrics()
