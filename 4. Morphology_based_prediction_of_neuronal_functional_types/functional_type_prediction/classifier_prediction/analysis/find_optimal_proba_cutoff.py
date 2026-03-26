"""Optimize probability cutoff threshold for predictions.

Tests cutoff thresholds and plots accuracy vs. coverage tradeoff.

Usage:
    python functional_type_prediction/classifier_prediction/analysis/find_optimal_proba_cutoff.py
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier

_SCRIPT_DIR = Path(__file__).resolve().parent
_CLASSIFIER_DIR = _SCRIPT_DIR.parent
_SRC = _CLASSIFIER_DIR.parent.parent / "src"
for _p in [str(_CLASSIFIER_DIR), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)  # Required for standalone execution

from src.util.get_base_path import get_base_path  # noqa: E402
from src.util.output_paths import get_output_dir  # noqa: E402

from core.class_predictor import class_predictor  # noqa: E402


def run_proba_cutoff(
    data_path=None,
    features_file="test",
    modalities=None,
    train_mod="all",
    test_mod="clem",
    rfe_metric="f1",
    cutoff_min=0.01,
    cutoff_max=0.99,
    cutoff_step=0.01,
):
    """Sweep probability cutoffs and plot accuracy vs. coverage.

    Parameters
    ----------
    data_path : Path, optional
        Base data directory. Uses get_base_path() if None.
    features_file : str
        HDF5 features file name (without _features.hdf5 suffix).
    modalities : list of str, optional
        Imaging modalities to load. Default: pa, clem, em, clem_predict.
    train_mod : str
        Training modality for RFE/CV. Default: 'all'.
    test_mod : str
        Test modality for RFE/CV. Default: 'clem'.
    rfe_metric : str
        Metric for RFE optimization. Default: 'f1'.
    cutoff_min : float
        Minimum probability cutoff to test. Default: 0.01.
    cutoff_max : float
        Maximum probability cutoff to test. Default: 0.99.
    cutoff_step : float
        Step size for cutoff sweep. Default: 0.01.
    """
    if data_path is None:
        data_path = get_base_path()
    if modalities is None:
        modalities = ["pa", "clem", "em", "clem_predict"]

    test = class_predictor(data_path)
    test.load_cells_df(modalities=modalities)
    test.calculate_metrics(features_file)
    test.load_cells_features(features_file, drop_neurotransmitter=False)

    test.select_features_RFE(
        train_mod, test_mod, save_features=True,
        estimator=AdaBoostClassifier(random_state=0),
        cv_method_rfe="ss", metric=rfe_metric,
        output_subdir="proba_cutoff",
    )

    cutoffs_success = []
    cutoffs_values = []
    n_used_cells = []
    for i in np.arange(cutoff_min, cutoff_max + cutoff_step / 2, cutoff_step):
        a = test.do_cv(
            method="lpo",
            clf=LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            feature_type="morph", train_mod=train_mod, test_mod=test_mod,
            idx=test.reduced_features_idx,
            plot=False, proba_cutoff=i,
        )
        cutoffs_success.append(a[0])
        cutoffs_values.append(i)
        n_used_cells.append(a[1])

    out_dir = get_output_dir("classifier_pipeline", "proba_cutoff")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("cutoff")
    ax1.set_ylabel("accuracy (%)", color=color)
    ax1.plot(cutoffs_values, cutoffs_success, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("used cells (n)", color=color)
    ax2.plot(cutoffs_values, n_used_cells, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_coverage.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    run_proba_cutoff()
