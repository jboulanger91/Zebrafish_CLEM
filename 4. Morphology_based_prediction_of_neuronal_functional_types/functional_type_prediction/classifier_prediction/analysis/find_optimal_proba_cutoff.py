"""Optimize probability cutoff threshold for predictions."""

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

if __name__ == "__main__":
    _data_path = get_base_path()
    test = class_predictor(_data_path)
    test.load_cells_df(
        modalities=['pa', 'clem241211', 'em', 'clem_predict241211'],
    )
    test.calculate_metrics('test')
    test.load_cells_features(
        'test',
        drop_neurotransmitter=False,
    )

    test.select_features_RFE(
        'all', 'clem', save_features=True,
        estimator=AdaBoostClassifier(random_state=0),
        cv_method_rfe='ss', metric='f1',
        output_subdir='proba_cutoff',
    )
    cutoffs_success = []
    cutoffs_values = []
    n_used_cells = []
    for i in np.arange(0.01, 1, 0.01):
        a = test.do_cv(
            method='lpo',
            clf=LinearDiscriminantAnalysis(
                solver='lsqr', shrinkage='auto'
            ),
            feature_type='morph', train_mod='all',
            test_mod='clem',
            idx=test.reduced_features_idx,
            plot=False, proba_cutoff=i,
        )
        cutoffs_success.append(a[0])
        cutoffs_values.append(i)
        n_used_cells.append(a[1])

    out_dir = get_output_dir("classifier_pipeline", "proba_cutoff")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('cutoff')
    ax1.set_ylabel('accuracy (%)', color=color)
    ax1.plot(cutoffs_values, cutoffs_success, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('used cells (n)', color=color)
    ax2.plot(cutoffs_values, n_used_cells, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_coverage.png", dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Results saved to {out_dir}")
