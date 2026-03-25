"""Analyze feature importance for cell type classification."""

import matplotlib
matplotlib.use("Agg")

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from tqdm import tqdm

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
    # load metrics and cells
    # Was: /Users/.../hindbrain_structure_function/nextcloud
    _data_path = get_base_path()
    test = class_predictor(_data_path)
    test.load_cells_df(
        modalities=[
            'pa', 'clem241211', 'em', 'clem_predict241211'
        ],
    )
    test.calculate_metrics('test')  #
    # test.calculate_published_metrics()
    test.load_cells_features(
        'test',
        drop_neurotransmitter=False,
    )

    test.select_features_RFE(
        'all', 'clem', save_features=True,
        estimator=AdaBoostClassifier(random_state=0),
        cv_method_rfe='ss', metric='f1',
    )

    np.random.seed(42)
    copy_features = copy.deepcopy(test.features)
    # Save RFE mask before the loop (load_cells_features resets it to None)
    rfe_mask = test.reduced_features_idx.copy()
    reference_value = test.do_cv(
        method='ss',
        clf=LinearDiscriminantAnalysis(
            solver='lsqr', shrinkage='auto'
        ),
        feature_type='morph',
        train_mod='all', test_mod='clem',
        idx=rfe_mask,
        plot=False, metric='f1',
    )[0]
    mean_permutated_accuracy = []
    importance = []

    K = 50
    for j in tqdm(range(copy_features.shape[1])):
        if rfe_mask[j]:
            permutated_accuracy = []
            for _i in range(K):
                test.load_cells_features(
                    'test',
                    drop_neurotransmitter=False,
                )
                test.reduced_features_idx = rfe_mask
                np.random.shuffle(test.features[:, j])
                a = test.do_cv(
                    method='ss',
                    clf=LinearDiscriminantAnalysis(
                        solver='lsqr', shrinkage='auto'
                    ),
                    feature_type='morph',
                    train_mod='all', test_mod='clem',
                    idx=rfe_mask,
                    plot=False, metric='f1',
                )
                permutated_accuracy.append(a[0])
            mean_permutated_accuracy.append(np.mean(permutated_accuracy))
            importance.append(reference_value - (1 / K) * np.sum(permutated_accuracy))
            test.features = copy_features
    out_dir = get_output_dir("classifier_pipeline", "feature_importance")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.title('mean permutated accuracy')
    plt.plot(mean_permutated_accuracy, marker='x')
    plt.axhline(reference_value, c='red', alpha=0.3)
    plt.xticks(ticks=range(len(mean_permutated_accuracy)),
               labels=np.array(test.column_labels)[test.reduced_features_idx],
               rotation=40, ha='right')
    plt.subplots_adjust(bottom=0.45)
    plt.savefig(out_dir / "mean_permutated_accuracy.png", dpi=200, bbox_inches='tight')
    plt.close()

    # permutation importance plot
    importance_df = pd.DataFrame({
        'features': np.array(
            test.column_labels
        )[test.reduced_features_idx],
        'importance': importance,
    }).sort_values('importance', ascending=False)

    plt.figure()
    plt.title(
        f'Permutation Importance\nPermutations per feature = {K}'
    )
    plt.bar(
        x=range(len(importance_df['importance'])),
        height=importance_df['importance'],
    )
    plt.xticks(
        ticks=range(len(importance_df['importance'])),
        labels=importance_df['features'],
        rotation=40, ha='right',
    )
    plt.subplots_adjust(bottom=0.45)
    plt.savefig(out_dir / "permutation_importance.png", dpi=200, bbox_inches='tight')
    plt.close()

    importance_df.to_csv(out_dir / "permutation_importance.csv", index=False)
    print(f"Results saved to {out_dir}")
