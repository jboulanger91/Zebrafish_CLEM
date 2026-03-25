"""Capture baseline values from current pipeline run.

Run this script once to establish expected values for regression tests.
The captured values are saved to baseline_values.json.

Usage:
    python capture_baseline.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add paths for imports
_TEST_DIR = Path(__file__).resolve().parent
_CLASSIFIER_DIR = _TEST_DIR.parent
_FUNCTIONAL_DIR = _CLASSIFIER_DIR.parent
_REPO_ROOT = _FUNCTIONAL_DIR.parent
_SRC = _REPO_ROOT / "src"

for path in [str(_CLASSIFIER_DIR), str(_SRC), str(_REPO_ROOT)]:
    if path not in sys.path:
        # _CLASSIFIER_DIR: needed for bare core.* imports
        # _SRC: needed for bare util.* fallback imports
        # Required for standalone execution (remove after pip install -e .)
        sys.path.insert(0, path)

# Set matplotlib to non-interactive backend
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.ensemble import AdaBoostClassifier  # noqa: E402

from core.class_predictor import class_predictor  # noqa: E402

try:
    from src.util.get_base_path import get_base_path  # noqa: E402
except ModuleNotFoundError:
    from util.get_base_path import get_base_path  # noqa: E402

matplotlib.use("Agg")

# Constants
DATA_PATH = get_base_path()
FEATURES_FILE = "test"
BASELINE_FILE = _TEST_DIR / "baseline_values.json"


def capture_baseline():
    """Run pipeline and capture all baseline values."""
    baseline = {
        "captured_at": datetime.now().isoformat(),
        "features_file": FEATURES_FILE,
    }

    print("=" * 60)
    print("CAPTURING BASELINE VALUES")
    print("=" * 60)

    # Stage 1: load_cells_df
    print("\n[1/7] Loading cells...")
    pred = class_predictor(DATA_PATH)
    pred.load_cells_df(
        modalities=["pa", "clem241211", "em", "clem_predict241211"],
    )

    baseline["after_load_cells"] = {
        "total_cells": len(pred.cells),
        "columns": list(pred.cells.columns),
        "modality_counts": pred.cells["imaging_modality"].value_counts().to_dict(),
        "function_counts": pred.cells["function"].value_counts().to_dict(),
    }
    print(f"  Total cells: {baseline['after_load_cells']['total_cells']}")

    # Stage 2: calculate_metrics and load_cells_features
    print("\n[2/7] Loading features...")
    pred.calculate_metrics(
        FEATURES_FILE, force_new=False, use_stored_features=True,
    )
    pred.load_cells_features(FEATURES_FILE, drop_neurotransmitter=False)

    baseline["after_load_features"] = {
        "total_cells": len(pred.all_cells_with_to_predict),
        "function_counts": pred.all_cells_with_to_predict["function"].value_counts().to_dict(),
        "modality_counts": pred.all_cells_with_to_predict["imaging_modality"]
        .value_counts()
        .to_dict(),
        "numeric_columns": list(
            pred.all_cells_with_to_predict.select_dtypes(include=[np.number]).columns
        ),
    }
    print(f"  Total cells with features: {baseline['after_load_features']['total_cells']}")

    # Stage 3: Morphology sync (now handled inside load_cells_features)
    baseline["after_preprocessing"] = {
        "total_cells": len(pred.all_cells_with_to_predict),
        "function_counts": pred.all_cells_with_to_predict["function"].value_counts().to_dict(),
        "modality_counts": pred.all_cells_with_to_predict["imaging_modality"]
        .value_counts()
        .to_dict(),
        "training_cells": len(
            pred.all_cells_with_to_predict[
                pred.all_cells_with_to_predict["function"] != "to_predict"
            ]
        ),
        "to_predict_cells": len(
            pred.all_cells_with_to_predict[
                pred.all_cells_with_to_predict["function"] == "to_predict"
            ]
        ),
    }
    print(f"  Cells after preprocessing: {baseline['after_preprocessing']['total_cells']}")
    print(f"  Training cells: {baseline['after_preprocessing']['training_cells']}")
    print(f"  To predict cells: {baseline['after_preprocessing']['to_predict_cells']}")

    # Stage 4: RFE Feature Selection
    print("\n[4/7] Running RFE feature selection...")

    # Capture RFE curve (F1 score vs number of features)
    from sklearn.feature_selection import RFE

    rfe_estimator = AdaBoostClassifier(random_state=0)
    rfe_curve = []
    clf_eval = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    print("   Calculating RFE curve (F1 vs n_features)...")
    for n_features in range(1, pred.features.shape[1] + 1):
        selector = RFE(rfe_estimator, n_features_to_select=n_features, step=1)
        selector.fit(pred.features, pred.labels)

        # Evaluate with LOO CV on CLEM
        from sklearn.metrics import f1_score
        from sklearn.model_selection import LeaveOneOut, cross_val_predict

        selected_features_temp = pred.features[pred.clem_idx][:, selector.support_]
        clem_labels = pred.labels[pred.clem_idx]
        y_pred = cross_val_predict(clf_eval, selected_features_temp, clem_labels, cv=LeaveOneOut())
        f1 = f1_score(clem_labels, y_pred, average="weighted")
        rfe_curve.append(float(f1))
        if n_features % 10 == 0:
            print(f"      n={n_features}: F1={f1:.4f}")

    optimal_n_features = int(np.argmax(rfe_curve) + 1)
    best_f1_rfe = float(rfe_curve[optimal_n_features - 1])
    print(
        f"   RFE curve complete: optimal = {optimal_n_features}"
        f" features, best F1 = {best_f1_rfe:.4f}"
    )

    pred.select_features_RFE(
        "all",
        "clem",
        save_features=True,  # Must be True to set reduced_features_idx for confusion_matrices
        cv_method_rfe="ss",
        estimator=AdaBoostClassifier(random_state=0),
        metric="f1",
    )

    # Capture selected features (boolean mask -> feature names)
    selected_features = list(np.array(pred.column_labels)[pred.reduced_features_idx])

    baseline["after_rfe"] = {
        "selected_features": selected_features,
        "n_features": len(selected_features) if selected_features else None,
        "rfe_curve": rfe_curve,
        "optimal_n_features": optimal_n_features,
        "best_f1_rfe": best_f1_rfe,
    }
    print(f"  Selected features: {baseline['after_rfe']['n_features']}")
    if selected_features:
        print(f"  Features: {selected_features[:10]}...")

    # Stage 5: Confusion Matrices
    print("\n[5/7] Generating confusion matrices...")
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    pred.confusion_matrices(
        clf, method="lpo", fraction_across_classes=True
    )

    # Capture confusion matrix and F1 score
    from sklearn.metrics import f1_score
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    cv_info = {}

    # Capture the confusion matrix values
    if hasattr(pred, "cm"):
        cv_info["confusion_matrix"] = pred.cm.tolist()
        cv_info["confusion_matrix_diagonal"] = np.diag(pred.cm).tolist()

    # Calculate F1 score using LOO CV on CLEM data (same as pipeline)
    clem_features = pred.features[pred.clem_idx][:, pred.reduced_features_idx]
    clem_labels = pred.labels[pred.clem_idx]
    y_pred = cross_val_predict(clf, clem_features, clem_labels, cv=LeaveOneOut())
    f1_clem = f1_score(clem_labels, y_pred, average="weighted")
    cv_info["f1_score_clem_loo"] = float(f1_clem)

    # Also calculate for ALL data
    all_features = pred.features[:, pred.reduced_features_idx]
    all_labels = pred.labels
    y_pred_all = cross_val_predict(clf, all_features, all_labels, cv=LeaveOneOut())
    f1_all = f1_score(all_labels, y_pred_all, average="weighted")
    cv_info["f1_score_all_loo"] = float(f1_all)

    baseline["after_confusion_matrices"] = cv_info
    print(f"  F1 Score (CLEM, LOO): {f1_clem:.4f} ({f1_clem * 100:.2f}%)")
    print(f"  F1 Score (ALL, LOO): {f1_all:.4f} ({f1_all * 100:.2f}%)")

    # Stage 6: Predictions
    print("\n[6/7] Running predictions...")
    pred.predict_cells(use_jon_priors=False, suffix="_optimize_all_predict", save_predictions=False)

    preds = pred.prediction_predict_df

    # Prediction distribution
    preds.groupby(["imaging_modality", "prediction"]).size().unstack(fill_value=0)

    baseline["after_predictions"] = {
        "total_predictions": len(preds),
        "em_count": len(preds[preds["imaging_modality"] == "EM"]),
        "clem_count": len(preds[preds["imaging_modality"] == "clem"]),
        "prediction_distribution": preds["prediction"].value_counts().to_dict(),
        "by_modality_prediction": {
            mod: preds[preds["imaging_modality"] == mod]["prediction"].value_counts().to_dict()
            for mod in preds["imaging_modality"].unique()
        },
    }
    print(f"  Total predictions: {baseline['after_predictions']['total_predictions']}")
    print(
        f"  EM: {baseline['after_predictions']['em_count']}, "
        f"CLEM: {baseline['after_predictions']['clem_count']}"
    )

    # Stage 7: Verification Metrics
    print("\n[7/7] Calculating verification metrics...")
    pred.calculate_verification_metrics(
        calculate_smat=False, with_kunst=False, required_tests=["IF", "LOF"], force_new=True
    )

    preds = pred.prediction_predict_df

    baseline["after_verification"] = {
        "total_passed": int(preds["passed_tests"].sum()),
        "em_passed": int(preds[preds["imaging_modality"] == "EM"]["passed_tests"].sum()),
        "clem_passed": int(preds[preds["imaging_modality"] == "clem"]["passed_tests"].sum()),
        "passed_by_prediction": {
            p: int(preds[preds["prediction"] == p]["passed_tests"].sum())
            for p in preds["prediction"].unique()
        },
    }
    print(f"  Total passed: {baseline['after_verification']['total_passed']}")
    print(f"  EM passed: {baseline['after_verification']['em_passed']}")
    print(f"  CLEM passed: {baseline['after_verification']['clem_passed']}")

    # Save baseline
    print("\n" + "=" * 60)
    print(f"Saving baseline to: {BASELINE_FILE}")

    with open(BASELINE_FILE, "w") as f:
        json.dump(baseline, f, indent=2, default=str)

    print("=" * 60)
    print("BASELINE CAPTURE COMPLETE")
    print("=" * 60)

    return baseline


if __name__ == "__main__":
    baseline = capture_baseline()
    print("\nBaseline values saved. You can now run pytest to verify.")
