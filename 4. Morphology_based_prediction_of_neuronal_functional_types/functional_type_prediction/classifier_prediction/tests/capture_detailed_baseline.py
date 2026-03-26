"""Capture detailed baseline values from current pipeline run.

This script captures detailed intermediate values for comprehensive regression tests.
Run this once to establish expected values for test_detailed_regression.py.

Usage:
    python capture_detailed_baseline.py
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
DETAILED_BASELINE_FILE = _TEST_DIR / "detailed_baseline_values.json"


def capture_detailed_baseline():
    """Run pipeline and capture detailed baseline values."""
    baseline = {
        "captured_at": datetime.now().isoformat(),
        "features_file": FEATURES_FILE,
    }

    print("=" * 60)
    print("CAPTURING DETAILED BASELINE VALUES")
    print("=" * 60)

    # Stage 1-3: Load and preprocess
    print("\n[1/6] Loading cells and features...")
    pred = class_predictor(DATA_PATH)
    pred.load_cells_df(
        modalities=["pa", "clem", "em", "clem_predict"],
    )

    pred.calculate_metrics(
        FEATURES_FILE, force_new=False, use_stored_features=True
    )
    pred.load_cells_features(FEATURES_FILE, drop_neurotransmitter=False)

    # Capture column names before preprocessing
    baseline["all_cells_columns"] = list(pred.all_cells_with_to_predict.columns)
    print(f"  Captured {len(baseline['all_cells_columns'])} column names")

    # Stage 3: RFE Feature Selection (morphology sync now handled inside load_cells_features)
    print("\n[3/6] Running RFE feature selection...")
    pred.select_features_RFE(
        "all",
        "clem",
        save_features=True,  # Must be True to set reduced_features_idx
        cv_method_rfe="ss",
        estimator=AdaBoostClassifier(random_state=0),
        metric="f1",
    )

    # Capture feature statistics
    baseline["feature_stats"] = {
        "shape": list(pred.features.shape),
        "overall_mean": float(pred.features.mean()),
        "overall_std": float(pred.features.std()),
        "overall_min": float(pred.features.min()),
        "overall_max": float(pred.features.max()),
    }

    # Capture reduced features index
    baseline["reduced_features_idx"] = pred.reduced_features_idx.tolist()

    # Capture selected feature names
    baseline["selected_feature_names"] = [
        pred.column_labels[i] for i, selected in enumerate(pred.reduced_features_idx) if selected
    ]

    print(f"  Feature shape: {baseline['feature_stats']['shape']}")
    print(f"  Selected {len(baseline['selected_feature_names'])} features")

    # Stage 5: Confusion Matrices
    print("\n[4/6] Generating confusion matrices...")
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    pred.confusion_matrices(
        clf, method="lpo", fraction_across_classes=True
    )

    # Capture confusion matrix
    baseline["confusion_matrix"] = {
        "values": pred.cm.tolist(),
        "diagonal": np.diag(pred.cm).tolist(),
        "shape": list(pred.cm.shape),
    }
    print(f"  Confusion matrix captured: {baseline['confusion_matrix']['shape']}")
    print(f"  Diagonal values: {baseline['confusion_matrix']['diagonal']}")

    # Stage 6: Predictions
    print("\n[5/6] Running predictions...")
    pred.predict_cells(use_jon_priors=False, suffix="_optimize_all_predict", save_predictions=False)

    preds = pred.prediction_predict_df

    # Capture probability statistics
    prob_cols = ["MON_proba", "cMI_proba", "iMI_proba", "SMI_proba"]
    baseline["probability_stats"] = {}
    for col in prob_cols:
        if col in preds.columns:
            baseline["probability_stats"][col] = {
                "mean": float(preds[col].mean()),
                "std": float(preds[col].std()),
                "min": float(preds[col].min()),
                "max": float(preds[col].max()),
            }
    print(f"  Captured probability stats for {len(baseline['probability_stats'])} columns")

    # Capture scaled prediction distribution
    baseline["prediction_scaled_distribution"] = preds["prediction_scaled"].value_counts().to_dict()
    print(f"  Scaled prediction distribution: {baseline['prediction_scaled_distribution']}")

    # Capture some specific cell predictions for regression testing
    sample_cells = preds.head(20)["cell_name"].tolist()
    baseline["specific_cell_predictions"] = {
        row["cell_name"]: row["prediction"]
        for _, row in preds[preds["cell_name"].isin(sample_cells)].iterrows()
    }
    print(f"  Captured {len(baseline['specific_cell_predictions'])} specific cell predictions")

    # Stage 7: Verification Metrics
    print("\n[6/6] Calculating verification metrics...")
    pred.calculate_verification_metrics(
        calculate_smat=False, with_kunst=False, required_tests=["IF", "LOF"], force_new=True
    )

    preds = pred.prediction_predict_df

    # Capture verification counts
    baseline["verification_counts"] = {
        "IF": int(preds["IF"].sum()) if "IF" in preds.columns else None,
        "LOF": int(preds["LOF"].sum()) if "LOF" in preds.columns else None,
        "OCSVM": int(preds["OCSVM"].sum()) if "OCSVM" in preds.columns else None,
        "IF_intra_class": int(preds["IF_intra_class"].sum())
        if "IF_intra_class" in preds.columns
        else None,
        "LOF_intra_class": int(preds["LOF_intra_class"].sum())
        if "LOF_intra_class" in preds.columns
        else None,
        "OCSVM_intra_class": int(preds["OCSVM_intra_class"].sum())
        if "OCSVM_intra_class" in preds.columns
        else None,
    }
    print(f"  Verification counts: {baseline['verification_counts']}")

    # Save baseline
    print("\n" + "=" * 60)
    print(f"Saving detailed baseline to: {DETAILED_BASELINE_FILE}")

    with open(DETAILED_BASELINE_FILE, "w") as f:
        json.dump(baseline, f, indent=2, default=str)

    print("=" * 60)
    print("DETAILED BASELINE CAPTURE COMPLETE")
    print("=" * 60)

    return baseline


if __name__ == "__main__":
    baseline = capture_detailed_baseline()
    print("\nDetailed baseline values saved. You can now run detailed regression tests.")
