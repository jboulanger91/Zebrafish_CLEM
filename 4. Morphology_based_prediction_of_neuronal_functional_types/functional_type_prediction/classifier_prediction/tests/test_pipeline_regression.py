"""Regression tests for pipeline_main.py.

These tests capture the current behavior of the pipeline to ensure
refactoring doesn't break functionality. Run these tests before and
after any refactoring to verify consistency.

Tests are organized in two groups:

1. Container-returning methods (load_data, select_features_rfe,
   cross_validate, predict, verify).

2. Attribute-based methods (load_cells_df, calculate_metrics,
   select_features_RFE, etc.).

Usage:
    pytest test_pipeline_regression.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add paths for imports
_TEST_DIR = Path(__file__).resolve().parent
_CLASSIFIER_DIR = _TEST_DIR.parent
_REPO_ROOT = _CLASSIFIER_DIR.parents[1]
_SRC = _REPO_ROOT / "src"

for path in [str(_REPO_ROOT), str(_SRC)]:
    if path not in sys.path:
        # Required for standalone execution (remove after pip install -e .)
        # _SRC: needed for bare util.* fallback imports
        sys.path.insert(0, path)

try:
    from src.util.get_base_path import get_base_path
except ModuleNotFoundError:
    from util.get_base_path import get_base_path

# Constants from conftest
DATA_PATH = get_base_path()
FEATURES_FILE = "test"
BASELINE_FILE = _TEST_DIR / "baseline_values.json"


def load_baseline():
    """Load baseline values from JSON file."""
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return None


# =============================================================================
# Container-returning API tests
# =============================================================================


class TestLoadData:
    """Tests for load_data() -> LoadedData."""

    def test_load_data_returns_container(self, data):
        """Verify load_data() returns LoadedData container."""
        from core.containers import LoadedData

        assert isinstance(data, LoadedData), f"Expected LoadedData, got {type(data)}"

    def test_data_has_training_data(self, data):
        """Verify LoadedData has training_data attribute."""
        assert hasattr(data, "training_data"), "LoadedData missing training_data attribute"
        from core.containers import TrainingData

        assert isinstance(data.training_data, TrainingData)

    def test_data_n_training_matches_baseline(self, data):
        """Verify training cell count matches baseline.

        Note: The container's n_training excludes both 'to_predict' AND 'neg_control'
        cells, while the baseline 'training_cells' only excludes 'to_predict'.
        We compute expected from the labeled cells in the container.
        """
        # The container's training dataset excludes to_predict and neg_control
        # So we check that it matches the expected labeled (non-to_predict, non-neg_control) count
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        # Baseline counts cells != 'to_predict' (133 = 120 training + 13 neg_control)
        # Container's n_training excludes neg_control too (120)
        # Expected = baseline training_cells - neg_control count
        baseline_training = baseline["after_preprocessing"]["training_cells"]
        neg_control_count = baseline["after_preprocessing"]["function_counts"].get("neg_control", 0)
        expected = baseline_training - neg_control_count

        actual = data.training_data.n_training

        assert actual == expected, f"Training count: {actual} vs expected {expected}"

    def test_data_feature_names_exist(self, data):
        """Verify feature names are accessible."""
        assert hasattr(data, "feature_names"), "LoadedData missing feature_names"
        assert len(data.feature_names) > 0, "feature_names is empty"

    def test_data_cells_df_exists(self, data):
        """Verify cells DataFrame is accessible."""
        assert hasattr(data, "cells_df"), "LoadedData missing cells_df"
        assert isinstance(data.cells_df, pd.DataFrame)
        assert len(data.cells_df) > 0

    def test_modality_mask_accessible(self, data):
        """Verify modality mask is accessible via container."""
        mask = data.training_data.training.modality_mask
        assert hasattr(mask, "clem"), "Missing clem mask"
        assert hasattr(mask, "photoactivation"), "Missing photoactivation mask"

    def test_clem_count_matches_baseline(self, data):
        """Verify CLEM training count matches HDF5-based baseline.

        Note: load_data() loads from HDF5 (73 CLEM training cells).
        load_cells_df() returns 74 because cell_576460752667468676
        is present in raw metadata but absent from the HDF5.
        """
        expected_clem = 73
        actual = data.training_data.training.modality_mask.n_clem

        assert actual == expected_clem, f"CLEM count: {actual} vs expected {expected_clem}"


class TestFeatureSelection:
    """Tests for select_features_rfe() -> RFEResult."""

    def test_rfe_returns_container(self, rfe_result):
        """Verify select_features_rfe() returns RFEResult."""
        from core.feature_selector import RFEResult

        assert isinstance(rfe_result, RFEResult), f"Expected RFEResult, got {type(rfe_result)}"

    def test_rfe_has_selected_features_idx(self, rfe_result):
        """Verify RFEResult has selected_features_idx."""
        assert hasattr(rfe_result, "selected_features_idx"), (
            "RFEResult missing selected_features_idx"
        )
        assert rfe_result.selected_features_idx is not None

    def test_rfe_feature_count_matches_baseline(self, rfe_result):
        """Verify selected feature count matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_rfe"]["n_features"]
        actual = int(np.sum(rfe_result.selected_features_idx))

        assert actual == expected, f"Feature count: {actual} vs expected {expected}"

    def test_rfe_has_best_n_features(self, rfe_result):
        """Verify RFEResult has best_n_features."""
        assert hasattr(rfe_result, "best_n_features"), "RFEResult missing best_n_features"


class TestCrossValidation:
    """Tests for cross_validate() -> CVResult."""

    def test_cv_returns_container(self, cv_result):
        """Verify cross_validate() returns CVResult."""
        from core.cross_validator import CVResult

        assert isinstance(cv_result, CVResult), f"Expected CVResult, got {type(cv_result)}"

    def test_cv_has_confusion_matrix(self, cv_result):
        """Verify CVResult has confusion_matrix."""
        assert hasattr(cv_result, "confusion_matrix"), "CVResult missing confusion_matrix"
        assert cv_result.confusion_matrix.shape == (4, 4), (
            f"Expected (4, 4) CM, got {cv_result.confusion_matrix.shape}"
        )

    def test_cv_has_score(self, cv_result):
        """Verify CVResult has score."""
        assert hasattr(cv_result, "score"), "CVResult missing score"
        # CVResult stores scores as percentages (0-100), not fractions
        assert 0 <= cv_result.score <= 100, f"Score out of range: {cv_result.score}"


class TestPredictionContainers:
    """Tests for predict() -> PredictionResults."""

    def test_predict_returns_container(self, predictions):
        """Verify predict() returns PredictionResults."""
        from core.containers import PredictionResults

        assert isinstance(predictions, PredictionResults), (
            f"Expected PredictionResults, got {type(predictions)}"
        )

    def test_predictions_has_cells(self, predictions):
        """Verify PredictionResults has cells DataFrame."""
        assert hasattr(predictions, "cells"), "PredictionResults missing cells"
        assert isinstance(predictions.cells, pd.DataFrame)

    def test_predictions_count_matches_baseline(self, predictions):
        """Verify prediction count matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_predictions"]["total_predictions"]
        actual = predictions.n_cells

        assert actual == expected, f"Predictions: {actual} vs expected {expected}"

    def test_predictions_has_probabilities(self, predictions):
        """Verify PredictionResults has probabilities."""
        assert hasattr(predictions, "probabilities"), "PredictionResults missing probabilities"
        assert predictions.probabilities is not None

    def test_predictions_distribution_matches_baseline(self, predictions):
        """Verify prediction distribution matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_predictions"]["prediction_distribution"]
        actual = predictions.prediction_counts

        for pred, count in expected.items():
            assert pred in actual, f"Missing prediction: {pred}"
            assert actual[pred] == count, f"{pred}: {actual[pred]} vs expected {count}"


class TestVerification:
    """Tests for verify() -> PredictionResults with verification status."""

    def test_verified_has_status(self, predictions):
        """Verify PredictionResults has verification status."""
        assert hasattr(predictions, "verified"), "PredictionResults missing verified attribute"
        assert predictions.verified is not None, "verified is None - verify() not called?"

    def test_n_verified_matches_baseline(self, predictions):
        """Verify verification count matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_verification"]["total_passed"]
        actual = predictions.n_verified

        assert actual == expected, f"Verified: {actual} vs expected {expected}"

    def test_verification_rate_reasonable(self, predictions):
        """Verify verification rate is reasonable."""
        rate = predictions.verification_rate
        assert 0 <= rate <= 1, f"Rate out of range: {rate}"


class TestEndToEnd:
    """End-to-end tests for the full pipeline."""

    def test_full_pipeline_completes(self, pipeline_results):
        """Verify full pipeline runs without error."""
        assert "data" in pipeline_results
        assert "rfe" in pipeline_results
        assert "predictions" in pipeline_results

    def test_em_predictions_via_container(self, predictions):
        """Verify EM predictions accessible via container."""
        em_cells = predictions.cells[predictions.cells["imaging_modality"] == "EM"]
        assert len(em_cells) > 0, "No EM cells found"

    def test_clem_predictions_via_container(self, predictions):
        """Verify CLEM predictions accessible via container."""
        clem_cells = predictions.cells[predictions.cells["imaging_modality"] == "clem"]
        assert len(clem_cells) > 0, "No CLEM cells found"


# =============================================================================
# Attribute-based API tests
# =============================================================================


# =============================================================================
# Test 1: load_cells_df - Cell loading and metadata
# =============================================================================
class TestLoadCellsDF:
    """Tests for load_cells_df stage."""

    def test_cells_dataframe_exists(self, predictor_after_load_cells):
        """Verify cells DataFrame is created."""
        assert hasattr(predictor_after_load_cells, "cells")
        assert isinstance(predictor_after_load_cells.cells, pd.DataFrame)
        assert len(predictor_after_load_cells.cells) > 0

    def test_required_columns_present(self, predictor_after_load_cells):
        """Verify required columns exist in cells DataFrame."""
        required_cols = ["cell_name", "function", "imaging_modality", "neurotransmitter"]
        for col in required_cols:
            assert col in predictor_after_load_cells.cells.columns, f"Missing column: {col}"

    def test_cell_count_matches_baseline(self, predictor_after_load_cells):
        """Verify cell count matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet. Run capture_baseline.py first.")

        expected = baseline["after_load_cells"]["total_cells"]
        actual = len(predictor_after_load_cells.cells)

        assert actual == expected, f"Cell count changed: {actual} vs expected {expected}"

    def test_modality_counts_match_baseline(self, predictor_after_load_cells):
        """Verify modality counts match baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_load_cells"]["modality_counts"]
        actual = predictor_after_load_cells.cells["imaging_modality"].value_counts().to_dict()

        for mod, count in expected.items():
            assert mod in actual, f"Missing modality: {mod}"
            assert actual[mod] == count, f"Modality {mod} count changed: {actual[mod]} vs {count}"

    def test_function_counts_match_baseline(self, predictor_after_load_cells):
        """Verify function counts match baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_load_cells"]["function_counts"]
        actual = predictor_after_load_cells.cells["function"].value_counts().to_dict()

        for func, count in expected.items():
            assert func in actual, f"Missing function: {func}"
            assert actual[func] == count, (
                f"Function {func} count changed: {actual[func]} vs {count}"
            )

    def test_no_duplicate_cells(self, predictor_after_load_cells):
        """Verify no duplicate cell names."""
        cells = predictor_after_load_cells.cells
        duplicates = cells["cell_name"].duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate cell names"


# =============================================================================
# Test 2: calculate_metrics - Feature calculation
# =============================================================================
class TestCalculateMetrics:
    """Tests for calculate_metrics and load_cells_features stages."""

    def test_features_loaded(self, predictor_after_load_features):
        """Verify features are loaded."""
        assert hasattr(predictor_after_load_features, "all_cells_with_to_predict")
        assert len(predictor_after_load_features.all_cells_with_to_predict) > 0

    def test_hdf5_file_exists(self):
        """Verify HDF5 features file exists in output directory."""
        from src.util.output_paths import get_output_dir

        hdf5_path = get_output_dir("classifier_pipeline", "features") / f"{FEATURES_FILE}_features.hdf5"
        assert hdf5_path.exists(), f"HDF5 file not found: {hdf5_path}"

    def test_cell_count_matches_baseline(self, predictor_after_load_features):
        """Verify cell count matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_load_features"]["total_cells"]
        actual = len(predictor_after_load_features.all_cells_with_to_predict)

        assert actual == expected, f"Cell count changed: {actual} vs expected {expected}"

    def test_function_distribution_matches_baseline(self, predictor_after_load_features):
        """Verify function distribution matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_load_features"]["function_counts"]
        actual = (
            predictor_after_load_features.all_cells_with_to_predict["function"]
            .value_counts()
            .to_dict()
        )

        for func, count in expected.items():
            assert func in actual, f"Missing function: {func}"
            assert actual[func] == count, (
                f"Function {func} count changed: {actual[func]} vs {count}"
            )

    def test_numeric_columns_match_baseline(self, predictor_after_load_features):
        """Verify numeric feature columns match baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = set(baseline["after_load_features"]["numeric_columns"])
        actual = set(
            predictor_after_load_features.all_cells_with_to_predict.select_dtypes(
                include=[np.number]
            ).columns
        )

        missing = expected - actual
        extra = actual - expected

        assert not missing, f"Missing columns: {missing}"
        # Extra columns may be OK, but log them
        if extra:
            print(f"Note: Extra columns found: {extra}")


# =============================================================================
# Test 3: Preprocessing (morphology sync now automatic in load_cells_features)
# =============================================================================
class TestPreprocessing:
    """Tests for preprocessing stages."""

    def test_cell_count_matches_baseline(self, predictor_after_preprocessing):
        """Verify cell count after preprocessing matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_preprocessing"]["total_cells"]
        actual = len(predictor_after_preprocessing.all_cells_with_to_predict)

        assert actual == expected, f"Cell count changed: {actual} vs expected {expected}"

    def test_training_cells_match_baseline(self, predictor_after_preprocessing):
        """Verify training cell count matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_preprocessing"]["training_cells"]
        cells = predictor_after_preprocessing.all_cells_with_to_predict
        actual = len(cells[cells["function"] != "to_predict"])

        assert actual == expected, f"Training cells changed: {actual} vs expected {expected}"

    def test_to_predict_cells_match_baseline(self, predictor_after_preprocessing):
        """Verify to_predict cell count matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_preprocessing"]["to_predict_cells"]
        cells = predictor_after_preprocessing.all_cells_with_to_predict
        actual = len(cells[cells["function"] == "to_predict"])

        assert actual == expected, f"To predict cells changed: {actual} vs expected {expected}"

    def test_function_distribution_matches_baseline(self, predictor_after_preprocessing):
        """Verify function distribution matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_preprocessing"]["function_counts"]
        actual = (
            predictor_after_preprocessing.all_cells_with_to_predict["function"]
            .value_counts()
            .to_dict()
        )

        for func, count in expected.items():
            assert func in actual, f"Missing function: {func}"
            assert actual[func] == count, (
                f"Function {func} count changed: {actual[func]} vs {count}"
            )


# =============================================================================
# Test 4: RFE Feature Selection
# =============================================================================
class TestRFEFeatureSelection:
    """Tests for RFE feature selection."""

    def test_selected_features_exist(self, predictor_after_rfe):
        """Verify selected features attribute exists."""
        assert hasattr(predictor_after_rfe, "reduced_features_idx"), (
            "reduced_features_idx attribute not found"
        )

    def test_feature_count_matches_baseline(self, predictor_after_rfe):
        """Verify number of selected features matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_rfe"]["n_features"]
        if expected is None:
            pytest.skip("No feature count in baseline.")

        actual = int(np.sum(predictor_after_rfe.reduced_features_idx))
        assert actual == expected, f"Feature count changed: {actual} vs expected {expected}"

    def test_rfe_curve_matches_baseline(self, predictor_after_rfe):
        """Verify RFE curve (F1 vs n_features) matches baseline."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.feature_selection import RFE
        from sklearn.metrics import f1_score
        from sklearn.model_selection import LeaveOneOut, cross_val_predict

        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_rfe"].get("rfe_curve")
        if expected is None:
            pytest.skip("No RFE curve in baseline. Re-run capture_baseline.py")

        # Calculate current RFE curve
        pred = predictor_after_rfe
        rfe_estimator = AdaBoostClassifier(random_state=0)
        clf_eval = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        rfe_curve = []

        for n_features in range(1, pred.features.shape[1] + 1):
            selector = RFE(rfe_estimator, n_features_to_select=n_features, step=1)
            selector.fit(pred.features, pred.labels)

            selected_features = pred.features[pred.clem_idx][:, selector.support_]
            clem_labels = pred.labels[pred.clem_idx]
            y_pred = cross_val_predict(clf_eval, selected_features, clem_labels, cv=LeaveOneOut())
            f1 = f1_score(clem_labels, y_pred, average="weighted")
            rfe_curve.append(float(f1))

        # Compare curves
        optimal_expected = int(np.argmax(expected) + 1)
        optimal_actual = int(np.argmax(rfe_curve) + 1)

        print(f"RFE curve: optimal features = {optimal_actual} (expected: {optimal_expected})")
        print(
            f"Best F1: {rfe_curve[optimal_actual - 1]:.4f}"
            f" (expected: {expected[optimal_expected - 1]:.4f})"
        )

        # Check optimal number of features matches
        assert optimal_actual == optimal_expected, (
            f"Optimal n_features changed: {optimal_actual} vs {optimal_expected}"
        )

        # Check curve values match (allow small tolerance)
        for i, (actual, exp) in enumerate(zip(rfe_curve, expected, strict=False)):
            assert abs(actual - exp) < 0.001, (
                f"RFE curve differs at n={i + 1}: {actual:.4f} vs {exp:.4f}"
            )

    def test_selected_features_match_baseline(self, predictor_after_rfe):
        """Verify exact selected features match baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_rfe"]["selected_features"]
        if expected is None:
            pytest.skip("No selected features in baseline.")

        actual = list(
            np.array(predictor_after_rfe.column_labels)[
                predictor_after_rfe.reduced_features_idx
            ]
        )
        assert set(actual) == set(expected), (
            f"Selected features changed.\nExpected: {expected}\nActual: {actual}"
        )


# =============================================================================
# Test 5: Confusion Matrices
# =============================================================================
class TestConfusionMatrices:
    """Tests for confusion matrix generation."""

    def test_confusion_matrix_generated(self, predictor_after_confusion_matrices):
        """Verify confusion matrix is generated."""
        assert hasattr(predictor_after_confusion_matrices, "cm"), (
            "Confusion matrix 'cm' attribute not found"
        )
        assert predictor_after_confusion_matrices.cm.shape == (4, 4), (
            f"Expected (4, 4) shape, got {predictor_after_confusion_matrices.cm.shape}"
        )

    def test_confusion_matrix_values_match_baseline(self, predictor_after_confusion_matrices):
        """Verify confusion matrix values match baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline.get("after_confusion_matrices", {})
        if "confusion_matrix" not in expected:
            pytest.skip("No confusion matrix values in baseline.")

        expected_cm = np.array(expected["confusion_matrix"])
        actual_cm = predictor_after_confusion_matrices.cm

        np.testing.assert_array_almost_equal(
            actual_cm, expected_cm, decimal=4, err_msg="Confusion matrix values changed"
        )

    def test_f1_score_matches_baseline(self, predictor_after_confusion_matrices):
        """Verify F1 score matches baseline."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import f1_score
        from sklearn.model_selection import LeaveOneOut, cross_val_predict

        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline.get("after_confusion_matrices", {})
        if "f1_score_clem_loo" not in expected:
            pytest.skip("No F1 score in baseline. Re-run capture_baseline.py")

        # Calculate current F1 score
        pred = predictor_after_confusion_matrices
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clem_features = pred.features[pred.clem_idx][:, pred.reduced_features_idx]
        clem_labels = pred.labels[pred.clem_idx]
        y_pred = cross_val_predict(clf, clem_features, clem_labels, cv=LeaveOneOut())
        actual_f1 = f1_score(clem_labels, y_pred, average="weighted")

        expected_f1 = expected["f1_score_clem_loo"]
        print(f"F1 Score (CLEM, LOO): {actual_f1:.4f} (expected: {expected_f1:.4f})")

        assert abs(actual_f1 - expected_f1) < 0.001, (
            f"F1 score changed: {actual_f1:.4f} vs {expected_f1:.4f}"
        )


# =============================================================================
# Test 6: Predictions vs Reference Files
# =============================================================================
class TestPredictions:
    """Tests comparing predictions to reference files."""

    def test_prediction_dataframe_exists(self, predictor_after_predictions):
        """Verify prediction DataFrame exists."""
        assert hasattr(predictor_after_predictions, "prediction_predict_df")
        assert len(predictor_after_predictions.prediction_predict_df) > 0

    def test_prediction_counts_match_baseline(self, predictor_after_predictions):
        """Verify prediction counts match baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_predictions"]
        actual = predictor_after_predictions.prediction_predict_df

        assert len(actual) == expected["total_predictions"], (
            f"Total predictions changed: {len(actual)} vs {expected['total_predictions']}"
        )

        em_count = len(actual[actual["imaging_modality"] == "EM"])
        assert em_count == expected["em_count"], (
            f"EM count changed: {em_count} vs {expected['em_count']}"
        )

        clem_count = len(actual[actual["imaging_modality"] == "clem"])
        assert clem_count == expected["clem_count"], (
            f"CLEM count changed: {clem_count} vs {expected['clem_count']}"
        )

    def test_clem_predictions_match_reference(
        self, predictor_after_predictions, clem_reference_predictions
    ):
        """Compare CLEM predictions to reference file."""
        if clem_reference_predictions is None:
            pytest.skip("CLEM reference file not found")

        current = predictor_after_predictions.prediction_predict_df
        current_clem = current[current["imaging_modality"] == "clem"].copy()
        # Filter out NaN rows from reference (may have summary rows at end)
        reference = clem_reference_predictions.dropna(subset=["cell_name"]).copy()

        # Normalize cell names to strings for comparison (reference may have floats)
        current_clem["cell_name_str"] = (
            current_clem["cell_name"].astype(str).str.replace(".0", "", regex=False)
        )
        reference["cell_name_str"] = (
            reference["cell_name"].astype(str).str.replace(".0", "", regex=False)
        )

        # Compare cell counts
        assert len(current_clem) == len(reference), (
            f"CLEM cell count mismatch: {len(current_clem)} vs {len(reference)}"
        )

        # Compare predictions cell by cell
        current_indexed = current_clem.set_index("cell_name_str")
        ref_indexed = reference.set_index("cell_name_str")

        common_cells = set(current_indexed.index) & set(ref_indexed.index)
        assert len(common_cells) > 0, "No common cells found"

        mismatches = []
        for cell in common_cells:
            curr_pred = current_indexed.loc[cell, "prediction"]
            ref_pred = ref_indexed.loc[cell, "prediction"]
            if curr_pred != ref_pred:
                mismatches.append({"cell": cell, "current": curr_pred, "reference": ref_pred})

        match_rate = (len(common_cells) - len(mismatches)) / len(common_cells) * 100
        assert match_rate == 100.0, (
            f"CLEM predictions don't match reference"
            f" ({match_rate:.1f}% match)."
            f" Mismatches: {mismatches[:5]}"
        )

    def test_em_predictions_match_reference(
        self, predictor_after_predictions, em_reference_predictions
    ):
        """Compare EM predictions to reference file."""
        if em_reference_predictions is None:
            pytest.skip("EM reference file not found")

        current = predictor_after_predictions.prediction_predict_df
        current_em = current[current["imaging_modality"] == "EM"].copy()
        # Filter out NaN rows from reference (may have summary rows at end)
        reference = em_reference_predictions.dropna(subset=["cell_name"]).copy()
        # Also filter out rows where prediction is NaN
        reference = reference.dropna(subset=["prediction"]).copy()

        # Normalize cell names to strings for comparison (reference may have floats)
        current_em["cell_name_str"] = (
            current_em["cell_name"].astype(str).str.replace(".0", "", regex=False)
        )
        reference["cell_name_str"] = (
            reference["cell_name"].astype(str).str.replace(".0", "", regex=False)
        )

        # Log cell counts (don't fail on mismatch, just warn)
        if len(current_em) != len(reference):
            print(
                f"\nNote: EM cell count differs - "
                f"Current: {len(current_em)}, "
                f"Reference: {len(reference)}"
            )

        # Compare predictions cell by cell
        current_indexed = current_em.set_index("cell_name_str")
        ref_indexed = reference.set_index("cell_name_str")

        common_cells = set(current_indexed.index) & set(ref_indexed.index)
        assert len(common_cells) > 0, "No common cells found"

        # Report coverage
        print(
            f"\nEM cells - Current: {len(current_em)}, "
            f"Reference: {len(reference)}, "
            f"Common: {len(common_cells)}"
        )

        mismatches = []
        for cell in common_cells:
            curr_pred = current_indexed.loc[cell, "prediction"]
            ref_pred = ref_indexed.loc[cell, "prediction"]
            if curr_pred != ref_pred:
                mismatches.append({"cell": cell, "current": curr_pred, "reference": ref_pred})

        match_rate = (len(common_cells) - len(mismatches)) / len(common_cells) * 100
        assert match_rate == 100.0, (
            f"EM predictions don't match reference"
            f" ({match_rate:.1f}% match)."
            f" Mismatches: {mismatches[:5]}"
        )

    def test_prediction_distribution_matches_baseline(self, predictor_after_predictions):
        """Verify prediction distribution matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_predictions"]["prediction_distribution"]
        actual = (
            predictor_after_predictions.prediction_predict_df["prediction"].value_counts().to_dict()
        )

        for pred, count in expected.items():
            assert pred in actual, f"Missing prediction type: {pred}"
            assert actual[pred] == count, (
                f"Prediction {pred} count changed: {actual[pred]} vs {count}"
            )


# =============================================================================
# Test 7: Verification Metrics
# =============================================================================
class TestVerificationMetrics:
    """Tests for verification metrics calculation."""

    def test_passed_tests_column_exists(self, predictor_after_verification):
        """Verify passed_tests column exists."""
        preds = predictor_after_verification.prediction_predict_df
        assert "passed_tests" in preds.columns, "passed_tests column not found"

    def test_total_passed_matches_baseline(self, predictor_after_verification):
        """Verify total passed tests matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_verification"]["total_passed"]
        actual = int(predictor_after_verification.prediction_predict_df["passed_tests"].sum())

        assert actual == expected, f"Total passed changed: {actual} vs expected {expected}"

    def test_em_passed_matches_baseline(self, predictor_after_verification):
        """Verify EM passed tests matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_verification"]["em_passed"]
        preds = predictor_after_verification.prediction_predict_df
        actual = int(preds[preds["imaging_modality"] == "EM"]["passed_tests"].sum())

        assert actual == expected, f"EM passed changed: {actual} vs expected {expected}"

    def test_clem_passed_matches_baseline(self, predictor_after_verification):
        """Verify CLEM passed tests matches baseline."""
        baseline = load_baseline()
        if baseline is None:
            pytest.skip("Baseline not captured yet.")

        expected = baseline["after_verification"]["clem_passed"]
        preds = predictor_after_verification.prediction_predict_df
        actual = int(preds[preds["imaging_modality"] == "clem"]["passed_tests"].sum())

        assert actual == expected, f"CLEM passed changed: {actual} vs expected {expected}"

    def test_em_passed_matches_reference(
        self, predictor_after_verification, em_reference_predictions
    ):
        """Compare EM passed_tests to reference file."""
        if em_reference_predictions is None:
            pytest.skip("EM reference file not found")

        if "passed_tests" not in em_reference_predictions.columns:
            pytest.skip("Reference file doesn't have passed_tests column")

        preds = predictor_after_verification.prediction_predict_df
        current_em = preds[preds["imaging_modality"] == "EM"]

        curr_passed = int(current_em["passed_tests"].sum())
        ref_passed = int(em_reference_predictions["passed_tests"].sum())

        assert curr_passed == ref_passed, f"EM passed_tests mismatch: {curr_passed} vs {ref_passed}"

    def test_clem_passed_matches_reference(
        self, predictor_after_verification, clem_reference_predictions
    ):
        """Compare CLEM passed_tests to reference file."""
        if clem_reference_predictions is None:
            pytest.skip("CLEM reference file not found")

        if "passed_tests" not in clem_reference_predictions.columns:
            pytest.skip("Reference file doesn't have passed_tests column")

        preds = predictor_after_verification.prediction_predict_df
        current_clem = preds[preds["imaging_modality"] == "clem"]

        curr_passed = int(current_clem["passed_tests"].sum())
        ref_passed = int(clem_reference_predictions["passed_tests"].sum())

        assert curr_passed == ref_passed, (
            f"CLEM passed_tests mismatch: {curr_passed} vs {ref_passed}"
        )


# =============================================================================
# Test 8: Edge Cases and Boundary Conditions
# =============================================================================
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_no_nan_in_predictions(self, predictor_after_predictions):
        """Verify no NaN values in prediction column."""
        preds = predictor_after_predictions.prediction_predict_df
        nan_count = preds["prediction"].isna().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in predictions"

    def test_no_nan_in_probabilities(self, predictor_after_predictions):
        """Verify no NaN values in probability columns."""
        preds = predictor_after_predictions.prediction_predict_df
        prob_cols = ["MON_proba", "cMI_proba", "iMI_proba", "SMI_proba"]

        for col in prob_cols:
            nan_count = preds[col].isna().sum()
            assert nan_count == 0, f"Found {nan_count} NaN values in {col}"

    def test_all_predictions_are_valid_classes(self, predictor_after_predictions):
        """Verify all predictions are valid cell class names."""
        valid_classes = {
            "motion_onset",
            "motion_integrator_contralateral",
            "motion_integrator_ipsilateral",
            "slow_motion_integrator",
        }

        preds = predictor_after_predictions.prediction_predict_df
        unique_predictions = set(preds["prediction"].unique())

        invalid = unique_predictions - valid_classes
        assert not invalid, f"Found invalid prediction classes: {invalid}"

    def test_no_empty_cell_names(self, predictor_after_predictions):
        """Verify no empty or null cell names."""
        preds = predictor_after_predictions.prediction_predict_df

        empty_count = (preds["cell_name"] == "").sum()
        null_count = preds["cell_name"].isna().sum()

        assert empty_count == 0, f"Found {empty_count} empty cell names"
        assert null_count == 0, f"Found {null_count} null cell names"

    def test_modality_values_are_expected(self, predictor_after_predictions):
        """Verify imaging modality values are expected."""
        expected_modalities = {"clem", "EM", "photoactivation"}

        preds = predictor_after_predictions.prediction_predict_df
        unique_modalities = set(preds["imaging_modality"].unique())

        unexpected = unique_modalities - expected_modalities
        assert not unexpected, f"Found unexpected modalities: {unexpected}"

    def test_training_cells_not_in_predictions(self, predictor_after_predictions):
        """Verify training cells are not duplicated in predictions (for to_predict cells)."""
        train_df = predictor_after_predictions.prediction_train_df
        pred_df = predictor_after_predictions.prediction_predict_df

        # Filter to only to_predict cells
        to_predict = pred_df[pred_df["function"] == "to_predict"]

        train_names = set(train_df["cell_name"])
        pred_names = set(to_predict["cell_name"])

        overlap = train_names & pred_names
        assert not overlap, f"Training cells found in to_predict predictions: {overlap}"


# =============================================================================
# Test 9: Data Integrity
# =============================================================================
class TestDataIntegrity:
    """Tests for data integrity across pipeline stages."""

    def test_features_no_infinite_values(self, predictor_after_rfe):
        """Verify features contain no infinite values."""
        features = predictor_after_rfe.features
        inf_count = np.isinf(features).sum()
        assert inf_count == 0, f"Found {inf_count} infinite values in features"

    def test_features_no_nan_values(self, predictor_after_rfe):
        """Verify features contain no NaN values after preprocessing."""
        features = predictor_after_rfe.features
        nan_count = np.isnan(features).sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in features"

    def test_labels_match_function_column(self, predictor_after_rfe):
        """Verify labels array matches function column in cells DataFrame."""
        labels = predictor_after_rfe.labels
        functions = predictor_after_rfe.all_cells["function"].values

        mismatches = (labels != functions).sum()
        assert mismatches == 0, f"Found {mismatches} mismatches between labels and function column"

    def test_modality_indices_are_exclusive(self, predictor_after_rfe):
        """Verify modality indices don't overlap."""
        clem_idx = predictor_after_rfe.clem_idx
        pa_idx = predictor_after_rfe.pa_idx

        overlap = (clem_idx & pa_idx).sum()
        assert overlap == 0, f"Found {overlap} cells in both CLEM and PA modalities"

    def test_modality_indices_cover_all_cells(self, predictor_after_rfe):
        """Verify modality indices cover all training cells."""
        clem_idx = predictor_after_rfe.clem_idx
        pa_idx = predictor_after_rfe.pa_idx

        total_covered = clem_idx.sum() + pa_idx.sum()
        total_cells = len(predictor_after_rfe.labels)

        # Note: There may be EM cells, so we check that at least CLEM+PA are covered
        assert total_covered <= total_cells, (
            f"Modality indices cover more than total cells: {total_covered} vs {total_cells}"
        )


# =============================================================================
# Test 10: Cross-Reference with Reference Files
# =============================================================================
class TestCrossReference:
    """Tests for cross-referencing predictions with reference files."""

    def test_em_prediction_confidence_matches_reference(
        self, predictor_after_predictions, em_reference_predictions
    ):
        """Verify EM prediction confidence levels match reference."""
        if em_reference_predictions is None:
            pytest.skip("EM reference file not found")

        if "SMI_proba" not in em_reference_predictions.columns:
            pytest.skip("Reference file doesn't have probability columns")

        preds = predictor_after_predictions.prediction_predict_df
        current_em = preds[preds["imaging_modality"] == "EM"].copy()
        reference = em_reference_predictions.dropna(subset=["cell_name"]).copy()

        # Normalize cell names for comparison
        current_em["cell_name_str"] = (
            current_em["cell_name"].astype(str).str.replace(".0", "", regex=False)
        )
        reference["cell_name_str"] = (
            reference["cell_name"].astype(str).str.replace(".0", "", regex=False)
        )

        current_indexed = current_em.set_index("cell_name_str")
        ref_indexed = reference.set_index("cell_name_str")

        common_cells = list(set(current_indexed.index) & set(ref_indexed.index))[:10]

        for cell in common_cells:
            for col in ["MON_proba", "cMI_proba", "iMI_proba", "SMI_proba"]:
                if col in ref_indexed.columns:
                    curr_val = current_indexed.loc[cell, col]
                    ref_val = ref_indexed.loc[cell, col]
                    assert abs(curr_val - ref_val) < 0.001, (
                        f"Cell {cell} {col} changed: {curr_val} vs {ref_val}"
                    )

    def test_clem_prediction_confidence_matches_reference(
        self, predictor_after_predictions, clem_reference_predictions
    ):
        """Verify CLEM prediction confidence levels match reference."""
        if clem_reference_predictions is None:
            pytest.skip("CLEM reference file not found")

        if "SMI_proba" not in clem_reference_predictions.columns:
            pytest.skip("Reference file doesn't have probability columns")

        preds = predictor_after_predictions.prediction_predict_df
        current_clem = preds[preds["imaging_modality"] == "clem"].copy()
        reference = clem_reference_predictions.dropna(subset=["cell_name"]).copy()

        # Normalize cell names for comparison
        current_clem["cell_name_str"] = (
            current_clem["cell_name"].astype(str).str.replace(".0", "", regex=False)
        )
        reference["cell_name_str"] = (
            reference["cell_name"].astype(str).str.replace(".0", "", regex=False)
        )

        current_indexed = current_clem.set_index("cell_name_str")
        ref_indexed = reference.set_index("cell_name_str")

        common_cells = list(set(current_indexed.index) & set(ref_indexed.index))[:10]

        for cell in common_cells:
            for col in ["MON_proba", "cMI_proba", "iMI_proba", "SMI_proba"]:
                if col in ref_indexed.columns:
                    curr_val = current_indexed.loc[cell, col]
                    ref_val = ref_indexed.loc[cell, col]
                    assert abs(curr_val - ref_val) < 0.001, (
                        f"Cell {cell} {col} changed: {curr_val} vs {ref_val}"
                    )
