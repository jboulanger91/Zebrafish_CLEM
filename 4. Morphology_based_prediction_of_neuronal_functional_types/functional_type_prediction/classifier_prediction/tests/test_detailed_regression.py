"""Detailed regression tests for pipeline_main.py.

These tests provide comprehensive coverage of intermediate values, confusion matrix
details, probability scores, and DataFrame column verification to ensure refactoring
doesn't change any behavioral outputs.

Usage:
    pytest test_detailed_regression.py -v
    pytest test_detailed_regression.py -v -k "test_confusion"  # Run specific test group
"""

import json
import sys
from pathlib import Path

import numpy as np
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
DETAILED_BASELINE_FILE = _TEST_DIR / "detailed_baseline_values.json"


def load_baseline():
    """Load baseline values from JSON file."""
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return None


def load_detailed_baseline():
    """Load detailed baseline values from JSON file."""
    if DETAILED_BASELINE_FILE.exists():
        with open(DETAILED_BASELINE_FILE) as f:
            return json.load(f)
    return None


# =============================================================================
# Test: Confusion Matrix Values
# =============================================================================
class TestConfusionMatrixValues:
    """Tests for actual confusion matrix values."""

    def test_confusion_matrix_exists(self, predictor_after_confusion_matrices):
        """Verify confusion matrix attribute exists."""
        assert hasattr(predictor_after_confusion_matrices, "cm"), (
            "Confusion matrix 'cm' attribute not found"
        )

    def test_confusion_matrix_shape(self, predictor_after_confusion_matrices):
        """Verify confusion matrix has correct shape (4x4 for 4 classes)."""
        cm = predictor_after_confusion_matrices.cm
        assert cm.shape == (4, 4), f"Expected (4, 4) shape, got {cm.shape}"

    def test_confusion_matrix_values_match_baseline(self, predictor_after_confusion_matrices):
        """Verify confusion matrix values match baseline."""
        baseline = load_detailed_baseline()
        if baseline is None:
            pytest.skip(
                "Detailed baseline not captured yet. Run capture_detailed_baseline.py first."
            )

        expected_cm = np.array(baseline["confusion_matrix"]["values"])
        actual_cm = predictor_after_confusion_matrices.cm

        # Allow small floating point tolerance
        np.testing.assert_array_almost_equal(
            actual_cm, expected_cm, decimal=4, err_msg="Confusion matrix values changed"
        )

    def test_confusion_matrix_diagonal_values(self, predictor_after_confusion_matrices):
        """Verify diagonal values (true positive rates) are preserved."""
        baseline = load_detailed_baseline()
        if baseline is None:
            pytest.skip("Detailed baseline not captured yet.")

        expected_diag = baseline["confusion_matrix"]["diagonal"]
        actual_diag = np.diag(predictor_after_confusion_matrices.cm)

        np.testing.assert_array_almost_equal(
            actual_diag,
            expected_diag,
            decimal=4,
            err_msg="Confusion matrix diagonal values changed",
        )

    def test_confusion_matrix_row_sums(self, predictor_after_confusion_matrices):
        """Verify row sums are approximately 1 (normalized by true labels)."""
        cm = predictor_after_confusion_matrices.cm
        row_sums = cm.sum(axis=1)

        # For normalized confusion matrix, rows should sum to ~1
        np.testing.assert_array_almost_equal(
            row_sums,
            np.ones(4),
            decimal=2,
            err_msg="Confusion matrix rows don't sum to 1 (normalization issue)",
        )


# =============================================================================
# Test: Prediction Probability Columns
# =============================================================================
class TestPredictionProbabilities:
    """Tests for prediction probability values."""

    def test_probability_columns_exist(self, predictor_after_predictions):
        """Verify probability columns exist in prediction DataFrame."""
        required_cols = ["MON_proba", "cMI_proba", "iMI_proba", "SMI_proba"]
        preds = predictor_after_predictions.prediction_predict_df

        for col in required_cols:
            assert col in preds.columns, f"Missing probability column: {col}"

    def test_probability_columns_scaled_exist(self, predictor_after_predictions):
        """Verify scaled probability columns exist."""
        required_cols = ["MON_proba_scaled", "cMI_proba_scaled", "iMI_proba_scaled", "SMI_proba_scaled"]
        preds = predictor_after_predictions.prediction_predict_df

        for col in required_cols:
            assert col in preds.columns, f"Missing scaled probability column: {col}"

    def test_probabilities_sum_to_one(self, predictor_after_predictions):
        """Verify unscaled probabilities sum to approximately 1 for each cell."""
        preds = predictor_after_predictions.prediction_predict_df
        prob_cols = ["MON_proba", "cMI_proba", "iMI_proba", "SMI_proba"]

        prob_sums = preds[prob_cols].sum(axis=1)

        # Check that all sums are approximately 1
        assert (np.abs(prob_sums - 1.0) < 0.01).all(), (
            f"Some probability rows don't sum to 1. Range: [{prob_sums.min()}, {prob_sums.max()}]"
        )

    def test_probabilities_in_valid_range(self, predictor_after_predictions):
        """Verify probabilities are in [0, 1] range."""
        preds = predictor_after_predictions.prediction_predict_df
        prob_cols = ["MON_proba", "cMI_proba", "iMI_proba", "SMI_proba"]

        for col in prob_cols:
            assert preds[col].min() >= 0, f"{col} has negative values"
            assert preds[col].max() <= 1, f"{col} has values > 1"

    def test_prediction_matches_max_probability(self, predictor_after_predictions):
        """Verify prediction matches the class with max probability."""
        preds = predictor_after_predictions.prediction_predict_df
        prob_cols = ["MON_proba", "cMI_proba", "iMI_proba", "SMI_proba"]
        class_names = [
            "motion_onset",
            "motion_integrator_contralateral",
            "motion_integrator_ipsilateral",
            "slow_motion_integrator",
        ]

        # Get predicted class from max probability
        max_prob_idx = preds[prob_cols].values.argmax(axis=1)
        expected_predictions = [class_names[i] for i in max_prob_idx]

        mismatches = (preds["prediction"].values != expected_predictions).sum()
        assert mismatches == 0, f"{mismatches} predictions don't match max probability"

    def test_probability_statistics_match_baseline(self, predictor_after_predictions):
        """Verify probability statistics match baseline."""
        baseline = load_detailed_baseline()
        if baseline is None:
            pytest.skip("Detailed baseline not captured yet.")

        preds = predictor_after_predictions.prediction_predict_df
        prob_cols = ["MON_proba", "cMI_proba", "iMI_proba", "SMI_proba"]

        for col in prob_cols:
            expected_mean = baseline["probability_stats"][col]["mean"]
            expected_std = baseline["probability_stats"][col]["std"]

            actual_mean = preds[col].mean()
            actual_std = preds[col].std()

            assert abs(actual_mean - expected_mean) < 0.001, (
                f"{col} mean changed: {actual_mean} vs {expected_mean}"
            )
            assert abs(actual_std - expected_std) < 0.001, (
                f"{col} std changed: {actual_std} vs {expected_std}"
            )


# =============================================================================
# Test: Scaled Predictions
# =============================================================================
class TestScaledPredictions:
    """Tests for scaled prediction values."""

    def test_prediction_scaled_column_exists(self, predictor_after_predictions):
        """Verify prediction_scaled column exists."""
        preds = predictor_after_predictions.prediction_predict_df
        assert "prediction_scaled" in preds.columns, "Missing prediction_scaled column"

    def test_prediction_scaled_distribution_matches_baseline(self, predictor_after_predictions):
        """Verify scaled prediction distribution matches baseline."""
        baseline = load_detailed_baseline()
        if baseline is None:
            pytest.skip("Detailed baseline not captured yet.")

        preds = predictor_after_predictions.prediction_predict_df
        expected = baseline["prediction_scaled_distribution"]
        actual = preds["prediction_scaled"].value_counts().to_dict()

        for pred, count in expected.items():
            assert pred in actual, f"Missing scaled prediction type: {pred}"
            assert actual[pred] == count, (
                f"Scaled prediction {pred} count changed: {actual[pred]} vs {count}"
            )

    def test_scaled_matches_max_scaled_probability(self, predictor_after_predictions):
        """Verify prediction_scaled matches the class with max scaled probability."""
        preds = predictor_after_predictions.prediction_predict_df
        scaled_cols = ["MON_proba_scaled", "cMI_proba_scaled", "iMI_proba_scaled", "SMI_proba_scaled"]
        class_names = [
            "motion_onset",
            "motion_integrator_contralateral",
            "motion_integrator_ipsilateral",
            "slow_motion_integrator",
        ]

        max_prob_idx = preds[scaled_cols].values.argmax(axis=1)
        expected_predictions = [class_names[i] for i in max_prob_idx]

        mismatches = (preds["prediction_scaled"].values != expected_predictions).sum()
        assert mismatches == 0, (
            f"{mismatches} scaled predictions don't match max scaled probability"
        )


# =============================================================================
# Test: Feature Values
# =============================================================================
class TestFeatureValues:
    """Tests for feature values and statistics."""

    def test_feature_shape_matches_baseline(self, predictor_after_rfe):
        """Verify feature matrix shape matches baseline."""
        baseline = load_detailed_baseline()
        if baseline is None:
            pytest.skip("Detailed baseline not captured yet.")

        expected_shape = tuple(baseline["feature_stats"]["shape"])
        actual_shape = predictor_after_rfe.features.shape

        assert actual_shape == expected_shape, (
            f"Feature shape changed: {actual_shape} vs {expected_shape}"
        )

    def test_feature_statistics_match_baseline(self, predictor_after_rfe):
        """Verify feature statistics (mean, std, min, max) match baseline."""
        baseline = load_detailed_baseline()
        if baseline is None:
            pytest.skip("Detailed baseline not captured yet.")

        features = predictor_after_rfe.features
        expected = baseline["feature_stats"]

        # Test overall statistics
        assert abs(features.mean() - expected["overall_mean"]) < 0.01, (
            f"Overall mean changed: {features.mean()} vs {expected['overall_mean']}"
        )
        assert abs(features.std() - expected["overall_std"]) < 0.01, (
            f"Overall std changed: {features.std()} vs {expected['overall_std']}"
        )

    def test_reduced_features_idx_matches_baseline(self, predictor_after_rfe):
        """Verify reduced features index matches baseline."""
        baseline = load_detailed_baseline()
        if baseline is None:
            pytest.skip("Detailed baseline not captured yet.")

        expected = np.array(baseline["reduced_features_idx"])
        actual = predictor_after_rfe.reduced_features_idx

        np.testing.assert_array_equal(actual, expected, err_msg="Reduced features index changed")

    def test_selected_feature_names(self, predictor_after_rfe):
        """Verify selected feature names match baseline."""
        baseline = load_detailed_baseline()
        if baseline is None:
            pytest.skip("Detailed baseline not captured yet.")

        expected_names = baseline["selected_feature_names"]
        actual_names = [
            predictor_after_rfe.column_labels[i]
            for i, selected in enumerate(predictor_after_rfe.reduced_features_idx)
            if selected
        ]

        assert set(actual_names) == set(expected_names), (
            f"Selected feature names changed.\nExpected: {expected_names}\nActual: {actual_names}"
        )


# =============================================================================
# Test: Verification Metrics Details
# =============================================================================
class TestVerificationMetricsDetails:
    """Tests for detailed verification metrics."""

    def test_outlier_detection_columns_exist(self, predictor_after_verification):
        """Verify outlier detection columns exist."""
        required_cols = [
            "OCSVM",
            "IF",
            "LOF",
            "OCSVM_intra_class",
            "IF_intra_class",
            "LOF_intra_class",
        ]
        preds = predictor_after_verification.prediction_predict_df

        for col in required_cols:
            assert col in preds.columns, f"Missing outlier detection column: {col}"

    def test_outlier_detection_values_are_boolean(self, predictor_after_verification):
        """Verify outlier detection columns contain boolean values."""
        bool_cols = ["OCSVM", "IF", "LOF", "OCSVM_intra_class", "IF_intra_class", "LOF_intra_class"]
        preds = predictor_after_verification.prediction_predict_df

        for col in bool_cols:
            unique_vals = preds[col].unique()
            assert all(v in [True, False, 0, 1] for v in unique_vals), (
                f"{col} contains non-boolean values: {unique_vals}"
            )

    def test_outlier_detection_counts_match_baseline(self, predictor_after_verification):
        """Verify outlier detection pass counts match baseline."""
        baseline = load_detailed_baseline()
        if baseline is None:
            pytest.skip("Detailed baseline not captured yet.")

        preds = predictor_after_verification.prediction_predict_df

        for col in ["IF", "LOF"]:  # Required tests from pipeline
            expected = baseline["verification_counts"].get(col, None)
            if expected is not None:
                actual = int(preds[col].sum())
                assert actual == expected, f"{col} pass count changed: {actual} vs {expected}"

    def test_passed_tests_calculation_correct(self, predictor_after_verification):
        """Verify passed_tests column correctly reflects required tests."""
        preds = predictor_after_verification.prediction_predict_df

        # Based on pipeline_main.py: required_tests=['IF', 'LOF']
        expected_passed = preds["IF"] & preds["LOF"]

        if "passed_tests" in preds.columns:
            mismatches = (preds["passed_tests"] != expected_passed).sum()
            assert mismatches == 0, f"{mismatches} rows have incorrect passed_tests value"


# =============================================================================
# Test: DataFrame Column Verification
# =============================================================================
class TestDataFrameColumns:
    """Tests for DataFrame column completeness and types."""

    def test_prediction_df_required_columns(self, predictor_after_predictions):
        """Verify all required columns exist in prediction DataFrame."""
        required_cols = [
            "cell_name",
            "function",
            "imaging_modality",
            "morphology",
            "neurotransmitter",
            "prediction",
            "prediction_scaled",
            "MON_proba",
            "cMI_proba",
            "iMI_proba",
            "SMI_proba",
            "MON_proba_scaled",
            "cMI_proba_scaled",
            "iMI_proba_scaled",
            "SMI_proba_scaled",
        ]

        preds = predictor_after_predictions.prediction_predict_df
        missing = [col for col in required_cols if col not in preds.columns]

        assert not missing, f"Missing columns: {missing}"

    def test_prediction_df_column_types(self, predictor_after_predictions):
        """Verify column types are correct."""
        preds = predictor_after_predictions.prediction_predict_df

        # String columns
        string_cols = [
            "cell_name",
            "function",
            "imaging_modality",
            "prediction",
            "prediction_scaled",
        ]
        for col in string_cols:
            if col in preds.columns:
                assert preds[col].dtype == "object" or preds[col].dtype.name == "string", (
                    f"{col} should be string type, got {preds[col].dtype}"
                )

        # Numeric columns
        numeric_cols = ["MON_proba", "cMI_proba", "iMI_proba", "SMI_proba"]
        for col in numeric_cols:
            if col in preds.columns:
                assert np.issubdtype(preds[col].dtype, np.number), (
                    f"{col} should be numeric type, got {preds[col].dtype}"
                )

    def test_all_cells_df_columns_match_baseline(self, predictor_after_preprocessing):
        """Verify all_cells_with_to_predict columns match baseline."""
        baseline = load_detailed_baseline()
        if baseline is None:
            pytest.skip("Detailed baseline not captured yet.")

        expected_cols = set(baseline["all_cells_columns"])
        actual_cols = set(predictor_after_preprocessing.all_cells_with_to_predict.columns)

        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        assert not missing, f"Missing columns: {missing}"
        # Note: Extra columns may be OK, just log them
        if extra:
            print(f"Note: Extra columns found: {extra}")


# =============================================================================
# Test: Preprocessing Effects
# =============================================================================
class TestPreprocessingEffects:
    """Tests for preprocessing step effects."""

    def test_morphology_annotation_applied(self, predictor_after_preprocessing):
        """Verify morphology annotation is applied to MON cells."""
        cells = predictor_after_preprocessing.all_cells_with_to_predict
        mon_cells = cells[cells["function"] == "motion_onset"]

        # MON cells should have morphology values
        if len(mon_cells) > 0:
            morphology_values = mon_cells["morphology_clone"].unique()
            assert len(morphology_values) > 0, "MON cells have no morphology annotations"


# =============================================================================
# Test: Training Data Consistency
# =============================================================================
class TestTrainingDataConsistency:
    """Tests for training data consistency."""

    def test_training_features_shape_matches_labels(self, predictor_after_predictions):
        """Verify training features and labels have matching shapes."""
        features = predictor_after_predictions.prediction_train_features
        labels = predictor_after_predictions.prediction_train_labels

        assert features.shape[0] == len(labels), (
            f"Feature rows ({features.shape[0]}) don't match label count ({len(labels)})"
        )

    def test_training_df_matches_features(self, predictor_after_predictions):
        """Verify training DataFrame matches training features count."""
        df = predictor_after_predictions.prediction_train_df
        features = predictor_after_predictions.prediction_train_features

        assert len(df) == features.shape[0], (
            f"Training df ({len(df)}) doesn't match features ({features.shape[0]})"
        )

    def test_prediction_features_shape_matches_df(self, predictor_after_predictions):
        """Verify prediction features match prediction DataFrame."""
        df = predictor_after_predictions.prediction_predict_df
        features = predictor_after_predictions.prediction_predict_features

        assert len(df) == features.shape[0], (
            f"Prediction df ({len(df)}) doesn't match features ({features.shape[0]})"
        )


# =============================================================================
# Test: Cell Name Consistency
# =============================================================================
class TestCellNameConsistency:
    """Tests for cell name consistency across pipeline stages."""

    def test_no_cell_name_duplicates_in_predictions(self, predictor_after_predictions):
        """Verify no duplicate cell names in predictions."""
        preds = predictor_after_predictions.prediction_predict_df
        duplicates = preds["cell_name"].duplicated().sum()

        assert duplicates == 0, f"Found {duplicates} duplicate cell names in predictions"

    def test_cell_names_in_predictions_subset_of_all_cells(self, predictor_after_predictions):
        """Verify prediction cell names are subset of all cells."""
        all_cells = predictor_after_predictions.all_cells_with_to_predict
        preds = predictor_after_predictions.prediction_predict_df

        all_names = set(all_cells["cell_name"])
        pred_names = set(preds["cell_name"])

        not_in_all = pred_names - all_names
        assert len(not_in_all) == 0, f"Prediction contains cells not in all_cells: {not_in_all}"

    def test_specific_cells_predictions_match_baseline(self, predictor_after_predictions):
        """Verify specific known cells have correct predictions."""
        baseline = load_detailed_baseline()
        if baseline is None:
            pytest.skip("Detailed baseline not captured yet.")

        preds = predictor_after_predictions.prediction_predict_df.set_index("cell_name")
        expected_predictions = baseline.get("specific_cell_predictions", {})

        for cell_name, expected_pred in expected_predictions.items():
            if cell_name in preds.index:
                actual_pred = preds.loc[cell_name, "prediction"]
                assert actual_pred == expected_pred, (
                    f"Cell {cell_name} prediction changed: {actual_pred} vs {expected_pred}"
                )
