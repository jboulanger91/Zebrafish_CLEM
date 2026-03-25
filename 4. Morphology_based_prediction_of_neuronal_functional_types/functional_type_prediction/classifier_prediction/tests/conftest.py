"""Pytest configuration and shared fixtures for pipeline tests.

Defines fixtures for pipeline methods:
- ClassPredictor.load_data() -> LoadedData
- ClassPredictor.select_features_rfe() -> RFEResult
- ClassPredictor.predict() -> PredictionResults
- ClassPredictor.verify() -> PredictionResults
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.ensemble import AdaBoostClassifier  # noqa: E402

try:
    from src.util.get_base_path import get_base_path  # noqa: E402
except ModuleNotFoundError:
    from util.get_base_path import get_base_path  # noqa: E402

matplotlib.use("Agg")

# =============================================================================
# Constants
# =============================================================================
DATA_PATH = get_base_path()
FEATURES_FILE = "test"

# Reference prediction files
CLEM_REFERENCE = DATA_PATH / "baselines" / "clem_baseline_predictions.xlsx"
EM_REFERENCE = DATA_PATH / "baselines" / "em_baseline_predictions.xlsx"


# =============================================================================
# Container-returning fixtures
# =============================================================================
@pytest.fixture(scope="session")
def pipeline_results():
    """Run full pipeline and return result containers.

    - load_data() returns LoadedData container
    - select_features_rfe() returns RFEResult container
    - predict() returns PredictionResults container
    - verify() returns PredictionResults with verification status
    """
    from core.class_predictor import class_predictor

    predictor = class_predictor(DATA_PATH)

    # load_data() returns LoadedData container
    # Combines: load_cells_df, calculate_metrics, load_cells_features, preprocessing
    data = predictor.load_data(
        features_file=FEATURES_FILE,
        modalities=["pa", "clem241211", "em", "clem_predict241211"],
        use_stored_features=True,
    )

    # select_features_rfe() accepts data, returns RFEResult
    rfe_result = predictor.select_features_rfe(
        data=data,
        train_mod="all",
        test_mod="clem",
        cv_method_rfe="ss",
        estimator=AdaBoostClassifier(random_state=0),
        metric="f1",
    )

    # cross_validate() returns CVResult (optional step)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    cv_result = predictor.cross_validate(
        data=data,
        selected_features=rfe_result.selected_features_idx,
        classifier=clf,
        method="lpo",
    )

    # predict() accepts data and features, returns PredictionResults
    predictions = predictor.predict(
        data=data,
        selected_features=rfe_result.selected_features_idx,
        use_jon_priors=False,
        suffix="_optimize_all_predict",
    )

    # verify() updates PredictionResults with verification
    verified = predictor.verify(
        predictions=predictions,
        data=data,
        required_tests=["IF", "LOF"],
    )

    return {
        "predictor": predictor,
        "data": data,
        "rfe": rfe_result,
        "cv": cv_result,
        "predictions": verified,
    }


# Convenience fixtures for accessing container components
@pytest.fixture(scope="session")
def data(pipeline_results):
    """LoadedData container from pipeline."""
    return pipeline_results["data"]


@pytest.fixture(scope="session")
def rfe_result(pipeline_results):
    """RFEResult container from pipeline."""
    return pipeline_results["rfe"]


@pytest.fixture(scope="session")
def cv_result(pipeline_results):
    """CVResult container from pipeline."""
    return pipeline_results["cv"]


@pytest.fixture(scope="session")
def predictions(pipeline_results):
    """PredictionResults container from pipeline."""
    return pipeline_results["predictions"]


# =============================================================================
# Attribute-based fixtures
# =============================================================================
@pytest.fixture(scope="session")
def predictor_after_load_cells():
    """class_predictor after load_cells_df."""
    from core.class_predictor import class_predictor

    pred = class_predictor(DATA_PATH)
    pred.load_cells_df(
        modalities=["pa", "clem241211", "em", "clem_predict241211"],
    )
    return pred


@pytest.fixture(scope="session")
def predictor_after_load_features(predictor_after_load_cells):
    """class_predictor after calculate_metrics and load_cells_features."""
    pred = predictor_after_load_cells
    pred.calculate_metrics(
        FEATURES_FILE, force_new=False, use_stored_features=True,
    )
    pred.load_cells_features(FEATURES_FILE, drop_neurotransmitter=False)
    return pred


@pytest.fixture(scope="session")
def predictor_after_preprocessing(predictor_after_load_features):
    """class_predictor after load_cells_features (morphology sync is automatic)."""
    return predictor_after_load_features


@pytest.fixture(scope="session")
def predictor_after_rfe(predictor_after_preprocessing):
    """class_predictor after RFE feature selection."""
    pred = predictor_after_preprocessing
    pred.select_features_RFE(
        "all",
        "clem",
        save_features=True,
        cv_method_rfe="ss",
        estimator=AdaBoostClassifier(random_state=0),
        metric="f1",
    )
    return pred


@pytest.fixture(scope="session")
def predictor_after_confusion_matrices(predictor_after_rfe):
    """class_predictor after confusion matrices."""
    pred = predictor_after_rfe
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    pred.confusion_matrices(
        clf, method="lpo", fraction_across_classes=True
    )
    return pred


@pytest.fixture(scope="session")
def predictor_after_predictions(predictor_after_confusion_matrices):
    """class_predictor after predict_cells."""
    pred = predictor_after_confusion_matrices
    pred.predict_cells(use_jon_priors=False, suffix="_optimize_all_predict", save_predictions=False)
    return pred


@pytest.fixture(scope="session")
def predictor_after_verification(predictor_after_predictions):
    """class_predictor after calculate_verification_metrics."""
    pred = predictor_after_predictions
    pred.calculate_verification_metrics(
        calculate_smat=False, with_kunst=False, required_tests=["IF", "LOF"], force_new=True
    )
    return pred


# =============================================================================
# Reference data fixtures
# =============================================================================
@pytest.fixture(scope="session")
def clem_reference_predictions():
    """Load CLEM reference prediction file."""
    if CLEM_REFERENCE.exists():
        return pd.read_excel(CLEM_REFERENCE)
    return None


@pytest.fixture(scope="session")
def em_reference_predictions():
    """Load EM reference prediction file."""
    if EM_REFERENCE.exists():
        return pd.read_excel(EM_REFERENCE)
    return None

