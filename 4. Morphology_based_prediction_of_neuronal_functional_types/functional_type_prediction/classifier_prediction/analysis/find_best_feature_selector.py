"""Find Best Feature Selector for Cell Type Classification.

This script performs Recursive Feature Elimination (RFE) to identify the optimal
feature selector and number of features for classifying zebrafish hindbrain neurons
into functional types (MI, MON, SMI).

The script:
1. Loads morphological and functional data for neurons from multiple modalities
   (CLEM, photoactivation, EM)
2. Calculates morphological metrics if not already cached
3. Removes incomplete neurons (truncated, growth cones, exits volume)
4. Applies manual morphology annotations
5. Performs RFE with multiple classifiers to find the best feature set
6. Generates plots showing feature selection performance

Output:
    - RFE plots saved to: classifier_pipeline/find_feature_selector/
    - Console output showing selected features and performance metrics

Notes
-----
    - MI = Motion Integrator (formerly "integrator")
    - MON = Motion Onset (formerly "dynamic_threshold")
    - SMI = Slow Motion Integrator (formerly "motor_command")
    - Features include cable length, branching patterns, spatial extent, etc.

Author: Florian Kämpf
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

# Path setup for local imports
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLASSIFIER_DIR = _REPO_ROOT / "functional_type_prediction" / "classifier_prediction"
_SRC = _REPO_ROOT / "src"
if str(_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLASSIFIER_DIR))  # Needed for bare core.* and util.* imports
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))  # Required for standalone execution

# Local imports
from src.util.get_base_path import get_base_path  # noqa: E402

from core.class_predictor import class_predictor  # noqa: E402

# Configure matplotlib to use non-interactive backend
# This prevents plots from blocking execution and requiring manual closure
matplotlib.use('Agg')  # Use non-GUI backend that saves to files only

# NumPy print options
np.set_printoptions(suppress=True)


def main(
    data_path=None,
    features_file='final',
    modalities=None,
    train_mod="all",
    test_mod="clem",
    cv_method="ss",
    metric="f1",
):
    """Run RFE feature selection analysis with multiple classifiers.

    Parameters
    ----------
    data_path : Path, optional
        Base data directory. Uses get_base_path() if None.
    features_file : str, optional
        HDF5 features file name. Default: FINAL_CLEM_CLEMPREDICT_EM_with_clem241211_withgregor250220.
    modalities : list of str, optional
        Imaging modalities to load. Default: pa, clem, em, clem_predict.
    train_mod : str
        Training modality for RFE. Default: 'all'.
    test_mod : str
        Test modality for RFE. Default: 'clem'.
    cv_method : str
        CV method for RFE ('ss' or 'lpo'). Default: 'ss'.
    metric : str
        Optimization metric ('f1', 'accuracy'). Default: 'f1'.
    """
    if data_path is None:
        data_path = get_base_path()
    if modalities is None:
        modalities = ["pa", "clem", "em", "clem_predict"]
    if features_file is None:
        features_file = (
            "FINAL_CLEM_CLEMPREDICT_EM_with_clem241211"
            "_withgregor250220"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data path does not exist: {data_path}\n"
            f"Please check config/path_configuration.txt."
        )

    print("=" * 80)
    print("Feature Selection Analysis for Zebrafish Hindbrain Neurons")
    print("=" * 80)

    predictor = class_predictor(data_path)
    print("\nLoading cell data...")
    predictor.load_cells_df(modalities=modalities)

    print("\nCalculating/loading morphological metrics...")
    predictor.calculate_metrics(features_file)

    print("\nLoading features...")
    predictor.load_cells_features(features_file, drop_neurotransmitter=False)

    print("\nPerforming Recursive Feature Elimination (RFE)...")
    predictor.select_features_RFE(
        train_mod=train_mod,
        test_mod=test_mod,
        cv_method_rfe=cv_method,
        metric=metric,
        estimator=None,
        output_subdir="find_feature_selector",
    )

    print("\n" + "=" * 80)
    print("Feature selection analysis complete!")
    print("Results saved to: classifier_pipeline/find_feature_selector/")
    print("=" * 80)



if __name__ == "__main__":
    main()
