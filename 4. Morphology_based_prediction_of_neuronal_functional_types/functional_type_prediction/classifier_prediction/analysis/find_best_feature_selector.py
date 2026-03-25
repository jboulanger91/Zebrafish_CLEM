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


def main():
    """Main execution function for feature selection analysis.

    This function:
    1. Initializes the class predictor with data path
    2. Loads cell data from multiple imaging modalities
    3. Calculates or loads morphological metrics
    4. Filters out incomplete neurons
    5. Applies manual morphology annotations
    6. Performs RFE feature selection with multiple classifiers

    The analysis compares performance with and without neurotransmitter information
    to determine which features are most important for classification.
    """
    print("=" * 80)
    print("Feature Selection Analysis for Zebrafish Hindbrain Neurons")
    print("=" * 80)

    # Data path resolved from config/path_configuration.txt
    # Was: /Users/.../hindbrain_structure_function/nextcloud
    data_path = get_base_path()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data path does not exist: {data_path}\n"
            f"Please check config/path_configuration.txt."
        )

    # Feature set identifier
    # FINAL_CLEM_CLEMPREDICT_EM_with_clem241211_withgregor250220
    # comparison_test3_refactored
    feature_set_name = (
        'FINAL_CLEM_CLEMPREDICT_EM_with_clem241211'
        '_withgregor250220'
    )

    # ========================================================================
    # Analysis WITH neurotransmitter information
    # ========================================================================
    print("\n" + "=" * 80)
    print("Running Feature Selection WITH Neurotransmitter Information")
    print("=" * 80)

    # Initialize predictor
    with_neurotransmitter = class_predictor(data_path)

    # Load cell data from multiple modalities
    print("\nLoading cell data...")
    with_neurotransmitter.load_cells_df(
        modalities=['pa', 'clem241211', 'em', 'clem_predict241211']
    )

    # Calculate morphological metrics (cached if already computed)
    print("\nCalculating/loading morphological metrics...")
    with_neurotransmitter.calculate_metrics(feature_set_name)

    # Load computed features
    print("\nLoading features...")
    with_neurotransmitter.load_cells_features(
        feature_set_name,
        drop_neurotransmitter=False
    )

    # Perform RFE feature selection
    print("\nPerforming Recursive Feature Elimination (RFE)...")
    print("This will test multiple classifiers and save plots to classifier_pipeline/find_feature_selector/")
    with_neurotransmitter.select_features_RFE(
        train_mod='all',
        test_mod='clem',
        cv_method_rfe='ss',  # ShuffleSplit cross-validation
        metric='f1',          # Optimize for F1 score
        estimator=None, #AdaBoostClassifier(random_state=0)
        output_subdir='find_feature_selector',
    )

    print("\n" + "=" * 80)
    print("Feature selection analysis complete!")
    print("=" * 80)
    print("\nResults saved to: classifier_pipeline/find_feature_selector/")
    print("\nCheck the plots to see which estimator and number of features")
    print("gave the best F1 score for CLEM neuron classification.")



if __name__ == "__main__":
    main()
