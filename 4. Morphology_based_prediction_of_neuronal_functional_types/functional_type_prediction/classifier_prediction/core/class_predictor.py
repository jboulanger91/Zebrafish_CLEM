"""ClassPredictor - Orchestrator for cell type classification pipeline.

This module provides the main interface for:
- Loading morphological features from HDF5 files
- Feature selection via RFE (delegates to feature_selector.py)
- Cross-validation evaluation (delegates to cross_validator.py)
- Cell type prediction (delegates to predictor.py)
- Prediction verification (delegates to verification.py)

Usage:
    predictor = ClassPredictor()
    predictor.load_cells(cells_path)
    predictor.load_metrics(features_path)
    predictor.select_features_RFE(estimator)
    predictor.predict_cells(output_path)

See Also
--------
    - core/cross_validator.py: ModalityCrossValidator for CV evaluation
    - core/predictor.py: PredictionPipeline for predictions
    - core/feature_selector.py: RFESelector for feature selection
    - core/verification.py: VerificationCalculator for validation
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: E402 - explicit import replacing wildcard
import navis  # noqa: E402 - explicit import replacing wildcard
import numpy as np  # noqa: E402 - explicit import replacing wildcard
import pandas as pd  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Path setup for local imports (src/ on sys.path)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))  # Required for standalone execution

try:
    from src.util.output_paths import get_output_dir
except ModuleNotFoundError:
    from util.output_paths import get_output_dir

# Local imports (after sys.path setup for local module access)
from core.config import (  # noqa: E402
    CELL_TYPE_COLORS,
    CellTypePriors,
)
from core.containers import (  # noqa: E402
    CellDataset,
    LoadedData,
    ModalityMask,
    PredictionResults,
    TrainingData,
)
from core.cross_validator import CVResult, ModalityCrossValidator  # noqa: E402

from core.data_loader import DataLoader  # noqa: E402
from core.feature_selector import FeatureSelector, RFEResult, RFESelector  # noqa: E402
from core.predictor import PredictionPipeline  # noqa: E402
from core.utilities import (  # noqa: E402
    check_swc_validity,
)
from core.verification import VerificationCalculator  # noqa: E402
from utils.calculate_metric2df import calculate_metric2df  # noqa: E402

np.set_printoptions(suppress=True)


class ClassPredictor:
    """Orchestrator for morphological cell type classification pipeline.

    This class coordinates the full classification workflow:
    1. Data loading from HDF5 feature files
    2. Feature selection via RFE or SelectKBest
    3. Cross-validation evaluation
    4. Cell type prediction with probability scaling
    5. Prediction verification via NBLAST similarity

    The class acts as a thin wrapper, delegating to specialized modules:
    - DataLoader: HDF5 I/O and preprocessing
    - FeatureSelector/RFESelector: Feature selection algorithms
    - ModalityCrossValidator: Cross-validation evaluation
    - PredictionPipeline: Classification and probability scaling
    - VerificationCalculator: NBLAST-based verification

    Attributes
    ----------
    path : pathlib.Path
        Base path to data directory.
    features_with_to_predict : numpy.ndarray
        Morphological features including 'to_predict'.
        (Historical note: 'morph' prefix refers to original author initials.)
    labels_with_to_predict : numpy.ndarray
        Function labels including 'to_predict'.
    labels_imaging_modality_with_to_predict : numpy.ndarray
        Imaging modality labels including 'to_predict'.
    features : numpy.ndarray
        Morphological features for training cells only (excludes to_predict).
    labels : numpy.ndarray
        Function labels for training cells only.
    labels_imaging_modality : numpy.ndarray
        Imaging modality labels for training cells only.
    all_cells : pandas.DataFrame
        DataFrame containing training cells (excludes to_predict).
    cells : pandas.DataFrame
        DataFrame containing all cells present in the loaded feature file.
    clem_idx : numpy.ndarray
        Boolean mask for cells with 'clem' (correlative light-EM) imaging modality.
    pa_idx : numpy.ndarray
        Boolean mask for cells with 'photoactivation' imaging modality.
    em_idx : numpy.ndarray
        Boolean mask for cells with 'EM' (electron microscopy) imaging modality.
    reduced_features_idx : numpy.ndarray
        Boolean mask indicating selected features after RFE.
    cm : numpy.ndarray
        Most recent confusion matrix from cross-validation.
    prediction_predict_df : pandas.DataFrame
        DataFrame containing prediction results after predict_cells().

    Example:
    -------
    >>> from pathlib import Path
    >>> predictor = ClassPredictor(Path("/data/nextcloud"))
    >>> predictor.load_cells_df(modalities=["clem", "pa"])
    >>> predictor.load_cells_features("FINAL_features")
    >>> predictor.select_features_RFE("all", "clem", save_features=True)
    >>> predictor.predict_cells(use_jon_priors=True)
    >>> predictor.calculate_verification_metrics()
    >>>
    >>> # Access results
    >>> print(f"Predicted {len(predictor.prediction_predict_df)} cells")

    See Also
    --------
    core.data_loader.DataLoader : Data loading and preprocessing
    core.feature_selector.RFESelector : Recursive feature elimination
    core.cross_validator.ModalityCrossValidator : Cross-validation
    core.predictor.PredictionPipeline : Prediction pipeline
    core.verification.VerificationCalculator : NBLAST verification

    Notes
    -----
    The pipeline assumes cells have been preprocessed and resampled
    to 1 micron resolution. Feature files must be in HDF5 format
    with 'complete_df' dataset containing the feature matrix.

    The 'morph' prefix in attribute names is historical (original author initials)
    and refers to morphological features. The 'pa' abbreviation refers to
    photoactivation imaging modality.
    """

    def __init__(self, path: Path) -> None:
        """Initialize the class predictor with a data path.

        Parameters
        ----------
        path : pathlib.Path
            Base path to the data directory containing features, predictions,
            and output folders.
        """
        print("=" * 80)
        print("Initializing Class Predictor")
        print("=" * 80)
        print(f"Data path: {path}")

        self.path = path

        # Cell type colors from config module
        self.color_dict = CELL_TYPE_COLORS

        # Output paths
        self.path_to_save_confusion_matrices = get_output_dir(
            "classifier_pipeline", "confusion_matrices"
        )

        # Cell type priors (anatomical ratios from Jons CLEM counts)
        # Total: 539 neurons. MON: 22, cMI: 155.5, iMI: 155.5, SMI: 206
        # See CellTypePriors docstring in core.config for details
        self._priors = CellTypePriors()
        self.real_cell_class_ratio_dict = self._priors.as_dict()

    def load_data(
        self,
        features_file: str,
        modalities: list = None,
        use_stored_features: bool = False,
        force_recalculation: bool = False,
        label_column: str = "kmeans_function",
    ) -> LoadedData:
        """Load complete pipeline data and return as LoadedData container.

        This unified loading method combines load_cells_df(), calculate_metrics(),
        load_cells_features(), and preprocessing into a single call that returns
        all data in a clean container.

        Args:
            features_file: Name of the HDF5 features file (without path and
                _features.hdf5 suffix).
            modalities: List of imaging modalities to load. Default is
                ['pa', 'clem'] if None.
            use_stored_features: If True, use existing HDF5 features file
                without recalculation. Default False.
            force_recalculation: If True, force recalculation of cached
                metrics. Default False.

        Returns
        -------
            LoadedData container with:
                - training_data: TrainingData with train/predict split
                - feature_names: List of feature column names
                - cells_df: DataFrame of training cells
                - cells_with_to_predict_df: Full DataFrame including to_predict

        Example:
            >>> predictor = ClassPredictor(Path("/data"))
            >>> data = predictor.load_data(
            ...     features_file="FINAL_features",
            ...     modalities=["pa", "clem"],
            ... )
            >>> print(f"Training: {data.n_training} cells")
            >>> print(f"Features: {data.n_features}")

        See Also
        --------
            - LoadedData: Container class for the return value
            - load_cells_features: Attribute-based method for HDF5 loading
        """
        print("\n Loading complete pipeline data.")
        print(f"   File: {features_file}")
        print(f"   Modalities: {modalities}")

        # Step 1: Load cells DataFrame (legacy path for now)
        self.load_cells_df(
            modalities=modalities or ["pa", "clem"],
            label_column=label_column,
        )

        # Step 2: Calculate metrics if needed
        self.calculate_metrics(
            features_file,
            force_new=force_recalculation,
            use_stored_features=use_stored_features,
        )

        # Step 3: Load features from HDF5
        self.load_cells_features(
            features_file,  drop_neurotransmitter=False
        )

        # Step 4: Build LoadedData container from instance attributes
        training_data = self._build_training_data()

        return LoadedData(
            training_data=training_data,
            feature_names=self.column_labels.copy(),
            cells_df=self.cells.copy(),
            cells_with_to_predict_df=self.cells_with_to_predict.copy(),
        )

    def select_features_rfe(
        self,
        data: LoadedData,
        train_mod: str,
        test_mod: str,
        cv_method_rfe: str = "ss",
        estimator=None,
        metric: str = "f1",
        save_features: bool = True,
        output_subdir: str = "rfe",
    ) -> RFEResult:
        """Select features using Recursive Feature Elimination.

        This method performs RFE feature selection on the loaded data,
        returning results in an RFEResult container.

        Args:
            data: LoadedData container from load_data().
            train_mod: Training modality ('all', 'clem', 'pa', or 'photoactivation').
            test_mod: Test modality for evaluation.
            cv_method_rfe: CV method ('ss' for ShuffleSplit, 'lpo').
                Default 'ss'.
            estimator: Classifier for RFE. Default is AdaBoostClassifier.
            metric: Scoring metric ('f1', 'accuracy'). Default 'f1'.
            save_features: If True, save selected features to file. Default True.

        Returns
        -------
            RFEResult with selected_features_idx, n_features, scores, etc.

        Example:
            >>> data = predictor.load_data("FINAL_features")
            >>> rfe_result = predictor.select_features_rfe(
            ...     data=data,
            ...     train_mod="all",
            ...     test_mod="clem",
            ... )
            >>> print(f"Selected {rfe_result.n_features} features")

        See Also
        --------
            - RFEResult: Container class for the return value
            - select_features_RFE: Attribute-based method
        """
        print("\n Selecting features...")
        print(f"   Train modality: {train_mod}")
        print(f"   Test modality: {test_mod}")

        # Set these attributes early (needed for confusion_matrices callback)
        self.select_train_mod = train_mod
        self.select_test_mod = test_mod
        try:
            self.estimator = repr(estimator) if estimator is not None else "None"
            self.select_method = type(estimator).__name__ if estimator is not None else "RFE"
        except AttributeError:
            self.estimator = "unknown"
            self.select_method = "RFE"

        # Create RFESelector with callbacks to class_predictor methods
        rfe_selector = RFESelector(
            features=self.features,
            labels=self.labels,
            feature_names=self.column_labels,
            cv_callback=self.do_cv,
            confusion_matrix_callback=self.confusion_matrices,
        )

        # Run RFE and get RFEResult
        result = rfe_selector.select_features_rfe(
            train_mod=train_mod,
            test_mod=test_mod,
            estimator=estimator,
            cv_method_rfe=cv_method_rfe,
            metric=metric,
            save_features=save_features,
            output_subdir=output_subdir,
        )

        # Store results in instance variables
        self.select_train_mod = train_mod
        self.select_test_mod = test_mod
        self.select_method = result.estimator_name
        self.reduced_features_idx = result.selected_features_idx

        print(f" RFE complete: {result.best_n_features} features selected")

        return result

    def cross_validate(
        self,
        data: LoadedData,
        selected_features: np.ndarray,
        classifier,
        method: str = "lpo",
        train_mod: str = "all",
        test_mod: str = "clem",
        n_repeats: int = 100,
        test_size: float = 0.3,
        p: int = 1,
        metric: str = "accuracy",
        plot: bool = False,
    ) -> CVResult:
        """Perform cross-validation and return results as CVResult container.

        This method runs cross-validation using the specified classifier and
        returns structured results including the confusion matrix and score.

        Args:
            data: LoadedData container from load_data().
            selected_features: Boolean mask of features to use (from RFE).
            classifier: Classifier instance for training/prediction.
            method: CV method ('lpo' for LeavePOut, 'ss' for ShuffleSplit).
                Default 'lpo'.
            train_mod: Training modality ('all', 'clem', 'pa'). Default 'all'.
            test_mod: Test modality for evaluation. Default 'clem'.
            n_repeats: Number of ShuffleSplit repeats. Default 100.
            test_size: Test fraction for ShuffleSplit. Default 0.3.
            p: Samples to leave out for LeavePOut. Default 1.
            metric: Scoring metric ('accuracy', 'f1'). Default 'accuracy'.
            plot: If True, plot confusion matrix. Default True.

        Returns
        -------
            CVResult with:
                - score: Cross-validation score
                - confusion_matrix: Confusion matrix (4x4)
                - class_labels: Class label names
                - n_predictions: Total predictions made
                - n_splits: Number of CV splits
                - metric: Metric name used

        Example:
            >>> data = predictor.load_data("FINAL_features")
            >>> rfe = predictor.select_features_rfe(data, "all", "clem")
            >>> cv_result = predictor.cross_validate(
            ...     data=data,
            ...     selected_features=rfe.selected_features_idx,
            ...     classifier=LinearDiscriminantAnalysis(),
            ...     method="lpo",
            ... )
            >>> print(f"Score: {cv_result.score:.2%}")
            >>> print(f"CM shape: {cv_result.confusion_matrix.shape}")

        See Also
        --------
            - CVResult: Container class for return value
            - do_cv: Attribute-based method with more options
        """
        print("\n Running cross-validation...")
        print(f"   Method: {method}")
        print(f"   Train: {train_mod}, Test: {test_mod}")
        print(f"   Metric: {metric}")

        # Store selected features
        self.reduced_features_idx = selected_features

        # Build features dictionary for ModalityCrossValidator
        features_dict = {"morph": self.features}

        # Build modality indices dictionary
        modality_indices = {
            "all": np.full(len(self.pa_idx), True),
            "pa": self.pa_idx,
            "clem": self.clem_idx,
        }

        # Create validator
        validator = ModalityCrossValidator(
            features_dict=features_dict,
            labels=self.labels,
            modality_indices=modality_indices,
            reduced_features_idx=selected_features,
        )

        # Run evaluation
        cv_result = validator.evaluate(
            clf=classifier,
            method=method,
            feature_type="morph",
            train_mod=train_mod,
            test_mod=test_mod,
            n_repeats=n_repeats,
            test_size=test_size,
            p=p,
            ax=None,
            figure_label=f"{train_mod}_{test_mod}_cv",
            spines_red=False,
            fraction_across_classes=True,
            idx=None,
            plot=plot,
            return_cvresult=True,
            proba_cutoff=None,
            metric=metric,
        )

        # Store confusion matrix
        self.cm = cv_result.confusion_matrix

        print(f"   Score: {cv_result.score:.2%}")
        print(f"   Predictions: {cv_result.n_predictions}")
        print(" Cross-validation complete\n")

        return cv_result

    def predict(
        self,
        data: LoadedData,
        selected_features: np.ndarray,
        use_jon_priors: bool = False,
        suffix: str = "",
        train_modalities: list = None,
        predict_recorded: bool = False,
        save_predictions: bool = False,
    ) -> PredictionResults:
        """Predict cell types and return results as PredictionResults container.

        This method performs cell type prediction using the trained classifier
        and returns structured results including predictions, probabilities,
        and scaled values.

        Args:
            data: LoadedData container from load_data().
            selected_features: Boolean mask of features to use (from RFE).
            use_jon_priors: If True, apply Jon's anatomical priors. Default False.
            suffix: Suffix for output file names. Default ''.
            train_modalities: Modalities to train on. Default ['clem', 'photoactivation'].
            predict_recorded: If True, use LOO for training cells. Default False.
            save_predictions: If True, save predictions to files. Default False.

        Returns
        -------
            PredictionResults with:
                - cells: DataFrame with predictions added
                - predictions: Array of predicted class labels
                - probabilities: Probability matrix (n_cells, n_classes)
                - scaled_predictions: CM-scaled predictions
                - scaled_probabilities: CM-scaled probabilities
                - verified: None (call verify() to populate)

        Example:
            >>> data = predictor.load_data("FINAL_features")
            >>> rfe = predictor.select_features_rfe(data, "all", "clem")
            >>> predictions = predictor.predict(
            ...     data=data,
            ...     selected_features=rfe.selected_features_idx,
            ...     use_jon_priors=False,
            ...     suffix="_test",
            ... )
            >>> print(f"Predicted {predictions.n_cells} cells")
            >>> print(f"Distribution: {predictions.prediction_counts}")

        See Also
        --------
            - PredictionResults: Container class for return value
            - predict_cells: Attribute-based method
            - verify: Method to add verification status
        """
        if train_modalities is None:
            train_modalities = ["clem", "photoactivation"]

        print("\n Predicting cell types...")
        print(f"   Training modalities: {train_modalities}")
        print(f"   Use Jon's priors: {use_jon_priors}")
        print(f"   Suffix: {suffix}")

        # Store selected features
        self.reduced_features_idx = selected_features

        # Handle suffix
        if use_jon_priors:
            self.suffix = suffix + "_jon_prior"
            suffix = suffix + "_jon_prior"
        else:
            self.suffix = suffix

        # Normalize suffix to always start with "_" for consistent output filenames
        if suffix != "" and suffix[0] != "_":
            self.suffix = "_" + suffix
            suffix = "_" + suffix

        self.save_predictions = save_predictions

        # Create pipeline and prepare data
        pipeline = PredictionPipeline()

        modality_indices = {"clem": self.clem_idx, "photoactivation": self.pa_idx}
        prepared = pipeline.prepare_data(
            train_modalities=train_modalities,
            all_cells=self.all_cells,
            all_cells_with_to_predict=self.all_cells_with_to_predict,
            cells=self.cells,
            cells_with_to_predict=self.cells_with_to_predict,
            features=self.features,
            features_with_to_predict=self.features_with_to_predict,
            labels=self.labels,
            labels_with_to_predict=self.labels_with_to_predict,
            modality_indices=modality_indices,
            selected_features_idx=selected_features,
            predict_recorded=predict_recorded,
        )

        self.prediction_train_df = prepared.train_df
        self.prediction_train_features = prepared.train_features
        self.prediction_train_labels = prepared.train_labels
        self.prediction_predict_df = prepared.predict_df
        self.prediction_predict_features = prepared.predict_features
        self.prediction_predict_labels = prepared.predict_labels

        # Calculate priors
        priors = pipeline.prepare_priors(
            self.prediction_train_labels,
            use_jon_priors=use_jon_priors,
            jon_priors_dict=self.real_cell_class_ratio_dict if use_jon_priors else None,
        )

        # Train and predict
        if not predict_recorded:
            predictions, probabilities = pipeline.train_and_predict(
                self.prediction_train_features,
                self.prediction_train_labels,
                self.prediction_predict_features,
                priors,
            )
        else:
            predictions, probabilities = pipeline.train_and_predict_loo(
                self.prediction_train_features,
                self.prediction_train_labels,
                self.prediction_predict_features,
                self.prediction_predict_df,
                priors,
            )

        predict_df = self.prediction_predict_df.copy()
        predict_df.loc[:, ["MON_proba", "cMI_proba", "iMI_proba", "SMI_proba"]] = probabilities
        predict_df["prediction"] = predictions

        # Compute confusion matrix via Leave-One-Out CV (method="lpo", p=1) to
        # estimate systematic prediction biases. LDA matches the main classifier.
        # Trains on all modalities, tests on CLEM, mirroring the validation protocol.
        # n_repeats/test_size are ignored when method="lpo".
        saved_cm = getattr(self, "cm", None)
        cm = self.do_cv(
            method="lpo",
            clf=LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            feature_type="morph",
            train_mod="all",
            test_mod="clem",
            fraction_across_classes=False,
            n_repeats=100,
            test_size=0.3,
            p=1,
            return_cm=True,
        )
        # Restore the original CM to avoid overwriting cross-validation results
        if saved_cm is not None:
            self.cm = saved_cm

        # Adjust raw probabilities using the confusion matrix (Bayesian calibration).
        # Accounts for the classifier's tendency to over/under-predict certain classes.
        scaled_probabilities, scaled_predictions = pipeline.scale_by_confusion_matrix(
            probabilities, cm
        )

        predict_df.loc[
            :, ["MON_proba_scaled", "cMI_proba_scaled", "iMI_proba_scaled", "SMI_proba_scaled"]
        ] = scaled_probabilities
        predict_df["prediction_scaled"] = scaled_predictions

        self.prediction_predict_df = predict_df

        # Create and return container
        result = PredictionResults(
            cells=predict_df.copy(),
            predictions=predictions,
            probabilities=probabilities,
            class_names=[
                "motion_onset",
                "motion_integrator_contralateral",
                "motion_integrator_ipsilateral",
                "slow_motion_integrator",
            ],
            scaled_predictions=scaled_predictions,
            scaled_probabilities=scaled_probabilities,
            verified=None,  # Set by verify() method
        )

        print(f"   Predictions: {result.n_cells}")
        print(f"   Distribution: {result.prediction_counts}")
        print(" Prediction complete\n")

        return result

    def verify(
        self,
        predictions: PredictionResults,
        data: LoadedData,
        required_tests: list = None,
        calculate4recorded: bool = False,
        force_new: bool = False,
    ) -> PredictionResults:
        """Verify predictions and update PredictionResults with verification status.

        This method runs verification tests (NBLAST similarity, outlier detection,
        etc.) on the predictions and updates the PredictionResults container
        with the verification status.

        Args:
            predictions: PredictionResults from predict().
            data: LoadedData container from load_data().
            required_tests: Tests that must pass for verification.
                Default ['IF', 'LOF'] (Isolation Forest, Local Outlier Factor).
            calculate4recorded: Include recorded cells in NBLAST. Default False.
            force_new: Force save even if files exist. Default False.

        Returns
        -------
            Updated PredictionResults with:
                - verified: Boolean mask of cells that passed all tests
                - cells: Updated DataFrame with test results columns

        Example:
            >>> predictions = predictor.predict(data, rfe.selected_features_idx)
            >>> verified = predictor.verify(
            ...     predictions=predictions,
            ...     data=data,
            ...     required_tests=["IF", "LOF"],
            ... )
            >>> print(f"Verified: {verified.n_verified}/{verified.n_cells}")
            >>> print(f"Rate: {verified.verification_rate:.1%}")

        See Also
        --------
            - PredictionResults: Container class with verified attribute
            - calculate_verification_metrics: Attribute-based method
        """
        if required_tests is None:
            required_tests = ["IF", "LOF"]

        print("\n Verifying predictions...")
        print(f"   Required tests: {required_tests}")
        print(f"   Cells to verify: {predictions.n_cells}")

        self.prediction_predict_df = predictions.cells.copy()
        self.prediction_train_df = getattr(self, "prediction_train_df", None)
        self.prediction_train_features = getattr(self, "prediction_train_features", None)
        self.prediction_predict_features = getattr(self, "prediction_predict_features", None)
        self.prediction_train_labels = getattr(self, "prediction_train_labels", None)

        # Create verification calculator
        verifier = VerificationCalculator(
            base_path=self.path,
            train_df=self.prediction_train_df,
            predict_df=self.prediction_predict_df,
            train_features=self.prediction_train_features,
            predict_features=self.prediction_predict_features,
            train_labels=self.prediction_train_labels,
            suffix=getattr(self, "suffix", ""),
            estimator=getattr(self, "estimator", "unknown"),
        )

        # Run verification
        verified_df = verifier.run_verification(
            required_tests=required_tests,
            calculate4recorded=calculate4recorded,
            save_predictions=getattr(self, "save_predictions", False),
            force_new=force_new,
        )

        self.prediction_predict_df = verified_df

        # Compute verified mask
        if "passed_tests" in verified_df.columns:
            verified_mask = verified_df["passed_tests"].to_numpy()
        else:
            # Fall back to checking individual test columns
            verified_mask = np.ones(len(verified_df), dtype=bool)
            for test in required_tests:
                col = f"{test}_passed" if f"{test}_passed" in verified_df.columns else test
                if col in verified_df.columns:
                    verified_mask = verified_mask & verified_df[col].to_numpy()

        # Create updated PredictionResults
        result = PredictionResults(
            cells=verified_df.copy(),
            predictions=predictions.predictions.copy(),
            probabilities=predictions.probabilities.copy(),
            class_names=predictions.class_names.copy(),
            scaled_predictions=predictions.scaled_predictions.copy()
            if predictions.scaled_predictions is not None
            else None,
            scaled_probabilities=predictions.scaled_probabilities.copy()
            if predictions.scaled_probabilities is not None
            else None,
            verified=verified_mask,
        )

        print(f"   Verified: {result.n_verified}/{result.n_cells}")
        print(f"   Rate: {result.verification_rate:.1%}")
        print(" Verification complete\n")

        return result

    def _build_training_dataset(self) -> CellDataset:
        """Build a CellDataset from the current training data attributes.

        Returns
        -------
            CellDataset containing training cells (excluding to_predict)
        """
        return CellDataset(
            cells=self.all_cells.copy(),
            features=self.features.copy(),
            labels=self.labels.copy(),
            modality=self.labels_imaging_modality.copy(),
            modality_mask=ModalityMask(
                clem=self.clem_idx.copy(),
                photoactivation=self.pa_idx.copy(),
                em=self.em_idx.copy(),
            ),
            feature_names=self.column_labels.copy(),
        )

    def _build_training_data(self) -> TrainingData:
        """Build a TrainingData container with training and prediction datasets.

        Returns
        -------
            TrainingData with training and to_predict CellDatasets
        """
        training = self._build_training_dataset()

        # Build to_predict dataset directly from _with_to_predict attributes
        mask = self.labels_with_to_predict == "to_predict"
        tp_cells = self.all_cells_with_to_predict[mask].reset_index(drop=True)
        to_predict = CellDataset(
            cells=tp_cells,
            features=self.features_with_to_predict[mask],
            labels=self.labels_with_to_predict[mask],
            modality=self.labels_imaging_modality_with_to_predict[mask],
            modality_mask=ModalityMask.from_series(tp_cells["imaging_modality"]),
            feature_names=self.column_labels.copy(),
        )

        # Include selected features if available
        selected = self.reduced_features_idx

        return TrainingData(
            training=training,
            to_predict=to_predict,
            selected_features=selected,
        )

    def load_segclr_embeddings(self, h5_path: str | Path) -> LoadedData:
        """Load SegCLR embeddings from an HDF5 file and return as LoadedData.

        Reads the all_embeddings.hdf5 file containing per-cell SegCLR
        embeddings and wraps them in a LoadedData container for use with
        the classification pipeline.

        Parameters
        ----------
        h5_path : str or Path
            Path to the all_embeddings.hdf5 file. Expected datasets:
            embeddings (n_cells, 512), cell_names, labels, label_names.

        Returns
        -------
        LoadedData
            Container with SegCLR embeddings as features (512-d per cell).
        """
        import h5py

        h5_path = Path(h5_path)

        with h5py.File(h5_path, "r") as f:
            embeddings = f["embeddings"][:]
            cell_names = [name.decode() if isinstance(name, bytes) else name
                          for name in f["cell_names"][:]]
            labels = np.array([label.decode() if isinstance(label, bytes) else label
                               for label in f["labels"][:]])

        n_dim = embeddings.shape[1]
        feature_names = [f"segclr_{i:03d}" for i in range(n_dim)]

        # Build cells DataFrame
        cells_df = pd.DataFrame({
            "cell_name": cell_names,
            "function": labels,
            "imaging_modality": "clem",
        })
        modalities = cells_df["imaging_modality"].to_numpy()
        modality_mask = ModalityMask.from_series(cells_df["imaging_modality"])

        training_dataset = CellDataset(
            cells=cells_df,
            features=embeddings,
            labels=labels,
            modality=modalities,
            modality_mask=modality_mask,
            feature_names=feature_names,
        )

        # Empty to_predict dataset
        empty_df = pd.DataFrame(columns=cells_df.columns)
        empty_modality = np.array([], dtype=str)
        empty_mask = ModalityMask(
            clem=np.array([], dtype=bool),
            photoactivation=np.array([], dtype=bool),
            em=np.array([], dtype=bool),
        )
        to_predict_dataset = CellDataset(
            cells=empty_df,
            features=np.empty((0, n_dim)),
            labels=np.array([], dtype=str),
            modality=empty_modality,
            modality_mask=empty_mask,
            feature_names=feature_names,
        )

        training_data = TrainingData(
            training=training_dataset,
            to_predict=to_predict_dataset,
        )

        return LoadedData(
            training_data=training_data,
            feature_names=feature_names,
            cells_df=cells_df,
            cells_with_to_predict_df=cells_df.copy(),
        )

    def combine_features(
        self,
        morph_data: LoadedData,
        segclr_data: LoadedData,
        morph_feature_mask: np.ndarray,
    ) -> LoadedData:
        """Combine selected morphology features with SegCLR embeddings.

        Extracts the RFE-selected morphology features and concatenates them
        horizontally with the full SegCLR embedding vectors.

        Parameters
        ----------
        morph_data : LoadedData
            Morphology features from load_data().
        segclr_data : LoadedData
            SegCLR embeddings from load_segclr_embeddings().
        morph_feature_mask : np.ndarray
            Boolean mask indicating which morphology features to keep
            (e.g., from RFE selection).

        Returns
        -------
        LoadedData
            Container with horizontally stacked features:
            [selected_morph | segclr_embeddings].
        """
        # Extract selected morphology features
        morph_features = morph_data.training_data.training.features[:, morph_feature_mask]
        morph_names = [name for name, keep
                       in zip(morph_data.feature_names, morph_feature_mask, strict=True)
                       if keep]

        # Get SegCLR features
        segclr_features = segclr_data.training_data.training.features
        segclr_names = segclr_data.feature_names

        # Concatenate horizontally
        combined_features = np.hstack([morph_features, segclr_features])
        combined_names = morph_names + segclr_names

        # Reuse morph_data's metadata for the combined container
        training = morph_data.training_data.training
        combined_training = CellDataset(
            cells=training.cells.copy(),
            features=combined_features,
            labels=training.labels.copy(),
            modality=training.modality.copy(),
            modality_mask=training.modality_mask,
            feature_names=combined_names,
        )

        # Build to_predict with combined features if it has cells
        to_predict = morph_data.training_data.to_predict
        if to_predict.n_cells > 0:
            tp_morph = to_predict.features[:, morph_feature_mask]
            tp_segclr = segclr_data.training_data.to_predict.features
            tp_combined = np.hstack([tp_morph, tp_segclr])
        else:
            tp_combined = np.empty((0, len(combined_names)))

        combined_to_predict = CellDataset(
            cells=to_predict.cells.copy(),
            features=tp_combined,
            labels=to_predict.labels.copy(),
            modality=to_predict.modality.copy(),
            modality_mask=to_predict.modality_mask,
            feature_names=combined_names,
        )

        combined_training_data = TrainingData(
            training=combined_training,
            to_predict=combined_to_predict,
        )

        return LoadedData(
            training_data=combined_training_data,
            feature_names=combined_names,
            cells_df=morph_data.cells_df.copy(),
            cells_with_to_predict_df=morph_data.cells_with_to_predict_df.copy(),
        )

    def prepare_data_4_metric_calc(self, df):
        """Prepare data for metric calculation.

        This function prepares the data for metric calculation by cleaning up
        the data and adding relevant information.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe that requires data cleaning and formatting.

        Returns
        -------
        pandas.DataFrame
            Modified dataframe with cleaned data and additional information.

        Notes
        -----
        The function performs:
        - Dropping duplicate 'cell_name' records, keeping first occurrence
        """
        # Use the DataLoader module for clean, modular data preparation
        loader = DataLoader(self.path)

        return loader.prepare_data_for_metrics(df)

    def calculate_metrics(
        self, file_name, force_new=False, use_stored_features=False,
    ):
        """Calculate metrics for the data set and save to file.

        Parameters
        ----------
        file_name : str
            Name of file where metrics result should be saved.
            Include full path if not in the same directory.
        force_new : bool, optional
            If True, force new metrics calculation. Default: False.
        use_stored_features : bool, optional
            If True, use existing HDF5 features file without recalculation.
            Default: False.

        Returns
        -------
        None
            Result is saved to the specified file.

        Notes
        -----
        Uses 'train_or_predict'='train' for training data.
        """
        if use_stored_features and not force_new:
            # Use the same path resolution as load_metrics so we check
            # the file that will actually be read downstream.
            loader = DataLoader(self.path)
            try:
                hdf5_path = loader._resolve_features_path(file_name)
            except FileNotFoundError:
                hdf5_path = None
            if hdf5_path is not None and hdf5_path.exists():
                # HDF5 exists — skip recalculation.  Training cell selection
                # is now handled by used_for_training from xlsx, so HDF5
                # labels no longer need updating from metadata files.
                print(f"\n   Using existing HDF5: {hdf5_path}")
                return
            else:
                print(f"\n   HDF5 file not found, regenerating features...")
                force_new = True

        # Map modern nomenclature back to legacy names for feature calculation.
        # calculate_metric2df (in utils/) was written for the original 3-class labels;
        # this mapping is applied before feature extraction and reversed afterward.
        # Map modern 4-class names to legacy 3-class names for calculate_metric2df,
        # which groups ipsilateral/contralateral into a single "integrator" class.
        new_to_old = {
            "motion_integrator": "integrator",
            "motion_integrator_ipsilateral": "integrator",
            "motion_integrator_contralateral": "integrator",
            "motion_onset": "dynamic_threshold",
            "slow_motion_integrator": "motor_command",
        }
        self.cells_with_to_predict["function"] = (
            self.cells_with_to_predict["function"].replace(new_to_old)
        )

        print(f"\n Calculating metrics for: {file_name}")
        print(f"   Force recalculation: {force_new}")
        print(f"   Number of cells: {len(self.cells_with_to_predict)}")
        calculate_metric2df(self.cells_with_to_predict, file_name, self.path, force_new=force_new)
        print(" Metric calculation complete\n")

    def load_metrics(self, file_name, drop_neurotransmitter=False):
        """Load morphological metrics from HDF5 feature file.

        Args:
            file_name (str): The file name from which the metrics should be loaded.
            drop_neurotransmitter (bool, optional): If True, drop the neurotransmitter
                column from output. Default is False.

        Returns
        -------
            list: A list containing the features, labels, and labels_imaging_modality arrays.
            list: A list of the column labels.
            DataFrame: The preprocessed DataFrame 'all_cells'.
        """
        loader = DataLoader(self.path)
        return loader.load_metrics(
            file_name=file_name,
            drop_neurotransmitter=drop_neurotransmitter,
        )

    def load_cells_features(self, file, drop_neurotransmitter=False):
        """Load cell features from file and filter into categories by imaging modality.

        Args:
            file (str): The file name from which the cell features will be loaded.
            drop_neurotransmitter (bool, optional): If True, drop neurotransmitter column.
                Default is False.

        Raises
        ------
            ValueError: If the metrics have not been loaded.

        Attributes
        ----------
            features_with_to_predict: Loaded features including to_predict cells.
            labels_with_to_predict: Loaded labels including to_predict cells.
            labels_imaging_modality_with_to_predict: Modality labels including to_predict.
            labels, features, labels_imaging_modality, all_cells: Filtered data
                excluding 'to_predict' and 'neg_control' categories.
            cells: All cells with 'to_predict' attribute present in the loaded file.
            clem_idx, pa_idx, em_idx: Boolean masks for different imaging modalities.

        Notes
        -----
            'morph' in attribute names refers to the original author initials (Florian Kämpf).
        """
        print(f"\n Loading cell features from file: {file}")
        all_metric, self.column_labels, self.all_cells_with_to_predict = self.load_metrics(
            file,  drop_neurotransmitter=drop_neurotransmitter
        )
        (
            self.features_with_to_predict,
            self.labels_with_to_predict,
            self.labels_imaging_modality_with_to_predict,
        ) = all_metric
        
        # Filter to training cells using used_for_training from xlsx
        # self.cells_with_to_predict comes from load_cells_df (has xlsx columns);
        # self.all_cells_with_to_predict comes from HDF5 (features only).
        training_names = set(
            self.cells_with_to_predict
            .loc[self.cells_with_to_predict["used_for_training"] == True, "cell_name"]  # noqa: E712
        )
        training_mask = self.all_cells_with_to_predict["cell_name"].isin(training_names).to_numpy()

        # Standardize features: fit scaler on TRAINING cells only, then
        # transform all cells with the same scaler. This prevents data leakage
        # from to_predict/neg_control cells influencing the scaling statistics.
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.features_with_to_predict[training_mask])
        self.features_with_to_predict = scaler.transform(self.features_with_to_predict)

        self.labels = self.labels_with_to_predict[training_mask]
        self.features = self.features_with_to_predict[training_mask]
        self.labels_imaging_modality = self.labels_imaging_modality_with_to_predict[training_mask]
        self.all_cells = self.all_cells_with_to_predict[training_mask]

        # Filter HDF5 training set to cells also present in the DataFrame.
        # The incomplete CLEM filter in load_cells_df() may remove cells that
        # still exist in the HDF5 (generated before that filter was added).
        df_cell_names = set(self.cells_with_to_predict["cell_name"])
        in_df_mask = self.all_cells["cell_name"].isin(df_cell_names).to_numpy()
        if not in_df_mask.all():
            n_dropped = (~in_df_mask).sum()
            print(f"   Dropping {n_dropped} HDF5 cells not in DataFrame")
            self.all_cells = self.all_cells[in_df_mask].reset_index(drop=True)
            self.features = self.features[in_df_mask]
            self.labels = self.labels[in_df_mask]
            self.labels_imaging_modality = self.labels_imaging_modality[in_df_mask]

        # Align xlsx DataFrame to match HDF5 training cells (same rows, same order)
        self.cells = (
            self.cells_with_to_predict.set_index("cell_name")
            .loc[self.all_cells["cell_name"]]
            .reset_index()
        )

        self.clem_idx = (self.cells["imaging_modality"] == "clem").to_numpy()
        self.pa_idx = (self.cells["imaging_modality"] == "photoactivation").to_numpy()
        self.em_idx = (self.cells["imaging_modality"] == "EM").to_numpy()

        self.clem_idx_with_to_predict = (
            self.cells_with_to_predict["imaging_modality"] == "clem"
        ).to_numpy()
        self.pa_idx_with_to_predict = (
            self.cells_with_to_predict["imaging_modality"] == "photoactivation"
        ).to_numpy()
        self.em_idx_with_to_predict = (
            self.cells_with_to_predict["imaging_modality"] == "EM"
        ).to_numpy()

        self.reduced_features_idx = None  # set later by select_features_RFE

        # Sync HDF5 morphology with xlsx-corrected values (morphology_gregor)
        self._sync_morphology_features()

        print(f"   Training set size: {len(self.labels)} cells")
        print(f"   CLEM: {self.clem_idx.sum()}, PA: {self.pa_idx.sum()}, EM: {self.em_idx.sum()}")
        print(" Cell features loaded\n")

    def load_cells_df(
        self,
        modalities=None,
        label_column: str = "kmeans_function",
    ):
        """Load cells dataframe and apply preprocessing pipeline.

        Args:
            modalities (list): Imaging modalities to load. Default ['pa', 'clem'].
            label_column (str): xlsx column to use as functional labels.
                Default 'kmeans_function'.
        """
        # Handle mutable default
        if modalities is None:
            modalities = ["pa", "clem"]
        self.modalities = modalities

        # Delegate to DataLoader
        loader = DataLoader(self.path)
        self.cells_with_to_predict, self.cells = loader.load_cells_df(
            modalities=modalities,
            label_column=label_column,
        )

    def calculate_published_metrics(self):
        """Calculate published metrics over the cell data.

        Attributes
        ----------
        features_pv : ndarray
            Persistence vectors using navis library (300 samples).
            Reference: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182184
        features_ps : ndarray
            All computed persistence vectors.
        features_ff : ndarray
            Form factor of neurons computed using navis library.

        Notes
        -----
        Uses navis.form_factor with 15 cores and 300 samples.
        Method: https://link.springer.com/article/10.1007/s12021-017-9341-1
        """
        print("\n Calculating published metrics...")
        print(f"   Processing {len(self.cells)} neurons...")

        print("   Calculating persistence vectors (300 samples)...")
        self.features_pv = np.stack(
            [navis.persistence_vectors(x, samples=300)[0] for x in self.cells.swc]
        )[:, 0, :]  # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182184
        self.features_ps = np.stack(
            [navis.persistence_vectors(x, samples=300)[1] for x in self.cells.swc]
        )  # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182184

        print("   Calculating form factors (15 cores, 300 samples)...")
        self.features_ff = navis.form_factor(
            navis.NeuronList(self.cells.swc), n_cores=15, parallel=True, num=300
        )  # https://link.springer.com/article/10.1007/s12021-017-9341-1
        print(" Published metrics calculated\n")

    def select_features(
        self,
        train_mod: str,
        test_mod: str,
        plot=False,
        use_assessment_per_class=False,
        which_selection="SKB",
        use_std_scale=False,
    ):
        """Select optimal features using SelectKBest with f_classif or mutual_info_classif.

        This method uses FeatureSelector.select_k_best() which tests both f_classif
        and mutual_info_classif scorers to find the optimal feature subset. The best
        scorer and feature count are determined by cross-validation accuracy.

        Parameters
        ----------
        train_mod : str
            The training modality ('all', 'pa', 'clem').
        test_mod : str
            The testing modality ('all', 'pa', 'clem').
        plot : bool, optional
            If True, plots accuracy vs number of features. Default is False.
        use_assessment_per_class : bool, optional
            If True, uses per-class accuracy (balanced) for feature selection.
            If False, uses overall accuracy. Default is False.
        which_selection : str, optional
            Ignored. FeatureSelector always tests both f_classif and
            mutual_info_classif. Default is 'SKB'.
        use_std_scale : bool, optional
            If True, applies penalty scaling based on score variance.
            Higher variance is penalized. Default is False.

        Returns
        -------
        bool_features_2_use : numpy.ndarray
            Boolean array indicating the selected features.
        max_accuracy_key : str
            The evaluator that achieved best results ('f_classif' or 'mutual_info_classif').
        train_mod : str
            The training modality (pass-through).
        test_mod : str
            The testing modality (pass-through).

        Notes
        -----
        When train_mod == test_mod, uses Leave-One-Out cross-validation.
        Otherwise, trains on training set and evaluates on test set.
        """
        print(f"\n Selecting features for {train_mod} → {test_mod}")
        print("   Feature selection method: SelectKBest (f_classif + mutual_info_classif)")
        print(f"   Use assessment per class: {use_assessment_per_class}")
        print(f"   Use penalty scaling: {use_std_scale}")

        # Get modality indices
        mod2idx = {"all": np.full(len(self.pa_idx), True), "pa": self.pa_idx, "clem": self.clem_idx}
        features_train = self.features[mod2idx[train_mod]]
        labels_train = self.labels[mod2idx[train_mod]]
        features_test = self.features[mod2idx[test_mod]]
        labels_test = self.labels[mod2idx[test_mod]]

        # Determine if train and test are identical (requires LOO-CV)
        train_test_identical = train_mod == test_mod

        # Run feature selection using FeatureSelector module
        # Returns: (pred_correct_dict, pred_correct_dict_per_class, used_features_idx)
        pred_correct_dict, pred_correct_dict_per_class, used_features_idx = (
            FeatureSelector.select_k_best(
                features_train=features_train,
                labels_train=labels_train,
                features_test=features_test,
                labels_test=labels_test,
                train_test_identical=train_test_identical,
                use_std_scale=use_std_scale,
            )
        )

        # Choose which accuracy dict to use for finding optimal features
        scores_dict = pred_correct_dict_per_class if use_assessment_per_class else pred_correct_dict

        # Find the best evaluator and optimal number of features
        best_evaluator = None
        best_n_features = None
        best_score = -1

        for evaluator_name, scores in scores_dict.items():
            max_score = np.max(scores)
            if max_score > best_score:
                best_score = max_score
                best_evaluator = evaluator_name
                best_n_features = np.argmax(scores) + 1  # +1 because index starts at 0

        # Get the boolean feature mask for the optimal configuration
        bool_features_2_use = used_features_idx[best_evaluator][best_n_features]
        max_accuracy_key = best_evaluator

        # Plot if requested
        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            for i, (evaluator_name, scores) in enumerate(scores_dict.items()):
                ax[i].plot(scores)
                ax[i].title.set_text(
                    f"train: {train_mod}\n"
                    f"test: {test_mod}\n"
                    f"Max: {np.max(scores):.3f}\n"
                    f"Evaluator: {evaluator_name}"
                )
                ax[i].set_xlabel("no of features")
                ax[i].axvline(np.argmax(scores), ls="--", lw=2, c="r")
                ax[i].set_xticks(
                    np.arange(0, features_train.shape[1], 3),
                    np.arange(1, features_train.shape[1] + 2, 3),
                )
            plt.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.1)
            plt.show()

        # Store results as instance variables
        self.reduced_features_idx = bool_features_2_use
        self.select_train_mod = train_mod
        self.select_test_mod = test_mod
        self.select_method = "SKB"  # SelectKBest

        print(f"   Selected {np.sum(bool_features_2_use)} features using {max_accuracy_key}")
        print(f"   Optimal feature count: {best_n_features}")
        print(f"   Maximum accuracy: {best_score:.3f}")
        print(" Feature selection complete\n")

        return bool_features_2_use, max_accuracy_key, train_mod, test_mod

    def select_features_RFE(
        self,
        train_mod,
        test_mod,
        estimator=None,
        save_features=False,
        cv_method_rfe="lpo",
        metric="accuracy",
        output_subdir="rfe",
    ):
        """Select features using Recursive Feature Elimination (RFE).

        Args:
            train_mod (str): The training modality ('all', 'pa', 'clem').
            test_mod (str): The testing modality.
            estimator: The estimator(s) to use for RFE. Default is multiple classifiers.
            save_features (bool): If True, saves selected features to instance. Default is False.
            cv_method_rfe (str): CV method for RFE ('lpo' or 'ss').
            metric (str): Metric for scoring ('accuracy', 'f1', etc.).

        Returns
        -------
            None (sets self.reduced_features_idx if save_features=True)
        """
        # Set these attributes early (needed for confusion_matrices callback)
        self.select_train_mod = train_mod
        self.select_test_mod = test_mod
        # Use repr() to avoid triggering sklearn estimator __len__
        try:
            self.estimator = repr(estimator) if estimator is not None else "None"
            self.select_method = type(estimator).__name__ if estimator is not None else "RFE"
        except AttributeError:
            self.estimator = "unknown"
            self.select_method = "RFE"

        # Create RFESelector with callbacks to class_predictor methods
        rfe_selector = RFESelector(
            features=self.features,
            labels=self.labels,
            feature_names=self.column_labels,
            cv_callback=self.do_cv,
            confusion_matrix_callback=self.confusion_matrices,
        )

        # Run RFE
        result = rfe_selector.select_features_rfe(
            train_mod=train_mod,
            test_mod=test_mod,
            estimator=estimator,
            cv_method_rfe=cv_method_rfe,
            metric=metric,
            save_features=save_features,
            output_subdir=output_subdir,
        )

        # Store results in instance variables
        self.select_train_mod = train_mod
        self.select_test_mod = test_mod
        self.select_method = result.estimator_name

        if save_features:
            self.reduced_features_idx = result.selected_features_idx

        print(f" RFE complete: {result.best_n_features} features selected")

        return result

    def do_cv(
        self,
        method: str,
        clf,
        feature_type,
        train_mod,
        test_mod,
        n_repeats=100,
        test_size=0.3,
        p=1,
        ax=None,
        figure_label="error:no figure label",
        spines_red=False,
        fraction_across_classes=True,
        idx=None,
        plot=True,
        return_cm=False,
        proba_cutoff=None,
        metric="accuracy",
    ):
        """Perform cross-validation on the given classifier and dataset.

        Parameters
        ----------
        method : str
            The cross-validation method to use ('lpo' for LeavePOut or 'ss' for ShuffleSplit).
        clf : object
            The classifier to use for training and prediction.
        feature_type : str
            The type of features to use ('morph', 'pv', 'ps', or 'ff').
        train_mod : str
            The training modality.
        test_mod : str
            The testing modality.
        n_repeats : int, optional
            The number of repeats for ShuffleSplit. Default is 100.
        test_size : float, optional
            The test size for ShuffleSplit. Default is 0.3.
        p : int, optional
            The number of samples to leave out for LeavePOut. Default is 1.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the confusion matrix. Default is None.
        figure_label : str, optional
            The label for the figure. Default is 'error:no figure label'.
        spines_red : bool, optional
            If True, set the spines of the plot to red. Default is False.
        fraction_across_classes : bool, optional
            If True, normalize the confusion matrix by true labels. Default is True.
        idx : array-like, optional
            The indices of the features to use. Default is None.
        plot : bool, optional
            If True, plot the confusion matrix. Default is True.
        return_cm : bool, optional
            If True, return the confusion matrix. Default is False.
        proba_cutoff : float, optional
            Minimum probability for confident predictions. Default: None
        metric : str, optional
            Metric to use: 'accuracy' or 'f1'. Default: 'accuracy'

        Returns
        -------
        tuple or numpy.ndarray
            (score, n_predictions) tuple if return_cm is False, otherwise the confusion matrix.

        Notes
        -----
        The confusion matrix is always stored in self.cm regardless of return_cm setting.
        """
        # Build features dictionary for ModalityCrossValidator.
        # Feature types: "morph" = morphological (default, always present),
        # "pv" = persistence vectors, "ps" = persistence statistics,
        # "ff" = form factor. Optional types are added only if computed.
        features_dict = {"morph": self.features}
        if hasattr(self, "features_pv"):
            features_dict["pv"] = self.features_pv
        if hasattr(self, "features_ps"):
            features_dict["ps"] = self.features_ps
        if hasattr(self, "features_ff"):
            features_dict["ff"] = self.features_ff

        # Build modality indices dictionary
        modality_indices = {
            "all": np.full(len(self.pa_idx), True),
            "pa": self.pa_idx,
            "clem": self.clem_idx,
        }

        # Create validator
        validator = ModalityCrossValidator(
            features_dict=features_dict,
            labels=self.labels,
            modality_indices=modality_indices,
            reduced_features_idx=self.reduced_features_idx,
        )

        # Single evaluation returns both score AND confusion matrix via CVResult
        cv_result = validator.evaluate(
            clf=clf,
            method=method,
            feature_type=feature_type,
            train_mod=train_mod,
            test_mod=test_mod,
            n_repeats=n_repeats,
            test_size=test_size,
            p=p,
            ax=ax,
            figure_label=figure_label,
            spines_red=spines_red,
            fraction_across_classes=fraction_across_classes,
            idx=idx,
            plot=plot,
            return_cvresult=True,  # Get both score and CM in one pass
            proba_cutoff=proba_cutoff,
            metric=metric,
        )

        # Always store confusion matrix (computed in single pass)
        self.cm = cv_result.confusion_matrix

        # Return based on caller's request
        if return_cm:
            return cv_result.confusion_matrix
        return (cv_result.score, cv_result.n_predictions)

    def confusion_matrices(
        self,
        clf,
        method: str,
        n_repeats=100,
        test_size=0.3,
        p=1,
        fraction_across_classes=False,
        feature_type="morph",
        idx=None,
        output_subdir=None,
    ):
        """Generate and save confusion matrices for different training and testing modalities.

        Parameters
        ----------
        clf : object
            The classifier to use for training and prediction.
        method : str
            The cross-validation method to use ('lpo' for LeavePOut or 'ss' for ShuffleSplit).
        n_repeats : int, optional
            The number of repeats for ShuffleSplit. Default is 100.
        test_size : float, optional
            The test size for ShuffleSplit. Default is 0.3.
        p : int, optional
            The number of samples to leave out for LeavePOut. Default is 1.
        fraction_across_classes : bool, optional
            If True, normalize the confusion matrix by true labels. Default is False.
        feature_type : str, optional
            The type of features to use ('morph', 'pv', 'ps', or 'ff'). Default is 'morph'.

        Returns
        -------
        None
        """
        print("\n Generating confusion matrices...")
        print(f"   Method: {method}, Classifier: {str(clf)[:50]}...")
        print(f"   Feature type: {feature_type}, Repeats: {n_repeats}")

        suptitle = feature_type.upper() + "_features_"
        if method == "lpo":
            suptitle += f"lpo{p}"
        elif method == "ss":
            suptitle += f"ss_{int((1 - test_size) * 100)}_{int((test_size) * 100)}"

        if feature_type == "morph":
            suptitle += f"{self.select_train_mod}_{self.select_test_mod}_{self.select_method}"

        print("   Creating 3x3 confusion matrix grid...")
        target_train_test = ["ALLCLEM", "CLEMCLEM", "PAPA"]
        fig, ax = plt.subplots(3, 3, figsize=(20, 20))
        for train_mod, loc_x in zip(["ALL", "CLEM", "PA"], range(3), strict=True):
            for test_mod, loc_y in zip(["ALL", "CLEM", "PA"], range(3), strict=True):
                spines_red = train_mod + test_mod in target_train_test
                self.do_cv(
                    method,
                    clf,
                    feature_type,
                    train_mod.lower(),
                    test_mod.lower(),
                    figure_label=f"{train_mod}_{test_mod}",
                    ax=ax[loc_x, loc_y],
                    spines_red=spines_red,
                    fraction_across_classes=fraction_across_classes,
                    n_repeats=n_repeats,
                    test_size=test_size,
                    p=p,
                    idx=idx,
                )

        save_dir = (
            get_output_dir("classifier_pipeline", output_subdir)
            if output_subdir
            else self.path_to_save_confusion_matrices
        )
        print(f"   Saving confusion matrices to: {save_dir}")
        fig.suptitle(suptitle, fontsize="xx-large")
        plt.savefig(save_dir / f"{suptitle}.png")
        plt.savefig(save_dir / f"{suptitle}.pdf")
        plt.show()
        print(" Confusion matrices saved\n")

    def check_swc_validity(self, df):
        """Check and fix invalid SWC neuron files."""
        return check_swc_validity(df)

    def calculate_verification_metrics(
        self,
        calculate_smat=False,
        with_kunst=True,
        calculate4recorded=False,
        load_summit_matrix=True,
        required_tests=None,
        force_new=False,
    ):
        """Calculate verification metrics for cell type predictions.

        Parameters
        ----------
        calculate_smat : bool
            Unused parameter.
        with_kunst : bool
            Unused parameter.
        calculate4recorded : bool
            If True, include recorded cells in NBLAST calculations.
        load_summit_matrix : bool
            Unused parameter.
        required_tests : list
            List of test names that must all pass for a cell to be marked verified.
        force_new : bool
            If True, force saving new prediction files even if unchanged.
        """
        # Handle mutable default
        if required_tests is None:
            required_tests = ["NBLAST_g"]
        # Determine whether to save predictions
        save_predictions = getattr(self, "save_predictions", True)

        # Create verification calculator with required data
        verifier = VerificationCalculator(
            base_path=self.path,
            train_df=self.prediction_train_df,
            predict_df=self.prediction_predict_df,
            train_features=self.prediction_train_features,
            predict_features=self.prediction_predict_features,
            train_labels=self.prediction_train_labels,
            suffix=getattr(self, "suffix", ""),
            estimator=getattr(self, "estimator", "unknown"),
        )

        # Run verification
        self.prediction_predict_df = verifier.run_verification(
            required_tests=required_tests,
            calculate4recorded=calculate4recorded,
            save_predictions=save_predictions,
            force_new=force_new,
        )

        # Copy results back to instance
        if verifier.nblast_calc is not None:
            self.smat_fish = verifier.nblast_calc.smat_fish
        if verifier.nblast_matrices:
            self.nb_train = verifier.nblast_matrices.get("train")
            self.nb_train_nc = verifier.nblast_matrices.get("train_nc")
            self.nb_train_predict = verifier.nblast_matrices.get("train_predict")
        if verifier.nblast_matches:
            self.nb_matches_train = verifier.nblast_matches.get("train")
            self.nb_matches_nc = verifier.nblast_matches.get("nc")
            self.nb_matches_predict = verifier.nblast_matches.get("predict")
        if verifier.nblast_distributions:
            self.nblast_values_dt = verifier.nblast_distributions.get("dt")
            self.nblast_values_ii = verifier.nblast_distributions.get("ii")
            self.nblast_values_ci = verifier.nblast_distributions.get("ci")
            self.nblast_values_mc = verifier.nblast_distributions.get("mc")

    def predict_cells(
        self,
        train_modalities=None,
        use_jon_priors=True,
        suffix="",
        predict_recorded=False,
        save_predictions=True,
    ):
        """Predict cell types based on training modalities.

        Parameters
        ----------
        train_modalities : list of str, optional
            Training modalities. Default: ['clem', 'photoactivation'].
        use_jon_priors : bool, optional
            If True, use Jon's priors. Default: True.
        save_predictions : bool, optional
            If True, save predictions to files. Default: True.
        predict_recorded : bool, optional
            If True, use LOO for cells in training set. Default: False.

        Returns
        -------
        None
        """
        # Handle mutable default
        if train_modalities is None:
            train_modalities = ["clem", "photoactivation"]
        print("\n Predicting cell types...")
        print(f"   Training modalities: {train_modalities}")
        print(f"   Use Jon's priors: {use_jon_priors}")
        print(f"   Save predictions: {save_predictions}")
        print(f"   Suffix: {suffix}")

        # Store save_predictions as instance variable for use in calculate_verification_metrics
        self.save_predictions = save_predictions

        # Handle suffix
        if use_jon_priors:
            self.suffix = suffix + "_jon_prior"
            suffix = suffix + "_jon_prior"
        else:
            self.suffix = suffix

        if suffix != "" and suffix[0] != "_":
            self.suffix = "_" + suffix
            suffix = "_" + suffix

        print(f"   Final suffix: {self.suffix}")

        pipeline = PredictionPipeline()

        # Prepare data using pipeline method
        modality_indices = {"clem": self.clem_idx, "photoactivation": self.pa_idx}
        prepared = pipeline.prepare_data(
            train_modalities=train_modalities,
            all_cells=self.all_cells,
            all_cells_with_to_predict=self.all_cells_with_to_predict,
            cells=self.cells,
            cells_with_to_predict=self.cells_with_to_predict,
            features=self.features,
            features_with_to_predict=self.features_with_to_predict,
            labels=self.labels,
            labels_with_to_predict=self.labels_with_to_predict,
            modality_indices=modality_indices,
            selected_features_idx=self.reduced_features_idx,
            predict_recorded=predict_recorded,
        )

        self.prediction_train_df = prepared.train_df
        self.prediction_train_features = prepared.train_features
        self.prediction_train_labels = prepared.train_labels
        self.prediction_predict_df = prepared.predict_df
        self.prediction_predict_features = prepared.predict_features
        self.prediction_predict_labels = prepared.predict_labels

        # Calculate priors
        priors = pipeline.prepare_priors(
            self.prediction_train_labels,
            use_jon_priors=use_jon_priors,
            jon_priors_dict=self.real_cell_class_ratio_dict if use_jon_priors else None,
        )

        # Train and predict
        if not predict_recorded:
            predictions, probabilities = pipeline.train_and_predict(
                self.prediction_train_features,
                self.prediction_train_labels,
                self.prediction_predict_features,
                priors,
            )
        else:
            predictions, probabilities = pipeline.train_and_predict_loo(
                self.prediction_train_features,
                self.prediction_train_labels,
                self.prediction_predict_features,
                self.prediction_predict_df,
                priors,
            )

        # Store predictions
        self.prediction_predict_df.loc[:, ["MON_proba", "cMI_proba", "iMI_proba", "SMI_proba"]] = (
            probabilities
        )
        self.prediction_predict_df["prediction"] = predictions

        # Get confusion matrix for probability scaling (calls do_cv)
        # Preserve existing self.cm since we only need this CM for scaling
        saved_cm = getattr(self, "cm", None)
        cm = self.do_cv(
            method="lpo",
            clf=LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            feature_type="morph",
            train_mod="all",
            test_mod="clem",
            fraction_across_classes=False,
            n_repeats=100,
            test_size=0.3,
            p=1,
            return_cm=True,
        )
        # Restore the original CM to avoid overwriting cross-validation results
        if saved_cm is not None:
            self.cm = saved_cm

        # Scale probabilities
        scaled_probabilities, scaled_predictions = pipeline.scale_by_confusion_matrix(
            probabilities, cm
        )

        # Store scaled results
        self.prediction_predict_df.loc[
            :, ["MON_proba_scaled", "cMI_proba_scaled", "iMI_proba_scaled", "SMI_proba_scaled"]
        ] = scaled_probabilities
        self.prediction_predict_df["prediction_scaled"] = scaled_predictions

        # Print summary
        pipeline.print_summary(predictions, scaled_predictions)

        print(" Cell prediction complete\n")

    def _sync_morphology_features(self):
        """Sync HDF5 morphology with xlsx-corrected values.

        The xlsx DataFrame (cells_with_to_predict) may contain morphology
        corrections from the morphology_gregor column that aren't in the
        HDF5.  This updates morphology_clone and features[:, 0] in the
        HDF5-derived DataFrames to match.
        """
        if "morphology_gregor" not in self.cells_with_to_predict.columns:
            return

        gregor_cells = self.cells_with_to_predict[
            self.cells_with_to_predict["morphology_gregor"].notna()
        ].set_index("cell_name")["morphology_gregor"]

        if gregor_cells.empty:
            return

        # Get standardized feature values for each morphology type
        morph_feature_values = {}
        for morph_str in ["ipsilateral", "contralateral"]:
            mask = self.all_cells["morphology_clone"] == morph_str
            if mask.any():
                morph_feature_values[morph_str] = np.unique(
                    self.features[mask.to_numpy(), 0]
                )[0]

        # Update HDF5 DataFrames and feature matrices
        for df_attr, feat_attr in [
            ("all_cells", "features"),
            ("all_cells_with_to_predict", "features_with_to_predict"),
        ]:
            df = getattr(self, df_attr)
            features = getattr(self, feat_attr)
            for cell_name, correct_morph in gregor_cells.items():
                cell_mask = (df["cell_name"] == cell_name).to_numpy()
                if cell_mask.any():
                    df.loc[cell_mask, "morphology_clone"] = correct_morph
                    if correct_morph in morph_feature_values:
                        features[cell_mask, 0] = morph_feature_values[correct_morph]


class_predictor = ClassPredictor
