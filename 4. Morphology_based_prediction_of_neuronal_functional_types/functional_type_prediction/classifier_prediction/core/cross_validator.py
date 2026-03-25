"""Cross-Validation Module.

Handles all cross-validation strategies for model evaluation.

Classes:
    CVResult: Dataclass containing score and confusion matrix from CV
    CrossValidator: Manages cross-validation procedures
    ModalityCrossValidator: Handles complex train/test modality combinations
    ConfusionMatrixGenerator: Generates confusion matrices

Author: Florian Kämpf
"""

import copy
from dataclasses import dataclass
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeavePOut, ShuffleSplit
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CVResult - Result container for cross-validation
# =============================================================================


@dataclass
class CVResult:
    """Result container for cross-validation evaluation.

    Holds both the performance score and confusion matrix from a single
    CV run, eliminating the need to run evaluation twice.

    Attributes
    ----------
    score : float
        Performance score (accuracy or F1) from cross-validation.
        Range [0, 100] as percentage, rounded to 2 decimal places.
    confusion_matrix : np.ndarray
        Normalized confusion matrix as 2D numpy array.
        Rows are true labels, columns are predictions.
        Values are fractions (row-normalized by default).
    class_labels : List[str]
        List of class names in matrix order.
    n_predictions : int
        Number of predictions made during CV.
    n_splits : int
        Number of CV splits performed.
    metric : str
        Metric used for scoring ('accuracy' or 'f1').

    Examples
    --------
    >>> result = validator.evaluate(clf, X, y, return_cvresult=True)
    >>> print(f"F1: {result.score:.2f}%")
    >>> print(f"Classes: {result.class_labels}")
    >>> print(f"Confusion matrix shape: {result.confusion_matrix.shape}")

    Notes
    -----
    This dataclass was created to solve the double-evaluation problem
    where do_cv() had to run CV twice: once for score, once for CM.
    Now both are computed in a single pass and returned together.
    """

    score: float
    confusion_matrix: np.ndarray
    class_labels: list[str]
    n_predictions: int
    n_splits: int
    metric: str

    def __iter__(self):
        """Allow unpacking as (score, n_predictions)."""
        return iter((self.score, self.n_predictions))


# =============================================================================
# Shared Helper Functions
# =============================================================================


def _compute_score(true_labels: np.ndarray, pred_labels: np.ndarray, metric: str) -> float:
    """Compute classification score.

    Parameters
    ----------
    true_labels : np.ndarray
        Ground truth labels
    pred_labels : np.ndarray
        Predicted labels
    metric : str
        Metric name: 'accuracy' or 'f1'

    Returns
    -------
    float
        Score as percentage (0-100), rounded to 2 decimal places

    Raises
    ------
    ValueError
        If metric is not 'accuracy' or 'f1'
    """
    if metric == "accuracy":
        return round(accuracy_score(true_labels, pred_labels) * 100, 2)
    elif metric == "f1":
        return round(f1_score(true_labels, pred_labels, average="weighted") * 100, 2)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'accuracy' or 'f1'.")


def _check_duplicates(train: np.ndarray, test: np.ndarray) -> bool:
    """Check for duplicate rows between training and test datasets.

    Parameters
    ----------
    train : np.ndarray
        Training feature matrix
    test : np.ndarray
        Test feature matrix

    Returns
    -------
    bool
        True if duplicates found, False otherwise
    """
    train_set = set(map(tuple, train))
    test_set = set(map(tuple, test))
    common_rows = train_set.intersection(test_set)

    if common_rows:
        print(f"DEBUG: {len(common_rows)} duplicate rows found between Train and Test.")
        return True
    return False


# Acronym dictionary for confusion matrix labels (shared)
ACRONYM_DICT = {
    # Canonical nomenclature
    "motion_onset": "MON",
    "motion_integrator_contralateral": "cMI",
    "motion_integrator_ipsilateral": "iMI",
    "slow_motion_integrator": "SMI",
    "motion onset": "MON",
    "motion integrator contralateral": "cMI",
    "motion integrator ipsilateral": "iMI",
    "slow motion integrator": "SMI",
    # New nomenclature (used in current datasets)
    "MI": "MI",
    "MON": "MON",
    "SMI": "SMI",
    "neg_control": "NC",
    "to_predict": "TP",
}


# =============================================================================
# CrossValidator - Basic CV for single dataset
# =============================================================================


class CrossValidator:
    """Handles cross-validation for classification models.

    Provides clean interface for different CV strategies including
    Leave-P-Out and ShuffleSplit cross-validation.

    Methods
    -------
    evaluate(clf, X, y, method='lpo', p=1, n_repeats=100, ...)
        Perform cross-validation and return metrics
    """

    @staticmethod
    def evaluate(
        clf,
        features: np.ndarray,
        labels: np.ndarray,
        method: str = "lpo",
        p: int = 1,
        n_repeats: int = 100,
        test_size: float = 0.3,
        metric: str = "accuracy",
        return_cm: bool = False,
        proba_cutoff: float | None = None,
    ) -> tuple:
        """Perform cross-validation on classifier.

        Parameters
        ----------
        clf : sklearn classifier
            The classifier to evaluate
        features : np.ndarray
            Feature matrix (n_samples, n_features)
        labels : np.ndarray
            Label array (n_samples,)
        method : str, optional
            CV method: 'lpo' or 'ss'. Default: 'lpo'
        p : int, optional
            Number of samples to leave out for LPO. Default: 1
        n_repeats : int, optional
            Number of repeats for ShuffleSplit. Default: 100
        test_size : float, optional
            Test size for ShuffleSplit. Default: 0.3
        metric : str, optional
            Metric to return: 'accuracy' or 'f1'. Default: 'accuracy'
        return_cm : bool, optional
            If True, return confusion matrix. Default: False
        proba_cutoff : float, optional
            Minimum probability for confident predictions. Default: None

        Returns
        -------
        score : float or tuple
            Performance score, or (score, n_predictions) if proba_cutoff used
        confusion_matrix : np.ndarray, optional
            Returned if return_cm=True

        Notes
        -----
        - LPO (Leave-P-Out): Exhaustive cross-validation
        - SS (ShuffleSplit): Random sampling cross-validation
        - Probability cutoff filters low-confidence predictions
        """
        print(f"\n Cross-validation: method={method}, metric={metric}")
        print(f"   Features: {features.shape}, Repeats: {n_repeats}")

        # Create splitter based on method
        if method == "lpo":
            splitter = LeavePOut(p=p)
        elif method == "ss":
            splitter = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=42)
        else:
            raise ValueError(f"Unknown CV method: {method}. Use 'lpo' or 'ss'.")

        # Run cross-validation
        all_predictions, all_true_labels = CrossValidator._run_cv(
            clf, splitter, features, labels, proba_cutoff
        )

        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)

        # Return confusion matrix if requested
        if return_cm:
            cm = confusion_matrix(all_true_labels, all_predictions)
            print("   Returning confusion matrix")
            return cm

        # Calculate and return score
        score = _compute_score(all_true_labels, all_predictions, metric)
        print(f"   {metric.capitalize()}: {score}%")
        print(" Cross-validation complete\n")

        return score, len(all_predictions)

    @staticmethod
    def _run_cv(
        clf, splitter, features: np.ndarray, labels: np.ndarray, proba_cutoff: float | None
    ) -> tuple[list, list]:
        """Run cross-validation loop.

        Private helper that handles both LPO and ShuffleSplit methods.
        """
        all_predictions = []
        all_true_labels = []

        for train_idx, test_idx in splitter.split(features):
            X_train, X_test = features[train_idx], features[test_idx]  # noqa: N806
            y_train, y_test = labels[train_idx], labels[test_idx]

            # Train model
            clf.fit(X_train, y_train)

            # Predict with optional probability cutoff
            if proba_cutoff is not None:
                proba = clf.predict_proba(X_test)
                max_proba = np.max(proba, axis=1)

                # Only predict if confident enough
                for i, (prob, true_label) in enumerate(zip(max_proba, y_test, strict=True)):
                    if prob >= proba_cutoff:
                        prediction = clf.predict(X_test[i : i + 1])[0]
                        all_predictions.append(prediction)
                        all_true_labels.append(true_label)
            else:
                predictions = clf.predict(X_test)
                all_predictions.extend(predictions)
                all_true_labels.extend(y_test)

        return all_predictions, all_true_labels


# =============================================================================
# ModalityCrossValidator - Multi-modality CV
# =============================================================================


class ModalityCrossValidator:
    """Cross-validator for complex train/test modality combinations.

    Handles scenarios where training and testing data come from different
    modalities (e.g., train on CLEM, test on PA) with proper handling of
    overlapping indices.

    Parameters
    ----------
    features_dict : Dict[str, np.ndarray]
        Dictionary mapping feature type ('morph', 'pv', 'ps', 'ff') to feature arrays
    labels : np.ndarray
        Label array for all samples
    modality_indices : Dict[str, np.ndarray]
        Dictionary mapping modality name to boolean index array
    reduced_features_idx : np.ndarray, optional
        Boolean array for selected features (for 'morph' feature type)
    """

    # Use shared acronym dictionary
    ACRONYM_DICT = ACRONYM_DICT

    def __init__(
        self,
        features_dict: dict[str, np.ndarray],
        labels: np.ndarray,
        modality_indices: dict[str, np.ndarray],
        reduced_features_idx: np.ndarray | None = None,
    ):
        """Initialize ModalityCrossValidator with features, labels, and modality info."""
        self.features_dict = features_dict
        self.labels = labels
        self.modality_indices = modality_indices
        self.reduced_features_idx = reduced_features_idx

        # Add 'all' modality if not present
        if "all" not in self.modality_indices:
            n_samples = len(labels)
            self.modality_indices["all"] = np.full(n_samples, True)

    def _extract_features(
        self, feature_type: str, modality: str, idx: np.ndarray | None = None
    ) -> np.ndarray:
        """Extract features for a given modality and feature type."""
        mod_idx = self.modality_indices[modality]
        features = self.features_dict[feature_type][mod_idx]

        if feature_type == "morph" and idx is not None:
            features = features[:, idx]

        return features

    def evaluate(
        self,
        clf,
        method: str,
        feature_type: str,
        train_mod: str,
        test_mod: str,
        n_repeats: int = 100,
        test_size: float = 0.3,
        p: int = 1,
        ax: Any | None = None,
        figure_label: str = "error:no figure label",
        spines_red: bool = False,
        fraction_across_classes: bool = True,
        idx: np.ndarray | None = None,
        plot: bool = True,
        return_cm: bool = False,
        return_cvresult: bool = False,
        proba_cutoff: float | None = None,
        metric: str = "accuracy",
    ) -> Union[tuple, np.ndarray, "CVResult"]:
        """Perform cross-validation with different train/test modalities.

        Parameters
        ----------
        clf : sklearn classifier
            Classifier to use for training and prediction
        method : str
            CV method: 'lpo' (LeavePOut) or 'ss' (ShuffleSplit)
        feature_type : str
            Feature type: 'morph', 'pv', 'ps', or 'ff'
        train_mod : str
            Training modality ('all', 'pa', 'clem')
        test_mod : str
            Testing modality ('all', 'pa', 'clem')
        n_repeats : int
            Number of repeats for ShuffleSplit
        test_size : float
            Test size for ShuffleSplit
        p : int
            Number of samples to leave out for LPO
        ax : matplotlib.axes.Axes, optional
            Axes for plotting
        figure_label : str
            Label for the figure
        spines_red : bool
            If True, set plot spines to red
        fraction_across_classes : bool
            If True, normalize CM by true labels
        idx : np.ndarray, optional
            Feature indices to use
        plot : bool
            If True, plot confusion matrix
        return_cm : bool
            If True, return confusion matrix only (deprecated, use return_cvresult)
        return_cvresult : bool
            If True, return CVResult containing both score AND confusion matrix.
            This is preferred over return_cm as it avoids double evaluation.
        proba_cutoff : float, optional
            Minimum probability for predictions
        metric : str
            Metric: 'accuracy' or 'f1'

        Returns
        -------
        result : Union[Tuple, np.ndarray, CVResult]
            - If return_cvresult=True: CVResult with score and confusion matrix
            - If return_cm=True: confusion matrix only (np.ndarray)
            - Otherwise: (score, n_predictions) tuple
        """
        print(f"\n Cross-validation: {train_mod} → {test_mod}")
        print(f"   Method: {method}, Classifier: {str(clf)[:50]}...")
        print(f"   Feature type: {feature_type}, Repeats: {n_repeats}")

        # Determine feature index
        if idx is None:
            if self.reduced_features_idx is not None and feature_type == "morph":
                idx = self.reduced_features_idx
            else:
                # Use all features 
                idx = np.full(self.features_dict[feature_type].shape[1], True)

        # Extract features and labels
        mod2idx = self.modality_indices
        features_train = self._extract_features(feature_type, train_mod, idx)
        features_test = self._extract_features(feature_type, test_mod, idx)
        labels_train = self.labels[mod2idx[train_mod]]
        labels_test = self.labels[mod2idx[test_mod]]

        # Determine evaluation scenario
        check_test_equals_train = test_mod == train_mod
        check_train_in_test = test_mod == "all" and train_mod != "all"
        check_test_in_train = train_mod == "all" and test_mod != "all"

        # Scale features
        scaler = StandardScaler()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

        # Set up splitter
        if method == "lpo":
            splitter = LeavePOut(p=p)
        elif method == "ss":
            splitter = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=0)

        # Run cross-validation
        true_labels, pred_labels, clf_work = self._run_cv(
            clf,
            splitter,
            method,
            features_train,
            features_test,
            labels_train,
            labels_test,
            mod2idx,
            train_mod,
            test_mod,
            check_test_equals_train,
            check_test_in_train,
            check_train_in_test,
            proba_cutoff,
        )

        # Calculate confusion matrix
        normalize = "true" if fraction_across_classes else "pred"
        cm = confusion_matrix(true_labels, pred_labels, normalize=normalize).astype(float)

        # Plot if requested
        if plot:
            self._plot_confusion_matrix(
                cm,
                clf_work,
                method,
                test_size,
                n_repeats,
                p,
                figure_label,
                ax,
                spines_red,
                true_labels,
                pred_labels,
            )

        # Return results
        if return_cvresult:
            # Return CVResult with both score AND confusion matrix
            score = _compute_score(np.array(true_labels), np.array(pred_labels), metric)
            class_labels = (
                list(clf_work.classes_) if clf_work and hasattr(clf_work, "classes_") else []
            )
            n_splits = (
                n_repeats if method == "ss" else sum(1 for _ in splitter.split(features_train))
            )

            print(f"   {metric.capitalize()}: {score}%")
            print(" Cross-validation complete (returning CVResult)\n")

            return CVResult(
                score=score,
                confusion_matrix=cm,
                class_labels=class_labels,
                n_predictions=len(pred_labels),
                n_splits=n_splits,
                metric=metric,
            )

        if return_cm:
            print("   Returning confusion matrix")
            return cm

        return self._calculate_score(true_labels, pred_labels, metric, proba_cutoff)

    def _run_cv(
        self,
        clf,
        splitter,
        method: str,
        features_train: np.ndarray,
        features_test: np.ndarray,
        labels_train: np.ndarray,
        labels_test: np.ndarray,
        mod2idx: dict[str, np.ndarray],
        train_mod: str,
        test_mod: str,
        check_test_equals_train: bool,
        check_test_in_train: bool,
        check_train_in_test: bool,
        proba_cutoff: float | None,
    ) -> tuple[list, list, Any]:
        """Run cross-validation and return predictions."""
        true_labels = []
        pred_labels = []
        clf_work = None

        if check_test_equals_train:
            # Same dataset for train and test
            true_labels, pred_labels, clf_work = self._cv_same_modality(
                clf, splitter, features_train, features_test, labels_train, labels_test
            )

        elif check_test_in_train:
            # Test modality is subset of train (train='all')
            true_labels, pred_labels, clf_work = self._cv_test_in_train(
                clf,
                splitter,
                features_train,
                labels_train,
                mod2idx,
                train_mod,
                test_mod,
                proba_cutoff,
            )

        elif check_train_in_test:
            # Train modality is subset of test (test='all')
            true_labels, pred_labels, clf_work = self._cv_train_in_test(
                clf, splitter, features_test, labels_test, mod2idx, train_mod, test_mod
            )

        else:
            # Different datasets for train and test
            true_labels, pred_labels, clf_work = self._cv_different_modalities(
                clf, splitter, method, features_train, features_test, labels_train, labels_test
            )

        return true_labels, pred_labels, clf_work

    def _cv_same_modality(
        self,
        clf,
        splitter,
        features_train: np.ndarray,
        features_test: np.ndarray,
        labels_train: np.ndarray,
        labels_test: np.ndarray,
    ) -> tuple[list, list, Any]:
        """CV when train and test modalities are the same."""
        true_labels = []
        pred_labels = []
        clf_work = None

        for train_index, test_index in splitter.split(features_train):
            clf_work = clone(clf)
            X_train = features_train[train_index]  # noqa: N806
            X_test = features_test[test_index]  # noqa: N806
            y_train = labels_train[train_index]
            y_test = labels_test[test_index]

            _check_duplicates(X_train, X_test)
            clf_work.fit(X_train, y_train)

            try:
                true_labels.extend(y_test)
                pred_labels.extend(clf_work.predict(X_test))
            except Exception:
                pass

        return true_labels, pred_labels, clf_work

    def _cv_test_in_train(
        self,
        clf,
        splitter,
        features_train: np.ndarray,
        labels_train: np.ndarray,
        mod2idx: dict[str, np.ndarray],
        train_mod: str,
        test_mod: str,
        proba_cutoff: float | None,
    ) -> tuple[list, list, Any]:
        """CV when test modality is subset of train (train='all')."""
        true_labels = []
        pred_labels = []
        clf_work = None

        for train_index, test_index in splitter.split(features_train):
            bool_train = np.full_like(mod2idx[test_mod], False)
            bool_test = np.full_like(mod2idx[test_mod], False)
            bool_train[train_index] = True
            bool_test[test_index] = True

            clf_work = clone(clf)
            X_train = features_train[bool_train * mod2idx[train_mod]]  # noqa: N806
            X_test = features_train[bool_test * mod2idx[test_mod]]  # noqa: N806
            y_train = labels_train[bool_train * mod2idx[train_mod]]
            y_test = labels_train[bool_test * mod2idx[test_mod]]

            clf_work.fit(X_train, y_train)

            if y_test.size != 0:
                try:
                    if proba_cutoff is not None:
                        if (clf_work.predict_proba(X_test) >= proba_cutoff).any():
                            pred_labels.extend(clf_work.predict(X_test))
                            true_labels.extend(list(y_test))
                    else:
                        pred_labels.extend(clf_work.predict(X_test))
                        true_labels.extend(list(y_test))
                except Exception:
                    pass

        return true_labels, pred_labels, clf_work

    def _cv_train_in_test(
        self,
        clf,
        splitter,
        features_test: np.ndarray,
        labels_test: np.ndarray,
        mod2idx: dict[str, np.ndarray],
        train_mod: str,
        test_mod: str,
    ) -> tuple[list, list, Any]:
        """CV when train modality is subset of test (test='all')."""
        true_labels = []
        pred_labels = []
        clf_work = None

        for train_index, test_index in splitter.split(features_test):
            bool_train = np.full_like(mod2idx[test_mod], False)
            bool_test = np.full_like(mod2idx[test_mod], False)
            bool_train[train_index] = True
            bool_test[test_index] = True

            clf_work = clone(clf)
            X_train = features_test[bool_train * mod2idx[train_mod]]  # noqa: N806
            X_test = features_test[bool_test * mod2idx[test_mod]]  # noqa: N806
            y_train = labels_test[bool_train * mod2idx[train_mod]]
            y_test = labels_test[bool_test * mod2idx[test_mod]]

            clf_work.fit(X_train, y_train)

            if y_test.size != 0:
                try:
                    pred_labels.extend(clf_work.predict(X_test))
                    true_labels.extend(list(y_test))
                except Exception:
                    pass

        return true_labels, pred_labels, clf_work

    def _cv_different_modalities(
        self,
        clf,
        splitter,
        method: str,
        features_train: np.ndarray,
        features_test: np.ndarray,
        labels_train: np.ndarray,
        labels_test: np.ndarray,
    ) -> tuple[list, list, Any]:
        """CV when train and test modalities are different."""
        true_labels = []
        pred_labels = []
        clf_work = None

        if method == "lpo":
            for train_index, _test_index in splitter.split(features_train):
                clf_work = clone(clf)
                X_train = features_train[train_index]  # noqa: N806
                X_test = features_test  # noqa: N806
                y_train = labels_train[train_index]
                y_test = labels_test

                _check_duplicates(X_train, X_test)
                clf_work.fit(X_train, y_train)

                try:
                    pred_labels.extend(clf_work.predict(X_test))
                    true_labels.extend(list(y_test))
                except Exception:
                    pass

        elif method == "ss":
            ss_train = copy.deepcopy(splitter)
            ss_test = copy.deepcopy(splitter)

            for train_indices, test_indices in zip(
                ss_train.split(features_train), ss_test.split(features_test), strict=True
            ):
                clf_work = clone(clf)
                train_index = train_indices[0]
                test_index = test_indices[1]

                X_train = features_train[train_index]  # noqa: N806
                X_test = features_test[test_index]  # noqa: N806
                y_train = labels_train[train_index]
                y_test = labels_test[test_index]

                _check_duplicates(X_train, X_test)
                clf_work.fit(X_train, y_train)

                try:
                    pred_labels.extend(clf_work.predict(X_test))
                    true_labels.extend(y_test)
                except Exception:
                    pass

        return true_labels, pred_labels, clf_work

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        clf_work,
        method: str,
        test_size: float,
        n_repeats: int,
        p: int,
        figure_label: str,
        ax: Any | None,
        spines_red: bool,
        true_labels: list,
        pred_labels: list,
    ) -> None:
        """Plot confusion matrix."""
        # Reorder to paper class ordering: iMI, cMI, MON, SMI
        # sklearn alphabetical: 0=cMI, 1=iMI, 2=MON, 3=SMI
        # Paper ordering:        iMI, cMI, MON, SMI → [1, 0, 2, 3]
        new_order = [1, 0, 2, 3]
        cm_plot = cm[np.ix_(new_order, new_order)]

        split = f"{(1 - test_size) * 100}:{test_size * 100}"
        f1 = round(f1_score(true_labels, pred_labels, average="weighted"), 3)

        if method == "ss":
            title = f"Confusion Matrix (SS {split} x{n_repeats})\nF1 Score: {f1}\n{figure_label}"
        else:
            title = f"Confusion Matrix (LPO = {p})\nF1 Score: {f1}\n{figure_label}"

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        ConfusionMatrixDisplay(cm_plot).plot(ax=ax, cmap="Blues")
        im = ax.images[0]
        im.set_clim(0, 1)
        ax.set_title(title)

        # Set tick labels
        ax.set_xticklabels(["iMI", "cMI", "MON", "SMI"])
        ax.set_yticklabels(["iMI", "cMI", "MON", "SMI"])

        # Red spines if requested
        if spines_red:
            for spine in ["bottom", "top", "left", "right"]:
                ax.spines[spine].set_color("red")
                ax.spines[spine].set_linewidth(2)

    def _calculate_score(
        self, true_labels: list, pred_labels: list, metric: str, proba_cutoff: float | None
    ) -> tuple:
        """Calculate and return the score."""
        score = _compute_score(np.array(true_labels), np.array(pred_labels), metric)

        cutoff_msg = " (with probability cutoff)" if proba_cutoff else ""
        print(f"   {metric.capitalize()}: {score}%{cutoff_msg}")
        print(" Cross-validation complete\n")

        return score, len(pred_labels)


# =============================================================================
# ConfusionMatrixGenerator - Grid CM Generation
# =============================================================================


class ConfusionMatrixGenerator:
    """Generates confusion matrices for model evaluation.

    Simplified interface for creating confusion matrices across
    different training/test modality combinations.
    """

    @staticmethod
    def generate_grid(
        clf,
        features_dict: dict[str, np.ndarray],
        labels_dict: dict[str, np.ndarray],
        modalities: list = None,
        method: str = "lpo",
        **cv_kwargs,
    ) -> dict[str, np.ndarray]:
        """Generate confusion matrix grid for multiple modality combinations.

        Parameters
        ----------
        clf : sklearn classifier
            Classifier to evaluate
        features_dict : Dict[str, np.ndarray]
            Dictionary mapping modality name to features
        labels_dict : Dict[str, np.ndarray]
            Dictionary mapping modality name to labels
        modalities : list, optional
            List of modalities to test. Default: ['all', 'clem', 'pa']
        method : str, optional
            CV method. Default: 'lpo'
        **cv_kwargs
            Additional arguments for CrossValidator.evaluate()

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping '{train}_{test}' to confusion matrix

        Examples
        --------
        >>> cms = ConfusionMatrixGenerator.generate_grid(
        ...     clf, {"all": X_all, "clem": X_clem}, {"all": y_all, "clem": y_clem}
        ... )
        >>> cm_all_clem = cms["all_clem"]
        """
        # Handle mutable default
        if modalities is None:
            modalities = ["all", "clem", "pa"]
        print("\n Generating confusion matrix grid...")
        print(f"   Modalities: {modalities}")

        confusion_matrices = {}

        for train_mod in modalities:
            for test_mod in modalities:
                key = f"{train_mod}_{test_mod}"
                print(f"   Evaluating: {train_mod} → {test_mod}")

                cm = CrossValidator.evaluate(
                    clf,
                    features_dict[train_mod],
                    labels_dict[train_mod],
                    method=method,
                    return_cm=True,
                    **cv_kwargs,
                )

                confusion_matrices[key] = cm

        print(" Confusion matrix grid complete\n")
        return confusion_matrices
