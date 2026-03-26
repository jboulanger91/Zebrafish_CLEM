"""Feature selection utilities for morphological classification.

This module provides two complementary feature selection approaches:

FeatureSelector:
    Statistical feature selection using SelectKBest with f_classif
    or mutual_info_classif. Fast, good for initial filtering.

RFESelector:
    Recursive Feature Elimination with cross-validation scoring.
    Uses classifier performance to select optimal feature subset.
    Slower but often more accurate.

Typical workflow:
    1. Use FeatureSelector for quick dimensionality reduction
    2. Use RFESelector for final feature optimization

Example:
    >>> selector = RFESelector(
    ...     features=X,
    ...     labels=y,
    ...     feature_names=feature_names,
    ...     cv_callback=do_cv_func,
    ... )
    >>> result = selector.select_features_rfe(
    ...     train_mod="clem", test_mod="clem", metric="f1"
    ... )
    >>> print(f"Selected {result.best_n_features} features: score={result.best_score:.2%}")

Classes:
    FeatureSelector: Statistical feature selection with SelectKBest
    RFESelector: Recursive Feature Elimination (RFE)
    RFEResult: Dataclass containing RFE selection results

See Also
--------
    - class_predictor.py: Main pipeline that uses these selectors
    - sklearn.feature_selection: Underlying sklearn implementations

Author: Florian Kämpf
"""

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

try:
    from src.util.output_paths import get_output_dir
except ModuleNotFoundError:
    from util.output_paths import get_output_dir
from matplotlib.patches import Patch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


class FeatureSelector:
    """Handles feature selection using various algorithms.

    This class provides a clean interface for selecting optimal features
    for classification tasks.

    Methods
    -------
    select_k_best(features_train, labels_train, features_test, labels_test, ...)
        Find optimal features using SelectKBest with f_classif or mutual_info
    calculate_penalty(scores)
        Calculate penalty score for feature selection stability
    """

    @staticmethod
    def calculate_penalty(scores: list[float]) -> float:
        """Calculate penalty based on mean and stability of scores.

        Higher penalty means better and more stable performance.

        Parameters
        ----------
        scores : List[float]
            List of performance scores across classes

        Returns
        -------
        float
            Penalty score (higher is better)

        Notes
        -----
        Formula: mean(scores) / exp(std(scores)^2)
        This penalizes high variance (unstable) predictions.
        """
        p = 2  # Power for std penalty
        penalty = np.mean(scores) / np.exp(np.std(scores) ** p)
        return penalty

    @staticmethod
    def select_k_best(
        features_train: np.ndarray,
        labels_train: np.ndarray,
        features_test: np.ndarray,
        labels_test: np.ndarray,
        train_test_identical: bool,
        use_std_scale: bool = False,
    ) -> tuple[dict, dict, dict]:
        """Find optimal number of features using SelectKBest.

        Tests both f_classif and mutual_info_classif scorers to find
        the best feature subset.

        Parameters
        ----------
        features_train : np.ndarray
            Training features (n_samples, n_features)
        labels_train : np.ndarray
            Training labels (n_samples,)
        features_test : np.ndarray
            Test features (n_samples, n_features)
        labels_test : np.ndarray
            Test labels (n_samples,)
        train_test_identical : bool
            Whether train and test sets are identical (requires LOO)
        use_std_scale : bool, optional
            Whether to use penalty scaling. Default: False.

        Returns
        -------
        pred_correct_dict : Dict
            Prediction accuracy for each evaluator and feature count
        pred_correct_dict_per_class : Dict
            Per-class prediction accuracy
        used_features_idx : Dict
            Boolean indices of selected features for each feature count

        Notes
        -----
        When train_test_identical is True, uses Leave-One-Out CV.
        Otherwise, trains on full training set and tests on test set.
        """
        evaluators = {"f_classif": f_classif, "mutual_info_classif": mutual_info_classif}

        if train_test_identical:
            # Use Leave-One-Out cross-validation
            return FeatureSelector._select_k_best_loo(
                features_train, labels_train, evaluators, use_std_scale
            )
        else:
            # Train on full training set, test on separate test set
            return FeatureSelector._select_k_best_holdout(
                features_train, labels_train, features_test, labels_test, evaluators, use_std_scale
            )

    @staticmethod
    def _select_k_best_loo(
        features: np.ndarray, labels: np.ndarray, evaluators: dict, use_std_scale: bool
    ) -> tuple[dict, dict, dict]:
        """SelectKBest with Leave-One-Out cross-validation.

        Private helper for select_k_best when train/test are identical.
        """
        pred_correct_dict = {}
        pred_correct_dict_per_class = {}
        used_features_idx = {}

        for evaluator_name, evaluator in evaluators.items():
            pred_correct_dict[evaluator_name] = []
            pred_correct_dict_per_class[evaluator_name] = []
            used_features_idx[evaluator_name] = {}

            # Test each possible number of features
            for n_features in range(1, features.shape[1] + 1):
                np.random.seed(42)  # Reproducibility

                # Select k best features
                selector = SelectKBest(evaluator, k=n_features)
                selector.fit(features, labels)
                feature_idx = selector.get_support()

                used_features_idx[evaluator_name][n_features] = feature_idx
                pred_correct_list = []
                X_selected = features[:, feature_idx]  # noqa: N806

                # Leave-One-Out CV
                for i in range(X_selected.shape[0]):
                    # Leave out sample i
                    train_idx = [x for x in range(X_selected.shape[0]) if x != i]
                    X_train = X_selected[train_idx]  # noqa: N806
                    X_test = X_selected[i, :]  # noqa: N806
                    y_train = labels[[x for x in range(X_selected.shape[0]) if x != i]]
                    y_test = labels[i]

                    # Calculate class priors
                    priors = [len(y_train[y_train == c]) / len(y_train) for c in np.unique(y_train)]

                    # Train LDA classifier
                    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)
                    clf.fit(X_train, y_train.flatten())

                    # Predict only if confidence >= 0.5
                    if np.max(clf.predict_proba(X_test[np.newaxis, :])) >= 0.5:
                        pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                    else:
                        pred_correct_list.append(None)

                # Calculate overall accuracy
                correct_count = np.sum([x for x in pred_correct_list if x is not None])
                total_count = len(pred_correct_list)
                pred_correct_dict[evaluator_name].append(correct_count / total_count)

                # Calculate per-class accuracy
                class_accuracies = []
                for unique_label in np.unique(labels):
                    class_mask = labels == unique_label
                    class_predictions = np.array(pred_correct_list)[class_mask]
                    class_correct = np.sum([x for x in class_predictions if x is not None])
                    class_total = len(class_predictions)
                    class_accuracies.append(class_correct / class_total)

                if use_std_scale:
                    penalty = FeatureSelector.calculate_penalty(class_accuracies)
                    pred_correct_dict_per_class[evaluator_name].append(penalty)
                else:
                    pred_correct_dict_per_class[evaluator_name].append(np.mean(class_accuracies))

        return pred_correct_dict, pred_correct_dict_per_class, used_features_idx

    @staticmethod
    def _select_k_best_holdout(
        features_train: np.ndarray,
        labels_train: np.ndarray,
        features_test: np.ndarray,
        labels_test: np.ndarray,
        evaluators: dict,
        use_std_scale: bool,
    ) -> tuple[dict, dict, dict]:
        """SelectKBest with holdout validation.

        Private helper for select_k_best with separate train/test sets.
        """
        pred_correct_dict = {}
        pred_correct_dict_per_class = {}
        used_features_idx = {}

        for evaluator_name, evaluator in evaluators.items():
            pred_correct_dict[evaluator_name] = []
            pred_correct_dict_per_class[evaluator_name] = []
            used_features_idx[evaluator_name] = {}

            # Test each possible number of features
            for n_features in range(1, features_train.shape[1] + 1):
                np.random.seed(42)

                # Select k best features from training set
                selector = SelectKBest(evaluator, k=n_features)
                selector.fit(features_train, labels_train)
                feature_idx = selector.get_support()

                used_features_idx[evaluator_name][n_features] = feature_idx

                # Apply feature selection to both train and test
                X_train = features_train[:, feature_idx]  # noqa: N806
                X_test = features_test[:, feature_idx]  # noqa: N806

                # Calculate class priors
                priors = [
                    len(labels_train[labels_train == c]) / len(labels_train)
                    for c in np.unique(labels_train)
                ]

                # Train classifier
                clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)
                clf.fit(X_train, labels_train.flatten())

                # Predict on test set
                predictions = clf.predict(X_test)
                accuracy = np.mean(predictions == labels_test)
                pred_correct_dict[evaluator_name].append(accuracy)

                # Calculate per-class accuracy
                class_accuracies = []
                for unique_label in np.unique(labels_test):
                    class_mask = labels_test == unique_label
                    class_predictions = predictions[class_mask]
                    class_labels = labels_test[class_mask]
                    class_accuracy = np.mean(class_predictions == class_labels)
                    class_accuracies.append(class_accuracy)

                if use_std_scale:
                    penalty = FeatureSelector.calculate_penalty(class_accuracies)
                    pred_correct_dict_per_class[evaluator_name].append(penalty)
                else:
                    pred_correct_dict_per_class[evaluator_name].append(np.mean(class_accuracies))

        return pred_correct_dict, pred_correct_dict_per_class, used_features_idx


@dataclass
class RFEResult:
    """Results from Recursive Feature Elimination (RFE) feature selection.

    This dataclass encapsulates all outputs from RFESelector.select_features_rfe(),
    providing a structured way to access selection results and metrics.

    Attributes
    ----------
    selected_features_idx : np.ndarray
        Boolean mask indicating which features were selected.
        Shape: (n_features,). True for selected features.
    best_n_features : int
        Optimal number of features determined by the selection process.
    best_score : float
        Cross-validation score achieved with the optimal feature subset.
        Interpretation depends on the metric used (accuracy, f1, etc.).
    scores_by_n_features : List[float]
        Score achieved for each feature count tested (1 to n_features).
    estimator_name : str
        Name of the estimator that achieved the best results.
    method : str
        The CV method used: 'lpo' (Leave-P-Out) or 'ss' (ShuffleSplit).

    Example:
    -------
    >>> result = selector.select_features_rfe(train_mod="clem", test_mod="clem")
    >>> print(f"Selected {result.best_n_features} features")
    >>> print(f"Feature indices: {np.where(result.selected_features_idx)[0]}")
    >>> print(f"Best score: {result.best_score:.2%}")
    """

    selected_features_idx: np.ndarray
    best_n_features: int
    best_score: float
    scores_by_n_features: list[float]
    estimator_name: str
    method: str


class RFESelector:
    """Handles Recursive Feature Elimination (RFE) for feature selection.

    This class provides RFE functionality, extracting logic
    previously embedded in class_predictor.select_features_RFE. It tests
    each feature count and uses a CV callback to score each subset.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    labels : np.ndarray
        Label array (n_samples,)
    feature_names : List[str]
        Names of features for plotting and result interpretation
    cv_callback : Callable
        Function to call for cross-validation scoring. Should accept parameters:
        method, clf, feature_type, train_mod, test_mod, figure_label,
        fraction_across_classes, n_repeats, idx, plot, metric.
        Returns accuracy score (float) or tuple (accuracy, other).
    confusion_matrix_callback : Callable, optional
        Function to call for confusion matrix generation after optimal
        feature selection. Default: None.

    Class Attributes
    ----------------
    DEFAULT_ESTIMATORS : List
        Default estimators to test when none specified. Includes:
        LinearDiscriminantAnalysis, LogisticRegression, LinearSVC,
        RidgeClassifier, Perceptron, PassiveAggressiveClassifier,
        RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier, AdaBoostClassifier, DecisionTreeClassifier.
    Example:
    -------
    >>> selector = RFESelector(
    ...     features=X,
    ...     labels=y,
    ...     feature_names=["feat1", "feat2", "feat3"],
    ...     cv_callback=my_cv_function,
    ... )
    >>> result = selector.select_features_rfe(
    ...     train_mod="clem", test_mod="clem", metric="f1"
    ... )
    """

    # Default estimators for RFE when none specified.
    # These cover linear, tree-based, and ensemble methods.
    DEFAULT_ESTIMATORS = [
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        LogisticRegression(random_state=0),
        LinearSVC(random_state=0),
        RidgeClassifier(random_state=0),
        Perceptron(random_state=0),
        PassiveAggressiveClassifier(random_state=0),
        RandomForestClassifier(random_state=0),
        GradientBoostingClassifier(random_state=0),
        ExtraTreesClassifier(random_state=0),
        AdaBoostClassifier(random_state=0),
        DecisionTreeClassifier(random_state=0),
    ]

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str],
        cv_callback: Callable[..., float],
        confusion_matrix_callback: Callable | None = None,
    ):
        """Initialize RFESelector with features, labels, and callbacks."""
        self.features = features
        self.labels = labels
        self.feature_names = np.array(feature_names)
        self.cv_callback = cv_callback
        self.confusion_matrix_callback = confusion_matrix_callback

    def select_features_rfe(
        self,
        train_mod: str,
        test_mod: str,
        estimator=None,
        cv_method_rfe: str = "lpo",
        metric: str = "accuracy",
        save_features: bool = False,
        output_subdir: str = "rfe",
    ) -> RFEResult:
        """Select features using Recursive Feature Elimination.

        Parameters
        ----------
        train_mod : str
            Training modality ('all', 'pa', 'clem')
        test_mod : str
            Test modality ('all', 'pa', 'clem')
        estimator : estimator or list, optional
            Estimator(s) for RFE. Default: multiple classifiers.
        cv_method_rfe : str
            CV method for scoring ('lpo' or 'ss').
        metric : str
            Metric for scoring.
        save_features : bool
            Whether this is a final selection (affects return behavior).

        Returns
        -------
        RFEResult
            Results including selected feature indices and scores.
        """
        print(f"\n Recursive Feature Elimination (RFE): {train_mod} → {test_mod}")
        print(f"   CV method: {cv_method_rfe}")
        print(f"   Metric: {metric}")

        # Set up estimators
        if estimator is None:
            all_estimators = self.DEFAULT_ESTIMATORS
        elif not isinstance(estimator, list):
            all_estimators = [estimator]
        else:
            all_estimators = estimator

        return self._select_with_manual_rfe(
            train_mod, test_mod, all_estimators, cv_method_rfe, metric, save_features,
            output_subdir,
        )

    def _select_with_manual_rfe(
        self,
        train_mod: str,
        test_mod: str,
        estimators: list,
        cv_method_rfe: str,
        metric: str,
        save_features: bool,
        output_subdir: str = "rfe",
    ) -> RFEResult:
        """Perform RFE with manual iteration over feature counts."""
        best_result = None

        for est in estimators:
            acc_list = []
            n_features_total = self.features.shape[1]

            # Test each number of features
            for n_feat in range(1, n_features_total + 1):
                selector = RFE(est, n_features_to_select=n_feat, step=1)
                selector.fit(self.features, self.labels)

                # Score with CV callback
                acc, _ = self.cv_callback(
                    method=cv_method_rfe,
                    clf=LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                    feature_type="morph",
                    train_mod=train_mod,
                    test_mod=test_mod,
                    figure_label=f"{str(est)}_n{n_feat}",
                    fraction_across_classes=True,
                    n_repeats=100,
                    idx=selector.support_,
                    plot=False,
                    metric=metric,
                )
                acc_list.append(acc)

            # Find optimal number of features
            best_n = np.nanargmax(acc_list) + 1
            best_score = acc_list[best_n - 1]

            # Create final selector with optimal features
            final_selector = RFE(est, n_features_to_select=best_n, step=1)
            final_selector.fit(self.features, self.labels)

            # Plot results
            self._plot_rfe_curve(
                acc_list, best_n, est, metric, cv_method_rfe, final_selector.support_,
                output_subdir,
            )

            est_name = str(est).split("(")[0]
            print(f"   Estimator: {est_name}, Features: {best_n}, {metric}: {best_score:.2f}%")

            # Call confusion matrix callback if provided
            if self.confusion_matrix_callback is not None:
                self.confusion_matrix_callback(
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                    method="lpo",
                    idx=final_selector.support_,
                )

            # Track best result across estimators
            if best_result is None or best_score > best_result.best_score:
                best_result = RFEResult(
                    selected_features_idx=final_selector.support_,
                    best_n_features=best_n,
                    best_score=best_score,
                    scores_by_n_features=acc_list,
                    estimator_name=est_name,
                    method=cv_method_rfe,
                )

        return best_result

    def _plot_rfe_curve(
        self,
        scores: list[float],
        best_n: int,
        estimator,
        metric: str,
        cv_method: str,
        selected_features: np.ndarray,
        output_subdir: str = "rfe",
    ):
        """Plot RFE score vs number of features."""
        plt.figure()
        plt.plot(scores)
        plt.axvline(best_n - 1, c="red", alpha=0.3)
        plt.text(
            best_n,
            np.mean(plt.ylim()),
            f"n = {best_n}",
            fontsize=12,
            color="red",
            ha="left",
            va="bottom",
        )

        n_features = len(scores)
        plt.gca().set_xticks(np.arange(0, n_features, 3), np.arange(1, n_features + 1, 3))

        est_name = str(estimator).split("(")[0]
        plt.title(f"{est_name}\n{metric} {scores[best_n - 1]:.2f}% {cv_method}", fontsize="small")

        # Add feature legend
        selected_names = self.feature_names[selected_features]
        legend_patches = [
            Patch(facecolor="white", edgecolor="white", label=name) for name in selected_names
        ]
        plt.legend(
            handles=legend_patches,
            frameon=False,
            fontsize=6,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        plt.subplots_adjust(left=0.1, right=0.65, top=0.80, bottom=0.1)

        # Save plot
        rfe_path = get_output_dir("classifier_pipeline", output_subdir)

        filename = (
            f"{np.nanmax(scores):.2f}_{est_name}"
            f"_features_{best_n}_{cv_method}_{metric}"
        )
        print(f"    Saving RFE plot to: {rfe_path / filename}.png|pdf")
        plt.savefig(rfe_path / f"{filename}.png")
        plt.savefig(rfe_path / f"{filename}.pdf")
        plt.close()

