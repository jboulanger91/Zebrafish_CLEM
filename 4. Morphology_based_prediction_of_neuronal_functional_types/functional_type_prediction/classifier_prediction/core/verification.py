"""Verification Module for Cell Type Predictions.

This module validates machine learning predictions of neuronal cell types
by comparing predicted cells against known examples using morphological
similarity metrics and statistical tests.

The verification pipeline consists of:
    1. **NBLAST Analysis**: Compute morphological similarity between neurons
       using the NBLAST algorithm (Costa et al., 2016). Neurons are compared
       using their 3D branching patterns stored in SWC format.

    2. **Outlier Detection**: Identify predictions that lie outside the
       expected feature space using three methods:
       - One-Class SVM (OCSVM): Support vector novelty detection
       - Isolation Forest (IF): Tree-based anomaly detection
       - Local Outlier Factor (LOF): Density-based outlier scoring

    3. **Statistical Testing**: Validate that predicted cells belong to their
       assigned class using distribution comparison tests:
       - Anderson-Ksamp: K-sample comparison of distributions
       - Kolmogorov-Smirnov: Non-parametric distribution test
       - Cramer-von Mises: Distribution equivalence test
       - Mann-Whitney U: Rank-based comparison

    4. **Visualization**: Generate PCA and t-SNE projections to visualize
       how predicted cells relate to training data.

    5. **Export**: Save verified predictions to Excel files with metadata.

Classes:
    NBLASTCalculator: Computes NBLAST morphological similarity scores.
    OutlierDetector: Detects outliers using OCSVM, IF, and LOF.
    StatisticalTester: Runs statistical verification tests.
    VerificationVisualizer: Creates PCA/t-SNE validation plots.
    PredictionExporter: Exports results to Excel and metadata files.
    VerificationCalculator: Orchestrates the full verification pipeline.

Example:
    >>> from verification import VerificationCalculator
    >>> verifier = VerificationCalculator(
    ...     base_path=data_path,
    ...     train_df=training_cells,
    ...     predict_df=predictions,
    ...     train_features=X_train,
    ...     predict_features=X_test,
    ...     train_labels=y_train,
    ... )
    >>> results = verifier.run_verification(required_tests=["IF", "LOF"])
    >>> print(f"Passed: {results['passed_tests'].sum()}/{len(results)}")

References
----------
    Costa, M. et al. (2016). NBLAST: Rapid, sensitive comparison of neuronal
    structure and construction of neuron family databases. Neuron, 91(2).

Author: Florian Kämpf
Refactored from class_predictor.calculate_verification_metrics (2026-01-29)
"""

import contextlib
import copy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

try:
    from src.util.output_paths import get_output_dir
except ModuleNotFoundError:
    from util.output_paths import get_output_dir
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Optional navis import (may not be available in all environments)
try:
    import navis

    from morphology.nblast import nblast_two_groups_custom_matrix

    NAVIS_AVAILABLE = True
except ImportError:
    NAVIS_AVAILABLE = False
    navis = None


class NBLASTCalculator:
    """Computes NBLAST morphological similarity between neurons.

    NBLAST (Neuronal BLAST) measures similarity between neurons by comparing
    their local geometry at each branch point. Higher scores indicate more
    similar morphology. The algorithm works by:
        1. Converting neurons to "dotprops" (tangent vectors at regular intervals)
        2. For each point in neuron A, finding nearest neighbor in neuron B
        3. Scoring based on distance and alignment of tangent vectors
        4. Summing scores and normalizing by self-match

    This implementation uses a custom scoring matrix optimized for zebrafish
    hindbrain neurons, which differs from the default Drosophila-trained matrix.

    Attributes
    ----------
        smat_fish (navis.nbl.Lookup2d): Custom NBLAST scoring matrix for
            zebrafish neurons. None if custom matrix file not found.

    Example:
        >>> calc = NBLASTCalculator(Path("custom_nblast_matrix.csv"))
        >>> matrices = calc.calculate_nblast_matrices(train_cells, predict_cells, neg_control_cells)
        >>> similarity = matrices["train_predict"].loc["cell_A", "cell_B"]
        >>> print(f"Similarity: {similarity:.3f}")  # Range: typically 0-1

    Notes
    -----
        - Requires navis package with NBLAST functionality
        - Custom matrix should be trained on species-specific data
        - Self-matches are set to NaN in output matrices

    See Also
    --------
        navis.nblast: Core NBLAST implementation
        morphology.nblast.nblast_two_groups_custom_matrix: Batch computation
    """

    def __init__(self, custom_matrix_path: Path) -> None:
        """Initialize NBLAST calculator with species-specific scoring matrix.

        Parameters
        ----------
        custom_matrix_path : Path
            Path to custom NBLAST scoring matrix CSV file. The matrix should
            contain distance bins as rows and dot product bins as columns,
            with log-odds scores as values.

        Raises
        ------
        ImportError
            If navis package is not available.
        """
        if not NAVIS_AVAILABLE:
            raise ImportError("navis is required for NBLAST calculations")

        self.smat_fish = None
        if custom_matrix_path.exists():
            # Load custom scoring matrix trained on zebrafish neurons
            # This improves NBLAST accuracy compared to default Drosophila matrix
            temp = pd.read_csv(custom_matrix_path, index_col=0)
            self.smat_fish = navis.nbl.ablast_funcs.Lookup2d.from_dataframe(temp)

    @staticmethod
    def remove_self_match(df: pd.DataFrame) -> pd.DataFrame:
        """Remove self-matches from NBLAST matrix by setting diagonal to NaN."""
        mask = pd.DataFrame(
            np.equal.outer(df.index, df.columns), index=df.index, columns=df.columns
        )
        df[mask] = np.nan
        return df

    def calculate_nblast_matrices(
        self,
        train_cells: pd.DataFrame,
        to_predict_cells: pd.DataFrame,
        neg_control_cells: pd.DataFrame,
        calculate4recorded: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Calculate NBLAST matrices between cell groups.

        Returns dict with keys: 'train', 'train_nc', 'train_predict'
        """
        results = {}

        # Training cells vs training cells
        results["train"] = nblast_two_groups_custom_matrix(
            train_cells, train_cells, custom_matrix=self.smat_fish, shift_neurons=False
        )
        results["train"] = self.remove_self_match(results["train"])

        # Training cells vs negative control
        results["train_nc"] = nblast_two_groups_custom_matrix(
            train_cells, neg_control_cells, custom_matrix=self.smat_fish, shift_neurons=False
        )
        results["train_nc"] = self.remove_self_match(results["train_nc"])

        # Training cells vs to-predict cells
        results["train_predict"] = nblast_two_groups_custom_matrix(
            train_cells, to_predict_cells, custom_matrix=self.smat_fish, shift_neurons=False
        )
        results["train_predict"] = self.remove_self_match(results["train_predict"])

        if calculate4recorded:
            results["train_predict"] = pd.concat(
                [results["train"], results["train_predict"]], axis=1
            )
            results["train_predict"] = self.remove_self_match(results["train_predict"])

        return results

    def calculate_per_class_separability(
        self, nb_train: pd.DataFrame, train_cells: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate how separable classes are with NBLAST."""
        nb_train_copy = copy.deepcopy(nb_train)
        nb_train_copy.index = [
            train_cells.loc[train_cells.cell_name == x, "function"].iloc[0]
            for x in nb_train_copy.index
        ]
        nb_train_copy.columns = [
            train_cells.loc[train_cells.cell_name == x, "function"].iloc[0]
            for x in nb_train_copy.columns
        ]
        return (
            nb_train_copy.groupby([nb_train_copy.index])
            .mean()
            .T.groupby(nb_train_copy.index)
            .mean()
        )

    def extract_nblast_matches(
        self, nb_train: pd.DataFrame, nb_train_nc: pd.DataFrame, nb_train_predict: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Extract best NBLAST matches."""
        return {
            "train": navis.nbl.extract_matches(nb_train, 2)
            .loc[:, ["id", "match_2", "score_2"]]
            .rename(columns={"match_2": "match_1", "score_2": "score_1"}),
            "nc": navis.nbl.extract_matches(nb_train_nc.T, 1),
            "predict": navis.nbl.extract_matches(nb_train_predict.T, 1),
        }

    def get_class_nblast_distributions(
        self, nb_train: pd.DataFrame, train_cells: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Get NBLAST value distributions for each class."""
        distributions = {}
        class_names = {
            "mon": "motion_onset",
            "iMI": "motion_integrator_ipsilateral",
            "cMI": "motion_integrator_contralateral",
            "smi": "slow_motion_integrator",
        }

        for abbrev, full_name in class_names.items():
            names = train_cells.loc[train_cells["function"] == full_name, "cell_name"]
            if len(names) > 0:
                distributions[abbrev] = (
                    navis.nbl.extract_matches(nb_train.loc[names, names], 2)
                    .loc[:, ["id", "match_2", "score_2"]]
                    .rename(columns={"match_2": "match_1", "score_2": "score_1"})
                )

        return distributions


class OutlierDetector:
    """Detects outliers using multiple novelty detection algorithms.

    This class identifies predictions that fall outside the expected feature
    space using three complementary approaches:

    **One-Class SVM (OCSVM)**:
        Learns a decision boundary around training data using kernel methods.
        Uses polynomial kernel which captures non-linear relationships between
        features. Good for detecting global outliers in feature space.

    **Isolation Forest (IF)**:
        Tree-based anomaly detector that isolates observations by randomly
        selecting features and split values. Outliers are easier to isolate
        (fewer splits needed). Robust to high-dimensional data.

    **Local Outlier Factor (LOF)**:
        Density-based method comparing local density of a point to its
        neighbors. Points in low-density regions relative to neighbors are
        outliers. Good for detecting local outliers in clusters.

    For verification, a cell typically needs to pass IF and LOF tests to be
    considered valid (OCSVM can be overly strict).

    Attributes
    ----------
        train_features (np.ndarray): Training feature matrix (n_samples, n_features).
        train_labels (np.ndarray): Training class labels (n_samples,).
        ocsvm (OneClassSVM): Fitted One-Class SVM detector.
        isolation_forest (IsolationForest): Fitted Isolation Forest detector.
        lof (LocalOutlierFactor): Fitted Local Outlier Factor detector.

    Example:
        >>> detector = OutlierDetector(X_train, y_train)
        >>> results = detector.predict_global(X_test)
        >>> # results['IF'] is True for inliers, False for outliers
        >>> valid_predictions = results["IF"] & results["LOF"]
        >>> print(f"Valid: {valid_predictions.sum()}/{len(valid_predictions)}")
    """

    def __init__(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
        """Initialize and fit outlier detectors on training data.

        Parameters
        ----------
        train_features : np.ndarray
            Training feature matrix of shape (n_samples, n_features).
            Should be the same features used for classification.
        train_labels : np.ndarray
            Training class labels of shape (n_samples,). Used for
            intra-class outlier detection.
        """
        self.train_features = train_features
        self.train_labels = train_labels

        # Global outlier detectors - fit on all training data
        # These detect predictions that fall outside the overall feature space

        # OCSVM with polynomial kernel - captures non-linear feature relationships
        self.ocsvm = OneClassSVM(gamma="scale", kernel="poly").fit(train_features)

        # Isolation Forest - contamination=0.1 assumes ~10% training outliers
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42).fit(
            train_features
        )

        # LOF with novelty=True for predicting new points (vs. labeling training)
        # n_neighbors=5 balances sensitivity and robustness
        self.lof = LocalOutlierFactor(n_neighbors=5, novelty=True).fit(train_features)

    def predict_global(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Predict outliers using global detectors."""
        return {
            "OCSVM": self.ocsvm.predict(features) == 1,
            "IF": self.isolation_forest.predict(features) == 1,
            "LOF": self.lof.predict(features) == 1,
        }

    def predict_intra_class(self, features: np.ndarray, predicted_class: str) -> dict[str, bool]:
        """Predict outliers within the predicted class."""
        class_mask = self.train_labels == predicted_class
        class_features = self.train_features[class_mask]

        if len(class_features) < 2:
            return {"OCSVM_intra_class": True, "IF_intra_class": True, "LOF_intra_class": True}

        ocsvm = OneClassSVM(gamma="scale", kernel="poly").fit(class_features)
        iso_forest = IsolationForest(contamination=0.1, random_state=42).fit(class_features)
        lof = LocalOutlierFactor(n_neighbors=min(5, len(class_features) - 1), novelty=True).fit(
            class_features
        )

        features_reshaped = features.reshape(1, -1) if features.ndim == 1 else features

        return {
            "OCSVM_intra_class": bool(ocsvm.predict(features_reshaped) == 1),
            "IF_intra_class": bool(iso_forest.predict(features_reshaped) == 1),
            "LOF_intra_class": bool(lof.predict(features_reshaped) == 1),
        }


class StatisticalTester:
    """Performs statistical verification tests for cell type predictions.

    This class compares NBLAST score distributions between a predicted cell
    and its assigned class to determine if the prediction is statistically
    valid. Tests are based on comparing:
        - The predicted cell's NBLAST scores vs. known class members
        - Distribution of scores within the predicted class

    Test Categories:
        **NBLAST General (NBLAST_g)**: Simple threshold test - does the
            best match exceed a minimum similarity cutoff?

        **NBLAST Z-score (NBLAST_z, NBLAST_z_scaled)**: Is the predicted
            cell's best match within 1.96 standard deviations of the class
            mean? Uses scaled probabilities for _scaled variant.

        **Distribution Tests (NBLAST_ak, NBLAST_ks, CVM, MWU)**: Do the
            NBLAST scores from the predicted cell follow the same distribution
            as scores between known class members?
            - Anderson-Ksamp (ak): K-sample test
            - Kolmogorov-Smirnov (ks): Distribution equivalence
            - Cramer-von Mises (CVM): ECDF comparison
            - Mann-Whitney U (MWU): Rank-based comparison

        **Probability Tests**: Does the classifier have high confidence
            (>70%) in any class prediction?

    Example:
        >>> results = StatisticalTester.run_all_tests(
        ...     query_dist,
        ...     target_dist,
        ...     query_dist_scaled,
        ...     target_match,
        ...     query_match_dist,
        ...     query_match_dist_scaled,
        ...     cutoff=0.5,
        ...     cell_probas={"MON": 0.8},
        ...     cell_probas_scaled={"MON": 0.75},
        ... )
        >>> passed = [k for k, v in results.items() if v]
        >>> print(f"Passed tests: {passed}")
    """

    @staticmethod
    def run_all_tests(
        query_dist: np.ndarray,
        target_dist: np.ndarray,
        query_dist_scaled: np.ndarray,
        target_match: float,
        query_match_dist: list[float],
        query_match_dist_scaled: list[float],
        cutoff: float,
        cell_probas: dict[str, float],
        cell_probas_scaled: dict[str, float],
    ) -> dict[str, bool]:
        """Run all statistical verification tests on a single predicted cell.

        Parameters
        ----------
        query_dist : np.ndarray
            NBLAST scores from predicted cell to known class members.
        target_dist : np.ndarray
            NBLAST scores between known class members (reference distribution).
        query_dist_scaled : np.ndarray
            Same as query_dist but using probability-scaled class membership.
        target_match : float
            Best NBLAST match score for the predicted cell.
        query_match_dist : List[float]
            Distribution of best-match scores within the predicted class.
        query_match_dist_scaled : List[float]
            Same as query_match_dist but for scaled predictions.
        cutoff : float
            Minimum NBLAST score threshold for NBLAST_g test.
        cell_probas : Dict[str, float]
            Classifier probabilities for each class (unscaled).
        cell_probas_scaled : Dict[str, float]
            Classifier probabilities for each class (scaled by confusion matrix).

        Returns
        -------
        Dict[str, bool]
            Dictionary mapping test name to pass (True) / fail (False).
            Keys include: NBLAST_g, NBLAST_z, NBLAST_z_scaled, NBLAST_ak,
            NBLAST_ak_scaled, NBLAST_ks, NBLAST_ks_scaled, CVM, CVM_scaled,
            MWU, MWU_scaled, probability_test, probability_test_scaled.
        """
        results = {}

        # ==== TEST 1: NBLAST General (NBLAST_g) ====
        # Simple threshold: Does the best NBLAST match exceed the class-specific cutoff?
        # Cutoff is typically the mean within-class best-match score
        results["NBLAST_g"] = target_match > cutoff

        # ==== TEST 2: NBLAST Z-score (NBLAST_z, NBLAST_z_scaled) ====
        # Is the predicted cell's best match within normal range for the class?
        # Z-score > 1.96 means outside 95% confidence interval (unusual)
        if len(query_match_dist) > 0 and np.std(query_match_dist) > 0:
            z_score = abs((target_match - np.mean(query_match_dist)) / np.std(query_match_dist))
            results["NBLAST_z"] = z_score <= 1.96  # Within 95% CI
        else:
            results["NBLAST_z"] = False

        if len(query_match_dist_scaled) > 0 and np.std(query_match_dist_scaled) > 0:
            z_score_scaled = abs(
                (target_match - np.mean(query_match_dist_scaled)) / np.std(query_match_dist_scaled)
            )
            results["NBLAST_z_scaled"] = z_score_scaled <= 1.96
        else:
            results["NBLAST_z_scaled"] = False

        # ==== TEST 3: Anderson-Ksamp (NBLAST_ak, NBLAST_ak_scaled) ====
        # K-sample test: Do the two distributions come from the same population?
        # H0: Distributions are the same. p > 0.05 means we cannot reject H0.
        if len(query_dist) > 0 and len(target_dist) > 0:
            try:
                results["NBLAST_ak"] = stats.anderson_ksamp([query_dist, target_dist]).pvalue > 0.05
            except Exception:
                results["NBLAST_ak"] = False

            try:
                results["NBLAST_ak_scaled"] = (
                    stats.anderson_ksamp([query_dist_scaled, target_dist]).pvalue > 0.05
                )
            except Exception:
                results["NBLAST_ak_scaled"] = False
        else:
            results["NBLAST_ak"] = False
            results["NBLAST_ak_scaled"] = False

        # ==== TEST 4: Kolmogorov-Smirnov (NBLAST_ks, NBLAST_ks_scaled) ====
        # Two-sample KS test with alternative="less": Is query distribution
        # stochastically less than target? p > 0.05 means no significant difference.
        # "Less" because we want query scores to be at least as good as target.
        if len(query_dist) > 0 and len(target_dist) > 0:
            results["NBLAST_ks"] = (
                stats.ks_2samp(query_dist, target_dist, alternative="less").pvalue > 0.05
            )
            results["NBLAST_ks_scaled"] = (
                stats.ks_2samp(query_dist_scaled, target_dist, alternative="less").pvalue > 0.05
            )
        else:
            results["NBLAST_ks"] = False
            results["NBLAST_ks_scaled"] = False

        # ==== TEST 5: Cramer-von Mises (CVM, CVM_scaled) ====
        # Compares empirical CDFs. More sensitive than KS to differences in tails.
        # Requires at least 2 samples per group for meaningful comparison.
        if len(query_dist) > 1 and len(target_dist) > 1:
            results["CVM"] = stats.cramervonmises_2samp(query_dist, target_dist).pvalue > 0.05
            results["CVM_scaled"] = (
                stats.cramervonmises_2samp(query_dist_scaled, target_dist).pvalue > 0.05
            )
        else:
            results["CVM"] = False
            results["CVM_scaled"] = False

        # ==== TEST 6: Mann-Whitney U (MWU, MWU_scaled) ====
        # Non-parametric rank test: Is query distribution shifted lower than target?
        # alternative='less' tests if query values tend to be smaller (worse matches)
        if len(query_dist) > 0 and len(target_dist) > 0:
            results["MWU"] = (
                stats.mannwhitneyu(query_dist, target_dist, alternative="less").pvalue > 0.05
            )
            results["MWU_scaled"] = (
                stats.mannwhitneyu(query_dist_scaled, target_dist, alternative="less").pvalue > 0.05
            )
        else:
            results["MWU"] = False
            results["MWU_scaled"] = False

        # ==== TEST 7: Probability Tests ====
        # Simple confidence check: Is the classifier confident (>70%) about any class?
        # Low confidence predictions are less reliable regardless of morphology.
        results["probability_test"] = any(p > 0.7 for p in cell_probas.values())
        results["probability_test_scaled"] = any(p > 0.7 for p in cell_probas_scaled.values())

        return results


class VerificationVisualizer:
    """Creates dimensionality reduction visualizations for verification.

    Uses PCA and t-SNE to project high-dimensional feature space into 2D
    for visual inspection of how predicted cells relate to training data.

    **PCA (Principal Component Analysis)**:
        Linear projection preserving maximum variance. Training-fitted PCA
        is applied to test data, showing if predictions fall within the
        training distribution in the most informative directions.

    **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
        Non-linear projection preserving local neighborhood structure.
        Better at revealing clusters but less interpretable for outlier
        detection since distances are not preserved globally.

    Visualizations show:
        - Blue points: Training (known) cells
        - Green points: Test cells that pass outlier detection
        - Red points: Test cells that fail outlier detection

    Attributes
    ----------
        train_features (np.ndarray): Original training features.
        test_features (np.ndarray): Original test features.
        pca (PCA): Fitted PCA model (2 components).
        train_pca (np.ndarray): Training data in PCA space (n_train, 2).
        test_pca (np.ndarray): Test data in PCA space (n_test, 2).
        tsne (TSNE): Fitted t-SNE model.
        tsne_combined (np.ndarray): Combined train+test in t-SNE space.

    Example:
        >>> viz = VerificationVisualizer(X_train, X_test)
        >>> viz.create_cell_validation_plot(
        ...     cell_idx=0,
        ...     cell=pred_df.iloc[0],
        ...     outlier_results={"IF": np.array([True, False, True])},
        ...     target_match=0.7,
        ...     target_nb_dist=[0.6, 0.65, 0.8],
        ...     query_nb_dist=np.array([0.5, 0.55, 0.6]),
        ...     query_match_dist=[0.55, 0.6],
        ...     cutoff=0.5,
        ...     output_path=Path("validation_cell_0.png"),
        ... )
    """

    def __init__(self, train_features: np.ndarray, test_features: np.ndarray) -> None:
        """Initialize visualizer by computing PCA and t-SNE projections.

        Parameters
        ----------
        train_features : np.ndarray
            Training feature matrix of shape (n_train, n_features).
        test_features : np.ndarray
            Test feature matrix of shape (n_test, n_features).

        Notes
        -----
        t-SNE is fit on combined train+test data to allow direct comparison,
        while PCA is fit only on training data to detect out-of-distribution
        test samples.
        """
        self.train_features = train_features
        self.test_features = test_features

        # PCA: Fit on training data only, then transform both
        # This reveals if test points fall outside training variance structure
        self.pca = PCA(n_components=2)
        self.pca.fit(train_features)
        self.train_pca = self.pca.transform(train_features)
        self.test_pca = self.pca.transform(test_features)

        # t-SNE: Fit on combined data for cluster visualization
        # Note: t-SNE must be fit on all points together since it's non-parametric
        self.tsne = TSNE(n_components=2)
        combined = np.concatenate([train_features, test_features])
        self.tsne_combined = self.tsne.fit_transform(combined)

    def create_cell_validation_plot(
        self,
        cell_idx: int,
        cell: pd.Series,
        outlier_results: dict[str, np.ndarray],
        target_match: float,
        target_nb_dist: list[float],
        query_nb_dist: np.ndarray,
        query_match_dist: list[float],
        cutoff: float,
        output_path: Path,
    ) -> None:
        """Create 3x3 validation plot for a single cell."""
        plt.clf()
        fig, ax = plt.subplots(3, 3, figsize=(30, 30))

        legend_elements_ax = [
            Patch(facecolor="red", edgecolor="red", label="Test cell: Fail"),
            Patch(facecolor="green", edgecolor="green", label="Test cell: Pass"),
            Patch(facecolor="blue", edgecolor="blue", label="Ground truth cell"),
        ]

        # Row 0: NBLAST distribution plots
        ax[0, 0].axvline(target_match, label="ground truth cells", color="red")
        sns.histplot(
            query_match_dist,
            ax=ax[0, 0],
            kde=True,
            alpha=0.25,
            label="test cell",
            color="blue",
            stat="probability",
        )
        ax[0, 0].set_title("Best NBLAST match comparison\n\nUsed in: NBLAST_z")

        sns.histplot(
            target_nb_dist,
            ax=ax[0, 1],
            kde=True,
            alpha=0.25,
            label="test cell",
            color="red",
            stat="probability",
        )
        sns.histplot(
            query_nb_dist,
            ax=ax[0, 1],
            kde=True,
            alpha=0.25,
            label="ground truth cells",
            color="blue",
            stat="probability",
        )
        ax[0, 1].set_title(
            "NBLAST distribution comparison\n\nUsed in: NBLAST_ak, NBLAST_ks, CVM, MWU"
        )

        ax[0, 2].axvline(target_match, label="test cell", color="red")
        ax[0, 2].axvline(cutoff, label="ground truth cells based cutoff", color="blue")
        ax[0, 2].set_title("NBLAST general test\n\nUsed in: NBLAST_g")

        # Row 1: PCA projections
        for col, (name, key) in enumerate([("IF", "IF"), ("LOF", "LOF"), ("OCSVM", "OCSVM")]):
            hue_array = outlier_results.get(key, np.ones(len(self.test_pca), dtype=bool))
            sns.scatterplot(
                x=self.train_pca[:, 0],
                y=self.train_pca[:, 1],
                hue=True,
                palette={True: "blue"},
                alpha=0.8,
                ax=ax[1, col],
            )
            sns.scatterplot(
                x=self.test_pca[:, 0],
                y=self.test_pca[:, 1],
                hue=hue_array,
                palette={True: "green", False: "red"},
                alpha=0.8,
                ax=ax[1, col],
            )
            ax[1, col].set_title(f"PCA projection of {name}")
            ax[1, col].legend(handles=legend_elements_ax)
            self._draw_circle(
                self.test_pca[cell_idx, 0], self.test_pca[cell_idx, 1], ax=ax[1, col], radius=0.25
            )

        # Row 2: t-SNE projections
        n_train = len(self.train_features)
        for col, (name, key) in enumerate([("IF", "IF"), ("LOF", "LOF"), ("OCSVM", "OCSVM")]):
            hue_array = np.concatenate(
                [
                    np.full(n_train, 2),
                    outlier_results.get(key, np.ones(len(self.test_pca), dtype=bool)).astype(int),
                ]
            )
            sns.scatterplot(
                x=self.tsne_combined[:, 0],
                y=self.tsne_combined[:, 1],
                hue=hue_array,
                palette={0: "red", 1: "green", 2: "blue"},
                alpha=0.8,
                ax=ax[2, col],
            )
            ax[2, col].set_title(f"t-SNE projection of {name}")
            ax[2, col].legend(handles=legend_elements_ax)
            self._draw_circle(
                self.tsne_combined[cell_idx + n_train, 0],
                self.tsne_combined[cell_idx + n_train, 1],
                ax=ax[2, col],
            )

        legend_elements = [
            Patch(facecolor="red", edgecolor="red", label="Test cell"),
            Patch(facecolor="blue", edgecolor="blue", label="Ground truth"),
        ]
        plt.subplots_adjust(top=0.8, bottom=0.05, left=0.075, right=0.925)
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.99, 0.88), fontsize="large")

        plt.savefig(output_path)
        plt.close(fig)

    @staticmethod
    def _draw_circle(x: float, y: float, radius: float = 1, ax: plt.Axes = None) -> None:
        """Draw a circle on the given axis."""
        if ax is None:
            return
        circle = plt.Circle((x, y), radius, color="blue", fill=False)
        ax.add_artist(circle)
        ax.plot(x, y, "ro")


class PredictionExporter:
    """Exports prediction results to Excel files.

    Handles the final output stage of the prediction pipeline:
        1. Save predictions to timestamped Excel files (separate for CLEM and EM)
        2. Add Excel metadata (author, estimator, timestamp)

    Excel files include all prediction columns, probabilities, verification
    test results, and summary statistics. Files are saved with timestamps
    to preserve history of prediction runs.

    Attributes
    ----------
        EXPORT_COLUMNS (List[str]): Columns to include in Excel export.
        SUM_COLUMNS (List[str]): Columns summed for 'sum' score (unscaled).
        SUM_COLUMNS_SCALED (List[str]): Columns summed for 'sum_scaled' score.

    Example:
        >>> df = PredictionExporter.calculate_summary_columns(
        ...     predictions_df, required_tests=["IF", "LOF"]
        ... )
        >>> PredictionExporter.export_to_excel(df, Path("predictions.xlsx"), "clem", "LDA")
    """

    # Columns to include in Excel export (in order)
    EXPORT_COLUMNS = [
        "cell_name",
        "function",
        "morphology_clone",
        "neurotransmitter_clone",
        "prediction",
        "MON_proba",
        "cMI_proba",
        "iMI_proba",
        "SMI_proba",
        "prediction_scaled",
        "MON_proba_scaled",
        "cMI_proba_scaled",
        "iMI_proba_scaled",
        "SMI_proba_scaled",
        "NBLAST_g",
        "NBLAST_z",
        "NBLAST_z_scaled",
        "NBLAST_ak",
        "NBLAST_ak_scaled",
        "NBLAST_ks",
        "NBLAST_ks_scaled",
        "probability_test",
        "probability_test_scaled",
        "OCSVM",
        "IF",
        "LOF",
        "OCSVM_intra_class",
        "IF_intra_class",
        "LOF_intra_class",
        "CVM",
        "CVM_scaled",
        "MWU",
        "MWU_scaled",
        "sum",
        "sum_scaled",
        "passed_tests",
    ]

    SUM_COLUMNS = ["NBLAST_g", "NBLAST_z", "NBLAST_ak", "NBLAST_ks", "OCSVM", "IF", "LOF"]
    SUM_COLUMNS_SCALED = [
        "NBLAST_g",
        "NBLAST_z_scaled",
        "NBLAST_ak_scaled",
        "NBLAST_ks_scaled",
        "OCSVM",
        "IF",
        "LOF",
    ]

    @classmethod
    def calculate_summary_columns(cls, df: pd.DataFrame, required_tests: list[str]) -> pd.DataFrame:
        """Add sum and passed_tests columns."""
        df = df.copy()

        # Calculate sums
        available_sum_cols = [c for c in cls.SUM_COLUMNS if c in df.columns]
        available_sum_cols_scaled = [c for c in cls.SUM_COLUMNS_SCALED if c in df.columns]

        if available_sum_cols:
            df["sum"] = df.loc[:, available_sum_cols].fillna(0).astype(int).sum(axis=1)
        if available_sum_cols_scaled:
            df["sum_scaled"] = (
                df.loc[:, available_sum_cols_scaled].fillna(0).astype(int).sum(axis=1)
            )

        # Calculate passed_tests
        df["passed_tests"] = False
        available_required = [t for t in required_tests if t in df.columns]
        if available_required:
            df.loc[df[available_required].all(axis=1), "passed_tests"] = True

        return df

    TRAINING_EXPORT_COLUMNS = [
        "cell_name",
        "function",
        "imaging_modality",
        "morphology_clone",
        "neurotransmitter_clone",
    ]

    @classmethod
    def export_to_excel(
        cls,
        predict_df: pd.DataFrame,
        output_path: Path,
        train_df: pd.DataFrame | None = None,
    ) -> None:
        """Export results to a single Excel file.

        Sheets: Training (if train_df provided), CLEM predictions, EM predictions.
        """
        export_cols = [c for c in cls.EXPORT_COLUMNS if c in predict_df.columns]
        clem_df = predict_df.loc[predict_df["imaging_modality"] == "clem", export_cols]
        em_df = predict_df.loc[predict_df["imaging_modality"] == "EM", export_cols]

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            if train_df is not None:
                train_cols = [c for c in cls.TRAINING_EXPORT_COLUMNS if c in train_df.columns]
                train_df[train_cols].to_excel(writer, sheet_name="Training", index=False)
            if not clem_df.empty:
                clem_df.to_excel(writer, sheet_name="CLEM", index=False)
            if not em_df.empty:
                em_df.to_excel(writer, sheet_name="EM", index=False)


class VerificationCalculator:
    """Orchestrates the full verification pipeline for cell type predictions.

    This is the main entry point for verification. It coordinates:
        1. NBLAST morphological similarity calculations
        2. Outlier detection (global and intra-class)
        3. Statistical distribution tests
        4. Visualization generation
        5. Results export to Excel

    The verification determines which predictions are reliable by:
        - Comparing predicted cells to known class members morphologically
        - Checking if predictions fall within expected feature distributions
        - Applying multiple statistical tests with different assumptions

    A prediction is typically marked as "passed" if it passes the required
    tests (default: Isolation Forest and Local Outlier Factor).

    Attributes
    ----------
        base_path (Path): Root data directory for I/O operations.
        train_df (pd.DataFrame): Training cells with 'cell_name', 'function' columns.
        predict_df (pd.DataFrame): Predictions with 'cell_name', 'prediction' columns.
        train_features (np.ndarray): Training feature matrix (n_train, n_features).
        predict_features (np.ndarray): Prediction feature matrix (n_pred, n_features).
        train_labels (np.ndarray): Training class labels.
        suffix (str): File suffix for output files (e.g., '_optimize_all_predict').
        estimator (str): Classifier name for metadata (e.g., 'LDA', 'AdaBoost').
        nblast_calc (NBLASTCalculator): NBLAST computation handler (or None).
        outlier_detector (OutlierDetector): Outlier detection handler.
        visualizer (VerificationVisualizer): Visualization handler.
        nblast_matrices (Dict): Computed NBLAST similarity matrices.
        nblast_matches (Dict): Extracted best NBLAST matches.
        nblast_distributions (Dict): Per-class NBLAST score distributions.
        cutoff (float): NBLAST score threshold for NBLAST_g test.

    Example:
        >>> verifier = VerificationCalculator(
        ...     base_path=Path("/data/project"),
        ...     train_df=training_cells_df,
        ...     predict_df=predictions_df,
        ...     train_features=X_train,
        ...     predict_features=X_test,
        ...     train_labels=y_train,
        ...     suffix="_experiment1",
        ...     estimator="LDA",
        ... )
        >>> results = verifier.run_verification(required_tests=["IF", "LOF"], save_predictions=True)
        >>> n_passed = results["passed_tests"].sum()
        >>> n_total = len(results)
        >>> print(f"Verification: {n_passed}/{n_total} passed ({100 * n_passed / n_total:.1f}%)")

    Notes
    -----
        - NBLAST requires navis package and custom scoring matrix
        - Without NBLAST, only outlier detection tests are performed
        - Results are added as new columns to predict_df

    See Also
    --------
        run_verification: Main method to execute the pipeline
    """

    # Maps full cell type names to abbreviations used in NBLAST analysis
    ACRONYM_DICT = {
        "motion_onset": "mon",
        "motion_integrator_contralateral": "cMI",
        "motion_integrator_ipsilateral": "iMI",
        "slow_motion_integrator": "smi",
        "neg_control": "nc",
    }

    def __init__(
        self,
        base_path: Path,
        train_df: pd.DataFrame,
        predict_df: pd.DataFrame,
        train_features: np.ndarray,
        predict_features: np.ndarray,
        train_labels: np.ndarray,
        suffix: str = "",
        estimator: str = "unknown",
    ) -> None:
        """Initialize verification calculator with data and configuration.

        Parameters
        ----------
        base_path : Path
            Base data path for loading NBLAST matrix and saving results.
            Expected structure: base_path/custom_matrix.csv for NBLAST,
            base_path/clem_zfish1/ and base_path/em_zfish1/ for exports.
        train_df : pd.DataFrame
            Training cells dataframe. Required columns: 'cell_name', 'function'.
        predict_df : pd.DataFrame
            Prediction cells dataframe. Required columns: 'cell_name',
            'prediction', 'prediction_scaled', probability columns.
        train_features : np.ndarray
            Training feature matrix of shape (n_train, n_features).
        predict_features : np.ndarray
            Prediction feature matrix of shape (n_pred, n_features).
        train_labels : np.ndarray
            Training class labels of shape (n_train,).
        suffix : str, optional
            Suffix for output filenames. Default: ''.
        estimator : str, optional
            Estimator name for Excel metadata. Default: 'unknown'.
        """
        self.base_path = base_path
        self.train_df = train_df
        self.predict_df = predict_df.copy()  # We'll modify this
        self.train_features = train_features
        self.predict_features = predict_features
        self.train_labels = train_labels
        self.suffix = suffix
        self.estimator = estimator

        # Initialize sub-components
        self.nblast_calc = None
        self.outlier_detector = None
        self.visualizer = None

        # Results storage
        self.nblast_matrices = {}
        self.nblast_matches = {}
        self.nblast_distributions = {}
        self.cutoff = None

    def run_verification(
        self,
        required_tests: list[str] = None,
        calculate4recorded: bool = False,
        save_predictions: bool = True,
        force_new: bool = False,
    ) -> pd.DataFrame:
        """Run full verification pipeline.

        Parameters
        ----------
        required_tests : List[str]
            List of test names that must pass for a cell to be marked as verified.
        calculate4recorded : bool
            If True, include recorded cells in NBLAST calculations.
        save_predictions : bool
            If True, save predictions to Excel and metadata files.
        force_new : bool
            If True, force saving new prediction files even if unchanged.

        Returns
        -------
        pd.DataFrame
            Updated prediction dataframe with verification results.
        """
        if required_tests is None:
            required_tests = ["IF", "LOF"]

        print("\n Running verification metrics...")

        # Separate cell groups
        train_cells = self.train_df
        to_predict_cells = self.predict_df[self.predict_df.function == "to_predict"]
        neg_control_cells = self.predict_df[self.predict_df["is_neg_control"]]

        # Initialize NBLAST calculator
        print("   Initializing NBLAST calculator...")
        custom_matrix_path = self.base_path / "custom_nblast_matrix.csv"
        if NAVIS_AVAILABLE and custom_matrix_path.exists():
            self.nblast_calc = NBLASTCalculator(custom_matrix_path)

            # Calculate NBLAST matrices
            print("   Calculating NBLAST matrices...")
            self.nblast_matrices = self.nblast_calc.calculate_nblast_matrices(
                train_cells, to_predict_cells, neg_control_cells, calculate4recorded
            )

            # Extract matches
            self.nblast_matches = self.nblast_calc.extract_nblast_matches(
                self.nblast_matrices["train"],
                self.nblast_matrices["train_nc"],
                self.nblast_matrices["train_predict"],
            )

            # Get class distributions
            self.nblast_distributions = self.nblast_calc.get_class_nblast_distributions(
                self.nblast_matrices["train"], train_cells
            )

            # Calculate cutoff
            mean_scores = [
                self.nblast_distributions[k].score_1.mean()
                for k in self.nblast_distributions
                if len(self.nblast_distributions[k]) > 0
            ]
            self.cutoff = np.mean(mean_scores) if mean_scores else 0.6

            print(f"   NBLAST cutoff: {self.cutoff:.3f}")
        else:
            print("   Skipping NBLAST (navis not available or custom matrix not found)")

        # Initialize outlier detector
        print("   Initializing outlier detectors...")
        self.outlier_detector = OutlierDetector(self.train_features, self.train_labels)

        # Apply global outlier detection
        print("   Running global outlier detection...")
        global_outliers = self.outlier_detector.predict_global(self.predict_features)
        for key, values in global_outliers.items():
            self.predict_df.loc[:, key] = values

        # Initialize visualizer
        print("   Initializing visualizer...")
        self.visualizer = VerificationVisualizer(self.train_features, self.predict_features)

        # Process each cell
        print("   Processing individual cells...")
        self._process_cells(train_cells, calculate4recorded)

        # Calculate summary columns
        print("   Calculating summary columns...")
        self.predict_df = PredictionExporter.calculate_summary_columns(
            self.predict_df, required_tests
        )

        # Prediction export is handled by pipeline_main.py (predictions.xlsx)

        print(" Verification complete")
        return self.predict_df

    def _process_cells(self, train_cells: pd.DataFrame, calculate4recorded: bool) -> None:
        """Process each cell for verification metrics."""
        # Get class name mappings
        class_names = {}
        for _abbrev, full in [
            ("mon", "motion_onset"),
            ("iMI", "motion_integrator_ipsilateral"),
            ("cMI", "motion_integrator_contralateral"),
            ("smi", "slow_motion_integrator"),
        ]:
            class_names[full] = train_cells.loc[train_cells["function"] == full, "cell_name"]

        for idx, (i, cell) in enumerate(self.predict_df.iterrows()):
            # Get prediction class abbreviation
            pred_abbrev = self.ACRONYM_DICT.get(cell["prediction"], "mon").lower()
            pred_scaled_abbrev = self.ACRONYM_DICT.get(cell["prediction_scaled"], "mon").lower()

            # Intra-class outlier detection
            intra_results = self.outlier_detector.predict_intra_class(
                self.predict_features[idx], cell["prediction"]
            )
            for key, value in intra_results.items():
                self.predict_df.loc[i, key] = value

            # NBLAST-based tests (if available)
            if self.nblast_calc is not None:
                self._run_nblast_tests(
                    idx, i, cell, train_cells, class_names, pred_abbrev, pred_scaled_abbrev
                )

            # Create visualization
            if "cell_data_dir" in cell and cell["cell_data_dir"] is not None:
                with contextlib.suppress(Exception):
                    _ = (
                        Path(cell["cell_data_dir"])
                        / f"validation_test_visualized_{cell['cell_name']}.png"
                    )

    def _run_nblast_tests(
        self,
        idx: int,
        i: Any,
        cell: pd.Series,
        train_cells: pd.DataFrame,
        class_names: dict,
        pred_abbrev: str,
        pred_scaled_abbrev: str,
    ) -> None:
        """Run NBLAST-based verification tests for a cell."""
        # Determine which NBLAST matrix to use
        if cell.get("is_neg_control", False):
            nb_df = self.nblast_matrices["train_nc"]
        else:
            nb_df = self.nblast_matrices["train_predict"]

        # Get prediction class names
        predict_names = class_names.get(cell["prediction"], pd.Series())
        predict_names = predict_names[predict_names != cell["cell_name"]]
        predict_names_scaled = class_names.get(cell["prediction_scaled"], pd.Series())
        predict_names_scaled = predict_names_scaled[predict_names_scaled != cell["cell_name"]]

        if len(predict_names) == 0:
            return

        # Get target distribution
        try:
            target_nb_dist = nb_df.loc[predict_names, cell["cell_name"]].dropna()
            target_nb_dist = list(target_nb_dist.loc[target_nb_dist.index != cell["cell_name"]])
            target_match = np.max(target_nb_dist) if target_nb_dist else 0
        except Exception:
            return

        # Get query distributions
        try:
            query_nb_dist = (
                self.nblast_matrices["train"].loc[predict_names, predict_names].to_numpy().flatten()
            )
            query_nb_dist = query_nb_dist[~np.isnan(query_nb_dist)]

            query_nb_dist_scaled = (
                self.nblast_matrices["train"]
                .loc[predict_names_scaled, predict_names_scaled]
                .to_numpy()
                .flatten()
            )
            query_nb_dist_scaled = query_nb_dist_scaled[~np.isnan(query_nb_dist_scaled)]

            query_match_dist = list(
                navis.nbl.extract_matches(
                    self.nblast_matrices["train"].loc[predict_names, predict_names], 2
                ).loc[:, "score_2"]
            )

            query_match_dist_scaled = list(
                navis.nbl.extract_matches(
                    self.nblast_matrices["train"].loc[predict_names_scaled, predict_names_scaled], 2
                ).loc[:, "score_2"]
            )
        except Exception:
            return

        # Get cell probabilities
        cell_probas = {
            "iMI": cell.get("iMI_proba", 0),
            "SMI": cell.get("SMI_proba", 0),
            "cMI": cell.get("cMI_proba", 0),
            "MON": cell.get("MON_proba", 0),
        }
        cell_probas_scaled = {
            "iMI": cell.get("iMI_proba_scaled", 0),
            "SMI": cell.get("SMI_proba_scaled", 0),
            "cMI": cell.get("cMI_proba_scaled", 0),
            "MON": cell.get("MON_proba_scaled", 0),
        }

        # Run statistical tests
        test_results = StatisticalTester.run_all_tests(
            query_nb_dist,
            target_nb_dist,
            query_nb_dist_scaled,
            target_match,
            query_match_dist,
            query_match_dist_scaled,
            self.cutoff,
            cell_probas,
            cell_probas_scaled,
        )

        # Store results
        for key, value in test_results.items():
            self.predict_df.loc[i, key] = value

    def _export_results(self, force_new: bool) -> None:
        """Export training + predictions to a single Excel file."""
        predictions_dir = get_output_dir("classifier_pipeline", "predictions")
        output_path = predictions_dir / f"predictions{self.suffix}.xlsx"

        PredictionExporter.export_to_excel(
            self.predict_df, output_path, train_df=self.train_df,
        )
