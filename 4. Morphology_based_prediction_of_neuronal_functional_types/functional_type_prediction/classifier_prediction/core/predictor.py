"""Prediction Pipeline Module.

Handles model training, prediction, and probability scaling for cell type classification.

Classes:
    PredictionPipeline: Complete prediction workflow for cell type classification

Author: Florian Kämpf
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@dataclass
class PreparedData:
    """Container for prepared training and prediction data.

    Attributes
    ----------
    train_df : pd.DataFrame
        Training cells DataFrame with metadata
    train_features : np.ndarray
        Training feature matrix (n_train, n_features)
    train_labels : np.ndarray
        Training labels
    predict_df : pd.DataFrame
        Prediction cells DataFrame with metadata
    predict_features : np.ndarray
        Prediction feature matrix (n_predict, n_features)
    predict_labels : np.ndarray
        Prediction labels (may be 'to_predict' or 'neg_control')
    """

    train_df: pd.DataFrame
    train_features: np.ndarray
    train_labels: np.ndarray
    predict_df: pd.DataFrame
    predict_features: np.ndarray
    predict_labels: np.ndarray


class PredictionPipeline:
    """Complete prediction pipeline for cell type classification.

    Handles the full workflow: training, prediction, probability scaling,
    and result formatting.

    This is the main prediction class used by class_predictor.predict_cells().

    Methods
    -------
    prepare_priors(labels, use_jon_priors, jon_priors_dict)
        Calculate class priors for LDA training
    train_and_predict(train_features, train_labels, predict_features, priors)
        Train LDA and predict on new data
    train_and_predict_loo(train_features, train_labels, predict_features, predict_df, priors)
        Train and predict using Leave-One-Out for recorded cells
    scale_by_confusion_matrix(probabilities, confusion_matrix)
        Scale probabilities by true positive rates
    format_results(df, predictions, probabilities, scaled_predictions, scaled_probabilities)
        Format prediction results into DataFrame
    print_summary(predictions, scaled_predictions)
        Print prediction summary

    Examples
    --------
    >>> pipeline = PredictionPipeline()
    >>> priors = pipeline.prepare_priors(labels)
    >>> preds, probs = pipeline.train_and_predict(X_train, y_train, X_test, priors)
    >>> scaled_probs, scaled_preds = pipeline.scale_by_confusion_matrix(probs, cm)
    >>> result_df = pipeline.format_results(df, preds, probs, scaled_preds, scaled_probs)
    """

    def __init__(self, class_names: list[str] = None):
        """Initialize prediction pipeline.

        Parameters
        ----------
        class_names : List[str], optional
            Names of classes in order. Default: ['motion_onset',
            'motion_integrator_contralateral',
            'motion_integrator_ipsilateral', 'slow_motion_integrator']
        """
        # Order must match sklearn's alphabetical class ordering
        self.class_names = class_names or [
            "motion_integrator_contralateral",
            "motion_integrator_ipsilateral",
            "motion_onset",
            "slow_motion_integrator",
        ]
        self.proba_columns = ["cMI_proba", "iMI_proba", "MON_proba", "SMI_proba"]
        self.proba_scaled_columns = [
            "cMI_proba_scaled",
            "iMI_proba_scaled",
            "MON_proba_scaled",
            "SMI_proba_scaled",
        ]
        self.clf = None

    def prepare_data(
        self,
        train_modalities: list[str],
        all_cells: pd.DataFrame,
        all_cells_with_to_predict: pd.DataFrame,
        cells: pd.DataFrame,
        cells_with_to_predict: pd.DataFrame,
        features: np.ndarray,
        features_with_to_predict: np.ndarray,
        labels: np.ndarray,
        labels_with_to_predict: np.ndarray,
        modality_indices: dict[str, np.ndarray],
        selected_features_idx: np.ndarray,
        predict_recorded: bool = False,
    ) -> PreparedData:
        """Prepare training and prediction data for cell type classification.

        This method handles all data preparation: modality selection, DataFrame
        alignment, exclusion filtering, and feature/label extraction.

        Parameters
        ----------
        train_modalities : List[str]
            Modalities to use for training (e.g., ['clem', 'photoactivation'])
        all_cells : pd.DataFrame
            All training cells DataFrame (excludes to_predict/neg_control)
        all_cells_with_to_predict : pd.DataFrame
            All cells including to_predict and neg_control
        cells : pd.DataFrame
            Training cells with SWC data
        cells_with_to_predict : pd.DataFrame
            All cells with SWC data
        features : np.ndarray
            Training features matrix
        features_with_to_predict : np.ndarray
            All features including to_predict
        labels : np.ndarray
            Training labels
        labels_with_to_predict : np.ndarray
            All labels including to_predict
        modality_indices : Dict[str, np.ndarray]
            Dict mapping modality names to boolean indices
        selected_features_idx : np.ndarray
            Boolean mask for selected features
        predict_recorded : bool
            If True, include recorded cells in prediction using LOO

        Returns
        -------
        PreparedData
            Container with train/predict DataFrames, features, and labels

        Example:
        -------
        >>> pipeline = PredictionPipeline()
        >>> data = pipeline.prepare_data(
        ...     train_modalities=["clem", "photoactivation"],
        ...     all_cells=predictor.all_cells,
        ...     all_cells_with_to_predict=predictor.all_cells_with_to_predict,
        ...     cells=predictor.cells,
        ...     cells_with_to_predict=predictor.cells_with_to_predict,
        ...     features=predictor.features,
        ...     features_with_to_predict=predictor.features_with_to_predict,
        ...     labels=predictor.labels,
        ...     labels_with_to_predict=predictor.labels_with_to_predict,
        ...     modality_indices={"clem": clem_idx, "photoactivation": pa_idx},
        ...     selected_features_idx=predictor.reduced_features_idx,
        ... )
        >>> print(f"Train: {len(data.train_df)}, Predict: {len(data.predict_df)}")
        """
        # === MODALITY SELECTION ===
        selected_indices = None
        for modality in train_modalities:
            if modality in modality_indices:
                if selected_indices is None:
                    selected_indices = modality_indices[modality]
                else:
                    selected_indices = selected_indices | modality_indices[modality]

        if selected_indices is None:
            raise ValueError(f"No valid modalities found in {train_modalities}")

        # === ALIGN DATAFRAMES ===
        # Merge feature data with SWC/metadata
        # Select columns to merge - only those not already in all_cells_with_to_predict
        merge_cols = ["cell_name", "swc"]
        for col in ["cell_data_dir", "comment", "is_neg_control", "used_for_training"]:
            if (
                col in cells_with_to_predict.columns
                and col not in all_cells_with_to_predict.columns
            ):
                merge_cols.append(col)

        super_df = pd.merge(
            all_cells_with_to_predict,
            cells_with_to_predict.loc[:, merge_cols],
            on=["cell_name"],
            how="inner",
            suffixes=("", "_y"),
        )
        # Drop any duplicate _y columns
        super_df = super_df.loc[:, ~super_df.columns.str.endswith("_y")]

        # Ensure 'comment' column exists (some DataFrames may not have it)
        if "comment" not in super_df.columns:
            super_df["comment"] = ""

        # === PREPARE TRAINING DATA ===
        # Select columns to merge for training data
        train_merge_cols = ["cell_name", "swc"]
        for col in ["cell_data_dir", "comment"]:
            if col in cells.columns and col not in all_cells.columns:
                train_merge_cols.append(col)

        train_df = pd.merge(
            all_cells,
            cells.loc[:, train_merge_cols],
            on=["cell_name"],
            how="inner",
            suffixes=("", "_y"),
        )
        # Drop any duplicate _y columns
        train_df = train_df.loc[:, ~train_df.columns.str.endswith("_y")]

        # Ensure 'comment' column exists
        if "comment" not in train_df.columns:
            train_df["comment"] = ""
        train_df = train_df[train_df.imaging_modality.isin(train_modalities)]
        train_features = features[selected_indices][:, selected_features_idx]
        train_labels = labels[selected_indices]

        # === ALIGN FEATURES/LABELS TO SUPER_DF ===
        # The inner join may drop cells, so align features and labels to super_df
        aligned_mask = all_cells_with_to_predict.cell_name.isin(super_df.cell_name).to_numpy()
        features_aligned = features_with_to_predict[aligned_mask]
        labels_aligned = labels_with_to_predict[aligned_mask]

        # === BUILD EXCLUSION FILTERS (all from super_df) ===
        # Exclude cells with 'axon' in name
        exclude_axon = super_df.cell_name.apply(
            lambda x: "axon" not in x
        ).to_numpy()

        # Exclude cells already in training set
        exclude_train = (~super_df.cell_name.isin(train_df.cell_name)).to_numpy()

        # Include only to_predict or neg_control cells
        is_to_predict = (
            (super_df.function == "to_predict")
            | super_df["is_neg_control"]
        ).to_numpy()

        # Exclude cells with all NaN features
        has_valid_features = np.any(~np.isnan(features_aligned), axis=1)

        # Exclude reticulospinal cells
        exclude_reticulospinal = np.array(
            ["reticulospinal" not in str(x) for x in super_df.comment]
        )

        # Exclude myelinated cells
        exclude_myelinated = np.array(["myelinated" not in str(x) for x in super_df.comment])

        # === COMBINE FILTERS ===
        if not predict_recorded:
            # Standard prediction: exclude training cells
            include_mask = (
                exclude_train
                & exclude_axon
                & is_to_predict
                & has_valid_features
                & exclude_reticulospinal
                & exclude_myelinated
            )
        else:
            # Predict recorded: include training cells for LOO
            include_mask = (
                exclude_axon & has_valid_features & exclude_reticulospinal & exclude_myelinated
            )

        # === PREPARE PREDICTION DATA ===
        predict_df = super_df.loc[include_mask, :].copy()
        predict_features = features_aligned[include_mask][:, selected_features_idx]
        predict_labels = labels_aligned[include_mask]

        print(f"   Training set: {len(train_df)} cells")
        print(f"   Prediction set: {len(predict_df)} cells")
        print(f"   Number of features: {train_features.shape[1]}")

        return PreparedData(
            train_df=train_df,
            train_features=train_features,
            train_labels=train_labels,
            predict_df=predict_df,
            predict_features=predict_features,
            predict_labels=predict_labels,
        )

    def prepare_priors(
        self,
        labels: np.ndarray,
        use_jon_priors: bool = False,
        jon_priors_dict: dict[str, float] | None = None,
    ) -> list[float]:
        """Calculate class priors for LDA.

        Parameters
        ----------
        labels : np.ndarray
            Training labels
        use_jon_priors : bool
            If True, use provided jon_priors_dict
        jon_priors_dict : Dict[str, float], optional
            Dictionary mapping class names to prior probabilities

        Returns
        -------
        List[float]
            Prior probabilities in class order
        """
        unique_classes = np.unique(labels)

        if use_jon_priors and jon_priors_dict is not None:
            priors = [jon_priors_dict[x] for x in unique_classes]
            print(f"   Using Jon's priors: {dict(zip(unique_classes, priors, strict=True))}")
        else:
            priors = [len(labels[labels == x]) / len(labels) for x in unique_classes]
            print(f"   Using empirical priors: {dict(zip(unique_classes, priors, strict=True))}")

        return priors

    def train_and_predict(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        predict_features: np.ndarray,
        priors: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train LDA classifier and predict on new data.

        Parameters
        ----------
        train_features : np.ndarray
            Training feature matrix
        train_labels : np.ndarray
            Training labels
        predict_features : np.ndarray
            Features to predict
        priors : List[float]
            Class priors

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels
        probabilities : np.ndarray
            Prediction probabilities (n_samples, n_classes)
        """
        print("   Fitting LDA classifier...")
        self.clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)
        self.clf.fit(train_features, train_labels.flatten())

        print("   Predicting probabilities...")
        probabilities = self.clf.predict_proba(predict_features)
        predicted_int = np.argmax(probabilities, axis=1)
        predictions = np.array([self.clf.classes_[x] for x in predicted_int])

        print(f"   Predicted {len(predictions)} cells")
        return predictions, probabilities

    def train_and_predict_loo(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        predict_features: np.ndarray,
        predict_df: pd.DataFrame,
        priors: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train and predict using Leave-One-Out for recorded cells.

        For cells in training set, leaves them out during training.
        For to_predict/neg_control cells, uses full training set.

        Parameters
        ----------
        train_features : np.ndarray
            Training feature matrix
        train_labels : np.ndarray
            Training labels
        predict_features : np.ndarray
            Features to predict
        predict_df : pd.DataFrame
            DataFrame with 'function' column to identify cell types
        priors : List[float]
            Class priors

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels
        probabilities : np.ndarray
            Prediction probabilities
        """
        print("   Running Leave-One-Out prediction...")
        n_samples = len(predict_df)
        probabilities = np.zeros((n_samples, 4))
        predictions = []

        for i, (_idx, item) in enumerate(predict_df.iterrows()):
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)

            if item["function"] == "to_predict" or item.get("is_neg_control", False):
                # Use all training data
                clf.fit(train_features, train_labels.flatten())
                proba = clf.predict_proba(predict_features[i].reshape(1, -1))
            else:
                # Leave this cell out of training
                bool_copy = np.full(len(train_labels), True, dtype=bool)
                bool_copy[i] = False
                clf.fit(train_features[bool_copy], train_labels[bool_copy].flatten())
                proba = clf.predict_proba(train_features[~bool_copy].reshape(1, -1))

            probabilities[i] = proba[0]
            predicted_int = np.argmax(proba)
            predictions.append(clf.classes_[predicted_int])

        self.clf = clf  # Store last classifier for class names
        print(f"   Predicted {len(predictions)} cells with LOO")
        return np.array(predictions), probabilities

    def scale_by_confusion_matrix(
        self, probabilities: np.ndarray, confusion_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scale probabilities by true positive rates from confusion matrix.

        Parameters
        ----------
        probabilities : np.ndarray
            Unscaled probabilities (n_samples, n_classes)
        confusion_matrix : np.ndarray
            Confusion matrix from cross-validation

        Returns
        -------
        scaled_probabilities : np.ndarray
            Scaled probabilities
        scaled_predictions : np.ndarray
            Predictions based on scaled probabilities
        """
        print("   Scaling probabilities by true positive rates...")

        # Extract true positive rates (diagonal of CM)
        true_positive_rates = np.diag(confusion_matrix)
        tp_dict = dict(
            zip(self.clf.classes_, true_positive_rates, strict=True)
        )
        print(f"   True positive rates: {tp_dict}")

        # Scale probabilities
        scaled_probabilities = probabilities * true_positive_rates

        # Get predictions from scaled probabilities
        predicted_int = np.argmax(scaled_probabilities, axis=1)
        scaled_predictions = np.array([self.clf.classes_[x] for x in predicted_int])

        return scaled_probabilities, scaled_predictions

    def format_results(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        scaled_predictions: np.ndarray,
        scaled_probabilities: np.ndarray,
    ) -> pd.DataFrame:
        """Format prediction results into DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Original DataFrame to add predictions to
        predictions : np.ndarray
            Unscaled predictions
        probabilities : np.ndarray
            Unscaled probabilities
        scaled_predictions : np.ndarray
            Scaled predictions
        scaled_probabilities : np.ndarray
            Scaled probabilities

        Returns
        -------
        pd.DataFrame
            DataFrame with prediction columns added
        """
        # Add probability columns
        df.loc[:, self.proba_columns] = probabilities
        df["prediction"] = predictions

        # Add scaled probability columns
        df.loc[:, self.proba_scaled_columns] = scaled_probabilities
        df["prediction_scaled"] = scaled_predictions

        return df

    def print_summary(self, predictions: np.ndarray, scaled_predictions: np.ndarray):
        """Print prediction summary."""
        print("\n   Prediction Summary:")
        for class_name in self.clf.classes_:
            count = np.sum(predictions == class_name)
            count_scaled = np.sum(scaled_predictions == class_name)
            print(f"      {class_name}: {count} (unscaled), {count_scaled} (scaled)")

