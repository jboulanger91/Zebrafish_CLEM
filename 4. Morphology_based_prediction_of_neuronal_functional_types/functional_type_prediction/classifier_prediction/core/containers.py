"""Data container dataclasses for the classification pipeline.

These containers provide clean, documented, type-safe storage for pipeline
data. They replace scattered instance attributes with organized namespaces.

The containers follow the principle of "Explicit is better than implicit"
from the Zen of Python, making data flow clear and self-documenting.

Classes:
    ModalityMask: Boolean masks for filtering by imaging modality
    CellDataset: Container for cells with features and labels
    TrainingData: Separates training and prediction datasets
    PredictionResults: Container for prediction outputs

Example:
    >>> # Create dataset from loaded data
    >>> mask = ModalityMask.from_series(df["imaging_modality"])
    >>> dataset = CellDataset(
    ...     cells=df,
    ...     features=X,
    ...     labels=y,
    ...     modality=modality_array,
    ...     modality_mask=mask,
    ...     feature_names=column_labels,
    ... )
    >>> print(f"Training on {dataset.n_cells} cells")
    >>> print(f"Features shape: {dataset.features.shape}")

    >>> # Filter by modality
    >>> clem_data = dataset.filter_by_modality("clem")
    >>> print(f"CLEM cells: {clem_data.n_cells}")

Notes
-----
    The 'morph' suffix previously used in legacy code (features_fk, labels_fk)
    stood for 'Florian Kämpf' and has been removed throughout the codebase.

See Also
--------
    - class_predictor.py: Uses these containers for data organization
    - config.py: Configuration dataclasses
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# =============================================================================
# Modality Mask Container
# =============================================================================


@dataclass
class ModalityMask:
    """Boolean masks for filtering by imaging modality.

    Provides convenient access to modality-specific subsets of data
    via boolean indexing arrays.

    Attributes
    ----------
        clem: Boolean mask for CLEM (correlative light-electron microscopy) cells
        photoactivation: Boolean mask for photoactivation cells
        em: Boolean mask for EM-only cells (no functional data)
        all: Boolean mask selecting all cells (all True)

    Example:
        >>> mask = ModalityMask.from_series(df["imaging_modality"])
        >>> clem_features = features[mask.clem]
        >>> pa_labels = labels[mask.photoactivation]

        >>> # Dict-like access also works
        >>> clem_features = features[mask["clem"]]
    """

    clem: np.ndarray
    photoactivation: np.ndarray
    em: np.ndarray
    all: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize the 'all' mask after other fields are set."""
        self.all = np.ones(len(self.clem), dtype=bool)

    @classmethod
    def from_series(cls, modality_series: pd.Series) -> "ModalityMask":
        """Create masks from a pandas Series of modality values.

        Args:
            modality_series: Series containing modality strings
                ('clem', 'photoactivation', 'EM')

        Returns
        -------
            ModalityMask with boolean arrays for each modality

        Example:
            >>> mask = ModalityMask.from_series(df["imaging_modality"])
        """
        return cls(
            clem=(modality_series == "clem").to_numpy(),
            photoactivation=(modality_series == "photoactivation").to_numpy(),
            em=(modality_series == "EM").to_numpy(),
        )

    def __getitem__(self, key: str) -> np.ndarray:
        """Allow dict-like access to masks.

        Args:
            key: Modality name ('clem', 'pa', 'photoactivation', 'em', 'all')
                'pa' is automatically converted to 'photoactivation'

        Returns
        -------
            Boolean numpy array for the requested modality

        Example:
            >>> mask["clem"]  # Same as mask.clem
            >>> mask["pa"]  # Same as mask.photoactivation
        """
        # Normalize key
        key = key.lower()
        if key == "pa":
            key = "photoactivation"
        return getattr(self, key)

    @property
    def n_clem(self) -> int:
        """Number of CLEM cells."""
        return int(self.clem.sum())

    @property
    def n_photoactivation(self) -> int:
        """Number of photoactivation cells."""
        return int(self.photoactivation.sum())

    @property
    def n_em(self) -> int:
        """Number of EM cells."""
        return int(self.em.sum())

    def summary(self) -> dict[str, int]:
        """Return count summary for all modalities."""
        return {
            "clem": self.n_clem,
            "photoactivation": self.n_photoactivation,
            "em": self.n_em,
            "total": len(self.all),
        }


# =============================================================================
# Cell Dataset Container
# =============================================================================


@dataclass
class CellDataset:
    """Container for a dataset of cells with features and labels.

    Encapsulates all data needed for a set of cells: the metadata DataFrame,
    feature matrix, class labels, and modality information.

    Attributes
    ----------
        cells: DataFrame with cell metadata (cell_name, function, etc.)
        features: Feature matrix of shape (n_cells, n_features)
        labels: Class label array of shape (n_cells,)
        modality: Imaging modality array of shape (n_cells,)
        modality_mask: ModalityMask for filtering by modality
        feature_names: List of feature column names

    Example:
        >>> dataset = CellDataset(
        ...     cells=df,
        ...     features=X,
        ...     labels=y,
        ...     modality=df["imaging_modality"].to_numpy(),
        ...     modality_mask=ModalityMask.from_series(df["imaging_modality"]),
        ...     feature_names=column_labels,
        ... )
        >>> print(f"Dataset has {dataset.n_cells} cells, {dataset.n_features} features")

        >>> # Filter to CLEM only
        >>> clem_data = dataset.filter_by_modality("clem")
    """

    cells: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    modality: np.ndarray
    modality_mask: ModalityMask
    feature_names: list[str] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        """Number of cells in dataset."""
        return len(self.labels)

    @property
    def n_features(self) -> int:
        """Number of features."""
        if self.features.ndim > 1:
            return self.features.shape[1]
        return 0

    @property
    def class_counts(self) -> dict[str, int]:
        """Count of cells per class label."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts, strict=True))

    def filter_by_modality(self, modality: str) -> "CellDataset":
        """Return new dataset filtered to a single modality.

        Args:
            modality: Modality to filter to ('clem', 'pa', 'photoactivation', 'em', 'all')

        Returns
        -------
            New CellDataset containing only cells of the specified modality

        Example:
            >>> clem_only = dataset.filter_by_modality("clem")
            >>> print(f"CLEM cells: {clem_only.n_cells}")
        """
        mask = self.modality_mask[modality]
        filtered_cells = self.cells[mask].reset_index(drop=True)

        return CellDataset(
            cells=filtered_cells,
            features=self.features[mask],
            labels=self.labels[mask],
            modality=self.modality[mask],
            modality_mask=ModalityMask.from_series(filtered_cells["imaging_modality"]),
            feature_names=self.feature_names,
        )

    def filter_by_labels(self, exclude_labels: list[str]) -> "CellDataset":
        """Return new dataset excluding specified labels.

        Args:
            exclude_labels: List of labels to exclude (e.g., ['to_predict', 'neg_control'])

        Returns
        -------
            New CellDataset with specified labels removed

        Example:
            >>> training_data = dataset.filter_by_labels(["to_predict", "neg_control"])
        """
        mask = ~np.isin(self.labels, exclude_labels)
        filtered_cells = self.cells[mask].reset_index(drop=True)

        return CellDataset(
            cells=filtered_cells,
            features=self.features[mask],
            labels=self.labels[mask],
            modality=self.modality[mask],
            modality_mask=ModalityMask.from_series(filtered_cells["imaging_modality"]),
            feature_names=self.feature_names,
        )

    def select_features(self, feature_mask: np.ndarray) -> "CellDataset":
        """Return new dataset with only selected features.

        Args:
            feature_mask: Boolean array indicating which features to keep

        Returns
        -------
            New CellDataset with reduced feature set

        Example:
            >>> reduced = dataset.select_features(selected_features_idx)
        """
        selected_names = [n for n, m in zip(self.feature_names, feature_mask, strict=True) if m]

        return CellDataset(
            cells=self.cells.copy(),
            features=self.features[:, feature_mask],
            labels=self.labels.copy(),
            modality=self.modality.copy(),
            modality_mask=self.modality_mask,
            feature_names=selected_names,
        )

    def summary(self) -> dict:
        """Return summary statistics for the dataset."""
        return {
            "n_cells": self.n_cells,
            "n_features": self.n_features,
            "class_counts": self.class_counts,
            "modality_counts": self.modality_mask.summary(),
        }


# =============================================================================
# Training Data Container
# =============================================================================


@dataclass
class TrainingData:
    """Container for training and prediction datasets.

    Separates cells into:
    - training: Cells with known labels (for model training)
    - to_predict: Cells to be classified (unknown labels)

    Also tracks feature selection state.

    Attributes
    ----------
        training: CellDataset of cells with known labels
        to_predict: CellDataset of cells to classify
        selected_features: Boolean mask of selected features (after RFE)
        feature_names: Names of all features (before selection)

    Example:
        >>> data = TrainingData(training=train_dataset, to_predict=predict_dataset)
        >>> print(f"Training: {data.n_training}, To predict: {data.n_to_predict}")

        >>> # Get training features for a specific modality
        >>> clem_features = data.get_training_features("clem")
    """

    training: CellDataset
    to_predict: CellDataset
    selected_features: np.ndarray | None = None

    @property
    def n_training(self) -> int:
        """Number of training cells."""
        return self.training.n_cells

    @property
    def n_to_predict(self) -> int:
        """Number of cells to predict."""
        return self.to_predict.n_cells

    @property
    def n_selected_features(self) -> int:
        """Number of selected features (after RFE)."""
        if self.selected_features is None:
            return self.training.n_features
        return int(self.selected_features.sum())

    @property
    def feature_names(self) -> list[str]:
        """Feature names from training dataset."""
        return self.training.feature_names

    @property
    def selected_feature_names(self) -> list[str]:
        """Names of selected features (after RFE)."""
        if self.selected_features is None:
            return self.feature_names
        return [n for n, m in zip(self.feature_names, self.selected_features, strict=True) if m]

    def get_training_features(
        self, modality: str = "all", apply_selection: bool = True
    ) -> np.ndarray:
        """Get training features, optionally filtered by modality and feature selection.

        Args:
            modality: Filter to specific modality ('clem', 'pa', 'all')
            apply_selection: If True and features are selected, return only selected

        Returns
        -------
            Feature matrix for training

        Example:
            >>> X_train = data.get_training_features("clem")
            >>> X_all = data.get_training_features("all", apply_selection=False)
        """
        features = self.training.features

        # Filter by modality
        if modality != "all":
            mask = self.training.modality_mask[modality]
            features = features[mask]

        # Apply feature selection
        if apply_selection and self.selected_features is not None:
            features = features[:, self.selected_features]

        return features

    def get_training_labels(self, modality: str = "all") -> np.ndarray:
        """Get training labels, optionally filtered by modality."""
        labels = self.training.labels
        if modality != "all":
            mask = self.training.modality_mask[modality]
            labels = labels[mask]
        return labels

    def get_prediction_features(self, apply_selection: bool = True) -> np.ndarray:
        """Get features for cells to predict.

        Args:
            apply_selection: If True and features are selected, return only selected

        Returns
        -------
            Feature matrix for prediction
        """
        features = self.to_predict.features
        if apply_selection and self.selected_features is not None:
            features = features[:, self.selected_features]
        return features

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            "n_training": self.n_training,
            "n_to_predict": self.n_to_predict,
            "n_features": self.training.n_features,
            "n_selected_features": self.n_selected_features,
            "training_class_counts": self.training.class_counts,
            "training_modality_counts": self.training.modality_mask.summary(),
        }


# =============================================================================
# Prediction Results Container
# =============================================================================


@dataclass
class PredictionResults:
    """Container for prediction results.

    Stores all outputs from the prediction pipeline including
    raw predictions, probabilities, scaled values, and verification status.

    Attributes
    ----------
        cells: DataFrame with cell info and predictions added
        predictions: Predicted class labels array
        probabilities: Class probability matrix (n_cells, n_classes)
        class_names: Names of classes in probability column order
        scaled_predictions: Predictions after probability scaling (optional)
        scaled_probabilities: Probabilities after CM-based scaling (optional)
        verified: Boolean mask of cells that passed verification (optional)

    Example:
        >>> results = PredictionResults(
        ...     cells=df,
        ...     predictions=y_pred,
        ...     probabilities=proba,
        ...     class_names=["MON", "cMI", "iMI", "SMI"],
        ... )
        >>> print(f"Predicted {results.n_cells} cells")
        >>> print(f"Distribution: {results.prediction_counts}")

        >>> # After verification
        >>> results.verified = verification_mask
        >>> print(f"Verified: {results.n_verified}")
    """

    cells: pd.DataFrame
    predictions: np.ndarray
    probabilities: np.ndarray
    class_names: list[str] = field(
        default_factory=lambda: [
            "motion_onset",
            "motion_integrator_contralateral",
            "motion_integrator_ipsilateral",
            "slow_motion_integrator",
        ]
    )
    scaled_predictions: np.ndarray | None = None
    scaled_probabilities: np.ndarray | None = None
    verified: np.ndarray | None = None

    @property
    def n_cells(self) -> int:
        """Number of predicted cells."""
        return len(self.predictions)

    @property
    def n_verified(self) -> int:
        """Number of cells that passed verification."""
        if self.verified is None:
            return 0
        return int(self.verified.sum())

    @property
    def verification_rate(self) -> float:
        """Fraction of cells that passed verification."""
        if self.verified is None or self.n_cells == 0:
            return 0.0
        return self.n_verified / self.n_cells

    @property
    def prediction_counts(self) -> dict[str, int]:
        """Count of cells per predicted class."""
        unique, counts = np.unique(self.predictions, return_counts=True)
        return dict(zip(unique, counts, strict=True))

    @property
    def scaled_prediction_counts(self) -> dict[str, int]:
        """Count of cells per scaled predicted class."""
        if self.scaled_predictions is None:
            return {}
        unique, counts = np.unique(self.scaled_predictions, return_counts=True)
        return dict(zip(unique, counts, strict=True))

    def get_verified_cells(self) -> pd.DataFrame:
        """Return DataFrame of cells that passed verification."""
        if self.verified is None:
            return pd.DataFrame()
        return self.cells[self.verified].reset_index(drop=True)

    def get_by_prediction(self, prediction: str, use_scaled: bool = True) -> pd.DataFrame:
        """Get cells with a specific prediction.

        Args:
            prediction: Class label to filter by
            use_scaled: If True and available, use scaled predictions

        Returns
        -------
            DataFrame of cells with the specified prediction
        """
        preds = (
            self.scaled_predictions
            if (use_scaled and self.scaled_predictions is not None)
            else self.predictions
        )
        mask = preds == prediction
        return self.cells[mask].reset_index(drop=True)

    def summary(self) -> dict:
        """Return summary statistics."""
        result = {
            "n_cells": self.n_cells,
            "prediction_counts": self.prediction_counts,
        }
        if self.scaled_predictions is not None:
            result["scaled_prediction_counts"] = self.scaled_prediction_counts
        if self.verified is not None:
            result["n_verified"] = self.n_verified
            result["verification_rate"] = f"{self.verification_rate:.1%}"
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        verified_str = f", verified={self.n_verified}" if self.verified is not None else ""
        return f"PredictionResults(n_cells={self.n_cells}{verified_str})"


# =============================================================================
# Loaded Data Container (Complete Pipeline Data)
# =============================================================================


@dataclass
class LoadedData:
    """Complete data loaded from HDF5 for classification pipeline.

    This container bundles all data needed for the full classification pipeline,
    providing a single return value from the load_data() method. It replaces the
    pattern of setting multiple instance attributes.

    Attributes
    ----------
        training_data: TrainingData container with train/predict split.
        feature_names: List of feature column names.
        cells_df: DataFrame of training cells with metadata.
        cells_with_to_predict_df: DataFrame including to_predict cells.

    Example:
        >>> predictor = ClassPredictor(path)
        >>> data = predictor.load_data("FINAL_features")
        >>> print(f"Training cells: {data.training_data.n_training}")
        >>> print(f"Features: {len(data.feature_names)}")

        >>> # Access training features
        >>> X_train = data.training_data.get_training_features()
        >>> y_train = data.training_data.get_training_labels()

    See Also
    --------
        - TrainingData: The nested container for train/predict split
        - ClassPredictor.load_data: Method that returns this container
    """

    training_data: TrainingData
    feature_names: list[str]
    cells_df: pd.DataFrame
    cells_with_to_predict_df: pd.DataFrame

    @property
    def n_training(self) -> int:
        """Number of training cells."""
        return self.training_data.n_training

    @property
    def n_to_predict(self) -> int:
        """Number of cells to predict."""
        return self.training_data.n_to_predict

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.feature_names)

    def summary(self) -> dict:
        """Return summary statistics for the loaded data."""
        return {
            "n_training": self.n_training,
            "n_to_predict": self.n_to_predict,
            "n_features": self.n_features,
            "training_class_counts": self.training_data.training.class_counts,
            "modality_counts": self.training_data.training.modality_mask.summary(),
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"LoadedData(n_training={self.n_training}, "
            f"n_to_predict={self.n_to_predict}, n_features={self.n_features})"
        )
