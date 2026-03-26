"""Data Loading and Preprocessing Module.

This module handles all data loading, preprocessing, and preparation tasks
for the cell type classifier. It follows the Single Responsibility Principle
by focusing solely on data operations.

Zen of Python principles applied:
    - Explicit is better than implicit
    - Simple is better than complex
    - Readability counts

Classes:
    DataLoader: Handles loading and preprocessing of cell data

Functions:
    get_encoding: Detect file encoding for reading metadata

Module Constants:
    DEFAULT_MODALITIES: Default imaging modalities to load
    NEUROTRANSMITTER_MAP: Mapping from neurotransmitter names to integer codes
    MORPHOLOGY_MAP: Mapping from morphology types to integer codes
    NEGATIVE_CONTROL_LABELS: Labels treated as negative controls

Author: Florian Kämpf
"""

import sys
from pathlib import Path

import chardet
import numpy as np
import pandas as pd


# Path setup for load_cells_predictor_pipeline import
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))  # Required for standalone execution

# Import the cell loading function (used by load_cells_df)
try:
    from utils.calculate_metric2df import load_cells_predictor_pipeline
except ImportError:
    # Fallback: function will raise error if called without proper setup
    load_cells_predictor_pipeline = None


# =============================================================================
# Module-Level Constants
# =============================================================================

#: Default imaging modalities supported by the pipeline
DEFAULT_MODALITIES: list[str] = ["pa", "clem", "em", "clem_predict"]

#: Mapping from neurotransmitter names to integer codes for ML features
NEUROTRANSMITTER_MAP: dict[str, int] = {"excitatory": 0, "inhibitory": 1, "na": 2, "nan": 2, "unknown": 2}

#: Mapping from morphology types to integer codes for ML features
MORPHOLOGY_MAP: dict[str, int] = {"contralateral": 0, "ipsilateral": 1}

#: Labels that should be treated as negative controls
NEGATIVE_CONTROL_LABELS: list[str] = [
    "no response",
    "off-response",
    "noisy, little modulation",
    "neg_control",
]

# Import the canonical label normalization map.
# FUNCTION_NAME_MAP is the single source of truth for old→new nomenclature.
try:
    from src.myio.load_cells2df import FUNCTION_NAME_MAP
except ImportError:
    from myio.load_cells2df import FUNCTION_NAME_MAP


def get_encoding(file_path: Path) -> str:
    """Detect the character encoding of a file.

    Parameters
    ----------
    file_path : Path
        Path to the file to analyze

    Returns
    -------
    str
        Detected encoding (e.g., 'utf-8', 'ascii')

    Examples
    --------
    >>> encoding = get_encoding(Path("metadata.txt"))
    >>> with open("metadata.txt", "r", encoding=encoding) as f:
    ...     content = f.read()
    """
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result["encoding"]


class DataLoader:
    """Handles loading and preprocessing of neuronal cell data.

    This class centralizes all data loading operations, making the codebase
    more maintainable and testable. It provides methods for loading cell data
    from various sources (HDF5 files, metadata files) and preparing them for
    machine learning pipelines.

    Attributes
    ----------
    path : Path
        Base path to the data directory (typically the nextcloud root)
    modalities : List[str]
        List of imaging modalities to load

    Methods
    -------
    prepare_data_for_metrics(df)
        Check for duplicate cell names
    load_cells_df(modalities)
        Load cell dataframe from multiple modalities
    load_metrics(file_name, drop_neurotransmitter=False)
        Load morphological metrics from HDF5 and return standardized features

    Examples
    --------
    >>> from pathlib import Path
    >>> loader = DataLoader(Path("/data/nextcloud"))
    >>>
    >>> # Load metrics for ML pipeline
    >>> (features, labels, modalities), cols, df = loader.load_metrics(
    ...     "FINAL_CLEM_EM"
    ... )
    """

    def __init__(self, path: Path):
        """Initialize the DataLoader.

        Parameters
        ----------
        path : Path
            Base path to the data directory
        """
        self.path = path
        self.modalities = None

    def _resolve_features_path(self, file_name: str) -> Path:
        """Find the HDF5 features file.

        Checks output directory first, then baselines/ under data root.

        Args:
            file_name: Base name of the features file (without ``_features.hdf5``).

        Returns
        -------
            Path to the HDF5 features file.

        Raises
        ------
        FileNotFoundError
            If the features file is not found.
        """
        hdf5_name = f"{file_name}_features.hdf5"
        # Check output directory first
        try:
            from src.util.output_paths import get_output_dir

            output_path = get_output_dir(
                "classifier_pipeline", "features"
            ) / hdf5_name
            if output_path.exists():
                return output_path
        except (ImportError, OSError):
            pass
        # Check baselines/ under data root
        baselines_path = self.path / "baselines" / hdf5_name
        if baselines_path.exists():
            return baselines_path
        raise FileNotFoundError(
            f"Features file not found: {hdf5_name}. "
            f"Expected in classifier_pipeline/features/ or baselines/."
        )

    def prepare_data_for_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for duplicate cell names and reset index.

        Kmeans labels, neurotransmitter, and PA regressor fields are now
        loaded during initial cell loading in ``load_cells_predictor_pipeline()``.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with cell data

        Returns
        -------
        pd.DataFrame
            Deduplicated dataframe
        """
        print("\n Preparing data for metric calculation...")
        print(f"   Input dataframe shape: {df.shape}")

        # Safety check — should not find duplicates with xlsx loading
        df = df.drop_duplicates(keep="first", inplace=False, subset="cell_name")
        df = df.reset_index(drop=True)
        print(f"   After removing duplicates: {df.shape}")

        print(" Data preparation complete\n")
        return df

    # =========================================================================
    # HDF5 Operations
    # =========================================================================

    def load_metrics(
        self, file_name: str, drop_neurotransmitter: bool = False
    ) -> tuple[list[np.ndarray], list[str], pd.DataFrame]:
        """Load morphological metrics from HDF5 feature file.

        Reads cell features from HDF5, applies preprocessing (sorting, NaN imputation,
        label standardization), and returns standardized feature arrays ready for ML.

        Args:
            file_name: Name of the HDF5 features file (without path and _features.hdf5
                suffix). Example: 'FINAL_CLEM_CLEMPREDICT_EM_with_clem241211'
            drop_neurotransmitter: If True, drop the neurotransmitter column from
                the output DataFrame. Default False.

        Returns
        -------
            Tuple containing:
                - features_list: [features, labels, labels_imaging_modality] where:
                    - features: np.ndarray of shape (n_cells, n_features), standardized
                    - labels: np.ndarray of function labels
                    - labels_imaging_modality: np.ndarray of imaging modality labels
                - column_labels: List of feature column names
                - all_cells: Preprocessed DataFrame with all cell data

        Raises
        ------
            FileNotFoundError: If HDF5 file doesn't exist at expected path
            KeyError: If 'complete_df' dataset not found in HDF5

        Example:
            >>> loader = DataLoader(Path("/data/nextcloud"))
            >>> (features, labels, modalities), cols, df = loader.load_metrics(
            ...     "FINAL_CLEM_EM"
            ... )
            >>> print(f"Loaded {features.shape[0]} cells with {features.shape[1]} features")
            Loaded 513 cells with 63 features

        Notes
        -----
            Preprocessing steps applied:
            1. Remove cells where 'function' is NaN
            2. Sort by function, morphology, imaging_modality, neurotransmitter
            3. Remap certain labels to 'neg_control' and add is_neg_control column
            4. Impute NaNs in angle/cross columns with 0
            5. Update 'motion_integrator' labels to include morphology suffix
            6. Convert neurotransmitter/morphology strings to integer codes
            7. Return raw features (standardization deferred to caller)
        """
        print(f"\n Loading metrics from: {file_name}")
        file_path = self._resolve_features_path(file_name)
        print(f"   File path: {file_path}")

        all_cells = pd.read_hdf(file_path, "complete_df")
        print(f"   Loaded {len(all_cells)} cells")

        # Data Preprocessing
        all_cells = all_cells.sort_values(
            by=["function", "morphology", "imaging_modality", "neurotransmitter"]
        )
        all_cells = all_cells.reset_index(drop=True)

        # Impute NaNs in angle/cross columns
        columns_possible_nans = ["angle", "angle2d", "x_cross", "y_cross", "z_cross"]
        all_cells.loc[:, columns_possible_nans] = all_cells[columns_possible_nans].fillna(0)

        # Normalize function labels (HDF5 may contain old names from calculate_metrics)
        # DO NOT REMOVE — calculate_metrics writes legacy names to HDF5, this maps them back
        all_cells.loc[:, "function"] = all_cells["function"].str.replace(" ", "_")
        all_cells.loc[:, "function"] = all_cells["function"].replace(FUNCTION_NAME_MAP)

        # Append morphology suffix to motion_integrator
        # (e.g. → motion_integrator_contralateral).
        # Only motion_integrator has directional subtypes.
        def update_integrator(df: pd.DataFrame) -> None:
            integrator_mask = df["function"] == "motion_integrator"
            df.loc[integrator_mask, "function"] += "_" + df.loc[integrator_mask, "morphology"]

        update_integrator(all_cells)

        # Convert categorical columns to integer codes for the classifier.
        # Original string values are preserved in *_clone columns for reporting.
        columns_replace_string = ["neurotransmitter", "morphology"]
        all_cells.loc[:, "neurotransmitter"] = all_cells["neurotransmitter"].fillna("nan")

        for work_column in columns_replace_string:
            all_cells.loc[:, work_column + "_clone"] = all_cells[work_column]  # preserve original
            mapping = NEUROTRANSMITTER_MAP if work_column == "neurotransmitter" else MORPHOLOGY_MAP
            for key, value in mapping.items():
                all_cells.loc[all_cells[work_column] == key, work_column] = value

        if drop_neurotransmitter:
            all_cells = all_cells.drop(columns="neurotransmitter")

        # Extract labels and features.
        # Columns 0-2 are metadata (cell_name, function, imaging_modality).
        # The last 2 columns are the *_clone copies. Everything in between is features.
        labels = all_cells["function"].to_numpy()
        labels_imaging_modality = all_cells["imaging_modality"].to_numpy()
        column_labels = list(all_cells.columns[3 : -len(columns_replace_string)])

        features = all_cells.iloc[:, 3 : -len(columns_replace_string)].to_numpy()

        print(f"   Features shape: {features.shape}")
        print(f"   Labels: {np.unique(labels)}")
        print(" Metrics loaded successfully\n")

        return [features, labels, labels_imaging_modality], column_labels, all_cells

    # =========================================================================
    # Cell Loading Operations
    # =========================================================================

    def load_cells_df(
        self,
        modalities: list[str] = None,
        label_column: str = "kmeans_function",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load cells dataframe and apply preprocessing pipeline.

        This method loads cell data from the predictor pipeline (which now
        handles kmeans labels, neurotransmitter, and PA regressor fields),
        filters out axon cells, resamples neurons to 1 micron, and creates
        standardized class labels.

        Args:
            modalities: List of imaging modalities to load. Default is
                ['pa', 'clem'] if None.

        Returns
        -------
            Tuple of (cells_with_to_predict, cells):
                - cells_with_to_predict: Full DataFrame including to_predict
                  and neg_control cells
                - cells: Training DataFrame excluding to_predict and neg_control

        Raises
        ------
            ImportError: If load_cells_predictor_pipeline is not available

        Example:
            >>> loader = DataLoader(Path("/data/nextcloud"))
            >>> all_cells, train_cells = loader.load_cells_df(
            ...     modalities=["pa", "clem"]
            ... )
            >>> print(f"Loaded {len(all_cells)} total, {len(train_cells)} training")

        Notes
        -----
            Preprocessing steps:
            1. Load cells using load_cells_predictor_pipeline
            2. Check for duplicates
            3. Remove cells with 'axon' in name
            4. Resample neurons to 1 micron resolution
            5. Create standardized class labels (function + morphology)
            6. Split into all cells vs training-only cells
        """
        if load_cells_predictor_pipeline is None:
            raise ImportError(
                "load_cells_predictor_pipeline not available. "
                "Ensure utils/calculate_metric2df.py is in the Python path."
            )

        if modalities is None:
            modalities = ["pa", "clem"]

        print("\n Loading cells dataframe...")
        print(f"   Modalities: {modalities}")

        self.modalities = modalities

        print("   Loading cells from predictor pipeline...")
        cells_with_to_predict = load_cells_predictor_pipeline(
            path_to_data=Path(self.path),
            modalities=modalities,
            label_column=label_column,
        )
        print(f"   Loaded {len(cells_with_to_predict)} cells")

        # Check for duplicates
        cells_with_to_predict = self.prepare_data_for_metrics(cells_with_to_predict)

        # Remove cells with 'axon' in name
        print("   Removing cells with 'axon' in name...")
        cells_with_to_predict = cells_with_to_predict.loc[
            cells_with_to_predict.cell_name.apply(lambda x: "axon" not in x), :
        ]
        print(f"   After filtering: {len(cells_with_to_predict)} cells")

        # Resample neurons to 1 micron
        print("   Resampling neurons to 1 micron...")
        cells_with_to_predict["swc"] = cells_with_to_predict["swc"].apply(
            lambda x: x.resample("1 micron")
        )

        # Create class labels (MI splits into ipsi/contra subtypes)
        print("   Creating class labels...")
        is_mi = cells_with_to_predict["function"] == "motion_integrator"
        cells_with_to_predict["class"] = cells_with_to_predict["function"]
        cells_with_to_predict.loc[is_mi, "class"] = (
            "motion_integrator_" + cells_with_to_predict.loc[is_mi, "morphology"]
        )

        # Split into training cells (only used_for_training=True from xlsx)
        cells = cells_with_to_predict[cells_with_to_predict["used_for_training"].fillna(False)]
        print(f"   Training cells (excluding to_predict): {len(cells)}")
        print(" Cells dataframe loaded\n")

        return cells_with_to_predict, cells

