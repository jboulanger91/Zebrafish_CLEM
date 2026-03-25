"""Core modules for the cell type classification pipeline.

Public API
----------
ClassPredictor
    Main pipeline orchestrator for cell type classification.
    (Also available as class_predictor)
LoadedData : dataclass
    Container for loaded pipeline data.
CVResult : dataclass
    Cross-validation results container.
PredictionResults : dataclass
    Prediction output container.

Modules
-------
class_predictor : Main predictor class for cell type classification
data_loader : Data loading and preprocessing utilities
feature_selector : Feature selection algorithms (SelectKBest, RFE)
cross_validator : Cross-validation strategies (LPO, ShuffleSplit)
predictor : Model training and prediction classes
visualizer : Plotting and visualization utilities
verification : NBLAST-based prediction verification
metrics_calculator : Morphological metrics calculation

Example
-------
>>> from core import ClassPredictor
>>> from pathlib import Path
>>> predictor = ClassPredictor(Path("/data"))
>>> data = predictor.load_data("FINAL_features", modalities=["clem", "pa"])
>>> rfe = predictor.select_features_rfe(data, "all", "clem")
>>> predictions = predictor.predict(data, rfe.selected_features_idx)
>>> verified = predictor.verify(predictions, data)

Example (attribute-based)
-------------------------
>>> from core import class_predictor
>>> predictor = class_predictor(Path("/data"))
>>> predictor.load_cells_df(modalities=["clem", "pa"])
>>> predictor.load_cells_features("FINAL_features")
>>> predictor.select_features_RFE("all", "clem", save_features=True)
>>> predictor.predict_cells(use_jon_priors=True)
"""

from .class_predictor import ClassPredictor
from .config import (
    ACRONYM_DICT,
    CELL_TYPE_COLORS,
    CellTypePriors,
    CrossValidationConfig,
)
from .containers import (
    CellDataset,
    LoadedData,
    ModalityMask,
    PredictionResults,
    TrainingData,
)
from .cross_validator import (
    CVResult,
    ModalityCrossValidator,
)
from .data_loader import DataLoader, get_encoding
from .feature_selector import FeatureSelector, RFEResult, RFESelector
from .metrics_calculator import (
    BranchMetrics,
    HemisphericMetrics,
    MetricsBatchCalculator,
    MorphologyMetrics,
)
from .predictor import PredictionPipeline
from .verification import VerificationCalculator
from .visualizer import (
    ConfusionMatrixPlotter,
)

class_predictor = ClassPredictor

__all__ = [
    # Main orchestrator
    "ClassPredictor",
    "class_predictor",
    # Configuration
    "CellTypePriors",
    "CrossValidationConfig",
    "CELL_TYPE_COLORS",
    "ACRONYM_DICT",
    # Data containers
    "LoadedData",
    "TrainingData",
    "PredictionResults",
    "CellDataset",
    "ModalityMask",
    # Cross-validation
    "CVResult",
    "ModalityCrossValidator",
    # Data loading
    "DataLoader",
    "get_encoding",
    # Feature selection
    "FeatureSelector",
    "RFESelector",
    "RFEResult",
    # Prediction
    "PredictionPipeline",
    # Verification
    "VerificationCalculator",
    # Visualization
    "ConfusionMatrixPlotter",
    # Metrics
    "MorphologyMetrics",
    "HemisphericMetrics",
    "BranchMetrics",
    "MetricsBatchCalculator",
]
