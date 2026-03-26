# core

Core library modules for the cell type classification pipeline.

## Overview

This package contains reusable library code for classifying zebrafish hindbrain neurons into functional types from morphological features. It is structured around a main orchestrator class (`ClassPredictor`) that delegates to specialized modules for data loading, feature selection, cross-validation, prediction, verification, and metrics calculation.

Public classes are re-exported from `__init__.py`:

```python
from core import ClassPredictor, LoadedData, CVResult, RFEResult
```

### Abbreviations

| Abbreviation | Full name |
|---|---|
| iMI | `motion_integrator_ipsilateral` |
| cMI | `motion_integrator_contralateral` |
| MON | `motion_onset` |
| SMI | `slow_motion_integrator` |
| LDA | Linear Discriminant Analysis |
| NBLAST | Neuron BLAST -- morphological similarity scoring between neuron skeletons (Costa et al., 2016) |
| RFE | Recursive Feature Elimination -- iteratively removes the least important features |
| SelectKBest | Univariate statistical feature selection (scikit-learn) |

## Architecture

*Pipeline data flow: `ClassPredictor` orchestrates the full classification pipeline, delegating each stage to a specialized module.*

```
ClassPredictor (orchestrator)
├── DataLoader          → HDF5 I/O, xlsx metadata, feature preprocessing
├── RFESelector         → Recursive Feature Elimination with 11 estimators
├── FeatureSelector     → SelectKBest (f_classif / mutual_info_classif)
├── ModalityCrossValidator → CV with modality-aware train/test splits
├── PredictionPipeline  → LDA training, prediction, probability scaling
├── VerificationCalculator → NBLAST similarity, outlier detection, statistical tests
└── config / containers → Dataclasses for configuration and data flow
```

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Public API surface. Re-exports major classes, containers, and configuration objects. Provides `class_predictor` as an alias for `ClassPredictor`. |
| `config.py` | Configuration dataclasses and constants. Exported: `CellTypePriors`, `CrossValidationConfig`, `CELL_TYPE_COLORS`, `ACRONYM_DICT`. Internal (used by `PipelineConfig`): `VerificationConfig`, `FeatureConfig`. Also defines `PipelineConfig` (composed internally by `ClassPredictor`), `CELL_TYPE_COLORS_SOLID`, and valid modality definitions. |
| `containers.py` | Typed data containers flowing between pipeline stages. `ModalityMask` (boolean arrays for clem/pa/em), `CellDataset` (cells + features + labels with filtering), `TrainingData` (train/predict split), `PredictionResults` (predictions + probabilities + verification), `LoadedData` (complete pipeline data bundle). |
| `class_predictor.py` | Main orchestrator. Methods: `load_data()` -> `LoadedData`, `select_features_rfe()` -> `RFEResult`, `cross_validate()` -> `CVResult`, `predict()` -> `PredictionResults`, `verify()` -> `PredictionResults`, `load_cells_df()`, `calculate_metrics()`, `load_cells_features()`, `select_features_RFE()`, `do_cv()`, `confusion_matrices()`, `predict_cells()`, `calculate_verification_metrics()`. |
| `data_loader.py` | `DataLoader` for loading cell data from HDF5 features files and metadata xlsx. `load_cells_df()` builds the cell metadata DataFrame with modality filtering and label assignment. `load_metrics()` loads precomputed features. Constants: `NEUROTRANSMITTER_MAP`, `MORPHOLOGY_MAP`, `NEGATIVE_CONTROL_LABELS`. |
| `feature_selector.py` | Two complementary selection approaches. `FeatureSelector`: fast statistical selection via SelectKBest. `RFESelector`: RFE testing up to 11 default estimators (LDA, AdaBoost, RandomForest, etc.) with per-estimator RFE curve plots. Returns `RFEResult` with selected feature mask and scores. |
| `cross_validator.py` | Cross-validation with modality awareness. `CVResult` dataclass (score + confusion matrix). `CrossValidator` for basic LPO/ShuffleSplit. `ModalityCrossValidator` handles all train/test modality combinations (e.g., train on all, test on CLEM). Confusion matrix display uses paper ordering (iMI, cMI, MON, SMI). |
| `predictor.py` | `PredictionPipeline` for LDA-based cell type prediction. `PreparedData` dataclass for aligned train/predict splits. `train_and_predict()` fits LDA (lsqr solver, auto-shrinkage). `train_and_predict_loo()` for leave-one-out on training cells. `scale_by_confusion_matrix()` calibrates probabilities using true positive rates. |
| `metrics_calculator.py` | Morphological feature computation from neuron skeletons. `MorphologyMetrics`: cable length, bounding box, tortuosity, Strahler index, persistence, Sholl analysis (10--190 um). `HemisphericMetrics`: ipsilateral/contralateral split, IC index. `BranchMetrics`: first-branch-point properties. `MetricsBatchCalculator`: batch processing with progress bars. Constant: `BRAIN_WIDTH = 495.56` um. |
| `verification.py` | Post-prediction validation pipeline. `NBLASTCalculator`: morphological similarity using custom zebrafish scoring matrix. `OutlierDetector`: One-Class SVM, Isolation Forest, Local Outlier Factor. `StatisticalTester`: 13 verification tests (see below). `VerificationVisualizer`: PCA + t-SNE projections. `PredictionExporter`: Excel output. `VerificationCalculator`: orchestrates full pipeline. |
| `visualizer.py` | `ConfusionMatrixPlotter`: single and 3x3 grid confusion matrices with normalization. |
| `utilities.py` | Helper functions. `check_swc_validity()`: validates SWC node ordering. Requires navis (conditionally imported). |

### Verification tests

The `StatisticalTester` runs 13 named tests on each predicted cell:

| Test | Description |
|------|-------------|
| `NBLAST_g` | General threshold -- is the best NBLAST match above a cutoff? |
| `NBLAST_z`, `NBLAST_z_scaled` | Z-score -- is the predicted cell within the 95% CI of its assigned class? |
| `NBLAST_ak`, `NBLAST_ak_scaled` | Anderson-Ksamp distribution comparison |
| `NBLAST_ks`, `NBLAST_ks_scaled` | Kolmogorov-Smirnov distribution test |
| `CVM`, `CVM_scaled` | Cramer-von Mises ECDF comparison |
| `MWU`, `MWU_scaled` | Mann-Whitney U rank-based comparison |
| `probability_test`, `probability_test_scaled` | Any class probability > 0.7? |

`_scaled` variants use confusion-matrix-calibrated probabilities.

### Morphological features

`MetricsBatchCalculator` computes the following per neuron:

- **Basic morphology** (`MorphologyMetrics`): cable length, bounding box volume, spatial extents, tortuosity, graph topology (leafs, branches, ends, edges)
- **Strahler index**: branch hierarchy complexity
- **Persistence**: topological persistence points (birth/death deltas)
- **Sholl analysis**: branching intersections at radii 10--190 um from soma
- **Hemispheric** (`HemisphericMetrics`): ipsilateral/contralateral cable split, IC index
- **Branch** (`BranchMetrics`): first branch point distance, cable length to first branch

## Key Exports

All items below are importable from `core`:

**Orchestrator:** `ClassPredictor` (alias: `class_predictor`)

**Configuration:** `CellTypePriors`, `CrossValidationConfig`, `CELL_TYPE_COLORS`, `ACRONYM_DICT`

**Data Containers:** `LoadedData`, `TrainingData`, `PredictionResults`, `CellDataset`, `ModalityMask`

**Cross-Validation:** `CVResult`, `ModalityCrossValidator`

**Data Loading:** `DataLoader`, `get_encoding`

**Feature Selection:** `FeatureSelector`, `RFESelector`, `RFEResult`

**Prediction:** `PredictionPipeline`

**Verification:** `VerificationCalculator`

**Visualization:** `ConfusionMatrixPlotter`

**Metrics:** `MorphologyMetrics`, `HemisphericMetrics`, `BranchMetrics`, `MetricsBatchCalculator`

## Dependencies

- **`src/myio/`** -- HDF5 read/write, SWC file loading, cell metadata
- **`src/viz/`** -- Visualization submodules (`colors`, `classification_plots`, `feature_plots`, `figure_helper`)
- **`src/util/`** -- Path resolution (`get_output_dir`)
- **External:** scikit-learn, navis, pandas, numpy, matplotlib, seaborn, scipy, openpyxl, h5py, tqdm

## Output Location

Modules that produce output files write to subdirectories under `~/Desktop/morph2func_output/classifier_pipeline/` via `get_output_dir()`.
