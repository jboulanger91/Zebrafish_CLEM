# classifier_prediction

LDA-based cell type classification pipeline with RFE feature selection, cross-validation, and NBLAST verification.

## Overview

This package implements the complete cell type prediction pipeline for zebrafish hindbrain neurons. It classifies neurons into four functional types -- motion_onset (MON), motion_integrator_contralateral (cMI), motion_integrator_ipsilateral (iMI), and slow_motion_integrator (SMI) -- using morphological features extracted from neuron tracings. The pipeline supports three imaging modalities (CLEM, photoactivation, EM) and includes post-prediction verification through outlier detection and morphological similarity analysis.

The package is organized into five subpackages: `core/` contains the library modules, `pipelines/` holds production scripts, `utils/` provides feature computation utilities, `analysis/` contains research scripts, and `tests/` holds regression tests.

This directory does not contain an `__init__.py` file. Modules are imported by adding the `classifier_prediction/` directory to `sys.path` or by using relative imports from within the package.

## Subpackages

| Directory | Purpose |
|-----------|---------|
| `core/` | Core library: predictor orchestrator, data loading, feature selection, cross-validation, prediction, visualization, verification, metrics |
| `pipelines/` | Production pipeline scripts (`pipeline_main.py`) |
| `utils/` | Morphological metric computation and HDF5 data preparation |
| `analysis/` | Research scripts for model evaluation, feature importance, and prediction comparison |
| `tests/` | Regression tests, baseline capture scripts, and test runner |

## Key Classes and Functions

From `core/`:

- **`ClassPredictor`** -- Main orchestrator class. Methods: `load_data()`, `select_features_rfe()`, `cross_validate()`, `predict()`, `verify()`, `load_cells_df()`, `calculate_metrics()`, `select_features_RFE()`, `predict_cells()`, etc.
- **`DataLoader`** -- Loads cell metadata and features from HDF5 files, handles k-means annotations and neurotransmitter data.
- **`RFESelector`** / **`RFEResult`** -- Recursive Feature Elimination with configurable estimators and result container.
- **`ModalityCrossValidator`** / **`CVResult`** -- Cross-validation across modality combinations, returns score and confusion matrix.
- **`PredictionPipeline`** -- LDA training, prediction, confusion matrix probability scaling.
- **`VerificationCalculator`** -- Orchestrates NBLAST, outlier detection, and statistical testing.
- **`LoadedData`**, **`PredictionResults`**, **`TrainingData`** -- Immutable data containers for pipeline state.

From `pipelines/`:

- **`run_pipeline()`** -- Executes the full 6-step production pipeline.

From `utils/`:

- **`calculate_metric2df()`** -- Computes ~60 morphological features and writes to HDF5.

## Pipeline Flow

```
load_data() --> select_features_rfe() --> cross_validate() --> predict() --> verify()
     |                  |                       |                 |             |
  LoadedData        RFEResult              CVResult      PredictionResults  PredictionResults
                                                          (with predictions) (with verification)
```

## Dependencies

- **`src/myio/`** -- Cell table loading (PA, CLEM, EM), mesh loading, CAVE pipeline, metadata writing
- **`src/morphology/`** -- Metric computation functions (cable length, Strahler order, Sholl analysis, persistence, branch angles, hemisphere analysis)
- **`src/viz/`** -- Visualization submodules (`colors`, `classification_plots`, `feature_plots`, `figure_helper`)
- **`src/util/`** -- `get_base_path()` for data path, `get_output_dir()` for output directory, notifications

External dependencies: scikit-learn, navis, pandas, numpy, matplotlib, seaborn, scipy, openpyxl, h5py, tqdm.

## Output Location

All outputs are written under `~/Desktop/hbsf_output/classifier_pipeline/` via `get_output_dir("classifier_pipeline", ...)`. Subdirectories include `confusion_matrices/`, `RFE/`, `predictions/`, `verification/`, and `test_verification_metrics/`.

## How to Run

```bash
# Production pipeline
python pipelines/pipeline_main.py

# Run all tests
cd tests && python run_tests.py --all

# Run specific test classes
pytest tests/test_pipeline_regression.py -v -k "TestNewAPI"
```
