# functional_type_prediction

Predicts functional cell types from morphological features in zebrafish hindbrain neurons.

## Overview

This top-level package contains the classifier prediction pipeline used to assign functional types (motion_onset, motion_integrator_contralateral, motion_integrator_ipsilateral, slow_motion_integrator) to reconstructed neurons. The pipeline operates on morphological features computed from neuron tracings (SWC files) across multiple imaging modalities (CLEM, photoactivation, EM), using Linear Discriminant Analysis (LDA) with Recursive Feature Elimination (RFE) for feature selection. Predictions are verified via NBLAST morphological similarity analysis and outlier detection.

## Directory Structure

```
functional_type_prediction/
  classifier_prediction/        Main classifier package
    core/                       Core library modules (predictor, data loading, config, etc.)
    pipelines/                  Production pipeline scripts
    utils/                      Utility functions for metric computation and data prep
    analysis/                   Research and evaluation scripts
    tests/                      Regression and integration tests
```

This directory does not contain an `__init__.py` file; individual modules are imported via direct path references or by adding the appropriate directories to `sys.path`.

## Pipeline Flow

The production pipeline (`classifier_prediction/pipelines/pipeline_main.py`) executes the following steps:

1. **Data Loading** -- Load cell metadata and morphological features from HDF5 files via `DataLoader`. Filter by imaging modality and apply neurotransmitter annotations.
2. **Preprocessing** -- Remove incomplete neurons (truncated, exiting, growth cone) and apply manual morphology annotations.
3. **Feature Selection** -- Use RFE with AdaBoost as the wrapper estimator to identify the optimal feature subset, evaluated by weighted F1 on CLEM leave-one-out cross-validation.
4. **Cross-Validation** -- Run Leave-P-Out or ShuffleSplit cross-validation with LDA (shrinkage='auto') to generate confusion matrices and accuracy metrics.
5. **Prediction** -- Train LDA on all labeled neurons, predict unlabeled (`to_predict`) cells. Optionally scale probabilities using anatomical priors from Jon Lindsey's EM cell counts (539 neurons) or confusion matrix calibration.
6. **Verification** -- Apply outlier detection (Isolation Forest, Local Outlier Factor, One-Class SVM) and statistical tests (NBLAST similarity, Anderson-Ksamp, KS, CVM) to flag unreliable predictions.

## Dependencies

- **`src/myio/`** -- Cell table loading (PA, CLEM, EM), mesh loading, CAVE pipeline, metadata writing
- **`src/morphology/`** -- Morphological metric functions (cable length, Strahler, Sholl analysis, branch angles)
- **`src/viz/`** -- Visualization submodules (`colors`, `classification_plots`, `feature_plots`, `figure_helper`)
- **`src/util/`** -- `get_base_path()` for data path resolution, `get_output_dir()` for output path resolution, notification utilities

## Output Location

All pipeline outputs are written to subdirectories under:

```
~/Desktop/morph2func_output/classifier_pipeline/
```

This path is resolved by `get_output_dir("classifier_pipeline", ...)` from `src/util/output_paths.py`.

## How to Run

```bash
# Run the production pipeline
python functional_type_prediction/classifier_prediction/pipelines/pipeline_main.py

# Run regression tests
cd functional_type_prediction/classifier_prediction/tests
python run_tests.py --all

# Capture new test baselines (before refactoring)
python run_tests.py --capture-baselines
```
