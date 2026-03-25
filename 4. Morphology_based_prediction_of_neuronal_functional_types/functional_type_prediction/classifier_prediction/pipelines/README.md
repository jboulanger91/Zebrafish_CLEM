# pipelines

Production pipeline scripts for running the cell type classification workflow.

## Overview

This package contains production-ready scripts that wire together the `core/` library modules into executable pipelines. The main script (`pipeline_main.py`) runs the full classification pipeline, where each step returns a typed dataclass container that flows into the next step.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package docstring describing `pipeline_main` and its usage. Exports `pipeline_main`. |
| `pipeline_main.py` | Main production pipeline script. Defines `PipelineConfig` (local configuration class with all pipeline settings), `run_pipeline()` (executes the 6-step pipeline), `print_data_summary()`, `print_prediction_summary()`, and `compare_to_reference()` (validates predictions against saved Excel reference files). Entry point: `if __name__ == "__main__"`. |
| `extract_training_cell_names.py` | Utility script that replicates the data loading steps of `pipeline_main` and saves the training cell names to a `.npy` file for use by other analysis scripts. |

## Key Functions

- **`run_pipeline()`** -- Executes the full prediction pipeline in six steps:
  1. Initialize `ClassPredictor` with data path
  2. `load_data()` -- returns `LoadedData` container
  3. `select_features_rfe()` -- returns `RFEResult` container
  4. `cross_validate()` -- returns `CVResult` container
  5. `predict()` -- returns `PredictionResults` container
  6. `verify()` -- returns `PredictionResults` with verification status

- **`PipelineConfig`** -- Local dataclass holding all configuration parameters: features file name, modalities list, classifier settings (LDA with shrinkage='auto'), RFE estimator (AdaBoost, random_state=0), cross-validation method ('lpo'), metric ('f1'), and verification required tests (['IF', 'LOF']).

- **`compare_to_reference()`** -- Loads saved CLEM and EM Excel reference files and compares current predictions cell-by-cell to detect regressions.

## Pipeline Configuration

The default configuration in `PipelineConfig`:

- **Features file:** `FINAL_CLEM_CLEMPREDICT_EM_with_clem241211_withgregor250220`
- **Modalities:** `['pa', 'clem241211', 'em', 'clem_predict241211']`
- **Classifier:** `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`
- **RFE estimator:** `AdaBoostClassifier(random_state=0)`
- **CV method:** Leave-P-Out (`'lpo'`)
- **Metric:** Weighted F1 (`'f1'`)
- **Verification tests:** Isolation Forest + Local Outlier Factor (`['IF', 'LOF']`)

## Dependencies

- **`core/`** -- `ClassPredictor`, `LoadedData`, `RFEResult`, `CVResult`, `PredictionResults`
- **`src/util/`** -- `get_base_path()` for data path resolution, `get_output_dir()` for output paths
- External: scikit-learn, pandas, numpy, matplotlib

## Output Location

Pipeline outputs (prediction Excel files, confusion matrix plots, RFE curves, verification results) are written to `~/Desktop/hbsf_output/classifier_pipeline/` via `get_output_dir("classifier_pipeline", ...)`.

## How to Run

```bash
# Run the production pipeline
python pipelines/pipeline_main.py

# Extract training cell names only
python pipelines/extract_training_cell_names.py
```
