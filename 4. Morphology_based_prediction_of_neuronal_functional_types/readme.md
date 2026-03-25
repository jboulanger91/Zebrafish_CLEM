## Overview

This module predicts functional cell types from neuronal morphology using supervised classification. It takes SWC skeleton tracings from three imaging modalities (photoactivation, CLEM, EM) and classifies neurons into four functional types:

- **iMI** — motion integrator ipsilateral
- **cMI** — motion integrator contralateral
- **MON** — motion onset
- **SMI** — slow motion integrator

The classifier uses Linear Discriminant Analysis (LDA) with Recursive Feature Elimination (RFE), selecting 13 optimal features from 68 morphological measurements. Leave-one-out cross-validation on CLEM neurons achieves 83.6% accuracy.

---

## Environment Setup

```bash
conda env create -f config/environment.yml
conda activate hbsf
```

**Important:** Requires `scikit-learn==1.5.2`. Version 1.6+ produces different RFE results due to internal changes in feature ranking.

---

## Data Setup

### 1. Download structural data

Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.19220629) and extract it locally.

### 2. Configure data path

```bash
python scripts/setup_data_paths.py
```

This maps your username to the local directory containing the structural data in `config/path_configuration.txt`.

---

## Running the Pipeline

```bash
python functional_type_prediction/classifier_prediction/pipelines/pipeline_main.py
```

The pipeline executes six steps:

1. **Data loading** — Load cell metadata from `metadata.xlsx` and SWC skeletons from all modalities
2. **Feature extraction** — Compute 68 morphological features (cable length, Strahler order, Sholl analysis, branch angles, spatial extent, etc.)
3. **Feature selection** — RFE with AdaBoost selects 13 optimal features, evaluated by weighted F1
4. **Cross-validation** — Leave-one-out CV with LDA (shrinkage='auto') on CLEM neurons
5. **Prediction** — Train on 120 labeled neurons (47 PA + 73 CLEM), predict 337 unlabeled neurons (215 EM + 122 CLEM)
6. **Verification** — NBLAST morphological similarity and outlier detection (Isolation Forest, Local Outlier Factor) flag unreliable predictions

---

## Input Data

The structural data is hosted on [Zenodo](https://doi.org/10.5281/zenodo.19220629). Required files:

| File/Directory | Description |
|---|---|
| `metadata.xlsx` | Cell inventory with functional labels (`kmeans_function`), training flags, morphology and neurotransmitter annotations |
| `paGFP/<cell_name>/` | 47 photoactivation neurons: SWC skeletons, functional dynamics (dF/F traces), metadata |
| `clem_zfish1/.../<cell_dir>/mapped/` | 301 CLEM neurons: registered SWC skeletons, OBJ meshes (soma, axon, dendrite), synapse CSVs |
| `em_zfish1/.../<cell_dir>/mapped/` | 215 EM neurons: registered SWC skeletons and meshes |
| `custom_nblast_matrix.csv` | Custom NBLAST scoring matrix for morphological similarity verification |

All coordinates are registered to the Z-Brain atlas (Randlett et al., 2015). SWC node labels: 1=soma, 2=axon, 3=dendrite, 4=presynapse, 5=postsynapse.

---

## Output Data

All outputs are written to `~/Desktop/hbsf_output/classifier_pipeline/` (configurable via `HBSF_OUTPUT_ROOT` environment variable).

| Output | Description |
|---|---|
| `predictions/predictions.xlsx` | Functional type predictions with probability scores (MON_proba, cMI_proba, iMI_proba, SMI_proba) and verification status |
| `rfe/` | RFE feature selection plots (F1 vs. number of features) |
| `confusion_matrices/` | 3x3 confusion matrix grids (train/test modality combinations) |
| `features/` | Pre-computed morphological feature matrices (HDF5) |

---

## Directory Structure

| Directory | Description |
|---|---|
| `functional_type_prediction/` | LDA classifier pipeline: orchestrator, data loading, feature selection, cross-validation, prediction, verification |
| `src/myio/` | Data I/O: load PA/CLEM/EM cell tables, SWC skeletons, meshes |
| `src/morphology/` | Morphology processing: NBLAST similarity, SWC repair, branch finding, neurite fragmentation |
| `src/viz/` | Visualization: color palettes, classification plots, feature plots, figure helpers |
| `src/util/` | Utilities: data path resolution, output paths, cell path construction |
| `tests/` | Unit tests (no data required) |
| `config/` | Conda environment and data path configuration |

---

## Testing

```bash
# Unit tests (no data required)
pytest tests/ -v

# Regression tests (requires data + computed baselines)
pytest functional_type_prediction/classifier_prediction/tests/ -v
```

---

## Functional Type Nomenclature

| Abbreviation | Full Name | Legacy Name |
|---|---|---|
| iMI | `motion_integrator_ipsilateral` | integrator |
| cMI | `motion_integrator_contralateral` | integrator |
| MON | `motion_onset` | dynamic_threshold |
| SMI | `slow_motion_integrator` | motor_command |

Legacy names in metadata files are automatically mapped to modern nomenclature during data loading.
