# Morphology-Based Prediction of Neuronal Functional Types

Code for predicting functional cell types from neuronal morphology in the zebrafish hindbrain, as described in [Boulanger-Weill, Kämpf et al. (2025)](https://www.biorxiv.org/content/10.1101/2025.03.14.643363v2). This module is part of the [Zebrafish_CLEM](https://github.com/jboulanger91/Zebrafish_CLEM) repository.

The pipeline classifies neurons into four functional types — motion integrator ipsilateral (iMI), motion integrator contralateral (cMI), motion onset (MON), and slow motion integrator (SMI) — using morphological features extracted from SWC skeleton tracings across three imaging modalities: photoactivation (PA), correlative light-electron microscopy (CLEM), and electron microscopy (EM).

## Overview

The classifier uses Linear Discriminant Analysis (LDA) with Recursive Feature Elimination (RFE) to select 13 optimal morphological features from a set of 68, achieving 83.6% leave-one-out cross-validation accuracy on CLEM neurons. Predictions are verified via NBLAST morphological similarity analysis and outlier detection.

The EM connectome can be explored interactively via [Neuroglancer](https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6294688695844864) (requires Gmail login). For proofreading details and programmatic access see: https://jboulanger91.github.io/fish1.5-release/

## Directory Structure

| Directory | Description |
|---|---|
| `src/myio/` | Data I/O: load PA/CLEM/EM cell tables, SWC morphologies, meshes |
| `src/morphology/` | Morphology processing: NBLAST similarity, SWC repair, branch finding, neurite fragmentation |
| `src/viz/` | Visualization: color palettes, classification plots, feature plots, figure helpers |
| `src/util/` | Utilities: data path resolution, output paths, cell path construction |
| `functional_type_prediction/` | LDA classifier pipeline: feature selection, cross-validation, prediction, verification |
| `scripts/` | Data path setup |
| `tests/` | Unit and integration tests (pytest) |
| `config/` | Conda environment and data path configuration |

## Setup

### 1. Install dependencies

```bash
# Conda (recommended)
conda env create -f config/environment.yml
conda activate hbsf

# Or pip
pip install -r requirements.txt
```

**Important:** The pipeline requires `scikit-learn==1.5.2`. Version 1.6+ produces different RFE results due to internal changes in feature ranking.

### 2. Configure data paths

```bash
python scripts/setup_data_paths.py
```

This creates `config/path_configuration.txt`, mapping your username to the local directory containing the structural data (available on [Zenodo](https://doi.org/10.5281/zenodo.19220629)). The path is resolved at runtime by `src.util.get_base_path.get_base_path()`.

### 3. Run the classifier pipeline

```bash
python functional_type_prediction/classifier_prediction/pipelines/pipeline_main.py
```

This executes the full pipeline:
1. Load cell metadata and SWC skeletons from all modalities
2. Extract 68 morphological features (cable length, Strahler order, Sholl analysis, branch angles, spatial extent, etc.)
3. Select 13 optimal features via RFE with AdaBoost
4. Train LDA classifier on 120 labeled neurons (47 PA + 73 CLEM)
5. Predict functional types for 337 unlabeled neurons (215 EM + 122 CLEM)
6. Verify predictions via NBLAST similarity and outlier detection

## Data

The structural data is hosted on Zenodo: [10.5281/zenodo.19220629](https://doi.org/10.5281/zenodo.19220629)

Contents:
- `metadata.xlsx` — Cell inventory with functional labels and training flags
- `paGFP/` — 47 photoactivation neurons (SWC skeletons, functional dynamics)
- `clem_zfish1/` — 301 CLEM neurons (registered SWC skeletons, OBJ meshes, synapse locations)
- `em_zfish1/` — 215 EM neurons (registered SWC skeletons, meshes)

All coordinates are registered to the Z-Brain atlas reference frame (Randlett et al., 2015). SWC node labels: 1=soma, 2=axon, 3=dendrite, 4=presynapse, 5=postsynapse.

## Output

All outputs are written to `~/Desktop/hbsf_output/classifier_pipeline/`, configurable via:

```bash
export HBSF_OUTPUT_ROOT=/custom/output/path
```

Output includes: RFE plots, confusion matrices, prediction spreadsheets with probability scores, and verification metrics.

## Testing

```bash
# Unit tests (no data required)
pytest tests/ -v

# Regression tests (requires data + baselines)
pytest functional_type_prediction/classifier_prediction/tests/ -v
```

## Functional Type Nomenclature

| Abbreviation | Full Name | Legacy Name |
|---|---|---|
| iMI | `motion_integrator_ipsilateral` | integrator |
| cMI | `motion_integrator_contralateral` | integrator |
| MON | `motion_onset` | dynamic_threshold |
| SMI | `slow_motion_integrator` | motor_command |

Legacy names in metadata files are automatically mapped to modern nomenclature during loading.

## Related Repositories

- [Zebrafish_CLEM](https://github.com/jboulanger91/Zebrafish_CLEM) — Parent repository: neuron downloading, registration, connectivity analysis, network modeling
- [fish1.5-release](https://jboulanger91.github.io/fish1.5-release/) — EM connectome documentation and CAVE API access

## Citation

If you use this code, please cite:

> Boulanger-Weill J\*, Kämpf F\*, et al. (2025). *A cellular-resolution structure-function map of the zebrafish hindbrain.* bioRxiv. [doi:10.1101/2025.03.14.643363](https://doi.org/10.1101/2025.03.14.643363)

## License

MIT License. See `LICENSE` for full text.
