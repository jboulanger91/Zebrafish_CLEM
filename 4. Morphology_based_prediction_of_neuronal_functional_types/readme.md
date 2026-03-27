# Morphology-Based Prediction of Neuronal Functional Types

Code for predicting functional cell types from neuronal morphology in the zebrafish hindbrain, as described in [Boulanger-Weill et al. (2025)](https://www.biorxiv.org/content/10.1101/2025.03.14.643363v2). This module is part of the [Zebrafish_CLEM](https://github.com/jboulanger91/Zebrafish_CLEM) repository.

The pipeline classifies neurons into four functional types: motion integrator ipsilateral (iMI), motion integrator contralateral (cMI), motion onset (MON), and slow motion integrator (SMI), using morphological features extracted from SWC skeleton tracings across three imaging modalities: photoactivation (PA), correlative light-electron microscopy (CLEM), and electron microscopy (EM).

## Quick Start

```bash
git clone https://github.com/jboulanger91/Zebrafish_CLEM.git
cd "Zebrafish_CLEM/4. Morphology_based_prediction_of_neuronal_functional_types"

python cli.py env --create        # 1. Create conda environment (one-time)
python cli.py setup --download    # 2. Download data from Zenodo (one-time, ~116 MB)
python cli.py run                 # 3. Train classifier and predict cell types
```

Running `python cli.py run` with no arguments reproduces the exact results reported in the paper. All parameters can be customized via CLI flags (see `python cli.py run --help`).

**Note:** The pipeline pins `scikit-learn==1.5.2`; version 1.6+ produces different RFE results. If conda is not available, `env --create` falls back to a Python venv.

## Overview

The classifier uses Linear Discriminant Analysis (LDA) with Recursive Feature Elimination (RFE) to select 13 optimal morphological features from a set of 68, achieving 82.1% leave-one-out cross-validation F1 score on CLEM neurons. Predictions are verified via NBLAST morphological similarity analysis and outlier detection.

## Pipeline

The `run` command executes these steps:

1. Load cell metadata and SWC skeletons from all modalities
2. Extract 68 morphological features (cable length, Strahler order, Sholl analysis, branch angles, spatial extent, etc.)
3. Select 13 optimal features via RFE with AdaBoost
4. Train LDA classifier on 120 labeled neurons (47 PA + 73 CLEM)
5. Predict functional types for 337 unlabeled neurons (215 EM + 122 CLEM)
6. Verify predictions via NBLAST similarity and outlier detection

## CLI Reference

All operations are available through `cli.py`:

```bash
python cli.py --help              # Show all commands
python cli.py <command> --help    # Show options for a specific command
```

| Command | Description |
|---|---|
| `python cli.py setup --download` | Download structural data from Zenodo (~116 MB) and configure paths |
| `python cli.py setup --verify` | Verify data paths are configured correctly |
| `python cli.py env --create` | Create conda environment with all dependencies |
| `python cli.py env --verify` | Check environment exists and has correct scikit-learn version |
| `python cli.py run` | Run the full classifier pipeline (feature selection, cross-validation, prediction, verification) |
| `python cli.py run --force-recalculation` | Recalculate morphological features from SWC files |
| `python cli.py run --features-file NAME` | Use a specific HDF5 features file |
| `python cli.py run --modalities pa clem` | Load only specific modalities |
| `python cli.py analysis published-metrics` | Reproduce published confusion matrices using persistence vectors ([Li et al., 2017](https://doi.org/10.1371/journal.pcbi.1005653)) and form factors ([Choi, Kim and Hyeon, 2023](https://doi.org/10.1016/j.celrep.2023.112746)) |
| `python cli.py analysis feature-importance` | Compute permutation importance for selected features |
| `python cli.py analysis feature-selector` | Find optimal feature selector via RFE with multiple estimators |
| `python cli.py analysis proba-cutoff` | Optimize probability cutoff (accuracy vs. coverage) |
| `python cli.py test --unit` | Run unit tests (no data required) |
| `python cli.py test --regression` | Run regression tests (requires data + baselines) |
| `python cli.py all` | Run everything: env check, data setup, pipeline, all analyses |

## Data

The structural data is hosted on Zenodo: [10.5281/zenodo.19235597](https://doi.org/10.5281/zenodo.19235597)

`python cli.py setup --download` downloads and extracts the data automatically. Default locations depend on your platform:

| Platform | Data path | Output path |
|---|---|---|
| macOS | `~/Desktop/morph2func/morph2func_input/` | `~/Desktop/morph2func/morph2func_output/classifier_pipeline/` |
| Linux | `~/morph2func/morph2func_input/` | `~/morph2func/morph2func_output/classifier_pipeline/` |
| Windows | `%USERPROFILE%\morph2func\morph2func_input\` | `%USERPROFILE%\morph2func\morph2func_output\classifier_pipeline\` |

Override with `MORPH2FUNC_ROOT` (both data and output) or `MORPH2FUNC_OUTPUT_ROOT` (output only).

Contents:
- `metadata.xlsx` - Cell inventory with functional labels and training flags (3 sheets: PA, CLEM, EM)
- `paGFP/` - 47 photoactivation neurons (registered SWC skeletons, functional dynamics HDF5)
- `clem_zfish1/` - 301 CLEM neurons (original and registered SWC skeletons, metadata)
- `em_zfish1/` - 215 EM neurons (original and registered SWC skeletons, metadata)
- `baselines/` - Reference predictions and pre-computed HDF5 features
- `custom_nblast_matrix.csv` - Zebrafish-trained NBLAST scoring matrix

CLEM and EM cells each have two SWC files: the original (`*.swc`, in native EM coordinates) and a registered version (`*_mapped.swc`, in [Z-Brain atlas](https://zebrafishexplorer.zib.de/home/) reference frame, [Randlett et al., 2015](https://www.nature.com/articles/nmeth.3581)). PA cells have a single SWC already in Z-Brain coordinates. The pipeline uses the registered skeletons. Units are microns. SWC node labels: 1=soma, 2=axon, 3=dendrite, 4=presynapse, 5=postsynapse.

Output includes: RFE plots, confusion matrices, prediction spreadsheets with probability scores, and verification metrics.

## Functional Type Nomenclature

| Abbreviation | Full Name | Legacy Name |
|---|---|---|
| iMI | `motion_integrator_ipsilateral` | integrator |
| cMI | `motion_integrator_contralateral` | integrator |
| MON | `motion_onset` | dynamic_threshold |
| SMI | `slow_motion_integrator` | motor_command |

Legacy names in metadata files are automatically mapped to modern nomenclature during loading.

## Directory Structure

| Directory | Description |
|---|---|
| `cli.py` | Unified command-line interface (run `python cli.py --help`) |
| `src/myio/` | Data I/O: load PA/CLEM/EM cell tables, SWC morphologies |
| `src/morphology/` | Morphology processing: NBLAST similarity, SWC repair, branch finding |
| `src/viz/` | Visualization: color palettes, classification plots, figure helpers |
| `src/util/` | Utilities: data path resolution, output paths, cell path construction |
| `functional_type_prediction/` | LDA classifier pipeline: feature selection, cross-validation, prediction, verification |
| `scripts/` | Data path setup and Zenodo download |
| `tests/` | Unit and integration tests (pytest) |
| `config/` | Environment specification and data path configuration |

## Related Repositories

- [Zebrafish_CLEM](https://github.com/jboulanger91/Zebrafish_CLEM) - Parent repository: neuron downloading, registration, connectivity analysis, network modeling
- [fish1.5-release](https://jboulanger91.github.io/fish1.5-release/) - EM connectome documentation and CAVE API access
- [Neuroglancer](https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6294688695844864) - Interactive EM connectome explorer (requires Gmail login)

## License

MIT License. See `LICENSE` for full text.
