# Zebrafish Hindbrain Functional Connectomics

A computational framework for analyzing the structure-function relationships of neurons in the zebrafish hindbrain. This codebase integrates correlative light-electron microscopy (CLEM), photoactivated GFP (PA) functional imaging, and electron microscopy (EM) datasets to predict functional cell types from neuronal morphology. It includes tools for neuron morphology loading and processing, NBLAST-based morphological similarity analysis, supervised classifier pipelines (LDA, AdaBoost, Random Forest) for predicting function from structure, and 2D/3D visualization of neurons across imaging modalities.

The dataset can be visualized with [Neuroglancer](https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6294688695844864) (requires Gmail login). For proofreading details and programmatic access see: https://jboulanger91.github.io/fish1.5-release/

## Directory Structure

| Directory | Description |
|---|---|
| `src/` | Core library modules: I/O loaders, morphology processing, visualization, and utility functions |
| `functional_type_prediction/` | Supervised classifier pipeline for predicting functional types (MI, MON, SMI) from morphological features |
| `contrastive_learning/` | SegCLR-SupCon deep learning pipeline for contrastive representation learning |
| `segclr_src/` | SegCLR core library: model definitions, training loops, data handling |
| `scripts/` | Standalone scripts: data path setup |
| `tests/` | Unit and integration test suite (pytest) |
| `config/` | Configuration files: data paths, conda environment |
| `PA/` | Legacy photoactivation registration docs |

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd hbsf_new

# Option A: Conda environment (recommended)
conda env create -f config/environment.yml
conda activate hbsf

# Option B: pip
pip install -r requirements.txt
```

### 2. Configure data paths

```bash
python scripts/setup_data_paths.py       # interactive terminal
python scripts/setup_data_paths.py --gui # GUI with folder browser
python scripts/setup_data_paths.py --verify
```

The data path is resolved at runtime by `src.util.get_base_path.get_base_path()`, which reads `config/path_configuration.txt` and matches against the current OS username.

### 3. Run the classifier pipeline

```bash
python -m functional_type_prediction.classifier_prediction.pipelines.pipeline_main
```

This runs the supervised functional type prediction pipeline, which extracts morphological features, trains classifiers, and predicts functional types (MI, MON, SMI) for unlabeled neurons.

## Data Availability

The dataset (~2 GB) includes CLEM, photoactivation, and EM connectome data for the zebrafish hindbrain.

The EM connectome can be explored interactively via [Neuroglancer](https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6294688695844864). For programmatic access, see the [CAVE API documentation](https://jboulanger91.github.io/fish1.5-release/).

## Code Availability

| Component | Details |
|---|---|
| **Language** | Python >= 3.12 |
| **Key dependency** | scikit-learn == 1.5.2 (pinned for reproducibility) |
| **Installation** | `conda env create -f config/environment.yml && conda activate hbsf` |
| **Tests** | `pytest tests/` (no data files required) |
| **Documentation** | Google-style docstrings, per-directory READMEs |
| **License** | MIT |

## Output Configuration

All scripts write output to a centralized output directory:

```
~/Desktop/hbsf_output/
```

This location is configurable via the `HBSF_OUTPUT_ROOT` environment variable:

```bash
export HBSF_OUTPUT_ROOT=/custom/output/path
```

On non-macOS systems (or if `~/Desktop` does not exist), the default falls back to `~/hbsf_output/`.

The output path system is implemented in `src/util/output_paths.py` via the `get_output_dir()` function.

## Data Paths

Data paths are configured via `config/path_configuration.txt`, a plain-text file mapping usernames to local filesystem paths where the `CLEM_paper_data` directory resides. The function `src.util.get_base_path.get_base_path()` reads this file and returns the appropriate `Path` for the current user.

Format:

```
username /absolute/path/to/CLEM_paper_data
```

Multiple users can be configured in the same file. If the current user is not found, a `NotSetup` exception is raised with instructions.

## Testing

Run the full test suite with pytest:

```bash
pytest tests/
```

The test suite includes unit tests organized across multiple test modules:

- `test_myio.py` -- I/O loader module: import smoke tests, function signatures, metadata read/write
- `test_cell_paths.py` -- Cell path resolution: path construction, directory existence, modality handling
- `test_color_unification.py` -- Color dictionaries: canonical source validation, GUI consistency, hex/RGBA conversions

No data files are required to run tests. Tests use mocks, synthetic data, and temporary files.

The `functional_type_prediction/` pipeline also has its own regression test suite under `functional_type_prediction/classifier_prediction/tests/`.

## Troubleshooting

**scikit-learn version mismatch**: The classifier pipeline requires `scikit-learn==1.5.2`. Version 1.6+ produces different Recursive Feature Elimination results due to internal changes in feature ranking. Create a fresh environment from `config/environment.yml` if you encounter version issues.

**Data path not configured**: If you see a `NotSetup` exception, run `python scripts/setup_data_paths.py` to configure the data path interactively.

**CAVE API errors**: The connectivity analysis can regenerate synapse tables from the CAVE cloud API, but this requires credentials. Pre-computed tables are included in the data archive, so CAVE access is optional.

## Key Modules

### `src/` -- Core Library

The shared library underpinning all analyses:

- **`src/myio/`** -- Data I/O: load PA/CLEM/EM cell tables, SWC morphologies, meshes, synapses. Includes CAVE pipeline integration for cloud-based EM data access and metadata read/write utilities.
- **`src/morphology/`** -- Morphology processing: NBLAST similarity scoring, SWC repair, branch finding, neurite fragmentation.
- **`src/viz/`** -- Visualization: unified color system (`colors`), classification plots (`classification_plots`), feature plots (`feature_plots`), figure helpers (`figure_helper`).
- **`src/util/`** -- Utilities: base path resolution, centralized output paths, cell path construction, constants.

### `functional_type_prediction/` -- Classifier Pipeline

Supervised machine learning pipeline for predicting neuron functional types from morphological and positional features. Supports multiple classifiers (LDA, AdaBoost, Random Forest, SVM, KNN, Naive Bayes) with cross-validation, hyperparameter optimization, and feature extraction from SWC morphologies.

### `contrastive_learning/` -- SegCLR-SupCon Pipeline

Deep learning pipeline for contrastive representation learning on neuron morphologies, including cluster submission scripts and experiment tracking.

### `segclr_src/` -- SegCLR Core Library

Core SegCLR library: model architecture definitions, training loops, and data handling utilities.

## Functional Type Nomenclature

The codebase uses two naming conventions for functional neuron types:

| Abbreviation | Full Name | Legacy Name |
|---|---|---|
| MI | motion_integrator | integrator |
| MON | motion_onset | dynamic_threshold |
| SMI | slow_motion_integrator | motor_command |

Subtypes include laterality: `motion_integrator_ipsilateral`, `motion_integrator_contralateral`.

## License

MIT License. Copyright (c) 2024 Armin Bahl. See `LICENSE` for full text.

## Authors

Florian Kampf, Armin Bahl, Jonathan Boulanger-Weill, and contributors.
