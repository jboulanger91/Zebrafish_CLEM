# src -- Hindbrain Structure-Function Core Library

This package provides the core analysis library for the zebrafish hindbrain
structure-function project (`morph2func`). It spans multiple imaging modalities
-- photoactivation (PA), correlative light and electron microscopy (CLEM),
and serial-section EM -- and covers data loading, morphological analysis,
and visualization.

## Module Overview

| Directory      | Purpose |
|----------------|---------|
| `morphology/`  | Neuron morphology processing: SWC repair (fixing broken nodes, re-rooting), NBLAST morphological similarity scoring, branch finding, and neurite fragmentation. |
| `myio/`        | Data I/O utilities: loading cell metadata into DataFrames (`load_cells2df`), mesh loading, PA/CLEM/EM table loaders, CAVE pipeline access. |
| `util/`        | General utilities: centralized output path resolution via `get_output_dir()`, base data path detection, cell path helpers. |
| `viz/`         | Visualization: color palette definitions, classification plots, feature plots, figure layout helpers. |

## Output Location

All generated files are written under a single root determined by
`src.util.output_paths.get_output_dir(module, *subdirs)`. The root is
resolved as:

1. The `MORPH2FUNC_OUTPUT_ROOT` environment variable, if set.
2. `~/Desktop/morph2func_output/` on macOS.
3. `~/morph2func_output/` elsewhere.

Each module passes its own name (e.g., `"classifier_pipeline"`)
as the first argument, producing paths like `~/Desktop/morph2func_output/classifier_pipeline/predictions/`.

## Key Dependencies

- **numpy**, **pandas** -- array and tabular data throughout
- **matplotlib** -- all visualization modules
- **navis** -- SWC/mesh I/O, morphological operations
- **h5py** -- HDF5 data I/O
- **scikit-learn** -- used by downstream classifier pipeline
