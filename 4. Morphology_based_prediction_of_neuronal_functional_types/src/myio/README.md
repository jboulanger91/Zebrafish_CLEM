# src.myio -- Data I/O Module

## Overview

The `src.myio` module provides all data loading and writing functions for the
zebrafish hindbrain structure-function project. It handles loading cell metadata
from multiple imaging modalities (photoactivation, CLEM, EM) into pandas
DataFrames, loading 3D morphology data (SWC skeletons and OBJ meshes),
and writing metadata files with classification results. It also
includes a CAVE download pipeline for retrieving neuronal segments, synapses,
meshes, and functional dynamics from the Lichtman/Engert zebrafish hindbrain
connectome, as well as a module for integrating LDA-based functional predictions
into the master metadata table.

## File Listing

| File | Description | Key Functions / Classes | External Dependencies |
|------|-------------|------------------------|-----------------------|
| `__init__.py` | Package init; re-exports all public API symbols. Conditionally imports CAVE pipeline functions (requires `caveclient`, `cloudvolume`, `navis`). | All `__all__` exports | -- |
| `_parsing.py` | Internal parsing utilities shared by multiple loaders. Parses key=value text format into single-row DataFrames. | `read_to_pandas_row` | `numpy`, `pandas` |
| `load_pa_table.py` | Loads photoactivation (PA) cell metadata from the cell inventory xlsx (PA sheet). Sets function/morphology from xlsx columns, filters for cells with valid type labels, and resolves per-cell metadata file paths. | `load_pa_table(path) -> DataFrame` | `pandas` |
| `load_clem_table.py` | Loads CLEM cell metadata from the cell inventory xlsx (CLEM sheet). Sets function/morphology from xlsx columns, computes `reconstruction_complete` boolean, resolves metadata file paths. | `load_clem_table(xlsx_path, clem_data_dir) -> DataFrame` | `pandas` |
| `load_em_table.py` | Loads EM cell metadata from the cell inventory xlsx (EM sheet). Resolves metadata file paths from the em_zfish1 data directory. | `load_em_table(xlsx_path, em_data_dir) -> DataFrame` | `pandas` |
| `load_mesh.py` | Loads 3D morphology data (OBJ meshes and SWC skeletons) for cells across PA, CLEM, and EM modalities. Resolves modality-specific directory structures. | `load_mesh(cell, swc, load_both, load_repaired)` | `navis`, `numpy` |
| `load_cells2df.py` | Main data loading pipeline. Combines cell tables from PA, CLEM, and EM modalities; loads meshes/skeletons; mirrors cells to left hemisphere; extracts functional type, morphology, and neurotransmitter labels; filters incomplete reconstructions. | `load_cells_predictor_pipeline(modalities, mirror, keywords, ...) -> DataFrame` | `navis`, `numpy`, `pandas`, plus all loaders above and `src.util.get_base_path` |
| `cave_pipeline.py` | CAVE download pipeline for the `clem_zfish1` connectome. Retrieves neuronal segments, meshes, synapses, and functional dynamics from CloudVolume/CAVEclient. Writes per-neuron metadata, OBJ meshes, synapse CSV tables, and HDF5 dynamics files. | `CavePipelineContext`, `create_pipeline_context`, `run_pipeline`, `generate_metadata_files`, `save_mesh`, `write_synapse_file`, `process_functional_data`, `check_problematic_segments`, `normalize_functional_type` | `caveclient`, `cloudvolume`, `navis`, `h5py`, `numpy`, `pandas`, `matplotlib` (optional, for plots), `src.util.get_base_path`, `src.viz.colors` |

## Key Exports

```python
# Loading functions
load_clem_table       # CLEM cell metadata table
load_em_table         # EM cell metadata table
load_mesh             # 3D morphology (SWC/OBJ) for a single cell
load_pa_table         # Photoactivation cell metadata table

# CAVE download pipeline (requires caveclient, cloudvolume, navis)
CavePipelineContext
check_problematic_segments
create_pipeline_context
generate_metadata_files
normalize_functional_type
process_functional_data
run_pipeline
save_mesh
write_synapse_file
```

## Dependencies on Other `src/` Modules

- **`src.util.get_base_path`**: Used by `load_cells2df.py` and `cave_pipeline.py`
  to resolve the user-specific dataset root directory from `config/path_configuration.txt`.
- **`src.viz.colors`**: Used by `cave_pipeline.py` for functional-type color
  palettes when generating diagnostic PDF plots.

## Data Path Resolution

Default data paths are derived from `src.util.get_base_path()`, which reads the
user-specific base path from `config/path_configuration.txt`. All loaders accept
explicit path arguments to override the defaults.
