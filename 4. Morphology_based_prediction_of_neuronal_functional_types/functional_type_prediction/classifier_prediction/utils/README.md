# utils

Morphological metrics computation and HDF5 data preparation utilities.

## Overview

This package provides utility functions for computing morphological features from neuron tracings and loading the resulting feature DataFrames from HDF5 storage. It bridges the gap between raw SWC neuron reconstructions (loaded via `src/myio/`) and the feature matrices consumed by the classifier pipeline. The computed features (~60 total) include cable length, branching statistics, spatial coordinates, Sholl analysis profiles, hemisphere metrics, persistence homology features, and axon crossing angles.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package docstring. Exports `calculate_metric2df`. |
| `calculate_metric2df.py` | Main utility module. `calculate_metric2df()` computes ~60 morphological features from neuron SWC data and saves the result to an HDF5 file. |

## Key Functions

- **`calculate_metric2df(cells_df, output_name, data_path)`** -- Iterates over all cells in the DataFrame, loads each neuron's SWC file, computes morphological features (cable length, total branches, branch order statistics, Strahler metrics, Sholl crossings, hemispheric IC index, persistence diagrams, first branchpoint metrics, axon crossing angles), and writes the complete feature matrix to an HDF5 file at `{data_path}/prediction/features/{output_name}_features.hdf5`.

## Dependencies

- **`src/myio/`** -- Cell table loading (PA, CLEM, EM), mesh loading, CAVE pipeline, metadata writing
- **`src/morphology/`** -- Low-level metric functions (cable length, Strahler order, Sholl analysis, persistence, hemisphere metrics, branch angles)
- External: pandas, numpy, h5py, navis

## Output Location

The `calculate_metric2df()` function writes HDF5 feature files to:

```
{data_path}/prediction/features/{output_name}_features.hdf5
```

where `data_path` is resolved by `get_base_path()` from `src/util/`.

## How to Run

These functions are typically called by the pipeline rather than run standalone:

```python
from classifier_prediction.utils.calculate_metric2df import calculate_metric2df

# Compute features (slow, run once)
calculate_metric2df(cells_df, "my_features", data_path)
```
