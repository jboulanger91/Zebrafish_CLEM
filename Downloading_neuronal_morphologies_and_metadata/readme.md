# clem_zfish1_neuroglancer_pipeline

This folder contains the pipeline used to retrieve neuronal segments and synapses from the **clem_zfish1** zebrafish hindbrain connectome, generate per-neuron metadata, export mesh components, and optionally extract functional imaging dynamics.

The pipeline is built around the script:

```
clem_zfish1_neuroglancer_pipeline.py
```

and relies on CloudVolume, Neuroglancer, CAVEclient, and the processed functional imaging dataset.

Before running this pipeline, CAVE credentials must be configured using the accompanying notebook:

```
CAVE_setup.ipynb
```

---

## Overview of Functionality

- Retrieve nucleus, soma, axon, and dendrite meshes using CloudVolume  
- Query pre- and postsynaptic synapses via CAVEclient  
- Merge predicted and manually annotated synapses  
- Generate per-neuron metadata files  
- Export mesh components (.obj) and synapse tables  
- Optionally extract ΔF/F functional dynamics for imaged neurons  

All output is organized into per-neuron directories under the folder provided via `--root-path`.

---

## Example Usage

Run the full pipeline from the command line:

```bash
python clem_zfish1_neuroglancer_pipeline.py \
    --excel-file /path/to/rgc_axons_output_020525.csv \
    --root-path /path/to/traced_axons_neurons/ \
    --manual-synapses-path /path/to/manual_synapses/ \
    --hdf5-path /path/to/all_cells.h5 \
    --size-cutoff 44
```

Replace the paths with values appropriate for your system.

---

## Environment Setup

The recommended environment for running this script can be created via:

```bash
conda env create --file pull_from_neuroglancer.yaml
conda activate pull_from_neuroglancer
```

---

## Pipeline Diagram

```
                +-----------------------------+
                |     Input neuron table      |
                |    (CSV with segment IDs)   |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |   Initialize CloudVolume &   |
                |         CAVEclient           |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |   Retrieve neuron segments   |
                |  (nucleus, soma, axon, dend) |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |    Download mesh files      |
                |        (.obj format)        |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |  Query pre/post synapses    |
                |    from materialized table   |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                | Merge predicted synapses     |
                |  with manual annotations     |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |     Export synapse tables    |
                | (NG resolution, CSV files)   |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |   Build metadata file per    |
                |             neuron           |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                | Extract ΔF/F dynamics (opt.) |
                |    for functionally imaged   |
                |             neurons          |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |     Final per-neuron folder  |
                | (metadata, meshes, synapses, |
                |       functional data)       |
                +-----------------------------+
```

---

## Notes

- Access to the datastack requires a valid CloudVolume token and configured CAVE credentials.  
- This script is part of the zebrafish hindbrain functional connectomics analysis pipeline.  
- All outputs adhere to the directory structure defined by `--root-path`.