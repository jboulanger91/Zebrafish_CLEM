# clem_zfish1_neuroglancer_pipeline

This folder contains the pipeline used to retrieve neuronal segments and synapses from the **clem_zfish1** zebrafish hindbrain connectome, generate per-neuron metadata, export mesh components, and optionally extract functional imaging dynamics.

The pipeline is built around the script:

```
clem_zfish1_neuroglancer_pipeline.py
```

and relies on CloudVolume, Neuroglancer, CAVEclient, and the processed functional imaging dataset.

Before running this pipeline, CAVE credentials must be configured using the accompanying notebook in this folder:

```
CAVE_setup.ipynb
```

---

## Overview of Functionality

- Retrieve nucleus, soma, axon, and dendrite meshes using CloudVolume  
- Query pre- and postsynaptic synapses via CAVEclient  
- Merge automatically predicted and manually annotated synapses  
- Generate per-neuron metadata files   
- Optionally extract ΔF/F functional dynamics for imaged neurons  

All output is organized into per-neuron directories under the folder provided via `--root-path`.

---

## Environment Setup

The recommended environment for running this script can be created via:

```bash
conda env create -f env_clem_zfish1_neuroglancer.yaml
conda activate clem_zfish1_neuroglancer
```

---

## Example Usage

Run the full pipeline from the command line:

```bash
  python3 clem_zfish1_neuroglancer_pipeline.py \
        --excel-file example_neuron.csv \
        --root-path traced_axons_neurons/ \
        --manual-synapses-path manual_synapses_example\
        --hdf5-path fish1.5_functional_data.h5 \
        --size-cutoff 44
```

## Input Data and Example Files

The CLEM zfish1 Neuroglancer pipeline relies on three main types of input:

1. A CSV file listing nuclei/soma/axons/dendrits segment IDs, you can use the whole dataset (all_reconstructed_neurons.csv) or the example file (example_neuron.csv) provided in the repository. 
2. An HDF5 file containing functional imaging data (optional). The functional ΔF/F data used in this study is provided as an HDF5 file: 
- **File name:** `fish1.5_functional_data.h5`  
- **Download:** [Zenodo record 16893093](https://zenodo.org/records/16893093)
3. (Optional) Manual synapse annotation files 

All filesystem paths in the example commands should be replaced with locations appropriate for your system.

---

## Output data

You can regenerate the whole dataset using the provided csv file: all_reconstructed_neurons.csv or download it here: 

- **File name:** `all_cells.h5`  
- **Download:** [Zenodo record 16893093](https://zenodo.org/records/16893093)

---


