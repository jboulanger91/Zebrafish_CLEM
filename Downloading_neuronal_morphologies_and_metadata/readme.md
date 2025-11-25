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
python3 clem_zfish1_neuroglancer_pipeline.py \
    --excel-file /path/to/reconstructed_neurons.csv \
    --root-path /path/to/traced_axons_neurons/ \
    --manual-synapses-path /path/to/manual_synapses/ \
    --hdf5-path /path/to/fish1.5_functional_data.h5 \
    --size-cutoff 44
```

Replace the paths with values appropriate for your system.

The functional data h5 file can be downloaded here: https://zenodo.org/records/16893093 

The full csv file with all the neurons and axons reconstructed in this study are found ...

The example csv to test the code and upload manual examples are provided direclty in the Github folder 

---

## Environment Setup

The recommended environment for running this script can be created via:

```bash
conda env create -f env_clem_zfish1_neuroglancer.yaml
conda activate clem_zfish1_neuroglancer
```

---

## Pipeline Diagram
```
[Input CSV] 
     |
     v
[Init CloudVolume + CAVEclient]
     |
     v
[Retrieve neuron segments]
     |
     v
[Download meshes (.obj)]
     |
     v
[Query pre/post synapses]
     |
     v
[Merge predicted + manual synapses]
     |
     v
[Export synapse tables]
     |
     v
[Write metadata file]
     |
     v
[Extract ΔF/F (optional)]
     |
     v
[Final per-neuron folder]
```

---

## Notes

- Access to the datastack requires a valid CloudVolume token and configured CAVE credentials.  
- This script is part of the zebrafish hindbrain functional connectomics analysis pipeline.  
- All outputs adhere to the directory structure defined by `--root-path`.