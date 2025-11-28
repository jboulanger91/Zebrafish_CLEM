# clem_zfish1_register_and_skeletonize_pipeline

This folder contains the pipeline used to register neuronal morphologies and synapses from the **clem_zfish1** zebrafish hindbrain connectome into a [standardized reference brain](https://zebrafishexplorer.zib.de/home/) and to generate skeletonized reconstructions.

The pipeline is built around the script:

```
02_map_and_skeletonize_cells.py
```

and relies on ANTs deformation fields, Trimesh, Navis, Skeletor, and the Neuroglancer‑generated morphology folders.

---

## Overview of Functionality

- Warp soma, axon, and dendrite meshes into a standardized reference brain  
- Map presynaptic and postsynaptic coordinates into reference space  
- Generate mesh files (`*_mapped.obj`) and synapse OBJ sphere files  
- Skeletonize neurons or isolated axons using TEASAR  
- Heal skeleton gaps and embed presynaptic & postsynaptic node annotations  
- Produce SWC files containing standardized node labels  

All output is saved inside each neuron’s folder under a `mapped/` subdirectory.

---

## Environment Setup

Create the recommended environment using:

```bash
conda env create -f map_and_skeletonize.yaml
conda activate map_and_skeletonize
```

This environment installs dependencies for ANTs, mesh processing, and skeletonization.

---

## Example Usage

Run the registration and skeletonization pipeline:

```bash
python3 02_map_and_skeletonize_cells.py \
    --cells-folder traced_axons_neurons/ \
    --transform-prefix ANTs_transforms/ANTs_dfield \
    --ants-bin-path /path/to/ants/bin \
    --ants-threads 11
```

**Notes**

- `--transform-prefix` must point to the prefix only.  
  The following files must exist:  
  - `ANTs_dfield.nii.gz`  
  - `ANTs_dfield_inverse.nii.gz`

- `--cells-folder` must contain per‑neuron subfolders such as:  
  `clem_zfish1_cell_XXXXX/` or `clem_zfish1_axon_XXXXX/`.

---

## Input Data and Example Files

Each neuron directory must contain:

- `<cell_name>_metadata.txt`  
- `<cell_name>_axon.obj`  
- `<cell_name>_dendrite.obj` *(optional)*  
- `<cell_name>_soma.obj` *(optional)*  
- `<cell_name>_presynapses.csv`  
- `<cell_name>_postsynapses.csv`  

Synapse CSVs must contain Neuroglancer‑resolution coordinate columns:
- `x (8 nm)`
- `y (8 nm)`
- `z (30 nm)`

These are mapped into reference space by the pipeline.

---

## Output Data

The pipeline produces:

### Original‑space outputs  
- `<cell_name>.obj` — merged axon+dendrite+soma mesh  
- `<cell_name>.swc` — healed skeleton in EM space  

### Mapped outputs (inside `mapped/`)  
- `<cell_name>_axon_mapped.obj`  
- `<cell_name>_dendrite_mapped.obj`  
- `<cell_name>_soma_mapped.obj`  
- `<cell_name>_mapped.obj`  
- `<cell_name>_presynapses_mapped.csv`  
- `<cell_name>_postsynapses_mapped.csv`  
- `<cell_name>_presynapses_mapped.obj`  
- `<cell_name>_postsynapses_mapped.obj`  
- `<cell_name>_mapped.swc`  

### SWC Node Labels  
- **1** — soma  
- **2** — axon  
- **3** — dendrite  
- **4** — presynapse  
- **5** — postsynapse  

---

This pipeline constitutes the second processing step following the Neuroglancer‑based morphology extraction and ensures all neurons are aligned to a unified anatomical reference.
