# 2. Reference_brain_registration

This folder contains the pipeline used to register neuronal morphologies and synapses from the **clem_zfish1** zebrafish hindbrain connectome into a [standardized reference brain](https://zebrafishexplorer.zib.de/home/) and to generate mapped meshes and skeletonized reconstructions.

The pipeline is built around the script:

```
register_and_skeletonize_cells.py
```

and relies on ANTs deformation fields, Trimesh, Navis, Skeletor, and the morphology folders produced during the Neuroglancer extraction step.

---

## Overview of Functionality

- Warp axon, soma, and dendrite meshes into reference brain space using ANTs
- Map presynaptic and postsynaptic coordinates into reference space
- Generate mapped mesh files (`*_mapped.obj`)
- Create OBJ synapse sphere files in both EM space and mapped space
- Skeletonize neurons or axons using TEASAR
- Heal skeleton gaps and embed presynaptic & postsynaptic annotations
- Export standardized SWC files

All mapped outputs are written into each neuron directory under `mapped/`.

---

## ANTs Requirements (Must Be Installed Separately)

This pipeline depends on **ANTs** for nonlinear registration. See the [Github repository](https://github.com/ANTsX/ANTs?tab=readme-ov-file) for installation options 

---

## Environment Setup

Create the dedicated registration environment using:

```bash
conda env create -f env_register_and_skeletonize.yaml
conda activate register_and_skeletonize
```

---

## Example Usage

Run the full registration and skeletonization pipeline:

```bash
python3 register_and_skeletonize.py \
    --cells-folder ".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/traced_axons_neurons" \
    --transform-prefix "ANTs_transforms/ANTs_dfield" \
    --ants-bin-path /Users/jonathanboulanger-weill/Packages/install/bin \
    --ants-threads 11
```

**Notes**
- `--transform-prefix` must point to the prefix only. Required files:
  - `ANTs_dfield_inverse.nii.gz`
- `--cells-folder` must contain subfolders such as: `clem_zfish1_cell_XXXXX/` or `clem_zfish1_axon_XXXXX/`.

---

## Input Data and Example Files

An example folder is provided under ".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/example_neuron_axon"

Each neuron/axon directory must contain:

- `<cell/axon_name>_metadata.txt`
- `<cell/axon_name>_axon.obj`
- `<cell/axon_name>_dendrite.obj` *(optional)*
- `<cell/axon_name>_soma.obj` *(optional)*
- `<cell/axon_name>_presynapses.csv`
- `<cell/axon_name>_postsynapses.csv` *(optional)*

Synapse CSVs must include the Neuroglancer‑resolution coordinates:
- `x (8 nm)`, `y (8 nm)`, `z (30 nm)`

These are mapped into reference space automatically.

---

## Output Data

The pipeline produces:

### Original‑space outputs
- `<cell/axon_name>.swc` — healed EM‑space skeleton
- `<ccell/axon_name>_presynapses.obj`
- `<cell/axon_name>_postsynapses.obj`

### Mapped outputs (`mapped/`)
- `<cell/axon_name>_axon_mapped.obj`
- `<cell/axon_name>_dendrite_mapped.obj`
- `<cell/axon_name>_soma_mapped.obj`
- `<cell/axon_name>_mapped.obj`
- `<cell/axon_name>_presynapses_mapped.csv`
- `<cell/axon_name>_postsynapses_mapped.csv`
- `<ccell/axon_name>_presynapses_mapped.obj`
- `<cell/axon_name>_postsynapses_mapped.obj`
- `<cell/axon_name>_mapped.swc`

### SWC Node Labels
- **1** — soma
- **2** — axon
- **3** — dendrite
- **4** — presynapse
- **5** — postsynapse

---
