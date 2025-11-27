# Zebrafish Hindbrain Functional Connectomics

![Description of the image](home_img.png)

This repository hosts collaborative analyses on the structure–function relationships in the zebrafish hindbrain, integrating synaptic connectivity, morphology-based predictions, and network modeling.

---

## Repository Structure

### 1. Downloading neuronal morphologies and metadata

Pipeline for reconstructing and organizing neuronal morphologies, synapse positions, and connected segments from the clem_zfish1 dataset.  
This includes:

- Downloading and organizing meshes (nucleus, soma, axon, dendrites)  
- Exporting NG-resolution synapse tables (8×8×30 nm)  
- Generating per-neuron metadata (IDs, reconstruction status, functional labels if available)  
- Optional extraction of ΔF/F dynamics for functionally imaged neurons  

Typical outputs are per-neuron folders containing:

- `*_metadata.txt` (metadata)  
- `*_axon.obj`, `*_dendrite.obj`, `*_soma.obj`, `*.obj` (meshes)  
- `*_presynapses.csv`, `*_postsynapses.csv` (synapse tables)  
- `*_dynamics.hdf5` (optional functional data)

**Main script:** `clem_zfish1_neuroglancer_pipeline.py`  
**Helper module:** `clem_zfish1_neuroglancer_helper.py`  
**Environment file:** `env_clem_zfish1_neuroglancer.yaml`  

---

### 2. Connectivity analysis

Tools for quantifying and visualizing connectomes derived from the traced neurons:

- Classify neurons by hemisphere (ipsilateral / contralateral)  
- Build directional connectivity matrices between functional neuron classes  
- Compute connectivity probabilities (cells and synapses, same- vs different-side)  
- Plot compact neural-network diagrams summarizing connectivity patterns  
- Generate mesh-based visualizations of input/output connectomes  
- Plot activity traces for functionally imaged neurons and their partners  

Typical outputs include:

- CSVs with updated metadata (e.g. hemisphere, LDA predictions)  
- Connectivity matrices and summary tables  
- PDF figures (neural network diagrams, mesh renderings, activity plots)

**Main scripts (examples):**

- `connectivity_analysis.py` (hemispheres, connectomes, matrices, plots)  
- `connectome_helpers_current.py` (helper functions for connectivity and plotting)

**Environment file:** `env.yaml`

---

### 3. Reference brain registration
Pipeline for registering neuronal morphologies to a standardized zebrafish reference brain coordinate framework.
 
**Environment file:** `env.yaml`

---

### 4. Morphology-based prediction of neuronal functional types
Includes scripts to predict functional properties (e.g., motion integrator, motion onset neurons) from morphology.

**Environment file:** `env.yaml`

---

### 5. Connectome-constrained network modeling
Computational models that simulate network dynamics under realistic connectome constraints.

**Environment file:** `env.yaml`

---
