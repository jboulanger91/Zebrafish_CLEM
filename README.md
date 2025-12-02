# Zebrafish Hindbrain Functional Connectomics

![Description of the image](home_img.png)

This repository hosts collaborative analyses on the structure–function relationships in the zebrafish hindbrain, integrating synaptic connectivity, morphology-based predictions, and network modeling.

---

## Repository Structure

### 1. Downloading neuronal morphologies and metadata

Pipeline for reconstructing and organizing neuronal morphologies, synapse positions, and connected segments from the clem_zfish1 dataset. This includes:

- Downloading and organizing meshes (nucleus, soma, axon, dendrites)  
- Exporting Neuroglancer-resolution synapse tables (8×8×30 nm)  
- Generating per-neuron metadata (IDs, reconstruction status, functional labels if available)  
- Optional extraction of ΔF/F dynamics for functionally imaged neurons  

Outputs are per-neuron/axon folders containing:

- `*_metadata.txt` (metadata)  
- `*_axon.obj`, `*_dendrite.obj`, `*_soma.obj`, `*.obj` (meshes)  
- `*_presynapses.csv`, `*_postsynapses.csv` (synapse tables)  
- , `*_dynamics.pdf`, `*_dynamics.hdf5` (optional functional data)

**Main script:** `clem_zfish1_neuroglancer_pipeline.py`  
**Helper module:** `clem_zfish1_neuroglancer_helper.py`  
**Environment file:** `env_clem_zfish1_neuroglancer.yaml`  

---

### 2. Reference brain registration

Pipeline for mapping neuronal morphologies and synapse coordinates from the **clem_zfish1** dataset into a standardized zebrafish reference brain. This step applies ANTs deformation fields to warp neuron meshes and synapse locations into a shared anatomical space, then generates skeletonized neuron reconstructions.

This includes:

- Mapping soma, axon, and dendrite meshes into reference space  
- Warping presynaptic and postsynaptic coordinates  
- Generating mapped OBJ meshes and synapse OBJ spheres  
- Skeletonizing neurons or axons using TEASAR  
- Healing skeleton gaps and embedding synapse node labels  
- Producing aligned SWC skeletons for further analysis  

Outputs are per-neuron folders containing:

- `*_axon_mapped.obj`, `*_dendrite_mapped.obj`, `*_soma_mapped.obj`, `*_mapped.obj` (mapped meshes)  
- `*_presynapses_mapped.csv`, `*_postsynapses_mapped.csv` (mapped synapse tables)  
- `*_presynapses_mapped.obj`, `*_postsynapses_mapped.obj` (synapse OBJ sphere files)  
- `*_mapped.swc`, `*.swc` (skeletonized neurons with synapse annotations)  

**Main script:** `register_and_skeletonize.py`  
**Helper module:** `register_and_skeletonize_cells_helpers.py`  
**Environment file:** `env_register_and_skeletonize.yaml`  

---

### 3. Connectivity Matrix and Diagrams Generation

Pipeline for computing and visualizing directional connectivity matrices from the **clem_zfish1** connectome.  This includes
- Import of Neuroglancer-resolution synapse tables
- Automatic hemisphere classification from mapped meshes (neurons need to be registered to the reference brain)   
- Grouping neurons into functional and morphological categories decribed in the manuscript. 
- Construction of two matrix types:  
  * **Pooled matrix** (across hemispheres)  
  * **Left/right-split matrix** with optional inhibitory/excitatory signed representation  
- Generation of matrix plots (heatmap or scatter) with functional-type sidebars

**Outputs:**
- Directional connectivity matrices (pre → post synapses) or pooled or Left/Right-split connectivity matrix PDFs  

**Main script:** `make_connectivity_matrices.py`  
**Helper module:** `clem_zfish1_connectivity_helper.py`  
**Environment file:** `env_clem_zfish1_connectivity.yaml`

---

### 4. Morphology-based prediction of neuronal functional types
Includes scripts to predict functional properties (e.g., motion integrator, motion onset neurons) from morphology.

**Environment file:** `env.yaml`

---

### 5. Connectome-constrained network modeling
Computational models that simulate network dynamics under realistic connectome constraints.

**Environment file:** `env.yaml`

---
