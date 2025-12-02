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

**Main script:** `download_axons_neurons_pipeline.py`  
**Helper module:** `download_axons_neurons_helper.py`  
**Environment file:** `env_clem_zfish1_global.yaml`  

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

### 3. Connectivity Matrices and Network Diagram Generation

This folder contains the pipelines used to compute **synaptic connectivity matrices** and to generate **two-layer network diagrams** from the **clem_zfish1** zebrafish hindbrain connectome. These analyses integrate CAVE-derived synapse tables, registered neuron meshes, and functional classifications.

Both pipelines rely on the mapped neuron outputs generated in **Step 2** and operate directly on per-neuron `*_presynapses.csv` and `*_postsynapses.csv` tables.

---

#### 3.1 Connectivity Matrices

This pipeline constructs **directional pre→post synaptic connectivity matrices** between functional/morphologcial neuron/axon classes. 

### **Outputs**
- Pooled connectivity matrix (PDF)  
- Left/right split connectivity matrix (PDF)  
- Optional raster or scatter representations  
- Optional inhibitory/excitatory signed representation (inhibitory rows × −1)

**Main script:** `make_connectivity_matrices.py`  
**Helper module:** `connectivity_matrices_helper.py`  
**Environment:** `env_clem_zfish1_global.yaml`

---

#### 3.2 Two-Layer Network Diagrams

This pipeline creates **compact schematic connectivity diagrams** for selected functional populations:

### **Outputs**
- Four-panel PDF connectivity diagram per population (cMI, MON, MC, iMI+, iMI−)

**Main script:** `make_connectome_diagrams.py`  
**Helper modules:**  
- `connectivity_diagrams_helper.py`  
- `connectivity_matrices_helper.py`  
**Environment:** `env_clem_zfish1_global.yaml`

---

### 4. Morphology-based prediction of neuronal functional types
Includes scripts to predict functional properties (e.g., motion integrator, motion onset neurons) from morphology.

**Environment file:** `env.yaml`

---

### 5. Connectome-constrained network modeling
Computational models that simulate network dynamics under realistic connectome constraints.

**Environment file:** `env.yaml`

---
