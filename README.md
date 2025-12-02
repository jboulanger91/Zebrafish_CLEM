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

This folder contains the pipelines used to compute **synaptic connectivity matrices** and to generate **two-layer network diagrams** from the **clem_zfish1** zebrafish hindbrain connectome. These analyses integrate Neuroglancer-derived synapse tables with functional classifications.

The pipelines operate on mapped neurons (from Step 2), using synapse tables and metadata to build population-level connectivity summaries.

---

## Connectivity Matrices (`make_connectivity_matrices.py`)

This script constructs **directional connectivity matrices** describing pre → post synaptic connections between functionally defined neuron groups.

It produces three matrix types:

#### **1. Pooled connectivity matrix (hemispheres merged)**  
Neurons and axons are grouped into the functional classes defined in the manuscript:

- `axon_rostral`  
- `ipsilateral_motion_integrator`  
- `contralateral_motion_integrator`  
- `motion_onset`  
- `slow_motion_integrator`  
- `myelinated`  
- `axon_caudal`

#### **2. Left–right split connectivity matrix**  
Same classes as above, expanded with hemisphere-specific suffixes:  
`*_left` and `*_right`.

#### **3. Left–right split connectivity matrix with raster display**  
Same as above with rste display. 

**Outputs include:**
- The matrices described bove, as PDFs. 

**Main script:** `make_connectivity_matrices.py`  
**Helper module:** `connectivity_matrices_helper.py`  
**Environment file:** `env_clem_zfish1_global.yaml`

---

## Two-Layer Network Diagrams (`make_connectome_diagrams.py`)

This script generates **compact two-layer connectivity schematics** for specific functional neuron populations:

- **cMI** — contralateral motion integrators  
- **MON** — motion onset neurons  
- **MC** — slow motion integrators (motor command–like)  
- **iMI+** — ipsilateral motion integrators (excitatory)  
- **iMI−** — ipsilateral motion integrators (inhibitory)

For each seed population, the pipeline:

- Extracts **same-side** and **cross-side** inputs & outputs  
- Computes **synapse-count probabilities**  
- Plots four diagrams per population:  
  - Same-side inputs  
  - Cross-side inputs  
  - Same-side outputs  
  - Cross-side outputs  

Node colors follow the functional color scheme defined in `clem_zfish1_connectivity_helper.py`, and connection thickness scales with synapse counts.

**Outputs include:**
- Four-panel network diagrams (PDFs) for each functional population

**Main script:** `make_connectome_diagrams.py`  
**Helper modules:**  
- `connectivity_diagrams_helper.py`  
- `connectivity_matrices_helper.py`
**Environment file:** `env_clem_zfish1_global.yaml`

---

### 4. Morphology-based prediction of neuronal functional types
Includes scripts to predict functional properties (e.g., motion integrator, motion onset neurons) from morphology.

**Environment file:** `env.yaml`

---

### 5. Connectome-constrained network modeling
Computational models that simulate network dynamics under realistic connectome constraints.

**Environment file:** `env.yaml`

---
