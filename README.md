# Zebrafish Hindbrain Functional Connectomics

This repository hosts collaborative analyses on the structure–function relationships in the zebrafish hindbrain, integrating anatomical connectivity, morphology-based predictions, and network modeling.

---


## Repository Structure

### 1. Downloading neuronal morphologies and metadata
Contains tools and scripts for uploading reconstructed neuronal morphologies and associated metadata. 

**Folder:** `Downloading_neuronal_morphologies_and_metadata/`  
**Environment file:** `env_clem_zfish1_neuroglancer.yaml`

---

### 2. Reference brain registration
Pipeline for registering neuronal morphologies to a standardized zebrafish reference brain coordinate framework.

**Folder:** `Reference_brain_registration/`  
**Environment file:** `env.yaml`

---

### 3. Connectivity analysis
Contains scripts for analyzing synaptic connectivity and generating connectivity matrices.

**Folder:** `Connectivity_analysis/`  
**Environment file:** `env.yaml`

---

### 4. Morphology-based prediction of neuronal functional types
Includes scripts to predict functional properties (e.g., motion integrator, motion onset threshold neurons) from morphology.

**Folder:** `Morphology_based_prediction_of_neuronal_functional_types/`  
**Environment file:** `env.yaml`

---

### 5. Connectome-constrained network modeling
Implements computational models that simulate or predict network dynamics under realistic connectome constraints.

**Folder:** `Connectome_constrained_network_modeling/`  
**Environment file:** `env.yaml`

---

## Setup
Each subproject includes an independent conda environment file (`env.yaml`).  
To create an environment:
```bash
conda env create -f env.yaml
conda activate <env_name>
```