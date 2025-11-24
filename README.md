# Zebrafish Hindbrain Functional Connectomics

This repository hosts collaborative analyses on the structure–function relationships in the zebrafish hindbrain, integrating anatomical connectivity, morphology-based predictions, and network modeling.

---

## 📂 Repository Structure

### 1. Connectivity analysis
Contains scripts and notebooks for analyzing synaptic connectivity, neuron–neuron relationships, and connectivity matrices.

**Folder:** `Connectivity_analysis/`  
**Environment file:** `env.yaml`

---

### 2. Morphology-based prediction of neuronal functional types
Includes models and data linking morphological features to functional classifications (e.g., integrator, dynamic threshold, motor command).

**Folder:** `Morphology_based_prediction_of_neuronal_functional_types/`  
**Environment file:** `env.yaml`

---

### 3. Connectome-constrained network modeling
Implements computational models that simulate or predict network dynamics under realistic connectome constraints.

**Folder:** `Connectome_constrained_network_modeling/`  
**Environment file:** `env.yaml`

---

## ⚙️ Setup
Each subproject includes an independent conda environment file (`env.yaml`).  
To create an environment:
```bash
conda env create -f env.yaml
conda activate <env_name>