## Overview of Functionality

### **A. Connectivity Matrices (`make_connectivity_matrices.py`)**

This pipeline builds directional connectivity matrices from synapse tables and a metadata CSV. It produces:

1. **Pooled connectivity matrix**, grouping neurons into:
   - axon_rostral  
   - ipsilateral_motion_integrator  
   - contralateral_motion_integrator  
   - motion_onset  
   - slow_motion_integrator  
   - myelinated  
   - axon_caudal  

2. **Left–Right split connectivity matrix**, using the same functional classes but expanded with hemisphere labels (e.g. `_left`, `_right`). To illustrate the different plotting options: a scatter and a raster version are produced. 

---

### **B. Two-Layer Network Diagrams (`make_connectome_diagrams.py`)**

This pipeline generates compact connectivity schematics for selected seed populations:

- **cMI**   — contralateral motion integrators  
- **MON**   — motion onset neurons  
- **MC**    — slow motion integrators (motor command–like)  
- **iMI+**  — ipsilateral motion integrators (excitatory)  
- **iMI−**  — ipsilateral motion integrators (inhibitory)

For each population, the script:

- Extracts **same-side** and **cross-side** inputs and outputs  
- Computes **synapse-count probabilities**  
- Plots **four 2-layer networks**:  
  - Same-side inputs  
  - Cross-side inputs  
  - Same-side outputs  
  - Cross-side outputs  

---

## Environment Setup

The global environment can be used:

```bash
conda env create -f clem_zfish1_global.yaml
conda activate clem_zfish1_global
```

---

## Example Usage

1. **Build connectivity matrices**

python3 make_connectivity_matrices.py \
    --metadata-csv ".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/all_reconstructed_neurons.csv" \
    --root-folder ".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/traced_axons_neurons" \
    --output-folder connectivity_matrices \
    --plot-type scatter \
    --suffix gt

1. **Builed two-layer connectivity diagrams**

python3 make_connectivity_diagrams.py \
    --lda-csv "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/all_reconstructed_neurons.csv" \
    --root-folder "/Users/jonathanboulanger-weill/Harvard University Dropbox/Jonathan Boulanger-Weill/Projects/Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/traced_axons_neurons" \
    --output-folder connectivity_diagrams \
    --suffix gt

---

## Input Data and Example Files

Both connectivity pipelines operate on two types of inputs:

### **1. Metadata Table (CSV)**
A single table containing one row per reconstructed neuron or axon, with the following required fields:

- `type` — `"cell"` or `"axon"`
- `nucleus_id`
- `axon_id`
- `dendrite_id`
- `functional classifier`  
  *(motion_integrator, motion_onset, slow_motion_integrator, myelinated)*
- `projection classifier` *(ipsilateral / contralateral)*
- `neurotransmitter classifier` *(inhibitory / excitatory)*
- `hemisphere` *(L / R; computed automatically if missing)*
- `comment` *(e.g. “axon exits the volume caudally/rostrally”)*

The whole table used in this study is provided under "".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/all_reconstructed_neurons"

---

### **2. Synapse Tables (per-neuron folder)**

Each traced neuron directory must contain presynaptic and postsynaptic synapse CSV files:

- `<cell/axon_name>_presynapses.csv`
- `<cell/axon_name>_postsynapses.csv` *(if neuron receives inputs)*

Each file must contain (at minimum):

- `presynaptic_ID` or `postsynaptic_ID`  
- `synapse_id`  
- `validation_status`  
- `x_(8_nm)`, `y_(8_nm)`, `z_(30_nm)`  *(Neuroglancer-resolution coordinates)*

These files are produced automatically by the download & NG-postprocessing pipeline.

---

## Output Data

### **A. Connectivity Matrices**

Running `make_connectivity_matrices.py` produces the following outputs:

- **Pooled connectivity matrix** (PDF)  
  `pooled_connectivity_matrix_<suffix>.pdf`

- **Left/Right split connectivity matrix** (PDF)  
  `lr_split_connectivity_matrix_inhibitory_excitatory_<suffix>.pdf`

- *(Optional for scatter mode)*  
  `left-right_matrix_e-i_raster_<suffix>.pdf`  
  `left-right_matrix_e-i_<suffix>.pdf`

All matrices are saved as PDFs inside the folder provided via `--output-folder`.

**Matrix contents**

Rows: presynaptic (source) neuron IDs  
Columns: postsynaptic (target) neuron IDs  
Values: number of valid synapses (signed if inhibitory/excitatory mode is used)

---

### **B. Two-Layer Network Diagrams**

Running `make_connectome_diagrams.py` produces one PDF per seed population:

- `ic_all_<suffix>.pdf`  
- `neural_network_visualization_with_lda_dt_all_<suffix>.pdf`  
- `neural_network_visualization_with_lda_mc_all_lda_<suffix>.pdf`  
- `neural_network_visualization_with_lda_ii_plus_all_lda_<suffix>.pdf`  
- `neural_network_visualization_with_lda_ii_minus_all_lda_<suffix>.pdf`

Each PDF contains a **2 × 2 panel figure**:

1. Same-side inputs  
2. Cross-side inputs  
3. Same-side outputs  
4. Cross-side outputs  

**Diagram contents**

- Node colors = functional class (loaded from `matrix_helpers.py`)  
- Line thickness = proportional to synapse counts  
- Labels = proportions or probabilities (configurable)  
- Seed functional category indicated by a large input circle  

All output files are written to the folder passed with `--output-folder`.

---