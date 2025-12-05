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

2. **Left–Right split connectivity matrix**, using the same functional classes but expanded with hemisphere labels (e.g. `_left`, `_right`).  
   To illustrate multiple display styles, both **scatter** and **raster** versions are produced.

---

### **B. Two-Layer Network Diagrams (`make_connectivity_diagrams.py`)**

This pipeline generates compact two-layer connectivity schematics for selected seed populations:

Note: Because hemispheric assignment depends on mapped neuron geometries, **registered OBJ meshes (mapped to the reference brain)** must be present for all neurons involved.

- **cMI**   — contralateral motion integrators  
- **MON**   — motion onset neurons  
- **SMI**   — slow motion integrators (motor-command–like)  
- **iMI+**  — ipsilateral motion integrators (excitatory)  
- **iMI−**  — ipsilateral motion integrators (inhibitory)

For each population, the script:

- Extracts **ipsilateral** (same-side) and **contralateral** (cross-side) inputs and outputs  
- Computes **synapse-count probabilities** and **cross-panel–normalized fractions**  
- Plots **four 2-layer connectivity diagrams**:  
  1. Ipsilateral input synapses  
  2. Contralateral input synapses  
  3. Ipsilateral output synapses  
  4. Contralateral output synapses  
- Exports a detailed text-based connectivity table for downstream quantitative analysis (`export_connectivity_tables_txt`)

---

## Environment Setup

Use the global environment for all connectivity analysis:

```bash
conda env create -f clem_zfish1_global.yaml
conda activate clem_zfish1_global
```

---

## Example Usage

### 1. Build connectivity matrices

```bash
python3 make_connectivity_matrices.py \
    --metadata-csv ".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/all_reconstructed_neurons.csv" \
    --root-folder ".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/traced_axons_neurons" \
    --output-folder connectivity_matrices \
    --plot-type scatter \
    --suffix gt
```

### 2. Build two-layer connectivity diagrams

```bash
python3 make_connectivity_diagrams.py \
    --metadata-csv ".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/all_reconstructed_neurons.csv" \
    --root-folder ".../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/traced_axons_neurons" \
    --output-folder connectivity_diagrams \
    --suffix gt
```

---

## Input Data and Example Files

Both connectivity pipelines operate on two types of inputs:

### **1. Metadata Table (CSV)**

A single table where each row corresponds to a reconstructed neuron or axon, containing:

- `type` — `"cell"` or `"axon"`  
- `nucleus_id`  
- `axon_id`  
- `dendrite_id`  
- `Functional Classifier`  
  *(motion_integrator, motion_onset, slow_motion_integrator, myelinated, axon, not functionally imaged)*  
- `Projection Classifier` *(ipsilateral / contralateral)*  
- `Neurotransmitter Classifier` *(inhibitory / excitatory)*  
- `hemisphere` *(L / R; computed automatically if missing)*  
- `comment` *(e.g., “axon exits the volume caudally/rostrally”)*  

This table is located at:

```
.../Zebrafish_CLEM/1. Downloading_neuronal_morphologies_and_metadata/all_reconstructed_neurons.csv
```

---

### **2. Synapse Tables (per-neuron folder)**

Each traced neuron directory must contain:

- `<cell_or_axon_name>_presynapses.csv`  
- `<cell_or_axon_name>_postsynapses.csv` *(if the neuron receives inputs)*  

Each table must include:

- `presynaptic_ID` or `postsynaptic_ID`  
- `synapse_id`  
- `validation_status`  
- Neuroglancer-resolution coordinates:  
  - `x_(8_nm)`  
  - `y_(8_nm)`  
  - `z_(30_nm)`

These files are produced automatically by the Neuroglancer postprocessing pipeline.

---

## Output Data

### **A. Connectivity Matrices**

Running `make_connectivity_matrices.py` produces:

- **Pooled connectivity matrix**  
  `pooled_connectivity_matrix_<suffix>.pdf`

- **Left/Right split connectivity matrix**  
  `lr_split_connectivity_matrix_inhibitory_excitatory_<suffix>.pdf`

- **Optional scatter/raster visualizations**  
  `left-right_matrix_e-i_raster_<suffix>.pdf`  
  `left-right_matrix_e-i_<suffix>.pdf`

**Matrix semantics**  
Rows = presynaptic neuron IDs  
Columns = postsynaptic neuron IDs  
Values = number of validated synapses (signed when inhibitory/excitatory mode is used)

---

### **B. Two-Layer Network Diagrams**

Running `make_connectivity_diagrams.py` generates one PDF per seed population, e.g.:

```
cMI_all_<suffix>.pdf
MON_all_<suffix>.pdf
SMI_all_<suffix>.pdf
iMI_plus_<suffix>.pdf
iMI_minus_<suffix>.pdf
iMI_<suffix>.pdf
```

Each PDF contains a **2×2 panel layout**:

1. Ipsilateral inputs  
2. Contralateral inputs  
3. Ipsilateral outputs  
4. Contralateral outputs  

**Diagram contents**

- Node colors = functional type (from `COLOR_CELL_TYPE_DICT`)  
- Node outline = neurotransmitter type (solid/dashed/dotted)  
- Edge thickness = fraction of synapses in that category  
- Marker glyphs = excitatory arrow, inhibitory inverted arrow, mixed dot, unknown line  
- Seed population shown as a large filled circle  

All outputs are saved in the folder specified via `--output-folder`.

---

### **C. Connectivity Tables (TXT)**

Each two-layer connectivity run also exports a machine-readable summary:

```
<population>_connectivity_details_<suffix>.txt
```

This table contains one row per connection category with:
- Connection type (inputs / outputs)
- Side (ipsilateral / contralateral)
- Unit type (cells / synapses)
- Functional classifier
- Neurotransmitter classifier
- Projection classifier
- Axon exit direction
- Raw synapse count
- Cross-panel–normalized probability
