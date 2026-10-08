"""
Overview
--------
This script generates perturbed structural connectivity masks from an empirical connectome
CSV file across multiple stochastic replicates. It allows evaluating the impact of topological
lesions, random edge shuffling, targeted cell-type neuron ablations, or pathway-specific
synapse removals on neural circuit dynamics.

Before running it, please make sure you have set up your .env with all the necessary variables:
- PATH_DIR="/path/to/project_root"  # Root project directory containing data/ and destination folders
Also, go through the following points:
- complete the line `path_save = path_dir / "PathToMyFolder"` so that it saves the manipulated connectivity
  matrix to the desired directory (the output files are not directly results, line)
- set up ablate_config_list according to the examples provided (from line 75)

Core Pipeline & Workflow:
1. Environment Resolution & Paths:
   - Resolves the target `.env` file either from command-line arguments (`sys.argv[1]`)
     or by falling back to the project root `../.env`.
   - Reads `PATH_DIR` and specifies paths for the source connectome matrix CSV
     (`data/connectome.csv`) and the destination export directory (`PathToMyFolder`).

2. Perturbation Configuration:
   - Sets perturbation mode flags (`do_shuffling`, `do_ablation`) and repeat count (`n_repeats`).
   - Configures targeted ablation specifications: supports pathway-specific synapse removal
     (e.g., knocking out a given count of recurrent ipsilateral LiMI-LiMI or RiMI-RiMI synapses),
     single-neuron silencing across populations, or global network-wide percentage ablations.

3. Stochastic Generation Loop:
   - Loads base adjacency matrix and population metadata via `get_W`.
   - For each repeat iteration:
     - Applies sign-preserving random edge permutation if `do_shuffling` is enabled.
     - Performs targeted index extraction, sub-block slicing (`np.ix_`), and zero-out ablations
       for designated neurons or active synaptic connections if `do_ablation` is enabled.
     - Serializes the altered structural mask to CSV via `update_connectome` with a unique timestamped filename.
"""

import numpy as np

from datetime import datetime
from pathlib import Path
from dotenv import dotenv_values

# Manually add root path for imports to improve interoperability across parent modules
import sys; sys.path.insert(0, "..")

from utils.load_connectome import get_W, update_connectome

# ------------------------------------------------
# Paths
# ------------------------------------------------
# When calling the script you can provide the path to the .env file as argument.
# If not, the default relative path ../.env is used.
try:
    env_path = sys.argv[1]
except IndexError:
    env_path = "../.env"
env = dotenv_values(env_path)

# Retrieve project root directory from environment variables and resolve IO paths
path_dir = Path(env["PATH_DIR"])
path_W_csv = path_dir / "data" / "connectome.csv"
path_save = path_dir / "data" / "PathToMyFolder"

# ------------------------------------------------
# Configuration
# ------------------------------------------------
# Feature toggles controlling structural randomization and targeted lesion protocols
do_shuffling = True
do_ablation = False

# Global scaling multiplier for ablation counts and number of stochastic lesion repeats
ablate_rate = 0.1
n_repeats = 300
# Registry of targeted ablation instructions specifying cell types, pathways, counts, and modes
ablate_config_list = [
  # {"population_label": "RMON",
  #  "population_index": 6,
  #  "n_ablate": 2,
  #  "mode": "neuron"},
  # {"population_label": "RcMI",
  #  "population_index": 5,
  #  "n_ablate": 2,
  #  "mode": "neuron"},
  # {"population_label": "LiMI-LiMI",
  #  "population_from": "LiMI",
  #  "population_to": "LiMI",
  #  "n_ablate": int(22*ablate_rate),
  #  "mode": "synapse"},
  # {"population_label": "RiMI-RiMI",
  #  "population_from": "RiMI",
  #  "population_to": "RiMI",
  #  "n_ablate": int(29*ablate_rate),
  #  "mode": "synapse"},
  {"population_label": "all",
   "ablate_rate": ablate_rate,
   "mode": "synapse"}
]

# ------------------------------------------------
# Utils
# ------------------------------------------------
# Randomly zeros out an entire row of a 2D array, representing the complete silencing of an incoming/outgoing unit
def zero_random_cells(arr, cell_idx_list, percentage, copy=True):
    out = arr.copy() if copy else arr
    num_cells = len(cell_idx_list)
    k = int(round(num_cells * (percentage / 100.0 if percentage > 1.0 else percentage)))

    rng = np.random.default_rng()
    zero_rows = rng.choice(cell_idx_list, size=k, replace=False)

    out[zero_rows, :] = 0
    return out

# Randomly selects a fraction of non-zero elements and sets them to zero, representing random synapse loss
def zero_random_nonzero_entries(arr, percentage, copy=True):
    out = arr.copy() if copy else arr

    # Locate all non-zero elements
    nz_indices = np.flatnonzero(out)
    num_nz = len(nz_indices)

    # Determine the number of entries to remove
    pct_fraction = percentage / 100.0 if percentage > 1.0 else percentage
    k = int(round(num_nz * pct_fraction))

    if k == 0:
        return out

    # Sample and zero out selected positions
    rng = np.random.default_rng()
    chosen_idx = rng.choice(nz_indices, size=k, replace=False)
    out.flat[chosen_idx] = 0

    return out

# ------------------------------------------------
# Loop over all repeats
# ------------------------------------------------
# Load reference connectivity matrix and anatomical neuron indices from the empirical CSV
W_norm, dict_neurons = get_W(path_W_csv, drop_axons=False, drop_unknown_cells=True,
                             do_symmetry_transform=False, is_W_csv_datavis_ready_transposed=True)

# Extract binary connectivity mask and initialize input weight mask
mask_W = dict_neurons["W_mask"]
mask_W_init = mask_W.copy()

# Extract indices corresponding to cells
cell_idx = np.array(dict_neurons["cell_indices"])
recurrent_sub_idx = np.ix_(cell_idx, cell_idx)

# Generate independent perturbed structural masks across replicates
for i_repeat in range(n_repeats):
    mask_W = mask_W_init.copy()
    model_label = i_repeat
    print(f"Manipulating connectome, repeat: {model_label}")

    # Randomly permute non-zero connection topology across the entire matrix while preserving biological Dale signs
    if do_shuffling:
        recurrent_sub = mask_W[recurrent_sub_idx]
        mask_W_shuffle = np.random.permutation(recurrent_sub.ravel()).reshape(recurrent_sub.shape)
        sign_sub = dict_neurons["W_sign"][np.ix_(cell_idx, cell_idx)]
        mask_W[np.ix_(cell_idx, cell_idx)] = np.abs(mask_W_shuffle) * sign_sub


    # Apply lesion protocols: transpose to align with [to, from] target indexing convention
    if do_ablation:
        mask_W = mask_W.T
        for ac in ablate_config_list:
            # Handle unselective global network-wide ablations strictly across somatic cells
            if ac["population_label"].lower() == "all":
                recurrent_sub = mask_W[recurrent_sub_idx]
                if ac["mode"] == "neuron":
                    mask_W = zero_random_cells(recurrent_sub, cell_idx, ac["ablate_rate"])
                elif ac["mode"] == "synapse":
                    mask_W[np.ix_(cell_idx, cell_idx)] = zero_random_nonzero_entries(recurrent_sub, ac["ablate_rate"])
            # Handle pathway- or population-specific targeted ablations
            else:
                # Knock out individual neurons: removes all incoming/outgoing connections and input drives
                if ac["mode"] == "neuron":
                    side = ac["population"][0]
                    cell = ac["population"][1:]
                    ablate_index_neuron = np.random.choice(dict_neurons["neurons"][side][cell]["idx_list"],
                                                           size=ac["n_ablate"], replace=False)
                    mask_W[ablate_index_neuron, :] = 0
                    mask_W[:, ablate_index_neuron] = 0
                # Knock out specific synaptic connections between designated source and target subpopulations
                elif ac["mode"] == "synapse":
                    side_from = ac["population_from"][0]
                    cell_from = ac["population_from"][1:]
                    pop_idx_from = dict_neurons["neurons"][side_from][cell_from]["idx_list"]
                    side_to = ac["population_to"][0]
                    cell_to = ac["population_to"][1:]
                    pop_idx_to = dict_neurons["neurons"][side_to][cell_to]["idx_list"]
                    # Extract submatrix for the specific pathway
                    mask_W_fromto = np.copy(mask_W[np.ix_(pop_idx_to, pop_idx_from)])
                    active_synapse_index = np.nonzero(mask_W_fromto)
                    if len(active_synapse_index[0]) == 0:
                        continue
                    # Sample non-zero synaptic entries to ablate
                    ablate_index_active_synapse_list = np.random.choice(np.arange(len(active_synapse_index[0])),
                                                                        size=ac["n_ablate"],
                                                                        replace=ac["n_ablate"] > len(active_synapse_index[0]))
                    for synapse_index in ablate_index_active_synapse_list:
                        mask_W_fromto[active_synapse_index[0][synapse_index], active_synapse_index[1][synapse_index]] = 0
                    # Reinsert lesioned submatrix back into full mask
                    mask_W[np.ix_(pop_idx_to, pop_idx_from)] = mask_W_fromto
        # Transpose back to original orientation
        mask_W = mask_W.T

    # Construct unique timestamped output filename
    model_name = f"connectivity_mask_{i_repeat:03}-{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.csv"

    # Export modified connectivity mask to destination CSV file
    update_connectome(path_W_csv, np.abs(mask_W.T), path_save=path_save / model_name, drop_axons=False)