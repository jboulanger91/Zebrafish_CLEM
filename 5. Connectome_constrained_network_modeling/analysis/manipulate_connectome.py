import numpy as np

from datetime import datetime
from pathlib import Path
from dotenv import dotenv_values

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "..")

from analysis.load_synapse_matrix import get_W, update_W

# ------------------------------------------------
# Paths
# ------------------------------------------------
try:
    env_path = sys.argv[1]
except IndexError:
    env_path = "../.env"
env = dotenv_values(env_path)

path_dir = Path(env["PATH_DIR"])
path_W_csv = path_dir / "data" / "connectome.csv"
path_save = path_dir / "PathToMyFolder"

# ------------------------------------------------
# Configuration
# ------------------------------------------------
do_shuffling = False
do_ablation = True

ablate_rate = 1
n_repeats = 300
ablate_config_list = [
  # {"population_label": "RMON",
  #  "population_index": 6,
  #  "n_ablate": 2,
  #  "mode": "neuron"},
  # {"population_label": "RcMI",
  #  "population_index": 5,
  #  "n_ablate": 2,
  #  "mode": "neuron"},
  {"population_label": "LiMI-LiMI",
   "population_from": "LiMI",
   "population_to": "LiMI",
   "n_ablate": int(22*ablate_rate),
   "mode": "synapse"},
  {"population_label": "RiMI-RiMI",
   "population_from": "RiMI",
   "population_to": "RiMI",
   "n_ablate": int(29*ablate_rate),
   "mode": "synapse"},
  # {"population_label": "all",
  #  "ablate_rate": ablate_rate,
  #  "mode": "synapse"}
]

# ------------------------------------------------
# Utils
# ------------------------------------------------
def zero_random_rows(arr, percentage, copy=True):
    out = arr.copy() if copy else arr
    num_rows = out.shape[0]
    k = int(round(num_rows * (percentage / 100.0 if percentage > 1.0 else percentage)))

    rng = np.random.default_rng()
    zero_rows = rng.choice(num_rows, size=k, replace=False)

    out[zero_rows, :] = 0
    return out

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
W_norm, dict_neurons = get_W(path_W_csv, do_symmetry_transform=False, is_W_csv_datavis_ready_transposed=True)

# Extract mask
mask_W = dict_neurons["W_mask"]
mask_W_init = mask_W.copy()
mask_U = np.ones(mask_W.shape[0])

for i_repeat in range(n_repeats):
    mask_W = mask_W_init.copy()
    model_label = i_repeat
    print(f"Manipulating connectome, repeat: {model_label}")

    if do_shuffling:
        mask_W_shuffle = np.random.permutation(mask_W.ravel()).reshape(mask_W.shape)
        mask_W_shuffle = np.abs(mask_W_shuffle) * dict_neurons["W_sign"]
        mask_W = mask_W_shuffle

    if do_ablation:
        mask_W = mask_W.T
        for ac in ablate_config_list:
            if ac["population_label"].lower() == "all":
                if ac["mode"] == "neuron":
                    mask_W = zero_random_rows(mask_W, ac["ablate_rate"])
                elif ac["mode"] == "synapse":
                    mask_W = zero_random_nonzero_entries(mask_W, ac["ablate_rate"])
            else:
                if ac["mode"] == "neuron":
                    side = ac["population"][0]
                    cell = ac["population"][1:]
                    ablate_index_neuron = np.random.choice(dict_neurons["neurons"][side][cell]["idx_list"],
                                                           size=ac["n_ablate"], replace=False)
                    mask_W[ablate_index_neuron, :] = 0
                    mask_W[:, ablate_index_neuron] = 0
                    mask_U[ablate_index_neuron] = 0
                elif ac["mode"] == "synapse":
                    side_from = ac["population_from"][0]
                    cell_from = ac["population_from"][1:]
                    pop_idx_from = dict_neurons["neurons"][side_from][cell_from]["idx_list"]
                    side_to = ac["population_to"][0]
                    cell_to = ac["population_to"][1:]
                    pop_idx_to = dict_neurons["neurons"][side_to][cell_to]["idx_list"]
                    mask_W_fromto = np.copy(mask_W[np.ix_(pop_idx_to, pop_idx_from)])
                    active_synapse_index = np.nonzero(mask_W_fromto)
                    if len(active_synapse_index) == 0:
                        continue
                    ablate_index_active_synapse_list = np.random.choice(np.arange(len(active_synapse_index[0])),
                                                                        size=ac["n_ablate"],
                                                                        replace=ac["n_ablate"] > len(active_synapse_index[0]))
                    for synapse_index in ablate_index_active_synapse_list:
                        mask_W_fromto[active_synapse_index[0][synapse_index], active_synapse_index[1][synapse_index]] = 0
                    mask_W[np.ix_(pop_idx_to, pop_idx_from)] = mask_W_fromto
        mask_W = mask_W.T

    model_name = f"connectivity_mask_{i_repeat:03}-{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.csv"

    update_W(path_W_csv, np.abs(mask_W.T), path_save=path_save / model_name)

