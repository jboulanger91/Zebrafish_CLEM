import pickle
from datetime import datetime

import torch
import numpy as np
from pathlib import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter

from plot.style import RNNDSStyle
from utils.ds_service import DSService
from utils.operators import integrate, get_hist, pid
from utils.rnn_service import RNNService
from utils.train_batch import TrainSignal
from utils.figure_helper import Figure

# ------------------------------------------------
# Paths
# ------------------------------------------------
path_dir = Path(r"C:\Users\Roberto\Desktop\highlights\clem_rnns\data")   # directory containing model_X.pkl
path_noise_estimation = path_dir / "noise_estimation" / "contralateral_motion_integrator_preferred_noise_estimation.pkl"
path_models = path_dir / "results" / "mask_traces_freeneurons_1_attempt2" / "RNNFreeNeurons_neurons102_tau0.1_input2step_softplus" / "top_5"  # RNNConstrainedMask_neurons86_tau0.2_input2step_elu"
path_traces = path_dir
path_save = path_models / "ablation" / f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}"

# ------------------------------------------------
# Configuration
# ------------------------------------------------
ablate_config_list = [
  # {"population_label": "RMON",
  #  "population_index": 6,
  #  "n_ablate": 2,
  #  "mode": "neuron"},
  {"population_label": "RcMI",
   "population_index": 5,
   "n_ablate": 2,
   "mode": "neuron"},
  # {"population_label": "LiMI-LiMI",
  #  "population_from_index": 0,
  #  "population_to_index": 0,
  #  "n_ablate": 5,
  #  "mode": "synapse"}
]

# ------------------------------------------------
# Loop over all trained models
# ------------------------------------------------
for path_model in path_models.glob(f"model_top*.pkl"):
    model_label = path_model.name.split("_")[1]
    print(f"Ablating model {model_label}")

    # Load model instance
    with open(path_model, 'rb') as f:
        model = pickle.load(f)
    model.eval()

    # Extract mask
    mask_W = model.mask_W.detach().clone().numpy()
    mask_U = model.mask_U.detach().clone().numpy()

    for ac in ablate_config_list:
        if ac["mode"] == "neuron":
            ablate_index_neuron = np.random.choice(model.anchor_indices_by_pop[ac["population_index"]],
                                            size=ac["n_ablate"], replace=False)
            mask_W[ablate_index_neuron, :] = 0
            mask_W[:, ablate_index_neuron] = 0
            mask_U[ablate_index_neuron] = 0
        elif ac["mode"] == "synapse":
            mask_W_fromto = np.copy(mask_W[np.ix_(model.anchor_indices_by_pop[ac["population_to_index"]], model.anchor_indices_by_pop[ac["population_from_index"]])])
            active_synapse_index = np.nonzero(mask_W_fromto)
            if len(active_synapse_index) == 0:
                continue
            ablate_index_active_synapse_list = np.random.choice(np.arange(len(active_synapse_index[0])), size=ac["n_ablate"], replace=False)
            for synapse_index in ablate_index_active_synapse_list:
                mask_W_fromto[active_synapse_index[0][synapse_index], active_synapse_index[1][synapse_index]] = 0
            mask_W[np.ix_(model.anchor_indices_by_pop[ac["population_to_index"]], model.anchor_indices_by_pop[ac["population_from_index"]])] = mask_W_fromto

    model.mask_W = torch.tensor(mask_W).type_as(model.mask_W)
    model.mask_U.data = torch.tensor(mask_U).type_as(model.mask_U)

    ac_label = "_".join([f"{ac['population_label']}-{ac['n_ablate']}-{ac['mode']}" for ac in ablate_config_list])
    model_name = f"model_{model_label}_{ac_label}.pkl"
    path_save.mkdir(parents=True, exist_ok=True)
    with open(path_save / model_name, 'wb') as f:
        pickle.dump(model, f)
