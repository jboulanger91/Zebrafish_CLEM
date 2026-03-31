import pickle

import torch
import numpy as np
from pathlib import Path

from dotenv import dotenv_values

# ------------------------------------------------
# Env variables and paths
# ------------------------------------------------
env = dotenv_values()
path_dir = Path(env["PATH_DIR"])
path_noise_estimation = Path("../noise_estimation.pkl")
path_models = Path(env["PATH_MODELS"])   # directory containing model_X.pkl
path_save = path_models

# ------------------------------------------------
# Configuration
# ------------------------------------------------
TOP_AMOUNT = 10
mode_select_top = "percentage"  # "count"  #
save_selected_models = True
select_models = "top"  # "median"

# ------------------------------------------------
# Loop over all trained models
# ------------------------------------------------
loss_list = []
model_path_list = []
i_model = 0
for path_model in path_models.glob(f"model_*.pkl"):
    print(f"Evaluating model {i_model}")
    i_model += 1

    model_path_list.append(path_model)
    # Load model instance
    with open(path_model, 'rb') as f:
        model = pickle.load(f)
    model.eval()

    x0 = torch.zeros(model.n_units)
    try:
        loss = model.loss_mse
    except AttributeError:
        continue

    loss_list.append(loss)


N_MODELS = len(loss_list)

# Select top-performant models
if mode_select_top.lower().startswith("perc"):
    perc_selected = TOP_AMOUNT / 100
    num_selected = int(perc_selected * N_MODELS)
elif mode_select_top.lower() == "count":
    perc_selected = TOP_AMOUNT / N_MODELS
    num_selected = TOP_AMOUNT
else:
    raise Exception(f"Configuration mode_select_top must have value 'percentage' or 'count'. {mode_select_top} was provided.")

if select_models == "median":
    loss_quantiles = np.quantile(loss_list, [0.5-perc_selected/2, 0.5+perc_selected/2])
    selected_indices = np.squeeze(np.argwhere(np.logical_and(loss_quantiles[0] <= np.array(loss_list), np.array(loss_list) <= loss_quantiles[1])))
else:
    selected_indices = np.argsort(loss_list)[:num_selected]  # by default take the top N models

top_label = 0
for i in selected_indices:
    with open(model_path_list[i], 'rb') as f:
        model = pickle.load(f)
    model.eval()

    # Save trained model
    model_name_split = model_path_list[i].name.replace(".pkl", "").split('_')
    model_name_top = f"model_top{top_label}_{model_name_split[1]}_{model_name_split[2]}.pkl"
    path_save_top_model = path_models / f"{select_models}_{num_selected}"
    path_save_top_model.mkdir(parents=True, exist_ok=True)
    with open(path_save_top_model / model_name_top, 'wb') as f:
        pickle.dump(model, f)
    top_label += 1
