import torch
import numpy as np
from pathlib import Path

from dotenv import dotenv_values

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "..")

from utils.load_model import load_model
from utils.services.rnn_service import RNNService


# ------------------------------------------------
# Configuration
# ------------------------------------------------
label_model_dir = "connectome_noLDA_matrix"
n_models_select = 5
mode_select_top = "count"  # Options are: "percentage", "count".  It is used to interpret the value in n_models_select
save_selected_models = True
select_models = "top"  # "median"  #

# ------------------------------------------------
# Env variables and paths
# ------------------------------------------------
env = dotenv_values()
path_dir = Path(env["PATH_DIR"])
path_noise_estimation = path_dir / "data" / "noise_estimation"
path_models = path_dir / "models" / label_model_dir   # directory containing model_X.pkl
path_save = path_models

# ------------------------------------------------
# Loop over all trained models
# ------------------------------------------------
loss_list = []
model_path_list = []
i_model = 0
for path_model in path_models.glob(f"model_*"):
    print(f"Evaluating model {i_model}")
    i_model += 1

    # Load model instance
    model = load_model(path_model)
    model.eval()

    x0 = torch.zeros(model.n_units)
    try:
        raw_loss = model.loss_mse
        if raw_loss is None:
            continue
        loss = float(raw_loss)
        if np.isnan(loss) or np.isinf(loss):
            continue
    except AttributeError:
        continue

    loss_list.append(loss)
    model_path_list.append(path_model)


N_MODELS = len(loss_list)

# Select top-performant models
if mode_select_top.lower().startswith("perc"):
    perc_selected = n_models_select / 100
    num_selected = int(perc_selected * N_MODELS)
elif mode_select_top.lower() == "count":
    perc_selected = n_models_select / N_MODELS
    num_selected = n_models_select
else:
    raise Exception(f"Configuration mode_select_top must have value 'percentage' or 'count'. {mode_select_top} was provided.")

if select_models == "median":
    loss_quantiles = np.quantile(loss_list, [0.5-perc_selected/2, 0.5+perc_selected/2])
    selected_indices = np.squeeze(np.argwhere(np.logical_and(loss_quantiles[0] <= np.array(loss_list), np.array(loss_list) <= loss_quantiles[1])))
else:
    selected_indices = np.argsort(loss_list)[:num_selected]  # by default take the top N models

top_label = 0
for i in selected_indices:
    path_model = model_path_list[i]
    model = load_model(path_model)
    model.eval()

    # ── Build checkpoint ────────────────────────────────────────────────────
    checkpoint = {
        "state_dict": model.state_dict(),
        "custom_attrs": RNNService.extract_custom_attrs(model),
        "class_name": type(model).__name__,  # useful as a sanity check
        "selection_loss_mse": model.loss_mse,
    }
    if type(model).__name__ in ["RNNFixedConnectivity", "RNNConnectome"]:
        checkpoint["dict_neurons"] = model.dict_neurons

    # Save trained model
    model_name_split = model_path_list[i].name.replace(".pkl", "").split('_')
    model_name_top = f"model_top{top_label}_{model_name_split[1]}_{model_name_split[2]}.pt"
    path_save_top_model = path_models / f"{select_models}_{num_selected}"
    path_save_top_model.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path_save_top_model / model_name_top)
    top_label += 1
