"""
Overview
--------
This script scans a directory of trained recurrent neural network (RNN) model checkpoints,
ranks them by training performance (mean squared error, loss_mse), and extracts a representative
subset—either the top-performing models or the median-performing cohort—into a dedicated
subfolder for downstream evaluation and figure generation.

Before running it, please make sure you have set up your .env with all the necessary variables:
- PATH_DIR="/path/to/project_root"  # Root project directory containing models/ and data/

Core Pipeline & Workflow:
1. Environment & Path Setup:
   - Loads the project root path (`PATH_DIR`) from the `.env` file.
   - Points to the target model folder specified by `label_model_dir` (e.g. `connectome_noLDA_matrix`).

2. Model Harvesting & Performance Parsing:
   - Iterates through all checkpoint files matching the pattern `model_*`.
   - Loads each model via `load_model`, validates numerical integrity, and extracts `loss_mse`
     (skipping invalid, NaN, infinite, or missing loss entries).
   - Records valid loss values alongside their respective file paths.

3. Model Selection:
   - Interprets `n_models_select` either as an absolute count ("count") or a percentage ("percentage").
   - Identifies candidate models:
     - "top": Selects the lowest-loss models via `np.argsort`.
     - "median": Selects models falling within a central quantile window around the median.

4. Checkpoint Packaging & Export:
   - Re-loads each selected model and extracts state dictionaries, custom attributes,
     neuron metadata (`dict_neurons`), and selection loss metrics.
   - Saves standard PyTorch checkpoints (`.pt`) into a structured subfolder
     (e.g. `top_5` or `median_5`) with standardized indexed filenames.
"""

import torch
import numpy as np
from pathlib import Path

from dotenv import dotenv_values

# Manually add root path for imports to improve interoperability across parent modules
import sys; sys.path.insert(0, "..")

from utils.load_model import load_model
from utils.services.rnn_service import RNNService


# ------------------------------------------------
# Configuration
# ------------------------------------------------
# Subdirectory within models/ containing trained model instances to evaluate
label_model_dir = "connectome"
# Number or percentage of models to select
n_models_select = 5
# Interpretation mode for n_models_select: "count" (fixed number) or "percentage" (fraction of total)
mode_select_top = "count"  # Options are: "percentage", "count".  It is used to interpret the value in n_models_select
# Flag controlling whether the selected model checkpoints are re-saved to disk
save_selected_models = True
# Selection criterion: "top" for best-performing models, "median" for typical models around the median loss
select_models = "top"  # "median"  #

# ------------------------------------------------
# Env variables and paths
# ------------------------------------------------
# Load root filesystem directory from environment configuration (.env)
env = dotenv_values()
path_dir = Path(env["PATH_DIR"])
path_noise_estimation = path_dir / "data" / "noise_estimation"
path_models = path_dir / "models" / label_model_dir   # directory containing model_X.pkl
path_save = path_models

# ------------------------------------------------
# Loop over all trained models
# ------------------------------------------------
# Collect loss values and matching file paths across all model checkpoints in the directory
loss_list = []
model_path_list = []
i_model = 0
for path_model in path_models.glob(f"model_*"):
    print(f"Evaluating model {i_model}")
    i_model += 1

    # Load model instance
    model = load_model(path_model)
    model.eval()

    # Verify model structure and extract mean squared error loss
    x0 = torch.zeros(model.n_units)
    try:
        raw_loss = model.loss_mse
        if raw_loss is None:
            continue
        loss = float(raw_loss)
        # Filter out corrupted or non-converged runs
        if np.isnan(loss) or np.isinf(loss):
            continue
    except AttributeError:
        continue

    loss_list.append(loss)
    model_path_list.append(path_model)


# Total number of successfully parsed models
N_MODELS = len(loss_list)

# ------------------------------------------------
# Select top-performant models
# ------------------------------------------------
# Resolve selection threshold based on percentage or explicit count
if mode_select_top.lower().startswith("perc"):
    perc_selected = n_models_select / 100
    num_selected = int(perc_selected * N_MODELS)
elif mode_select_top.lower() == "count":
    perc_selected = n_models_select / N_MODELS
    num_selected = n_models_select
else:
    raise Exception(f"Configuration mode_select_top must have value 'percentage' or 'count'. {mode_select_top} was provided.")

# Determine target index subset using either median-centered quantiles or ascending sort
if select_models == "median":
    # Identify models within symmetric bounds around the 50th percentile
    loss_arr = np.asarray(loss_list)
    loss_quantiles = np.quantile(loss_list, [0.5-perc_selected/2, 0.5+perc_selected/2])
    mask = (loss_quantiles[0] <= loss_arr) & (loss_arr <= loss_quantiles[1])
    selected_indices = np.flatnonzero(mask)
    # if the interval is too small, just take the median
    if selected_indices.size < 1:
        selected_indices = [np.argsort(loss_list)[len(loss_list)//2]]
else:
    # Default: take the lowest loss values (best training performance)
    selected_indices = np.argsort(loss_list)[:num_selected]  # by default take the top N models

# ------------------------------------------------
# Save selected model checkpoints
# ------------------------------------------------
top_label = 0
for i in selected_indices:
    path_model = model_path_list[i]
    model = load_model(path_model)
    model.eval()

    # ------------------------------------------------
    # Build checkpoint
    # ------------------------------------------------
    # Package parameters, state dictionaries, and metadata into a standard checkpoint dictionary
    checkpoint = {
        "state_dict": model.state_dict(),
        "custom_attrs": RNNService.extract_custom_attrs(model),
        "class_name": type(model).__name__,  # useful as a sanity check
        "selection_loss_mse": model.loss_mse,
    }
    # Include anatomical cell-type mapping dictionary if present on the architecture
    checkpoint["dict_neurons"] = model.dict_neurons

    # Format destination filename preserving original run identifiers and assigning a rank prefix
    model_name_split = model_path_list[i].name.replace(".pkl", "").split('_')
    model_name_top = f"model_top{top_label}_{model_name_split[1]}_{model_name_split[2]}.pt"
    # Destination directory named by selection method and sample count (e.g. top_5 or median_5)
    path_save_top_model = path_models / f"{select_models}_{num_selected}"
    path_save_top_model.mkdir(parents=True, exist_ok=True)

    # Save packaged PyTorch checkpoint to target destination
    torch.save(checkpoint, path_save_top_model / model_name_top)
    top_label += 1