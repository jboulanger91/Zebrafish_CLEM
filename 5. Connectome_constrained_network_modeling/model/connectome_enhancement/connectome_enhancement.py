import random
from datetime import datetime

import numpy as np
from math import comb
from pathlib import Path
from dotenv import dotenv_values

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "../..")

from analysis.load_synapse_matrix import get_W, update_W
from utils.config import ConfigurationRNN, ConfigurationNeural
from utils.services.rnn_service import RNNService


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
n_models = 499
cell_seed_recurrency = "iMI"
target_recurrency_rate_pop = 0.36

augment_sparsity_sidemax = True
augment_recurrency = False
augment_sparsity_scaleup = False

debug = False
verbose = False

env_path = "../../.env"
env = dotenv_values(env_path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":

    path_W_csv = Path(env["PATH_W_CSV"])
    W, dict_neurons = get_W(path_W_csv, do_symmetry_transform=False)
    mask = dict_neurons["W_mask"]
    W_binary = W!=0

    pop_idx_list = [dict_neurons["neurons"][side][cell]["idx_list"] \
                    for side in dict_neurons["neurons"].keys() for cell in dict_neurons["neurons"][side].keys() if
                    cell != "idx_list"]

    W_binary_img_init = np.zeros_like(W_binary, dtype=int)
    W_binary_img_init[W_binary] = 1

    if verbose:
        print("Before augmentation")
        for side in ConfigurationRNN.side_list:
            idx_pop = dict_neurons["neurons"][side][cell_seed_recurrency]["idx_list"]
            recurrency_pop = RNNService.check_connectivity_selected_neurons(W_binary_img_init, idx_pop)
            group_to_discover = recurrency_pop["presynaptic"] | recurrency_pop["seed"] | recurrency_pop["postsynaptic"]
            recurrency_rate_pop = len(recurrency_pop["rediscovered_group"]) / len(group_to_discover)
            print(f"{side}_{cell_seed_recurrency} | recurrency: {recurrency_rate_pop}\n")

    if debug:
        W_binary_img_init *= 255
        W_binary_img_init = W_binary_img_init.T

    for i_model in range(n_models):
        print(f"\nAugmenting model {i_model}...")
        W_binary_model = W_binary.copy()

        if augment_sparsity_sidemax:
            # Augment by side-maxing sparsity
            W_binary_model = RNNService.augment_connectivity_sparsity(W_binary_model, dict_neurons=dict_neurons, target_sparsity=ConfigurationNeural.P_balanced)

        if verbose:
            print("After sparsity-balanced augmentation")
            for side in ConfigurationRNN.side_list:
                idx_pop = dict_neurons["neurons"][side][cell_seed_recurrency]["idx_list"]
                recurrency_pop = RNNService.check_connectivity_selected_neurons(W_binary_model, idx_pop)
                group_to_discover = recurrency_pop["presynaptic"] | recurrency_pop["seed"] | recurrency_pop["postsynaptic"]
                recurrency_rate_pop = len(recurrency_pop["rediscovered_group"]) / len(group_to_discover)
                print(f"{side}_{cell_seed_recurrency} | recurrency: {recurrency_rate_pop}\n")

        if augment_sparsity_scaleup:
            # Compute baseline sparsity iMI before recurrency enhancement
            sparsity_iMI_init = RNNService.compute_sparsity_cell_sides(W_binary_model, dict_neurons, cell_seed_recurrency)

        if augment_recurrency:
            # augment by recurrency constraint
            W_binary_model = RNNService.augment_connectivity_recurrency(W_binary_model, dict_neurons=dict_neurons)

        if augment_sparsity_scaleup:
            sparsity_iMI = RNNService.compute_sparsity_cell_sides(W_binary_model, dict_neurons, cell_seed_recurrency)
            if np.any(sparsity_iMI > sparsity_iMI_init):
                scaleup_sparsity_iMI = sparsity_iMI / sparsity_iMI_init
                scaleup_sparsity = np.max(scaleup_sparsity_iMI)
                if verbose:
                    print("Scale-up connection probability: ", scaleup_sparsity)

                    print(f"Recurrency iMI after recurrency augmentation: {recurrency_rate_pop}")
                    print(f"Sparsity iMI after recurrency augmentation: {sparsity_iMI}")

                # augment scaleup sparsity
                W_binary_model = RNNService.augment_connectivity_sparsity(W_binary_model, dict_neurons=dict_neurons,
                                                                          target_sparsity=ConfigurationNeural.P_balanced * scaleup_sparsity)
            else:
                print("WARNING | After augmentation connection probability for iMI did not increase. Connection probability scale-up skipped.")

        W_binary_img = np.zeros_like(W_binary_model, dtype=int)
        W_binary_img[W_binary_model] = 1
        if debug:
            W_binary_img *= 255
            W_binary_img = W_binary_img.T

        # Store enhanced connectome
        if debug:
            W_binary_out = W_binary_img.T.copy().astype(float)
            W_binary_out /= np.max(W_binary_img)
        else:
            W_binary_out = W_binary_img.copy()
        update_W(path_W_csv, W_binary_out, path_save=Path(env["PATH_SAVE"]) / f"connectivity_mask_{i_model:03}-{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.csv")

        print(f"Augmented model {i_model} saved.")

