'''
Script used to compute the population-pair connection probability in the
recorded connectome. The results of this script can be found in config.py
'''

import numpy as np
from dotenv import dotenv_values
from pathlib import Path

# Manually add root path for imports to improve interoperability
import sys;

from matplotlib import pyplot as plt

sys.path.insert(0, "..")

from analysis.load_synapse_matrix import get_W
from utils.load_model import load_model
from utils.services.rnn_service import RNNService
from utils.config import ConfigurationRNN, ConfigurationNeural


# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
generate_random_matrices = False
use_connectome = False
use_synthetic = False
show_plot = True

env_path = "../.env"
env = dotenv_values(env_path)

# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
if __name__ == "__main__":
    if use_synthetic:
        W_list = []
        try:
            path_models = Path(env["PATH_MODELS"])
            for path_model in path_models.glob("model_*.pt"):
                model = load_model(path_model)
                W = model.W().detach().clone()
                W_list.append(W)
        except KeyError:
            path_model = Path(env["PATH_MODEL"])
            model = load_model(path_model)
            W = model.W().detach().clone()
            W_list.append(W)
    else:
        path_W_csv = Path(env["PATH_W_CSV"])
        W, dict_neurons = get_W(path_W_csv, do_symmetry_transform=False)
        mask = dict_neurons["W_mask"]

        pop_idx_list = [dict_neurons["neurons"][side][cell]["idx_list"] \
                        for side in dict_neurons["neurons"].keys() for cell in dict_neurons["neurons"][side].keys() if cell != "idx_list"]

        P_ = np.zeros((len(pop_idx_list), len(pop_idx_list)))
        sparsity_ = np.zeros((len(pop_idx_list), len(pop_idx_list)))
        W_binary = np.abs(np.sign(W))
        W_ = W_binary
        for i, pop_idx in enumerate(pop_idx_list):
            for i_, pop_idx_ in enumerate(pop_idx_list):
                block = W_[np.ix_(pop_idx, pop_idx_)]
                P_[i, i_] = np.mean(block)
                sparsity_[i, i_] = np.sum(block) / np.size(block)
        P = P_.T
        sparsity = sparsity_.T

        # Plot sparsity
        fig, ax = plt.subplots()
        im = ax.imshow(sparsity, cmap="Blues")
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(range(len(pop_idx_list)), labels=ConfigurationNeural.POP_META["name"])
        ax.set_yticks(range(len(pop_idx_list)), labels=ConfigurationNeural.POP_META["name"])
        # Loop over data dimensions and create text annotations.
        for i in range(len(pop_idx_list)):
            for j in range(len(pop_idx_list)):
                text = ax.text(j, i, f"{sparsity[i, j]:.02f}",
                               ha="center", va="center", color="w")
        ax.set_title("Connection probability")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.show()

        # Compute sparsity side-balanced (take side-max for each population pair)
        sparsity_balanced_ = np.zeros((len(pop_idx_list), len(pop_idx_list)))
        for i, pop_idx in enumerate(pop_idx_list[:4]):
            for i_, pop_idx_ in enumerate(pop_idx_list[:4]):
                block = W_[np.ix_(pop_idx, pop_idx_)]
                block_contra = W_[np.ix_(pop_idx_list[i+4], pop_idx_list[i_+4])]
                sparsity_block = np.sum(block) / np.size(block)
                sparsity_block_contra = np.sum(block_contra) / np.size(block_contra)
                sparsity_balanced_block = np.max((sparsity_block, sparsity_block_contra))
                sparsity_balanced_[i, i_] = sparsity_balanced_block
                sparsity_balanced_[i+4, i_+4] = sparsity_balanced_block
        for i, pop_idx in enumerate(pop_idx_list[:4]):
            for i_, pop_idx_ in enumerate(pop_idx_list[4:]):
                block = W_[np.ix_(pop_idx, pop_idx_)]
                block_contra = W_[np.ix_(pop_idx_list[i + 4], pop_idx_list[i_])]  # keep in mind i_ contains 0->3, while pop_idx_ contains correspondent to 4->7
                sparsity_block = np.sum(block) / np.size(block)
                sparsity_block_contra = np.sum(block_contra) / np.size(block_contra)
                sparsity_balanced_block = np.max((sparsity_block, sparsity_block_contra))
                sparsity_balanced_[i+4, i_] = sparsity_balanced_block
                sparsity_balanced_[i, i_+4] = sparsity_balanced_block
        sparsity_balanced = sparsity_balanced_.T

        # Plot sparsity balanced
        fig, ax = plt.subplots()
        im = ax.imshow(sparsity_balanced, cmap="Blues")
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(range(len(pop_idx_list)), labels=ConfigurationNeural.POP_META["name"])
        ax.set_yticks(range(len(pop_idx_list)), labels=ConfigurationNeural.POP_META["name"])
        # Loop over data dimensions and create text annotations.
        for i in range(len(pop_idx_list)):
            for j in range(len(pop_idx_list)):
                text = ax.text(j, i, f"{sparsity_balanced[i, j]:.02f}",
                               ha="center", va="center", color="w")
        ax.set_title("Connection probability balanced")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.show()
