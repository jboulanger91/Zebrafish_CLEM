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

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
n_models = 300

is_control = True
debug = False
verbose = False

env_path = "../../.env"
env = dotenv_values(env_path)

# --------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------
def count_loops_over_quadruplets(A):
    """
    Count loops of length 1 (self-loop), 2 (mutual edge pair), or 3
    (directed triangle) summed across all possible 4-node quadruplets
    of a directed 0/1 connectivity matrix A (N x N), without ever
    enumerating quadruplets explicitly.

    Returns a dict with raw elementary loop counts and the total
    weighted by how many quadruplets each loop participates in.
    """
    A = np.array(A, dtype=np.int64)
    N = A.shape[0]
    if N < 4:
        raise ValueError("Need at least 4 nodes to form quadruplets.")

    diag = np.diag(A).copy()
    A0 = A.copy()
    np.fill_diagonal(A0, 0)  # off-diagonal-only graph

    # Elementary loop counts (each computed once, not per-quadruplet)
    L1 = int(diag.sum())                                    # self-loops
    L2 = int(np.sum(np.triu(A0 * A0.T, k=1)))                # mutual (2-node) loops
    L3 = int(round(np.trace(A0 @ A0 @ A0) / 3))               # directed 3-cycles

    # Number of quadruplets containing a fixed 1-, 2-, or 3-node loop
    w1 = comb(N - 1, 3)
    w2 = comb(N - 2, 2)
    w3 = comb(N - 3, 1)
    total_quadruplets = comb(N, 4)

    incidence_total = L1 * w1 + L2 * w2 + L3 * w3

    return {
        "self_loops": L1,
        "mutual_pairs": L2,
        "directed_triangles": L3,
        "weight_per_self_loop": w1,
        "weight_per_pair": w2,
        "weight_per_triangle": w3,
        "total_distinct_loops": L1 + L2 + L3,
        "quadruplet_incidence_total": incidence_total,  # sum over all quadruplets of loops-per-quadruplet
        "total_quadruplets": total_quadruplets,
        "avg_loops_per_quadruplet": incidence_total / total_quadruplets,
    }

def analyze_selected_neurons(A, selected_indices):
    """
    For each selected neuron (row index) in a directed connectivity matrix A,
    where A[i, j] == 1 means neuron i projects onto neuron j:

      - presynaptic partners: neurons that project onto the selected neuron
      - postsynaptic partners: neurons the selected neuron projects onto (1st order)
      - second-order postsynaptic partners: postsynaptic partners of the 1st-order
        postsynaptic partners (repetitions kept, since a neuron can be reached
        via multiple 1st-order partners)

    Returns a dict keyed by neuron index, each containing the three identity
    lists plus counts of second-order partners and how many of those (with
    repetition) also belong to the presynaptic or 1st-order postsynaptic sets.
    """
    A = np.asarray(A)
    _results = {}

    for i in selected_indices:
        presynaptic = np.where(A[:, i])[0].tolist()       # j -> i
        postsynaptic = np.where(A[i, :])[0].tolist()       # i -> j

        second_order_postsynaptic = []
        for j in postsynaptic:
            second_order_postsynaptic.extend(np.where(A[j, :])[0].tolist())

        other_groups = set(presynaptic) | set(selected_indices) | set(postsynaptic)
        second_order_group = set(second_order_postsynaptic)
        overlap_count = sum(1 for x in second_order_postsynaptic if x in other_groups)
        rediscovered_group = second_order_group & other_groups

        _results[i] = {
            "presynaptic": presynaptic,
            "postsynaptic": postsynaptic,
            "second_order_postsynaptic": second_order_postsynaptic,
            "n_second_order_postsynaptic": len(second_order_postsynaptic),
            "n_second_order_overlap_with_other_groups": overlap_count,
            "rediscovered_group": rediscovered_group
        }
    # results["all"] = \
    results = {
        "presynaptic": set().union(*[_results[i]["presynaptic"] for i in _results.keys()]),
        "n_presynaptic": np.sum([len(_results[i]["presynaptic"]) for i in _results.keys()]),
        "seed": set(selected_indices),
        "n_seed": len(selected_indices),
        "postsynaptic": set().union(*[_results[i]["postsynaptic"] for i in _results.keys()]),
        "n_postsynaptic": np.sum([len(_results[i]["postsynaptic"]) for i in _results.keys()]),
        "second_order_postsynaptic": set().union(*[_results[i]["second_order_postsynaptic"] for i in _results.keys()]),
        "n_second_order_postsynaptic": np.sum([_results[i]["n_second_order_postsynaptic"] for i in _results.keys()]),
        "n_second_order_overlap_with_other_groups": np.sum([_results[i]["n_second_order_overlap_with_other_groups"] for i in _results.keys()]),
        "rediscovered_group": set().union(*[_results[i]["rediscovered_group"] for i in _results.keys()])
    }

    return results

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    path_W_csv = Path(env["PATH_W_CSV"])
    W, dict_neurons = get_W(path_W_csv, do_symmetry_transform=False)
    mask = dict_neurons["W_mask"]
    if is_control:
        W_binary = np.zeros_like(W, dtype=bool)
    else:
        W_binary = W != 0


    W_binary_img_init = np.zeros_like(W_binary, dtype=int)
    W_binary_img_init[W_binary] = 255
    if debug:
        W_binary_img_init *= 255
        W_binary_img_init = W_binary_img_init.T

    for i_model in range(n_models):
        i_pop = 0
        W_binary_model = W_binary.copy()
        for side in ConfigurationRNN.side_list:
            for cell in ConfigurationRNN.cell_list:
                idx_pop = dict_neurons["neurons"][side][cell]["idx_list"]

                # zoom in quadrant defining connectivity between this pop and each other pop
                _i_pop = 0
                for _side in ConfigurationRNN.side_list:
                    for _cell in ConfigurationRNN.cell_list:
                        _idx_pop = dict_neurons["neurons"][_side][_cell]["idx_list"]
                        W_binary_popX = W_binary_model[np.ix_(_idx_pop, idx_pop)]

                        # Compute sparsity
                        W_binary_popX_img = np.zeros_like(W_binary_popX, dtype=int)
                        W_binary_popX_img[W_binary_popX] = 1
                        sparsity_popX = np.sum(W_binary_popX_img) / W_binary_popX_img.size
                        target_sparsity_popX = ConfigurationNeural.P.T[i_pop, _i_pop]

                        while sparsity_popX < target_sparsity_popX:
                            group_inactive_edges = set(map(tuple, np.argwhere(~W_binary_popX)))
                            new_edge = random.choice(list(group_inactive_edges))
                            group_inactive_edges -= {new_edge}
                            W_binary_popX[new_edge] = True
                            W_binary_popX_img[new_edge] = 1
                            sparsity_popX = np.sum(W_binary_popX_img) / W_binary_popX_img.size

                            W_binary_model[np.ix_(_idx_pop, idx_pop)] = W_binary_popX
                        _i_pop += 1
                        print(f"{side}_{cell} - {_side}_{_cell} | sparsity: {sparsity_popX}")
                i_pop += 1

                # Compute network statistics
                recurrency_pop = analyze_selected_neurons(W_binary_model, idx_pop)
                group_to_discover = recurrency_pop["presynaptic"] | recurrency_pop["seed"] | recurrency_pop["postsynaptic"]
                recurrency_rate_pop = len(recurrency_pop["rediscovered_group"]) / len(group_to_discover)
                print(f"{side}_{cell} | recurrency: {recurrency_rate_pop}\n")

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

