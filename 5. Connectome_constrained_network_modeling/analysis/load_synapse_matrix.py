"""
Load a synapse-count connectivity matrix from CSV, drop all "axon..." rows/columns,
and produce a numpy array copy together with an index<->neuron-id mapping.

Expected CSV layout:
- First row: header with neuron identifiers for each column (first cell is a
  label like "presynaptic" for the row-index column).
- First column: neuron identifiers for each row (same set as the columns,
  since this is a square pre->post synapse-count matrix).
- Identifiers look like:
    "axon | ID: 576460752710808710"
    "cell | ID: 576460752631366630 | functional classifier: motion_integrator | ..."
"""
import itertools

import numpy as np
import pandas as pd

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "..")

from utils.config import ConfigurationRNN

def get_idx_side_change(df,
                        identifier_change="functional classifier: motion_integrator",
                        identifier_pre_change="functional classifier: slow_motion_integrator"):
    idx = df.index.astype(str)
    if identifier_pre_change is None:
        pre_positions = [0]
    else:
        pre_positions = [i for i, label in enumerate(idx) if identifier_pre_change in label]
    if not pre_positions:
        return None
    first_pre_pos = pre_positions[0]
    for i in range(first_pre_pos + 1, len(idx)):
        if identifier_change in idx[i]:
            return i  # positional value
    return None

def load_synapse_matrix(csv_path, drop_axons=True, drop_myelinated=True):
    # First column becomes the index automatically because it has no header
    # name aligned with a real column count (typical "presynaptic" style CSV).
    df = pd.read_csv(csv_path, index_col=0)

    # Sanity check: matrix should be square with matching row/col labels.
    # If rows and columns are not identically labeled/ordered, align them.
    if not df.index.equals(df.columns):
        print("WARNING | matrix should be square with matching row/col labels. Aligning them.")
        common = df.index.intersection(df.columns)
        df = df.loc[common, common]

    is_axon = df.index.str.strip().str.startswith("axon")
    axon_labels = df.index[is_axon]
    axon_rostral_idx = [i for i, is_axon_rostral in enumerate(axon_labels.str.strip().str.contains("rostral")) if not is_axon_rostral]
    axon_values = np.array(df.loc[axon_labels].sum(axis=0).to_numpy())
    axon_values[axon_rostral_idx] = 0

    # Identify nodes to discard: anything whose identifier starts with "axon"
    if drop_axons:
        non_axon_labels = df.index[~is_axon]
        non_axon_idx = df.index.get_indexer(non_axon_labels)
        df = df.loc[non_axon_labels, non_axon_labels]
        axon_values = axon_values[non_axon_idx]

    # Identify nodes to discard: anything whose identifier starts with "myelinated"
    if drop_myelinated:
        is_myelinated = df.index.str.strip().str.contains("myelinated")
        non_myelinated_labels = df.index[~is_myelinated]
        non_myelinated_idx = df.index.get_indexer(non_myelinated_labels)
        df = df.loc[non_myelinated_labels, non_myelinated_labels]
        axon_values = axon_values[non_myelinated_idx]

    df_clean = df

    # Identify index for side change after removing axons and myelinated
    idx_side_change = get_idx_side_change(df_clean)

    # Numpy array copy of the cleaned matrix
    matrix = df_clean.to_numpy(dtype=float).copy()

    # Index -> original neuron id mapping (and the reverse)
    idx_to_id = {i: label for i, label in enumerate(df_clean.index)}
    id_to_idx = {label: i for i, label in idx_to_id.items()}

    return matrix, axon_values, idx_to_id, id_to_idx, df_clean, idx_side_change

def process_synapse_matrix(W_raw, idx_to_id, idx_side_change=None):
    # normalize W_raw
    W_sum_neuron = np.sum(W_raw, axis=0)
    W_sum_neuron = np.array([1 if sum_neuron == 0 else sum_neuron for sum_neuron in W_sum_neuron])
    W_norm = W_raw / W_sum_neuron

    # get neurons info
    dict_neurons = {ConfigurationRNN.SIDE_LEFT: {}, ConfigurationRNN.SIDE_RIGHT: {}}
    for i_neuron, info_neuron_str in idx_to_id.items():
        # parse neuron info string
        info_neuron_list = info_neuron_str.split(" | ")
        # get side
        if "hemisphere" in info_neuron_str:
            side = info_neuron_list[2].replace("hemisphere: ", "")
        else:
            if i_neuron < idx_side_change: side = ConfigurationRNN.SIDE_LEFT
            else: side = ConfigurationRNN.SIDE_RIGHT
        # get other features
        function = info_neuron_list[3].replace("functional classifier: ", "")
        projection = info_neuron_list[4].replace("projection classifier: ", "")
        neurotransmitter = info_neuron_list[5].replace("neurotransmitter classifier: ", "")

        pop = ConfigurationRNN.classifier_to_pop_map[function][projection]
        if pop not in dict_neurons[side].keys():
            dict_neurons[side][pop] = {"excitatory": [],
                                       "inhibitory": [],
                                       "unknown": []}
        dict_neurons[side][pop][f"{neurotransmitter}"].append(i_neuron)
    for side in dict_neurons.keys():
        for pop in dict_neurons[side].keys():
            dict_neurons[side][pop]["n_neurons"] = len(dict_neurons[side][pop]["excitatory"]) + len(dict_neurons[side][pop]["inhibitory"]) + len(dict_neurons[side][pop]["unknown"])
            dict_neurons[side][pop]["idx_list"] = dict_neurons[side][pop]["excitatory"] + dict_neurons[side][pop]["inhibitory"] + dict_neurons[side][pop]["unknown"]
        dict_neurons[side]["idx_list"] = list(itertools.chain.from_iterable([dict_neurons[side][pop]["idx_list"] for pop in dict_neurons[side].keys()]))

    # get W_sign
    W_sign = np.zeros((len(idx_to_id), len(idx_to_id)))
    for side in dict_neurons.keys():
        for pop in ConfigurationRNN.cell_list:
            # compute E/I ratio from known E and I neurons, to infer neurotransmitter identity for unknown neurons
            ratio_pop_EI = len(dict_neurons[side][pop]["excitatory"]) / (len((dict_neurons[side][pop]["inhibitory"])) + len(dict_neurons[side][pop]["excitatory"]))
            n_unknown_E = ratio_pop_EI * len(dict_neurons[side][pop]["unknown"])
            # populate W_sign out of dict_neurons
            for idx_E in dict_neurons[side][pop]["excitatory"]:
                W_sign[idx_E] = 1
            for idx_I in dict_neurons[side][pop]["inhibitory"]:
                W_sign[idx_I] = -1
            # trivially split so that all the first ones are E and the others are I.
            # neurons order is random from the dataset, so it is no problem.
            for i, idx_U in enumerate(dict_neurons[side][pop]["unknown"]):
                W_sign[idx_U] = 1 if i<n_unknown_E else -1

    # compute W
    W = (W_norm * W_sign).T

    return W, dict_neurons

def get_W(path_W_csv, do_symmetry_transform=False):
    W_raw, U, idx_to_id, _, _, idx_side_change = load_synapse_matrix(path_W_csv)
    W, _dict_neurons = process_synapse_matrix(W_raw, idx_to_id, idx_side_change)
    if do_symmetry_transform:
        W, U_sim, _dict_neurons = symmetry_transform(W, U, _dict_neurons)
    U_norm = U / np.sum(U)
    dict_neurons = {"neurons": _dict_neurons,
                    "W": W,
                    "W_mask": np.sign(W),
                    "U": U,
                    "U_norm": U_norm,
                    "U_mask": np.sign(U),
                    "idx_side_change": idx_side_change,
                    "is_symmetry_transformed": do_symmetry_transform,
                    "symmetry_transform": ~do_symmetry_transform,}
    return W, dict_neurons

def update_W(path_csv, W_new, path_save=None, drop_axons=True, drop_myelinated=True):
    df = pd.read_csv(path_csv, index_col=0)
    # Identify neurons to discard: anything whose identifier starts with "axon"
    if drop_axons:
        is_axon = df.index.str.strip().str.startswith("axon")
        non_axon_labels = df.index[~is_axon]
        df = df.loc[non_axon_labels, non_axon_labels]

    # Identify neurons to discard: anything whose identifier starts with "myelinated"
    if drop_myelinated:
        is_myelinated = df.index.str.strip().str.contains("myelinated")
        non_myelinated_labels = df.index[~is_myelinated]
        df = df.loc[non_myelinated_labels, non_myelinated_labels]

    # Define row/col identifiers
    row_labels = df.index  # first-column metadata
    col_labels = df.columns  # first-row metadata (header)

    # Check shape of the upated W matches row/col size
    if W_new.shape != (len(row_labels), len(col_labels)):
        raise ValueError(
            f"Shape mismatch: data is {W_new.shape}, "
            f"expected ({len(row_labels)}, {len(col_labels)})"
        )

    # Create new df and save it as csv
    df_out = pd.DataFrame(W_new, index=row_labels, columns=col_labels)
    if path_save is None:
        path_save = path_csv
    df_out.to_csv(path_save)

def symmetry_transform(W, U, dict_neurons):
    # accept L or R as reference side depending on which one is the smallest one
    # (for which we have all information available to build a symmetric matrix)
    if len(dict_neurons[ConfigurationRNN.SIDE_LEFT]["idx_list"]) <= len(dict_neurons[ConfigurationRNN.SIDE_RIGHT]["idx_list"]):
        reference_side = ConfigurationRNN.SIDE_LEFT
        other_side = ConfigurationRNN.SIDE_RIGHT
        n_neurons_side = dict_neurons["idx_side_change"]
    else:
        reference_side = ConfigurationRNN.SIDE_RIGHT
        other_side = ConfigurationRNN.SIDE_LEFT
        n_neurons_side = W.shape[0] - dict_neurons["idx_side_change"]

    # mirror transform dict_neurons
    dict_neurons[other_side] = dict_neurons[reference_side]

    # define symmetric W mirroring reference side
    W_sim = np.zeros((2*n_neurons_side, 2*n_neurons_side))  # take left side (the first one) as reference
    idx_REF = dict_neurons[reference_side]["idx_list"]
    idx_OTHER = dict_neurons[other_side]["idx_list"]  # cut other_side neurons to size of reference_side
    W_sim[np.ix_(idx_OTHER[:n_neurons_side], idx_OTHER[:n_neurons_side])] = W[np.ix_(idx_REF, idx_REF)]
    W_sim[np.ix_(idx_REF, idx_OTHER[:n_neurons_side])] = W[np.ix_(idx_OTHER[:n_neurons_side], idx_REF)]

    # define symmetric U mirroring reference side
    U_sim = np.repeat(U[idx_REF], 2)

    return W_sim, U_sim, dict_neurons

# if __name__ == "__main__":
#     csv_path = r"C:\Users\Roberto\Desktop\left-right_matrix_e-i_raster_native.csv"
#     get_W(csv_path)

