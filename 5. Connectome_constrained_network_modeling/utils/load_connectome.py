"""
Overview
--------
This module provides functions for loading, filtering, normalizing, and structuring
connectomic synaptic connectivity matrices from CSV files into formats suitable for
recurrent neural network (RNN) modeling.

No environment variables are directly required by this utility module.

Expected layout of th eCSV to load:
- First row: header with neuron identifiers for each column (first cell is a
  label like "presynaptic" for the row-index column).
- First column: neuron identifiers for each row (same set as the columns,
  since this is a square pre->post synapse-count matrix).
- Identifiers look like:
    "axon | ID: 576460752710808710"
    "cell | ID: 576460752631366630 | functional classifier: motion_integrator | ..."

Core Pipeline & Workflow:
1. Matrix Ingestion & Node Filtering (`load_synapse_matrix` & `_drop_non_functionally_identified`):
   - Reads the square adjacency matrix CSV containing neuron and axon identifiers.
   - Drops non-somatic nodes (axons, rostral fibers) while optionally summing incoming axon
     synapse counts to serve as feedforward drive proxies (U).
   - Filters out non-functionally imaged cells, myelinated fibers, and optionally linear discriminant
     analysis (LDA) predicted neurons.
   - Aligns index labels, validates square topology, and identifies hemispheric partition boundaries.

2. Biological Annotations & Normalization (`process_synapse_matrix`):
   - Normalizes raw incoming synaptic weights per postsynaptic neuron column.
   - Parses structured metadata strings from row/column labels to extract hemisphere, functional
     identity (eMI, cMI, MON, sMI), projection target, and neurotransmitter type.
   - Implements Dale's law: builds signed weight matrices (`W_sign`, `W`), enforcing excitatory (+1)
     or inhibitory (-1) transmission, and infers unknown neurotransmitter identities based on
     population-level E/I empirical ratios.

3. Symmetry & Mask Packaging (`get_W`, `symmetry_transform`, & `update_connectome`):
   - Optionally enforces exact bilateral hemispheric symmetry by mirroring connectivity from the
     more completely reconstructed hemisphere.
   - Packages connectivity matrices, sign matrices, structural masks (`np.sign(W)`), and input vectors
     into a standardized metadata dictionary (`dict_neurons`).
   - Provides utilities to write updated or perturbed weight matrices back to formatted CSVs.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# Manually add root path for imports to improve interoperability across parent modules
import sys; sys.path.insert(0, "..")

from utils.config import ConfigurationRNN

def get_idx_side_change(df,
                        identifier_change="functional classifier: motion_integrator",
                        identifier_pre_change="functional classifier: slow_motion_integrator"):
    # Detect the transition point in the index separating left and right hemisphere neuron groups
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

def _drop_unknown_cells(df):
    """Filter out non-functionally imaged and/or myelinated cells."""
    searchfor = ["myelinated | not functionally imaged", "not available"]
    is_unknown = df.index.str.strip().str.contains(" | ".join(searchfor))
    valid_labels = df.index[~is_unknown]
    df = df.loc[valid_labels, valid_labels]
    return df

def _drop_lda_predicted(df):
    """Filter out LDA-predicted cells."""
    is_predicted = df.index.str.strip().str.contains("lda: predicted")
    valid_labels = df.index[~is_predicted]
    df = df.loc[valid_labels, valid_labels]
    return df

def _drop_axons(df, drop_axons=True, return_axon_values=True):
    """Aggregate incoming axon counts and optionally drop axon rows/columns."""
    is_axon = df.index.str.strip().str.startswith("axon")
    axon_labels = df.index[is_axon]

    # Calculate axon drive proxies across postsynaptic target columns
    axon_rostral_idx = [i for i, is_rostral in enumerate(axon_labels.str.strip().str.contains("rostral")) if not is_rostral]
    axon_values = np.array(df.loc[axon_labels].sum(axis=0).to_numpy())
    axon_values[axon_rostral_idx] = 0

    if drop_axons:
        non_axon_labels = df.index[~is_axon]
        non_axon_idx = df.index.get_indexer(non_axon_labels)
        df = df.loc[non_axon_labels, non_axon_labels]
        axon_values = axon_values[non_axon_idx]

    if return_axon_values:
        return df, axon_values
    return df

def load_synapse_matrix(csv_path, drop_axons=True, drop_unknown_cells=True, drop_lda_predicted=False, is_W_csv_datavis_ready_transposed=True):
    # First column becomes the index automatically because it has no header
    # name aligned with a real column count (typical "presynaptic" style CSV).
    df = pd.read_csv(csv_path, index_col=0)

    # Sanity check: matrix should be square with matching row/col labels.
    # If rows and columns are not identically labeled/ordered, align them.
    if not df.index.equals(df.columns):
        print("WARNING | matrix should be square with matching row/col labels. Aligning them.")
        common = df.index.intersection(df.columns)
        df = df.loc[common, common]

    # Filter out non-somatic or non-functionally identified elements
    if drop_unknown_cells:
        df = _drop_unknown_cells(df)
    if drop_lda_predicted:
        df = _drop_lda_predicted(df)

    # Filter out axons
    df, axon_values = _drop_axons(df, drop_axons=drop_axons)

    # Identify index for side change after removing axons and myelinated
    idx_side_change = get_idx_side_change(df)

    # Numpy array copy of the cleaned matrix
    matrix = df.to_numpy(dtype=float).copy()

    # Align matrix orientation to standard presynaptic -> postsynaptic convention if not pre-transposed
    if not is_W_csv_datavis_ready_transposed:
        matrix = matrix.T

    # Index -> original neuron id mapping (and the reverse)
    idx_to_id = {i: label for i, label in enumerate(df.index)}
    id_to_idx = {label: i for i, label in idx_to_id.items()}

    return matrix, axon_values, idx_to_id, id_to_idx, df, idx_side_change

def process_synapse_matrix(W_raw, idx_to_id, idx_side_change=None):
    # take absolute and normalize W_raw by column (postsynaptic input weight normalization)
    W_raw = np.abs(W_raw)
    W_sum_neuron = np.sum(W_raw, axis=0)
    W_sum_neuron = np.array([1 if sum_neuron == 0 else sum_neuron for sum_neuron in W_sum_neuron])
    W_norm = W_raw / W_sum_neuron

    # get neurons info: parse metadata strings into structured cell-type dictionaries
    dict_neurons = {ConfigurationRNN.SIDE_LEFT: {}, ConfigurationRNN.SIDE_RIGHT: {}}
    cell_indices, axon_indices = [], []
    for i_neuron, info_neuron_str in idx_to_id.items():
        # Skip axonal entries that lack cell functional annotations
        if info_neuron_str.strip().startswith("axon"):
            axon_indices.append(i_neuron)
            continue
        cell_indices.append(i_neuron)

        # parse neuron info string
        info_neuron_list = info_neuron_str.split(" | ")
        # get side: infer from label text or relative position to hemispheric transition index
        if "hemisphere" in info_neuron_str:
            side = info_neuron_list[2].replace("hemisphere: ", "").strip()
        else:
            if i_neuron < idx_side_change: side = ConfigurationRNN.SIDE_LEFT
            else: side = ConfigurationRNN.SIDE_RIGHT
        # get other features: functional class, projection target, and neurotransmitter
        function = info_neuron_list[3].replace("functional classifier: ", "").strip()
        projection = info_neuron_list[4].replace("projection classifier: ", "").strip()
        neurotransmitter = info_neuron_list[5].replace("neurotransmitter classifier: ", "").strip()

        # Map functional and projection combinations to canonical population categories
        pop = ConfigurationRNN.classifier_to_pop_map[function][projection]
        if pop not in dict_neurons[side].keys():
            dict_neurons[side][pop] = {"excitatory": [],
                                       "inhibitory": [],
                                       "unknown": []}
        dict_neurons[side][pop][f"{neurotransmitter}"].append(i_neuron)
    # Aggregate neuron index lists per subpopulation and per hemisphere
    for side in dict_neurons.keys():
        for pop in dict_neurons[side].keys():
            dict_neurons[side][pop]["n_neurons"] = len(dict_neurons[side][pop]["excitatory"]) + len(dict_neurons[side][pop]["inhibitory"]) + len(dict_neurons[side][pop]["unknown"])
            dict_neurons[side][pop]["idx_list"] = dict_neurons[side][pop]["excitatory"] + dict_neurons[side][pop]["inhibitory"] + dict_neurons[side][pop]["unknown"]
        dict_neurons[side]["idx_list"] = list(itertools.chain.from_iterable([dict_neurons[side][pop]["idx_list"] for pop in dict_neurons[side].keys()]))

    # get W_sign: construct sign matrix according to Dale's principle
    W_sign = np.zeros((len(idx_to_id), len(idx_to_id)))
    for side in dict_neurons.keys():
        for pop in ConfigurationRNN.cell_list:
            if pop not in dict_neurons[side]:
                continue
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
    # Preserve axonal inputs by assigning a positive drive sign (+1)
    for idx_A in axon_indices:
        W_sign[idx_A] = 1

    dict_neurons["W_sign"] = W_sign
    dict_neurons["cell_indices"] = cell_indices
    dict_neurons["axon_indices"] = axon_indices

    # compute W: combine normalized absolute weight magnitudes with presynaptic signs
    W = (W_norm * W_sign).T

    return W, W_sign.T, dict_neurons

def get_W(path_W_csv, do_symmetry_transform=False, is_W_csv_datavis_ready_transposed=True,
          drop_axons=True, drop_unknown_cells=True, drop_lda_predicted=False, flag_lda_predicted=False):
    # Pipeline orchestrator: load matrix, process biological signs, and apply optional hemispheric symmetry
    W_raw, U, idx_to_id, _, _, idx_side_change = load_synapse_matrix(path_W_csv,
                                                                     is_W_csv_datavis_ready_transposed=is_W_csv_datavis_ready_transposed,
                                                                     drop_axons=drop_axons,
                                                                     drop_unknown_cells=drop_unknown_cells,
                                                                     drop_lda_predicted=drop_lda_predicted)
    W, W_sign, _dict_neurons = process_synapse_matrix(W_raw, idx_to_id, idx_side_change)
    # Apply bilateral mirroring transformation if requested
    if do_symmetry_transform:
        W, U, _dict_neurons = symmetry_transform(W, U, _dict_neurons)
    # Ensure non-zero input drive vector and normalize to unit sum
    if U is None or np.sum(U) == 0:
        U = np.ones_like(U)
    U_norm = U / np.sum(U)

    # Package processed matrices and metadata into structured output dictionary
    dict_neurons = {"neurons": _dict_neurons,
                    "W": W,
                    "W_mask": np.sign(W),
                    "W_sign": W_sign,
                    "U": U,
                    "U_norm": U_norm,
                    "U_mask": np.sign(U),
                    "idx_side_change": idx_side_change,
                    "cell_indices": _dict_neurons["cell_indices"],
                    "axon_indices": _dict_neurons["axon_indices"],
                    "idx_to_id": idx_to_id,
                    "is_symmetry_transformed": do_symmetry_transform,
                    "symmetry_transform": ~do_symmetry_transform,}

    # Identify index partitions for native functionally identified vs. LDA-predicted cells
    if flag_lda_predicted:
        dict_neurons["lda_predicted_idx"] = [i for i in range(len(idx_to_id)) if "lda: predicted" in idx_to_id[i]]
        dict_neurons["lda_native_idx"] = [i for i in range(len(idx_to_id)) if "lda: native" in idx_to_id[i]]

    return W, dict_neurons

def update_connectome(path_csv, connectome_new, path_save=None, drop_axons=True, drop_unknown_cells=True, drop_lda_predicted=False):
    # Read reference matrix to preserve original row and column label annotations
    df = pd.read_csv(path_csv, index_col=0)

    if drop_axons:
        df, _ = _drop_axons(df)
    if drop_unknown_cells:
        df = _drop_unknown_cells(df)
    if drop_lda_predicted:
        df = _drop_lda_predicted(df)

    # Define row/col identifiers
    row_labels = df.index  # first-column metadata
    col_labels = df.columns  # first-row metadata (header)

    # Check shape of the upated W matches row/col size
    if connectome_new.shape != (len(row_labels), len(col_labels)):
        raise ValueError(
            f"Shape mismatch: data is {connectome_new.shape}, "
            f"expected ({len(row_labels)}, {len(col_labels)})"
        )

    # Create new df and save it as csv with preserved metadata index labels
    df_out = pd.DataFrame(connectome_new, index=row_labels, columns=col_labels)
    if path_save is None:
        path_save = path_csv
    Path(path_save).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(path_save)
    return df_out

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

    # mirror transform dict_neurons: duplicate reference side metadata
    dict_neurons[other_side] = dict_neurons[reference_side]

    # define symmetric W mirroring reference side: copy intra-hemispheric and swap cross-hemispheric submatrices
    W_sim = np.zeros((2*n_neurons_side, 2*n_neurons_side))  # take left side (the first one) as reference
    idx_REF = dict_neurons[reference_side]["idx_list"]
    idx_OTHER = dict_neurons[other_side]["idx_list"]  # cut other_side neurons to size of reference_side
    W_sim[np.ix_(idx_OTHER[:n_neurons_side], idx_OTHER[:n_neurons_side])] = W[np.ix_(idx_REF, idx_REF)]
    W_sim[np.ix_(idx_REF, idx_OTHER[:n_neurons_side])] = W[np.ix_(idx_OTHER[:n_neurons_side], idx_REF)]

    # define symmetric U mirroring reference side
    U_sim = np.repeat(U[idx_REF], 2)

    return W_sim, U_sim, dict_neurons