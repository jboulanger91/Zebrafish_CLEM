from datetime import datetime
from pathlib import Path
from dotenv import dotenv_values

import pandas as pd
import numpy as np
import pickle

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "..")

from model.core.RNNFixedConnectivity import RNNFixedConnectivity
from analysis.load_synapse_matrix import get_W
from utils.configuration_rnn import ConfigurationRNN
from utils.services.ds_service import DSService
from utils.math.operators import inv_softplus
from utils.math.train_batch import TrainSignal

if __name__ == '__main__':
    # Configurations
    save_model = True
    fit_model = True

    # Generate a 1D input and target
    activation = "softplus"
    dt = 0.01
    duration_rest_start = 20
    duration_stimulus = 40
    duration_rest_end = 20
    n_input_signal = 2
    tau_neuron = 0.1

    # Training
    do_symmetry_transform = False
    n_epochs = 3001
    seed = None

    # Resolve env
    # When calling the script you can provide the path to the .env file as argument.
    # If not, the root .env of the project is used.
    try:
        env_path = sys.argv[1]
    except IndexError:
        env_path = "../.env"
    env = dotenv_values(env_path)

    # Paths
    path_traces = Path(env["PATH_DATA"])
    path_save = Path(env["PATH_SAVE"])
    path_noise_estimation = Path(env["PATH_NOISE_ESTIMATION"])
    path_load = None
    path_load_mask = None

    # Initialize model
    if path_load is not None:
        with open(path_load, 'rb') as f:
            rnn_load = pickle.load(f)
        n_units = rnn_load.n_units
        n_units_hemi = int(n_units/2)
        rnn = rnn_load
    else:
        path_W_csv = Path(env["PATH_W_CSV"])
        W_norm, dict_neurons = get_W(path_W_csv, do_symmetry_transform=do_symmetry_transform)

        n_units_LiMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["iMI"]["n_neurons"]
        n_units_LcMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["cMI"]["n_neurons"]
        n_units_LMON = dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["MON"]["n_neurons"]
        n_units_LsMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["sMI"]["n_neurons"]
        n_units_RiMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["iMI"]["n_neurons"]
        n_units_RcMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["cMI"]["n_neurons"]
        n_units_RMON = dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["MON"]["n_neurons"]
        n_units_RsMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["sMI"]["n_neurons"]
        n_units = W_norm.shape[0]
        rnn = RNNFixedConnectivity(W_norm, dict_neurons, n_beta=1, sparsity_U=1, tau=tau_neuron, dt=dt,
                                   seed=seed, activation=activation, clamp_weights_min=1e-2)

    # Define input/output signals for training
    amplitude_input_signal_list = np.linspace(0.1, 1, n_input_signal)
    input_signal = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)), np.zeros(int(duration_rest_end / dt))))
    input_signal_list = []
    for i in range(n_input_signal):
        input_signal_list.append(input_signal * amplitude_input_signal_list[i])

    # Load traces to use as target signals
    cell_types_list = ["iMI", "cMI", "MON", "sMI"]
    side_list = ["preferred", "null"]
    traces_dict = {ct: {s: None for s in side_list} for ct in cell_types_list}
    min_traces_all = 0  # initialize offset
    for ct in cell_types_list:
        for s in side_list:
            filename = f"avgresponses_{ct}_{s}_constant.csv"
            data = np.loadtxt(path_traces / filename, dtype=float, delimiter=",", skiprows=1)
            downsample_time_list = data[:, 0]
            traces_dict[ct][s] = data[:, 1] / 100
            min_trace_here = np.min(data[:, 1] / 100)
            if min_trace_here < min_traces_all:
                min_traces_all = min_trace_here
    min_traces_all = np.abs(min_traces_all)

    with open(path_noise_estimation, 'rb') as f:
        p_noise = pickle.load(f)
        def noise_filter(x):
            return DSService.ou_noise(x, p_noise["tau"], p_noise["sigma"], 0.5, p_noise["scale"])

    train_list = []
    for i, amplitude in enumerate(amplitude_input_signal_list):
        input_signal = input_signal_list[i]
        target_signal_L = np.stack((traces_dict["iMI"]["preferred"],
                                    traces_dict["cMI"]["preferred"],
                                    traces_dict["MON"]["preferred"],
                                    traces_dict["sMI"]["preferred"],
                                    traces_dict["iMI"]["null"],
                                    traces_dict["cMI"]["null"],
                                    traces_dict["MON"]["null"],
                                    traces_dict["sMI"]["null"]
                                    ),
                                   axis=-1)
        # target_signal_L /= np.max(target_signal_L)
        if amplitude != 1:
            target_signal_L += noise_filter(target_signal_L)
        target_signal_L = target_signal_L * np.sqrt(amplitude)  # scaling
        target_signal_L += min_traces_all

        input_signal_neurons_L = np.concatenate((np.array([input_signal for _ in range(n_units_LiMI)]),
                                                 np.array([input_signal for _ in range(n_units_LcMI)]),
                                                 np.array([input_signal for _ in range(n_units_LMON)]),
                                                 np.array([input_signal for _ in range(n_units_LsMI)]),
                                                 np.array([np.zeros_like(input_signal) for _ in range(n_units_RiMI)]),
                                                 np.array([np.zeros_like(input_signal) for _ in range(n_units_RcMI)]),
                                                 np.array([np.zeros_like(input_signal) for _ in range(n_units_RMON)]),
                                                 np.array([np.zeros_like(input_signal) for _ in range(n_units_RsMI)]),)).T
        initial_value_L = np.concatenate((np.array([target_signal_L[0, 0] for _ in range(n_units_LiMI)]) + np.random.normal(0, np.abs(target_signal_L[0, 0])/5, n_units_LiMI),
                                          np.array([target_signal_L[0, 1] for _ in range(n_units_LcMI)]) + np.random.normal(0, np.abs(target_signal_L[0, 1])/5, n_units_LcMI),
                                          np.array([target_signal_L[0, 2] for _ in range(n_units_LMON)]) + np.random.normal(0, np.abs(target_signal_L[0, 2])/5, n_units_LMON),
                                          np.array([target_signal_L[0, 3] for _ in range(n_units_LsMI)]) + np.random.normal(0, np.abs(target_signal_L[0, 3])/5, n_units_LsMI),
                                          np.array([target_signal_L[0, 4] for _ in range(n_units_RiMI)]) + np.random.normal(0, np.abs(target_signal_L[0, 4])/5, n_units_RiMI),
                                          np.array([target_signal_L[0, 5] for _ in range(n_units_RcMI)]) + np.random.normal(0, np.abs(target_signal_L[0, 5])/5, n_units_RcMI),
                                          np.array([target_signal_L[0, 6] for _ in range(n_units_RMON)]) + np.random.normal(0, np.abs(target_signal_L[0, 6])/5, n_units_RMON),
                                          np.array([target_signal_L[0, 7] for _ in range(n_units_RsMI)]) + np.random.normal(0, np.abs(target_signal_L[0, 7])/5, n_units_RsMI),))
                                          # np.array([np.mean(target_signal_L[0]) for _ in range(n_units_free)]) + np.random.normal(0, np.abs(np.mean(target_signal_L[0])) / 5, n_units_free)))
        train_list.append(TrainSignal(input_signal_neurons_L, target_signal_L, inv_softplus(initial_value_L)))

        target_signal_R = np.stack((traces_dict["iMI"]["null"],
                                    traces_dict["cMI"]["null"],
                                    traces_dict["MON"]["null"],
                                    traces_dict["sMI"]["null"],
                                    traces_dict["iMI"]["preferred"],
                                    traces_dict["cMI"]["preferred"],
                                    traces_dict["MON"]["preferred"],
                                    traces_dict["sMI"]["preferred"]
                                    ),
                                   axis=-1)
        # target_signal_R /= np.max(target_signal_R)
        target_signal_R += noise_filter(target_signal_R)
        target_signal_R = target_signal_R * np.sqrt(amplitude)
        target_signal_R += min_traces_all
        input_signal_neurons_R = np.concatenate((np.array([np.zeros_like(input_signal) for _ in range(n_units_LiMI)]),
                                                 np.array([np.zeros_like(input_signal) for _ in range(n_units_LcMI)]),
                                                 np.array([np.zeros_like(input_signal) for _ in range(n_units_LMON)]),
                                                 np.array([np.zeros_like(input_signal) for _ in range(n_units_LsMI)]),
                                                 np.array([input_signal for _ in range(n_units_RiMI)]),
                                                 np.array([input_signal for _ in range(n_units_RcMI)]),
                                                 np.array([input_signal for _ in range(n_units_RMON)]),
                                                 np.array([input_signal for _ in range(n_units_RsMI)]),)).T
        initial_value_R = np.concatenate((np.array([target_signal_R[0, 0] for _ in range(n_units_LiMI)]) + np.random.normal(0, np.abs(target_signal_R[0, 0])/5, n_units_LiMI),
                                          np.array([target_signal_R[0, 1] for _ in range(n_units_LcMI)]) + np.random.normal(0, np.abs(target_signal_R[0, 1])/5, n_units_LcMI),
                                          np.array([target_signal_R[0, 2] for _ in range(n_units_LMON)]) + np.random.normal(0, np.abs(target_signal_R[0, 2])/5, n_units_LMON),
                                          np.array([target_signal_R[0, 3] for _ in range(n_units_LsMI)]) + np.random.normal(0, np.abs(target_signal_R[0, 3])/5, n_units_LsMI),
                                          np.array([target_signal_R[0, 4] for _ in range(n_units_RiMI)]) + np.random.normal(0, np.abs(target_signal_R[0, 4])/5, n_units_RiMI),
                                          np.array([target_signal_R[0, 5] for _ in range(n_units_RcMI)]) + np.random.normal(0, np.abs(target_signal_R[0, 5])/5, n_units_RcMI),
                                          np.array([target_signal_R[0, 6] for _ in range(n_units_RMON)]) + np.random.normal(0, np.abs(target_signal_R[0, 6])/5, n_units_RMON),
                                          np.array([target_signal_R[0, 7] for _ in range(n_units_RsMI)]) + np.random.normal(0, np.abs(target_signal_R[0, 7])/5, n_units_RsMI),))
                                          # np.array([np.mean(target_signal_R[0]) for _ in range(n_units_free)]) + np.random.normal(0, np.abs(np.mean(target_signal_R[0])) / 5, n_units_free)))
        train_list.append(TrainSignal(input_signal_neurons_R, target_signal_R, inv_softplus(initial_value_R)))

    if path_load_mask is not None:
        with open(path_load_mask, 'rb') as f:
            rnn_load = pickle.load(f)
        rnn.mask_W = rnn_load.mask_W

    # Train
    if fit_model:
        W_fit = rnn.fit(train_list, n_epochs=n_epochs, downsample_target_list=downsample_time_list)

    # Save trained model
    if save_model:
        label_model = f"RNNFreePop_neurons{n_units}_tau{tau_neuron}_input{n_input_signal}step_{activation}"
        label_model_instance = f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}"
        path_save_model = path_save / label_model
        path_save_model.mkdir(parents=True, exist_ok=True)
        with open(path_save_model / f"model_{label_model_instance}.pkl", 'wb') as f:
            pickle.dump(rnn, f)
