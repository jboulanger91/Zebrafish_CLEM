from datetime import datetime
from pathlib import Path
from dotenv import dotenv_values

import numpy as np
import pickle

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "..")

from model.RNNFreePop import RNNFreePop
from utils.ds_service import DSService
from utils.operators import inv_softplus
from utils.train_batch import TrainSignal


if __name__ == '__main__':
    # Configurations
    save_model = True
    fit_model = True

    # Generate a 1D input and target
    activation = "softplus"  # "relu"  #
    dt = 0.01
    duration_rest_start = 20
    duration_stimulus = 40
    duration_rest_end = 20
    n_input_signal = 2
    tau_neuron = 0.1
    n_free_neurons = 16
    n_slow_pops = 8

    # Training
    n_epochs = 5001
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
        n_units_A = 15
        n_units_B = 15
        n_units_C = 2
        n_units_D = 11
        n_units_hemi = n_units_A + n_units_B + n_units_C + n_units_D
        n_units = n_units_hemi * 2 + n_free_neurons
        rnn = RNNFreePop(nA=n_units_A, nB=n_units_B, nC=n_units_C, nD=n_units_D, nX=n_free_neurons,
                         E_frac_A=0.85, E_frac_B=0.1, E_frac_C=0.1, E_frac_D=1/3, E_frac_X=0.37,
                          # intra-hemispheric sparsity
            sparsity_AA=0.1125, sparsity_AB=0.18, sparsity_AC=0.475, sparsity_AD=0.065, sparsity_AX=0.07,
            sparsity_BA=0, sparsity_BB=0, sparsity_BC=0, sparsity_BD=0, sparsity_BX=0.1,
            sparsity_CA=0.08, sparsity_CB=0.45, sparsity_CC=0.09, sparsity_CD=0.04, sparsity_CX=0.13,
            sparsity_DA=0.02, sparsity_DB=0, sparsity_DC=0, sparsity_DD=0, sparsity_DX=0.02,
                          # inter-hemispheric sparsity
            sparsity_LA_RA=0, sparsity_LA_RB=0, sparsity_LA_RC=0, sparsity_LA_RD=0,
            sparsity_LB_RA=0.2, sparsity_LB_RB=0.3, sparsity_LB_RC=0.04, sparsity_LB_RD=0.06,
            sparsity_LC_RA=0.03, sparsity_LC_RB=0.15, sparsity_LC_RC=0, sparsity_LC_RD=0,
            sparsity_LD_RA=0.05, sparsity_LD_RB=0.05, sparsity_LD_RC=0.05, sparsity_LD_RD=0,
                         sparsity_XA=0.07, sparsity_XB=0.1, sparsity_XC=0.1, sparsity_XD=0.02, sparsity_XX=0.074,  # Free population X is just one for both hemispheres
                         sparsity_U=1,
                         tau=tau_neuron, dt=dt, seed=seed, activation=activation, clamp_weights_min=1e-2, n_slow_pops=n_slow_pops)

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
        input_signal_neurons_L = np.concatenate((np.repeat(input_signal[..., np.newaxis], n_units_hemi, axis=1),
                                                 np.zeros((len(input_signal), n_units_hemi)),
                                                 np.zeros((len(input_signal), n_free_neurons))), axis=1)
        initial_value_L = np.concatenate((np.array([target_signal_L[0, 0] for _ in range(n_units_A)]) + np.random.normal(0, np.abs(target_signal_L[0, 0])/5, n_units_A),
                                          np.array([target_signal_L[0, 1] for _ in range(n_units_B)]) + np.random.normal(0, np.abs(target_signal_L[0, 1])/5, n_units_B),
                                          np.array([target_signal_L[0, 2] for _ in range(n_units_C)]) + np.random.normal(0, np.abs(target_signal_L[0, 2])/5, n_units_C),
                                          np.array([target_signal_L[0, 3] for _ in range(n_units_D)]) + np.random.normal(0, np.abs(target_signal_L[0, 3])/5, n_units_D),
                                          np.array([target_signal_L[0, 4] for _ in range(n_units_A)]) + np.random.normal(0, np.abs(target_signal_L[0, 4])/5, n_units_A),
                                          np.array([target_signal_L[0, 5] for _ in range(n_units_B)]) + np.random.normal(0, np.abs(target_signal_L[0, 5])/5, n_units_B),
                                          np.array([target_signal_L[0, 6] for _ in range(n_units_C)]) + np.random.normal(0, np.abs(target_signal_L[0, 6])/5, n_units_C),
                                          np.array([target_signal_L[0, 7] for _ in range(n_units_D)]) + np.random.normal(0, np.abs(target_signal_L[0, 7])/5, n_units_D),
                                          np.array([np.mean(target_signal_L[0]) for _ in range(n_free_neurons)]) + np.random.normal(0, np.abs(np.mean(target_signal_L[0])) / 5, n_free_neurons)))
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
        input_signal_neurons_R = np.concatenate((np.zeros((len(input_signal), n_units_hemi)),
                                                 np.repeat(input_signal[..., np.newaxis], n_units_hemi, axis=1),
                                                 np.zeros((len(input_signal), n_free_neurons))), axis=1)
        initial_value_R = np.concatenate((np.array([target_signal_R[0, 0] for _ in range(n_units_A)]) + np.random.normal(0, np.abs(target_signal_R[0, 0])/5, n_units_A),
                                          np.array([target_signal_R[0, 1] for _ in range(n_units_B)]) + np.random.normal(0, np.abs(target_signal_R[0, 1])/5, n_units_B),
                                          np.array([target_signal_R[0, 2] for _ in range(n_units_C)]) + np.random.normal(0, np.abs(target_signal_R[0, 2])/5, n_units_C),
                                          np.array([target_signal_R[0, 3] for _ in range(n_units_D)]) + np.random.normal(0, np.abs(target_signal_R[0, 3])/5, n_units_D),
                                          np.array([target_signal_R[0, 4] for _ in range(n_units_A)]) + np.random.normal(0, np.abs(target_signal_R[0, 4])/5, n_units_A),
                                          np.array([target_signal_R[0, 5] for _ in range(n_units_B)]) + np.random.normal(0, np.abs(target_signal_R[0, 5])/5, n_units_B),
                                          np.array([target_signal_R[0, 6] for _ in range(n_units_C)]) + np.random.normal(0, np.abs(target_signal_R[0, 6])/5, n_units_C),
                                          np.array([target_signal_R[0, 7] for _ in range(n_units_D)]) + np.random.normal(0, np.abs(target_signal_R[0, 7])/5, n_units_D),
                                          np.array([np.mean(target_signal_R[0]) for _ in range(n_free_neurons)]) + np.random.normal(0, np.abs(np.mean(target_signal_R[0])) / 5, n_free_neurons)))
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
