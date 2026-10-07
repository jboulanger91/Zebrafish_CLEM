"""
Overview
--------
This script trains a biologically constrained recurrent neural network (RNNConnectome)
to reproduce empirical bilateral calcium imaging dynamics (e.g., zebrafish hindbrain
oculomotor integrator populations: iMI, cMI, MON, sMI) under unilateral and bilateral step
stimuli. The estimated completion time on a single cpu with the default configurations is ~2 hours.

Before running it, please make sure you have set up your .env with all the necessary variables:
- PATH_DATA="/path/to/project_root/data"  # Path to directory containing empirical response traces (avgresponses_*.csv)
- PATH_SAVE="/path/to/project_root/models"  # Path to directory where trained model checkpoints (.pt) will be exported
- PATH_NOISE_ESTIMATION="/path/to/project_root/data/noise_estimation/contralateral_motion_integrator_preferred_noise_estimation.pkl"  # Path to pickle file containing fitted OU noise parameters (tau, sigma, scale), you can generate one running analysis/noise_component_in_traces.py
- PATH_W_CSV="/path/to/project_root/data/connectome.csv"  # Path to the empirical connectome adjacency matrix CSV
- LABEL="connectome"  # Optional identifier tag appended to the output model filename (e.g. "_run1")

Core Pipeline & Workflow:
1. Environment Resolution & Configuration:
   - Resolves environment variable files (.env) passed via command-line arguments or defaults.
   - Sets network hyperparameters (integration dt, membrane time constant tau, activation
     function, training epochs, Dale's law non-negativity clamping).
   - Generates unique run timestamps and model instance tags.

2. Biological Connectome Matrix Loading:
   - Loads the empirical connectivity matrix (W_norm) and cell subpopulation metadata
     (`dict_neurons`) from CSV via `get_W`.
   - Supports linear discriminant analysis (LDA) predicted neuron inclusion/exclusion and
     bilateral hemispheric symmetry transforms.
   - Instantiates `RNNConnectome` with specified cell-type masks and feedforward input topologies.

3. Training Batch Construction:
   - Generates step input pulses across multiple stimulus amplitude levels.
   - Loads empirical calcium average traces (`avgresponses_*.csv`), normalizes units, and
     calculates non-negative baseline offsets.
   - Injects empirical Ornstein-Uhlenbeck (OU) colored noise into sub-maximal stimulus conditions.
   - Formulates mirror-symmetric bilateral training batches (Left stimulus driving left-hemisphere
     preferred targets; Right stimulus driving right-hemisphere preferred targets).
   - Initializes network membrane potentials via the inverse softplus transformation (`inv_softplus`).

4. Network Training & Final Checkpoint Export:
   - Optimizes recurrent synaptic weights (`rnn.fit`) against downsampled empirical target traces.
   - Extracts network parameters, state dictionaries, and custom model attributes.
   - Saves serialized PyTorch checkpoint dictionaries (`.pt`) to designated results directories.
"""

from datetime import datetime
from pathlib import Path
from dotenv import dotenv_values

import torch
import numpy as np
import pickle

# Manually add root path for imports to improve interoperability across parent modules
import sys; sys.path.insert(0, "..")

from model.core.RNNConnectome import RNNConnectome
from utils.load_connectome import get_W
from utils.config import ConfigurationRNN
from utils.services.ds_service import DSService
from utils.services.rnn_service import RNNService
from utils.math.operators import inv_softplus
from utils.math.train_batch import TrainSignal

if __name__ == '__main__':
    # ------------------------------------------------
    # Configurations
    # ------------------------------------------------
    # Flags controlling whether optimization is run and whether final checkpoint is written to disk
    save_model = True
    fit_model = True

    # Simulation and single-neuron biophysical parameters
    activation = "softplus"
    dt = 0.01
    duration_rest_start = 20
    duration_stimulus = 40
    duration_rest_end = 20
    n_input_signal = 2
    tau_neuron = 0.1

    # Training and structural connectivity options
    is_W_csv_datavis_ready_transposed = True
    flag_lda_predicted = False
    drop_lda_predicted = False
    do_symmetry_transform = False
    n_epochs = 5001
    seed = None

    # ------------------------------------------------
    # Resolve env
    # ------------------------------------------------
    # When calling the script you can provide the path to the .env file as argument.
    # If not, the root .env of the project is used.
    try:
        env_path = sys.argv[1]
    except IndexError:
        env_path = "../.env"
    env = dotenv_values(env_path)
    label_model_instance = "_" + env["LABEL"] if "LABEL" in env.keys() else ""
    if label_model_instance == "_":
        label_model_instance += f"{np.random.randint(0,999999):06d}"

    # ------------------------------------------------
    # Paths
    # ------------------------------------------------
    # Extract directory paths from environment definitions
    path_traces = Path(env["PATH_DATA"])
    path_save = Path(env["PATH_SAVE"])
    path_noise_estimation = Path(env["PATH_NOISE_ESTIMATION"])
    path_load = None
    path_load_mask = None

    # ------------------------------------------------
    # Initialize model
    # ------------------------------------------------
    # Either load a pre-existing pickled RNN instance or construct a new model from connectome data
    if path_load is not None:
        with open(path_load, 'rb') as f:
            rnn_load = pickle.load(f)
        n_units = rnn_load.n_units
        n_units_hemi = int(n_units/2)
        rnn = rnn_load
    else:
        path_W_csv = Path(env["PATH_W_CSV"])
        # Load empirical recurrent matrix W and neuron population metadata dictionary
        W_norm, dict_neurons = get_W(path_W_csv, do_symmetry_transform=do_symmetry_transform,
                                     is_W_csv_datavis_ready_transposed=is_W_csv_datavis_ready_transposed,
                                     flag_lda_predicted=flag_lda_predicted, drop_lda_predicted=drop_lda_predicted)

        # Extract population sizes across bilateral cell classes
        n_units_LiMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["iMI"]["n_neurons"]
        n_units_LcMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["cMI"]["n_neurons"]
        n_units_LMON = dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["MON"]["n_neurons"]
        n_units_LsMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["sMI"]["n_neurons"]
        n_units_RiMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["iMI"]["n_neurons"]
        n_units_RcMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["cMI"]["n_neurons"]
        n_units_RMON = dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["MON"]["n_neurons"]
        n_units_RsMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["sMI"]["n_neurons"]
        n_units = W_norm.shape[0]

        # Identify neurons whose dynamics are left unconstrained by target loss functions
        let_neurons_free_list = dict_neurons["lda_predicted_idx"] if flag_lda_predicted else []
        # Instantiate continuous-time connectome-constrained recurrent neural network
        rnn = RNNConnectome(dict_neurons, tau=tau_neuron, dt=dt, seed=seed,
                            slow_populations=[], use_connectome_mask_U=True,           # exclude MON and sMI cells
                            # slow_populations=[0, 1, 4, 5], use_connectome_mask_U=True,           # exclude MON and sMI cells
                            activation=activation, clamp_weights_min=0, let_neurons_free_list=let_neurons_free_list)

    # ------------------------------------------------
    # Define input/output signals for training
    # ------------------------------------------------
    # Construct square step driving currents across multiple amplitude scaling factors
    amplitude_input_signal_list = np.linspace(0.1, 1, n_input_signal)
    input_signal = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)), np.zeros(int(duration_rest_end / dt))))
    input_signal_list = []
    for i in range(n_input_signal):
        input_signal_list.append(input_signal * amplitude_input_signal_list[i])

    # Load traces to use as target signals across anatomical cell classes and orientations
    cell_types_list = ["iMI", "cMI", "MON", "sMI"]
    side_list = ["preferred", "null"]
    traces_dict = {ct: {s: None for s in side_list} for ct in cell_types_list}
    min_traces_all = 0  # initialize offset
    for ct in cell_types_list:
        for s in side_list:
            filename = f"avgresponses_{ct}_{s}_constant.csv"
            data = np.loadtxt(path_traces / filename, dtype=float, delimiter=",", skiprows=1)
            downsample_time_list = data[:, 0]
            # Convert percentage delta F/F to fractional values
            traces_dict[ct][s] = data[:, 1] / 100
            min_trace_here = np.min(data[:, 1] / 100)
            if min_trace_here < min_traces_all:
                min_traces_all = min_trace_here
    # Compute baseline shift to ensure non-negative signal activity
    min_traces_all = np.abs(min_traces_all)

    # Load empirical noise model parameters and define Ornstein-Uhlenbeck (OU) noise process
    with open(path_noise_estimation, 'rb') as f:
        p_noise = pickle.load(f)
        def noise_filter(x):
            return DSService.ou_noise(x, p_noise["tau"], p_noise["sigma"], 0.5, p_noise["scale"])

    # ------------------------------------------------
    # Assemble training batches
    # ------------------------------------------------
    train_list = []
    for i, amplitude in enumerate(amplitude_input_signal_list):
        input_signal = input_signal_list[i]
        # Assemble Left-hemisphere driving target signal matrix
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
        # Augment non-maximal amplitude target traces with colored noise
        if amplitude != 1:
            target_signal_L += noise_filter(target_signal_L)
        target_signal_L = target_signal_L * np.sqrt(amplitude)  # scaling
        target_signal_L += min_traces_all

        # Map step drive to left-hemisphere units while right-hemisphere units receive zero drive
        input_signal_neurons_L = np.concatenate((np.array([input_signal for _ in range(n_units_LiMI)]),
                                                 np.array([input_signal for _ in range(n_units_LcMI)]),
                                                 np.array([input_signal for _ in range(n_units_LMON)]),
                                                 np.array([input_signal for _ in range(n_units_LsMI)]),
                                                 np.array([np.zeros_like(input_signal) for _ in range(n_units_RiMI)]),
                                                 np.array([np.zeros_like(input_signal) for _ in range(n_units_RcMI)]),
                                                 np.array([np.zeros_like(input_signal) for _ in range(n_units_RMON)]),
                                                 np.array([np.zeros_like(input_signal) for _ in range(n_units_RsMI)]),)).T
        # Sample initial firing rates around target baseline and map to latent voltage states via inv_softplus
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

        # Assemble Right-hemisphere driving target signal matrix (reversing preferred and null orientations)
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
        target_signal_R += noise_filter(target_signal_R)
        target_signal_R = target_signal_R * np.sqrt(amplitude)
        target_signal_R += min_traces_all
        # Map step drive to right-hemisphere units while left-hemisphere units receive zero drive
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

    # Optional mask injection from an external saved network
    if path_load_mask is not None:
        with open(path_load_mask, 'rb') as f:
            rnn_load = pickle.load(f)
        rnn.mask_W = rnn_load.mask_W

    # ------------------------------------------------
    # Train
    # ------------------------------------------------
    # Fit recurrent weights via gradient descent against downsampled target timestamps
    if fit_model:
        W_fit = rnn.fit(train_list, n_epochs=n_epochs, downsample_target_list=downsample_time_list)

    # ------------------------------------------------
    # Save trained model
    # ------------------------------------------------
    # Assemble serialization dictionary and export trained PyTorch checkpoint
    if save_model:
        checkpoint = {
            # Define checkpoint to save
            "dict_neurons": dict_neurons,
            "state_dict": rnn.state_dict(),
            "custom_attrs": RNNService.extract_custom_attrs(rnn),
            "class_name": type(rnn).__name__,
        }

        # Formulate directory paths and timestamped output filenames
        label_model = f"RNNConnectome_neurons{n_units}"
        label_model_instance = label_model_instance + f"_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}"
        path_save_model = path_save / label_model
        path_save_model.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path_save_model / f"model{label_model_instance}.pt")