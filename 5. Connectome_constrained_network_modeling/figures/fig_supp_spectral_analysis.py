"""
Overview
--------
This script generates the following figure panels in Boulanger-Weiss et al. 2026:
- Supp. Fig. 10...

This script evaluates and statistically compares the dynamic response fidelity of recurrent
neural network (RNN) dynamical models trained across different biological and structural
connectivity conditions (e.g., intact connectome, shuffled connectivity mask, and ipsilateral
recurrent synapse lesions). Evaluation is performed by computing harmonic spectral similarity
scores between network population activity and empirical calcium imaging traces across multiple
testing protocols (step inputs, sinusoidal oscillations, and side switches).

Before running it, please make sure you have set up your .env with all the necessary variables:
- PATH_DIR="/path/to/project_root"  # Root project directory containing models/, data/, and results/

Core Pipeline & Workflow:
1. Environment and Architectural Setup:
   - Reads `PATH_DIR` from .env to establish relative paths for models, experimental calcium
     recordings, noise estimations, and output destinations.
   - Loads a reference checkpoint (`path_model_example`) to infer network topology (subpopulation
     unit allocations, step size dt, integration parameters).
   - Configures plot geometry, coordinate origins, and canvas dimensions via `RNNDSStyle` and `Figure`.

2. Test Signal Formulation & Noise Augmentation:
   - Defines unilateral and bilateral stimulus functions (step pulses, oscillating sine waves,
     and direction-switching drives).
   - Loads and formats empirical calcium target traces (`avgresponses_*.csv` and test datasets)
     for bilateral integrator populations (iMI, cMI, MON, sMI).
   - Incorporates Ornstein-Uhlenbeck (OU) colored noise models and aligns baseline activities.

3. Simulation & Spectral Distance Computation (Optional / Toggleable):
   - When `compute_spectral_distance_distributions` is enabled, loops over experimental test
     conditions and model condition cohorts ("Original", "Shuffle mask", "Remove 50% iMI-iMI").
   - Simulates continuous-time forward dynamics (`RNNService.plot_response_by_cell` with
     `compute_spectrum=True`) for each model instance.
   - Extracts harmonic spectral similarity scores from the frequency response spectra and stores
     the resulting metric distributions, serializing them into a timestamped `.pkl` dictionary.
   - When disabled, loads precomputed spectral distance distributions directly from `path_spectral_distance`.

4. Statistical Comparison & Visualization:
   - Arranges a 2x2 multi-panel layout of categorical distribution plots corresponding to each
     stimulation protocol.
   - Performs two-sided Mann-Whitney U non-parametric rank-sum tests comparing perturbed model
     cohorts against the intact baseline ("Original"), marking significant differences (p < 0.005)
     with asterisks.
   - Renders distribution violins across model cohorts and exports the assembled multi-panel
     vector graphic as a PDF.
"""

import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import dotenv_values
from scipy.stats import mannwhitneyu

# Manually add root path for imports to improve interoperability across parent modules
import sys; sys.path.insert(0, "..")

from style import RNNDSStyle
from utils.services.ds_service import DSService
from utils.services.rnn_service import RNNService
from utils.math.operators import inv_softplus, get_hist
from utils.figure_helper import Figure
from utils.load_model import load_model
from utils.load_connectome import get_W
from utils.config import ConfigurationRNN


# ------------------------------------------------
# Configuration
# ------------------------------------------------
# Toggle visualization of the loss comparison violin plot
compute_spectral_distance_distributions = False
show_spect_dist = True

# Suffix tag applied to the exported figure filename
label_figure = "_all"

# Load directory paths from environment configuration (.env)
env = dotenv_values()
path_dir = Path(env["PATH_DIR"])
path_models = path_dir / "models"
path_model_example = path_models / "connectome" / "model_000788_2026-10-04_17-38-30.pt"  # used to extract model configuration
path_data = path_dir / "data" # directory containing avgresponse_X.csv
path_noise_estimation = path_dir / "data" / "noise_estimation" / "contralateral_motion_integrator_preferred_noise_estimation.pkl"
path_save = path_dir / "results"
path_spectral_distance = path_models / "spectral_distance_dict_2026-10-12_00-03-45.pkl"

# Registry of model directories and display labels representing distinct experimental/ablation cohorts
condition_list = [
    {"path": path_models / "connectome",
     "label": "Original"},
    {"path": path_models / "U_full" / "connectome_shuffle",
     "label": "Shuffle mask"},
    {"path": path_models / "connectome_iMI_cut_50",
     "label": r"Remove 50% iMI $\leftrightarrow$ iMI"},
]

if compute_spectral_distance_distributions:
    # Fetch model configurations from model example
    model_example = load_model(path_model_example)
    dt = model_example.dt
    n_units = model_example.n_units
    n_units_hemi = model_example.n_units_hemi
    n_units_LiMI = len(model_example.idx_LiMI)
    n_units_LcMI = len(model_example.idx_LcMI)
    n_units_LMON = len(model_example.idx_LMON)
    n_units_LsMI = len(model_example.idx_LsMI)
    n_units_RiMI = len(model_example.idx_RiMI)
    n_units_RcMI = len(model_example.idx_RcMI)
    n_units_RMON = len(model_example.idx_RMON)
    n_units_RsMI = len(model_example.idx_RsMI)

    # Data configurations
    dt_data = ConfigurationRNN.dt_data
    dt_data_test = ConfigurationRNN.dt_data_test

    # Phase durations (in seconds) defining the baseline rest, active stimulus, and post-stimulus recovery epochs
    duration_rest_start = 20
    duration_stimulus = 40
    duration_rest_end = 20
    duration_simulation = duration_rest_start + duration_stimulus + duration_rest_end


# ----------------------------------------------------------------
# Plot configuration (layout, sizes, padding, etc.)
# ----------------------------------------------------------------
# Initialize custom canvas styling, plot dimensions, and layout coordinates
style = RNNDSStyle()

plot_height = style.plot_size_big * 3
plot_height_small = plot_height / 2.5

plot_width_small = style.plot_width_small

plot_width = style.plot_size_big * 3

padding = style.padding / 2
padding_big = style.padding * 2
padding_vertical = style.padding

xpos_start = style.xpos_start
ypos_start = style.ypos_start - plot_height/2
xpos = xpos_start
ypos = ypos_start - padding


# ----------------------------------------------------------------
# Initialize figure container
# ----------------------------------------------------------------
# Instantiate figure canvas container
fig = Figure()

if not compute_spectral_distance_distributions:
    assert path_spectral_distance is not None, "Please provide path_spectral_distance or set compute_spectral_distance_distributions to True"
    with open(path_spectral_distance, 'rb') as f:
        spect_dist_dict = pickle.load(f)
else:
    # ----------------------------------------------------------------
    # Define inputs
    # ----------------------------------------------------------------
    # Define input signals used in training: step current applied unilaterally or bilaterally
    def input_signal_constant(duration_rest_start, duration_stimulus, duration_rest_end, side="L", scale=1):
        input_signal_ = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)), np.zeros(int(duration_rest_end / dt)))) * scale
        if side[0].lower() == "l":
            input_signal = np.concatenate((np.repeat(input_signal_[..., np.newaxis], n_units_hemi, axis=1),
                                           np.zeros((len(input_signal_), n_units-n_units_hemi))), axis=1)
        elif side[0].lower() == "r":
            input_signal = np.concatenate((np.zeros((len(input_signal_), n_units_hemi)),
                                           np.repeat(input_signal_[..., np.newaxis], n_units-n_units_hemi, axis=1)), axis=1)
        elif side[0].lower == "a":
            input_signal = np.repeat(input_signal_[..., np.newaxis], n_units, axis=1)

        else:
            input_signal = None
        return input_signal

    # Define input signal for bilateral test with asymmetric scaling between hemispheres
    def input_signal_constant_bilateral(duration_rest_start, duration_stimulus, duration_rest_end, side="L", scale=1, ratio_LR=1):
        input_signal_ = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)), np.zeros(int(duration_rest_end / dt)))) * scale
        input_signal = np.repeat(input_signal_[..., np.newaxis], n_units, axis=1)
        if side[0] in ["l", "L"]:
            input_signal[:, n_units_hemi:] *= ratio_LR
        elif side[0] in ["r", "R"]:
            input_signal[:, :n_units_hemi] *= ratio_LR
        return input_signal

    # Define input signals used for testing (sine wave) to evaluate integration of smooth periodic stimuli
    def input_signal_sine(duration_rest_start, duration_stimulus, duration_rest_end, side="L", scale=1):
        sine = lambda t: 0.5 * np.sin(t-np.pi/2) + 0.5
        input_signal_ = np.concatenate((np.zeros(int(duration_rest_start / dt)), sine(np.arange(0, duration_stimulus, dt)), np.zeros(int(duration_rest_end / dt)))) * scale
        if side[0] in ["l", "L"]:
            input_signal = np.concatenate((np.repeat(input_signal_[..., np.newaxis], n_units_hemi, axis=1),
                                           np.zeros((len(input_signal_), n_units-n_units_hemi))), axis=1)
        elif side[0] in ["r", "R"]:
            input_signal = np.concatenate((np.zeros((len(input_signal_), n_units_hemi)),
                                           np.repeat(input_signal_[..., np.newaxis], n_units-n_units_hemi, axis=1)), axis=1)
        else:
            input_signal = None
        return input_signal

    # Define input signals used for testing (side switching) to test competition and mutual inhibition
    def input_signal_switch(duration_rest_start, duration_stimulus, duration_rest_end, side="L", scale=1):
        duration_stimulus_rest = duration_rest_end / 2
        input_signal_step_first = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)), np.zeros(int((duration_rest_end) / dt)))) * scale
        input_signal_step_second = np.concatenate((np.zeros(int((duration_rest_start + duration_stimulus) / dt)), np.ones(int(duration_stimulus_rest / dt)), np.zeros(int(duration_stimulus_rest / dt)))) * scale

        if side[0] in ["l", "L"]:
            input_signal = np.concatenate((np.repeat(input_signal_step_first[..., np.newaxis], n_units_hemi, axis=1),
                                           np.repeat(input_signal_step_second[..., np.newaxis], n_units-n_units_hemi, axis=1)), axis=1)
        elif side[0] in ["r", "R"]:
            input_signal = np.concatenate((np.repeat(input_signal_step_second[..., np.newaxis], n_units_hemi, axis=1),
                                           np.repeat(input_signal_step_first[..., np.newaxis], n_units-n_units_hemi, axis=1)), axis=1)
        else:
            input_signal = None
        return input_signal


    # ----------------------------------------------------------------
    # Load noise model
    # ----------------------------------------------------------------
    # Instantiate Ornstein-Uhlenbeck (OU) stochastic noise process based on empirical fits
    with open(path_noise_estimation, 'rb') as f:
        p_noise = pickle.load(f)
    def noise_filter(x):
        return DSService.ou_noise(x, p_noise["tau"], p_noise["sigma"], dt_data, 3)


    # ----------------------------------------------------------------
    # Define tests to show
    # ----------------------------------------------------------------
    # Registry of experimental protocols, data source locations, signal shapes, and plotting configurations
    test_list = [
        {"label": "Train high L",
         "duration_rest_start": 20,
         "duration_stimulus": 40,
         "duration_rest_end": 20,
         "path_traces": path_data,
         "path_noise": None,
         "filename_root": "avgresponses",
         "file_extension": "csv",
         "stimulus_name": "constant",
         # "filename": "avgresponses_*_constant.csv", # not used yet
         "combine_data": None,
         "scale_target": None, # Scale the target signal found at path_traces and the input signal. Set to None for no scaling.
         "input_signal": input_signal_constant,
         "time_target_array": None,
         "dt_data": dt_data,
         "side_list": ("preferred", "null"),
         "flip_side": False,
         "cell_type_list": ("iMI", "cMI", "MON", "sMI"),
         "fix_offset_response": True},
        {"label": "Test constant L",
         "duration_rest_start": 16,
         "duration_stimulus": 32,
         "duration_rest_end": 32,
         "path_traces": path_data / "test_dataset" / "cyto8s",
         "path_noise": None,
         "filename_root": "responses",
         "file_extension": "csv",
         "stimulus_name": "constant",
         "filename": "responses_*_constant_*.csv",
         "combine_data": "average",
         "scale_target": None, # Scale the target signal found at path_traces and the input signal. Set to None for no scaling.
         "input_signal": input_signal_constant,
         "time_target_array": np.arange(0, 80-dt_data_test, dt_data_test),
         "dt_data": dt_data_test,
         "side_list": ("left", "right"),
         "flip_side": False,
         "cell_type_list": ("MI", "MON", "SMI"),
         "fix_offset_response": False},
        {"label": "Test sine L",
         "duration_rest_start": 16,
         "duration_stimulus": 32,
         "duration_rest_end": 32,
         "path_traces": path_data / "test_dataset" / "cyto8s",
         "path_noise": None,
         "filename_root": "responses",
         "file_extension": "csv",
         "stimulus_name": "oscillating",
         # "filename": "responses_*_oscillating_*.csv",
         "combine_data": "average",
         "scale_target": None, # Scale the target signal found at path_traces and the input signal. Set to None for no scaling.
         "input_signal": input_signal_sine,
         "time_target_array": np.arange(0, 80-dt_data_test, dt_data_test),
         "dt_data": dt_data_test,
         "side_list": ("left", "right"),
         "flip_side": False,
         "cell_type_list": ("MI", "MON", "SMI"),
         "fix_offset_response": False},
        {"label": "Test switch L",
         "duration_rest_start": 16,
         "duration_stimulus": 32,
         "duration_rest_end": 32,
         "path_traces": path_data / "test_dataset" / "cyto8s",
         "path_noise": None,
         "filename_root": "responses",
         "file_extension": "csv",
         "stimulus_name": "switching",
         # "filename": "responses_*_switching_*.csv",
         "combine_data": "average",
         "scale_target": None, # None or 1: don't scale the target signal found at path_traces and the input signal
         "input_signal": input_signal_switch,
         "time_target_array": np.arange(0, 80-dt_data_test, dt_data_test),
         "dt_data": dt_data_test,
         "side_list": ("left", "right"),
         "flip_side": False,
         "cell_type_list": ("MI", "MON", "SMI"),
         "fix_offset_response": False},
    ]

    # --------------------------------------------------------------------------------
    # Discover and inspect all trained model solutions within the condition directory
    # --------------------------------------------------------------------------------
    model_dict = {}
    for cond in condition_list:
        model_dict[cond['label']] = []
        print(f"Loading models for condition {cond['label']}")
        for path_model in Path(cond["path"]).glob(f"model_*"):

            # Load model instance
            model = load_model(path_model)
            model_dict[cond['label']].append(model)


    # ----------------------------------------------------------------
    # Loop over all tests
    # ----------------------------------------------------------------
    # Preallocate list of loss containers and initialize tracking counters
    spect_dist_dict = {}
    MAX_LOSS = 10
    fix_offset_response = None
    for i_test, test in enumerate(test_list):
        # ----------------------------------------------------------------
        # Load traces to use as target signals
        # ----------------------------------------------------------------
        # Read biological calcium traces across cell populations and sides (handling format variations)
        traces_dict = {ct: {s: None for s in test["side_list"]} for ct in test["cell_type_list"]}
        all_signals = []
        min_traces_all = 0
        for ct in test["cell_type_list"]:
            for s in test["side_list"]:
                try:  # Jon's data naming convention
                    filename = f"{test['filename_root']}_{ct}_{s}_{test['stimulus_name']}.{test['file_extension']}"
                    data_raw = np.loadtxt(test['path_traces'] / filename, dtype=float, delimiter=",", skiprows=1)
                except FileNotFoundError:
                    # Alternative naming convention: swap stimulus name and side order
                    filename = f"{test['filename_root']}_{ct}_{test['stimulus_name']}_{s}.{test['file_extension']}"
                    try:
                        data_raw = np.loadtxt(test['path_traces'] / filename, dtype=float, delimiter=",", skiprows=1)
                    except ValueError:
                        # Fallback: load structured pickle file and locate the matching dictionary key
                        with open(test['path_traces'] / f"{test['filename_root']}.pkl", 'rb') as f:
                            data_dict_raw = pickle.load(f)
                        data_key = [k for k in data_dict_raw.keys() if f"{ct}_{test['stimulus_name']}_{s}" in k][
                            0]  # Extract the first key starting with identifier
                        data_raw = list(data_dict_raw[data_key].values())[0]

                # Average across trials/cells if requested; otherwise preserve raw trace matrix
                if test['combine_data'] == "average":
                    data = np.mean(data_raw, axis=1).copy()
                else:
                    data = data_raw.copy()
                # Parse timestamps and scale dF/F values from percentage to fractional units
                if test['time_target_array'] is None:
                    time_target_array = data[:, 0]
                    traces_dict[ct][s] = data[:, 1] / 100
                else:
                    time_target_array = test['time_target_array']
                    traces_dict[ct][s] = data / 100
                # Track global minimum activity to ensure correct non-negative baseline alignment
                min_trace_here = np.min(traces_dict[ct][s])
                if min_trace_here < min_traces_all:
                    min_traces_all = min_trace_here
        min_traces_all = np.abs(min_traces_all)

        # Define fixed offset based on one reference test, to keep all responses operating in the same range
        if fix_offset_response is None and test["fix_offset_response"]:
            fix_offset_response = min_traces_all
        if fix_offset_response is not None and not test["fix_offset_response"]:
            min_traces_all = fix_offset_response

        # Extract target signal keeping it strictly ordered across hemilateral cell populations
        sanity_check_list = []
        target_signal_list = []
        flip_side = 1 if test['flip_side'] else 0
        for i_s in range(len(test['side_list'])):
            for i_ct in range(len(test['cell_type_list'])):
                target_signal_list.append(
                    traces_dict[test['cell_type_list'][i_ct]][test['side_list'][int(np.abs(i_s - flip_side))]])
                # Handle datasets grouping iMI and cMI into a single unseparated MI population
                if i_ct == 0 and len(
                        test['cell_type_list']) == 3:  # do it again if there is no contra/ipsi differentiation of MI cells
                    target_signal_list.append(
                        traces_dict[test['cell_type_list'][i_ct]][test['side_list'][int(np.abs(i_s - flip_side))]])
                sanity_check_list.append((test['cell_type_list'][i_ct], test['side_list'][int(np.abs(i_s - flip_side))]))
        target_signal = np.stack(target_signal_list, axis=-1).copy()

        # Scale signal (sqrt is applied to scaling factor)
        scale_list = test['scale_target'] if test['scale_target'] is not None else [1]
        if not hasattr(scale_list, "__iter__"):
            scale_list = [scale_list]

            # ----------------------------------------------------------------
            # Load traces to use as target signals
            # ----------------------------------------------------------------
            # Read biological calcium traces across cell populations and sides (handling format variations)
            traces_dict = {ct: {s: None for s in test["side_list"]} for ct in test["cell_type_list"]}
            all_signals = []
            min_traces_all = 0
            for ct in test["cell_type_list"]:
                for s in test["side_list"]:
                    try:  # Jon's data naming convention
                        filename = f"{test['filename_root']}_{ct}_{s}_{test['stimulus_name']}.{test['file_extension']}"
                        data_raw = np.loadtxt(test['path_traces'] / filename, dtype=float, delimiter=",", skiprows=1)
                    except FileNotFoundError:
                        # Alternative naming convention: swap stimulus name and side order
                        filename = f"{test['filename_root']}_{ct}_{test['stimulus_name']}_{s}.{test['file_extension']}"
                        try:
                            data_raw = np.loadtxt(test['path_traces'] / filename, dtype=float, delimiter=",",
                                                  skiprows=1)
                        except ValueError:
                            # Fallback: load structured pickle file and locate the matching dictionary key
                            with open(test['path_traces'] / f"{test['filename_root']}.pkl", 'rb') as f:
                                data_dict_raw = pickle.load(f)
                            data_key = [k for k in data_dict_raw.keys() if f"{ct}_{test['stimulus_name']}_{s}" in k][
                                0]  # Extract the first key starting with identifier
                            data_raw = list(data_dict_raw[data_key].values())[0]

                    # Average across trials/cells if requested; otherwise preserve raw trace matrix
                    if test['combine_data'] == "average":
                        data = np.mean(data_raw, axis=1).copy()
                    else:
                        data = data_raw.copy()
                    # Parse timestamps and scale dF/F values from percentage to fractional units
                    if test['time_target_array'] is None:
                        time_target_array = data[:, 0]
                        traces_dict[ct][s] = data[:, 1] / 100
                    else:
                        time_target_array = test['time_target_array']
                        traces_dict[ct][s] = data / 100
                    # Track global minimum activity to ensure correct non-negative baseline alignment
                    min_trace_here = np.min(traces_dict[ct][s])
                    if min_trace_here < min_traces_all:
                        min_traces_all = min_trace_here
            min_traces_all = np.abs(min_traces_all)

            # Define fixed offset based on one reference test, to keep all responses operating in the same range
            if fix_offset_response is None and test["fix_offset_response"]:
                fix_offset_response = min_traces_all
            if fix_offset_response is not None and not test["fix_offset_response"]:
                min_traces_all = fix_offset_response

            # Extract target signal keeping it strictly ordered across hemilateral cell populations
            sanity_check_list = []
            target_signal_list = []
            flip_side = 1 if test['flip_side'] else 0
            for i_s in range(len(test['side_list'])):
                for i_ct in range(len(test['cell_type_list'])):
                    target_signal_list.append(
                        traces_dict[test['cell_type_list'][i_ct]][test['side_list'][int(np.abs(i_s - flip_side))]])
                    # Handle datasets grouping iMI and cMI into a single unseparated MI population
                    if i_ct == 0 and len(test[
                                             'cell_type_list']) == 3:  # do it again if there is no contra/ipsi differentiation of MI cells
                        target_signal_list.append(
                            traces_dict[test['cell_type_list'][i_ct]][test['side_list'][int(np.abs(i_s - flip_side))]])
                    sanity_check_list.append(
                        (test['cell_type_list'][i_ct], test['side_list'][int(np.abs(i_s - flip_side))]))
            target_signal = np.stack(target_signal_list, axis=-1).copy()

            # Scale signal (sqrt is applied to scaling factor)
            scale_list = test['scale_target'] if test['scale_target'] is not None else [1]
            if not hasattr(scale_list, "__iter__"):
                scale_list = [scale_list]

        # ------------------------------------------------
        # Loop over model conditions
        # ------------------------------------------------
        spect_dist_dict[test["label"]] = {}
        for i_cond, cond in enumerate(condition_list):
            model_list = model_dict[cond['label']]
            # --------------------------------------------------------------
            # Compute and plot responses to test signals
            # --------------------------------------------------------------
            duration_simulation = test['duration_rest_start'] + test['duration_stimulus'] + test['duration_rest_end']

            # Apply stimulus scaling and assemble multi-trial batches
            input_signal_list = []
            output_signal_list = []
            initial_value_list = []
            for s in scale_list:
                # Define input signal for simulation
                input_signal_ = test['input_signal'](test['duration_rest_start'], test['duration_stimulus'],
                                                     test['duration_rest_end'], side="R" if test['flip_side'] else "L",
                                                     scale=s)
                # Define target output signal with square-root amplitude scaling, noise, and baseline shift
                output_signal = target_signal * np.sqrt(s)  # scaling
                output_signal += noise_filter(output_signal)
                output_signal += min_traces_all
                output_signal_list.append(output_signal)
                # Define initial value for simulation: invert softplus to obtain initial latent state voltages
                initial_value = inv_softplus(np.concatenate((np.array([output_signal[0, 0] for _ in range(n_units_LiMI)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 0]) / 5, n_units_LiMI),
                                                             np.array([output_signal[0, 1] for _ in range(n_units_LcMI)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 1]) / 5, n_units_LcMI),
                                                             np.array([output_signal[0, 2] for _ in range(n_units_LMON)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 2]) / 5, n_units_LMON),
                                                             np.array([output_signal[0, 3] for _ in range(n_units_LsMI)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 3]) / 5, n_units_LsMI),
                                                             np.array([output_signal[0, 4] for _ in range(n_units_RiMI)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 4]) / 5, n_units_RiMI),
                                                             np.array([output_signal[0, 5] for _ in range(n_units_RcMI)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 5]) / 5, n_units_RcMI),
                                                             np.array([output_signal[0, 6] for _ in range(n_units_RMON)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 6]) / 5, n_units_RMON),
                                                             np.array([output_signal[0, 7] for _ in range(
                                                                 n_units_RsMI)]))))  # + np.random.normal(0, np.abs(target_signal[0, 7]) / 5, n_units_RsMI)))

                input_signal_list.append(input_signal_)
                initial_value_list.append(initial_value * np.sqrt(s))

            # Convert batched arrays to PyTorch float32 tensors for simulation forward pass
            input_signal = [torch.tensor(signal, dtype=torch.float32) for signal in input_signal_list]
            input_signal = torch.stack(input_signal)
            output_signal = [torch.tensor(signal, dtype=torch.float32) for signal in output_signal_list]
            output_signal = torch.stack(output_signal)
            # Add offset to make the whole signal positive
            x0 = [torch.tensor(iv, dtype=torch.float32) for iv in initial_value_list]
            x0 = torch.stack(x0)

            label = test['label']
            t_sim = np.linspace(0, duration_simulation, input_signal.shape[1])

            # ------------------------------------------------
            # Simulate models in condition
            # ------------------------------------------------
            print(f"Simulating models for condition {cond['label']}")
            res = RNNService.plot_response_by_cell(model_list, t_sim, input_signal,
                                                   t_exp=time_target_array, output_signal_array=output_signal, x0=x0,
                                                   compute_spectrum=True,
                                                   time_structure={"rest_start": test['duration_rest_start'],
                                                                   "stimulus": test['duration_stimulus'],
                                                                   "rest_end": test['duration_rest_end'],
                                                                   "duration": duration_simulation})
            spectrum = res["spectrum"]
            for input_key in spectrum.keys():
                spectral_similarity_list = [s["harmonic_spectral_score"] for s in spectrum[input_key]]
                print(f"{input_key} | Spectral similarity: {np.mean(spectral_similarity_list):.04f}" + r"\pm" + f"{np.std(spectral_similarity_list):.04f}\n")
                break  # only interested in 1 (baseline) scaling of each test

            spect_dist_dict[test["label"]][cond["label"]] = spectral_similarity_list

    # Store result
    with open(path_models / f"spectral_distance_dict_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.pkl", 'wb') as f:
        pickle.dump(spect_dist_dict, f)


# ------------------------------------------------
# Display distributions
# ------------------------------------------------
# Make one distribution per condition per test, organize in a 2x2 grid, of distributions,
# each with a test, containing the violins for the 3 model conditions.
N_COND = len(condition_list)
if show_spect_dist:
    # Set y-axis upper boundary for loss display

    min_value, max_value = 0, 1
    for i_test_label, test_label in enumerate(spect_dist_dict.keys()):
        spect_dist_list = [cond for cond in spect_dist_dict[test_label].values()]

        # Create categorical plot panel for loss distributions
        plot_dist = fig.create_plot(plot_title=f"Spectral similarity for {test_label}",
                                         xpos=xpos, ypos=ypos, plot_height=plot_height,
                                         plot_width=plot_width,
                                         xmin=0.5, xmax=N_COND+0.5, xticks=np.arange(N_COND)+1,
                                         xticklabels=[cond["label"] for cond in condition_list],
                                         xticklabels_rotation=45,
                                         ymin=min_value, ymax=max_value, yticks=[min_value, max_value])

        # Run non-parametric Mann-Whitney U test comparing each condition against baseline cohort (index 0)
        for i_test, loss in enumerate(spect_dist_list[1:]):
            res = mannwhitneyu(np.array(spect_dist_list[0]), np.array(loss))
            print(f"Test {i_test} | p-val Original vs {condition_list[i_test+1]['label']}: {res.pvalue}")
            # Annotate significant differences with an asterisk marker near top of panel
            if res.pvalue < 0.005:
                ytext = max_value*9/10
                plot_dist.draw_text((i_test+2), ytext, "*")

        # Render violin distribution plots for each cohort
        plot_dist.draw_violin(spect_dist_list, facecolor="k")

        # Advance coordinates for subsequent panels
        if i_test_label == 1:
            xpos = xpos_start
            ypos -= plot_height + padding
        else:
            xpos += plot_width + padding


# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
# Ensure target output directory exists and save vector figure as PDF
path_save.mkdir(parents=True, exist_ok=True)
filename = f"figure_spectral_analysis{label_figure}.pdf" if label_figure is not None else "figure_compare.pdf"
fig.save(path_save / filename, open_file=False, tight=style.page_tight)