"""
Overview
--------
This script generates the following figure panels in Boulanger-Weiss et al. 2026:
- Fig. 5...

This script evaluets trained recurrent neural network (RNN) dynamical models against
empirical calcium imaging traces (e.g., zebrafish hindbrain oculomotor / velocity storage
neural integrator populations).

Before running it, please make sure you have set up your .env with all the necessary variables:
- PATH_DIR="/path/to/project_root"  # Root project directory containing data/, models/, and results/

Core Pipeline & Workflow:
1. Configuration & Environment Setup:
   - Sets up label tags, feature toggle flags (loss distributions, weight matrices,
     connectivity statistics, dynamic activity simulations), and paths loaded from .env.
   - Configures coordinate layouts, plot styling, bounding boxes, and canvas origins
     via `RNNDSStyle` and custom `Figure` containers.

2. Diagnostic Visualizations (Optional Modules):
   - Loss Distributions: Reads loss values across saved models and renders an empirical
     distribution histogram.
   - Model Loading & Weight Inspection: Loads trained PyTorch RNN models, extracts recurrent
     weight matrices (W), input weights (U), anatomical cell-type boundaries, and mask
     topologies, and renders 2D connectivity heatmaps and connection probability / rediscovery
     rate statistics across neural populations.

3. Dynamic Testing & Simulation:
   - Constructs input driving functions (unilateral constant steps, bilateral differential
     steps, sinusoidal oscillations, and side-switching stimuli) with flexible timing
     (rest start, stimulus drive, rest end).
   - Iterates through a test suite defining experimental protocols (training conditions vs.
     generalization test conditions such as sine waves and side switches).
   - Dynamically loads empirical reference target traces from CSV or pickle files, handles
     cell-type mappings (iMI, cMI, MON, sMI across hemispheres), rescales traces, and applies
     noise or systematic baseline offsets.
   - Initializes network dynamical states by computing inverse softplus mappings on baseline
     activity, propagates inputs through PyTorch models, and renders population response
     trajectories juxtaposed against experimental data.

4. Figure Export:
   - Saves the assembled multi-panel composition as a vector PDF in the designated output
     directory.
"""

import pickle

import torch
import numpy as np
from pathlib import Path
from dotenv import dotenv_values

# Manually add root path for imports to improve interoperability across nested research subdirectories
import sys; sys.path.insert(0, "..")

from style import RNNDSStyle
from utils.services.ds_service import DSService
from utils.services.rnn_service import RNNService
from utils.math.operators import inv_softplus, get_hist
from utils.figure_helper import Figure
from utils.load_model import load_model

# ------------------------------------------------
# Configuration
# ------------------------------------------------
# ---- Paths -------------------------------------
# Base label descriptor for model selection, subfolder organization, and figure export names
label = "connectome"
label_load = "connectome" if label is None else label
label_save = None if label is None else f"_{label}"

# ---- Show -------------------------------------
# Feature flags controlling which diagnostic panels are computed and drawn on the figure canvas
show_loss_histograms = False
show_matrices = True
show_connectivity_stats = False
show_activity = True

remove_clamping = False
# If set to an integer N, restricts model evaluation to the top N pre-selected checkpoint models
show_only_n_top_models = None  # activating this requires to first run select_models on the specific models directory

# ------------------------------------------------
# Env and paths
# ------------------------------------------------
# Retrieve filesystem paths from environment configuration (.env file)
env = dotenv_values()
path_dir = Path(env["PATH_DIR"])
path_data = path_dir / "data" # directory containing avgresponse_X.csv
path_noise_estimation = path_dir / "data" / "noise_estimation" / "contralateral_motion_integrator_preferred_noise_estimation.pkl"
path_models = path_dir / "models" / label_load # directory containing model_X.pt
if show_only_n_top_models:
    path_models = path_models / f"top_{show_only_n_top_models}"
    label_save = f"{label_save}_top{show_only_n_top_models}"
path_save = path_dir / "results"

# ---- Simulate ----------------------------------
# Default temporal discretization (time-step size in seconds) for reference signals and simulation
dt_data = 0.5
dt_data_test = 0.1
# Phase durations (in seconds) defining the baseline rest, active stimulus, and post-stimulus recovery epochs
duration_rest_start = 20
duration_stimulus = 40
duration_rest_end = 20
duration_simulation = duration_rest_start + duration_stimulus + duration_rest_end

# ----------------------------------------------------------------
# Plot configuration (layout, sizes, padding, etc.)
# ----------------------------------------------------------------
# Initialize plotting style specifications, dimensions, margins, and canvas origin coordinates
style = RNNDSStyle()

plot_height = style.plot_height
plot_height_small = plot_height / 2.5

plot_width = style.plot_width
plot_width_small = style.plot_width_small

plot_size_matrix = style.plot_size_big * 2

padding = style.padding / 2
padding_big = style.padding * 2
padding_vertical = style.padding

xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start - padding

# ----------------------------------------------------------------
# Initialize figure container
# ----------------------------------------------------------------
# Construct custom canvas manager for assembling multi-axis vector graphics
fig = Figure()

# ----------------------------------------------------------------
# Plot histogram of loss functions
# ----------------------------------------------------------------
if show_loss_histograms:
    # Plot loss histogram for trained models across different structural masks or runs
    loss_list = []
    for path_model in path_models.glob(f"model_*.pt"):
        model = load_model(path_model)
        loss_list.append(model.loss_mse)

    # Determine optimal loss and compute histogram bin counts across predefined loss bounds
    best_loss = loss_list[np.argsort(loss_list)[0]]
    max_value = 0.03
    n_bins = 90
    h, b = get_hist(loss_list, bins=n_bins, hist_range=(0, max_value), center_bin=True)
    plot_loss_dist = fig.create_plot(plot_title="Loss distribution\nacross masks",
                                     xpos=xpos, ypos=ypos, plot_height=plot_height,
                                     plot_width=plot_size_matrix,
                                     xmin=0, xmax=max_value, xticks=[0, max_value],
                                     ymin=0, ymax=30, yticks=[0, 15, 30])
    width_bin = max_value/n_bins
    plot_loss_dist.draw_vertical_bars(b, h, vertical_bar_width=width_bin-width_bin/3)

    # Shift layout coordinates horizontally, then wrap to start of next vertical row
    xpos += plot_size_matrix + padding

    xpos = xpos_start
    ypos -= plot_height + padding_big

# ----------------------------------------------------------------
# Load model instance
# ----------------------------------------------------------------
# Load model solutions matching the specified pattern and configure evaluation states
i_model = 0
model_list = []
for path_model in path_models.glob(f"model_*.pt"):
    model_ = load_model(path_model)
    if model_ is None:
        print(f"Model {path_model} could not be loaded. Skipping evaluation.")
        continue
    model = model_ # two-steps assignment to protect from wrongly loaded models
    model.eval()
    print(f"Loading model {path_model}")
    if remove_clamping:
        # Remove clamping from model (i.e. lower bound constraints)
        model.clamp_weights_min = 0

    model_list.append(model)

# Extract core architecture parameters, time resolution, and population unit sizes from the last active model
dt = model.dt
n_units_hemi = model.n_units_hemi
n_units_LiMI = len(model.idx_LiMI)
n_units_LcMI = len(model.idx_LcMI)
n_units_LMON = len(model.idx_LMON)
n_units_LsMI = len(model.idx_LsMI)
n_units_RiMI = len(model.idx_RiMI)
n_units_RcMI = len(model.idx_RcMI)
n_units_RMON = len(model.idx_RMON)
n_units_RsMI = len(model.idx_RsMI)
n_units = model.n_units

# Plot U, W, and associated masks
if show_matrices or show_connectivity_stats:
    ypos = ypos - plot_size_matrix / 2 # give a bit of extra shift so that the big image doesn't overflow up

    colormap = RNNDSStyle.cmap_list["neurons_4"]

    # Construct categorical index array mapping each neuron index to its discrete anatomical cell type
    # (0: iMI, 1: cMI, 2: MON, 3: sMI for Left and Right hemispheres)
    neuron_identity_array = np.concatenate(
        (np.zeros((n_units_LiMI, 1)), np.ones((n_units_LcMI, 1)), 2 * np.ones((n_units_LMON, 1)),
         3 * np.ones((n_units_LsMI, 1)),
         np.zeros((n_units_RiMI, 1)), np.ones((n_units_RcMI, 1)), 2 * np.ones((n_units_RMON, 1)),
         3 * np.ones((n_units_RsMI, 1))))
    # Normalize population indices to [0, 1] for categorical colormap lookup
    neuron_identity_array /= 3

    # Extract input matrix U, recurrent matrix W, and binary connectivity masks from the PyTorch model
    U = model.U().detach().numpy()
    mask_U = model.mask_U.detach().numpy()
    W = model.W().detach().numpy().T
    try:
        mask_W = (model.mask_W).detach().numpy().T
    except AttributeError:
        mask_W = None

    # Compute boundary indices partitioning the connectivity matrix into distinct cell-type subpopulations
    grid_pop = np.array([n_units_LiMI, n_units_LiMI + n_units_LcMI, n_units_LiMI + n_units_LcMI + n_units_LMON,
                         n_units_LiMI + n_units_LcMI + n_units_LMON + n_units_LsMI,
                         n_units_hemi + n_units_RiMI, n_units_hemi + n_units_RiMI + n_units_RcMI,
                         n_units_hemi + n_units_RiMI + n_units_RcMI + n_units_RMON,
                         n_units_hemi + n_units_RiMI + n_units_RcMI + n_units_RMON + n_units_RsMI])

    if show_matrices:
        print("Plotting connectivity matrices")
        # Render structural mask heatmap if available
        if mask_W is not None:
            _, xpos, ypos = RNNService.plot_connectivity(mask_W, U=mask_U, neuron_identity_array=neuron_identity_array, grid_pop=grid_pop,
                                                         fig=fig, xpos=xpos, ypos=ypos, plot_size_matrix=plot_size_matrix,
                                                         padding=padding, value_lim=[-1, 1], plot_title=f"Mask W", cmap_pop=colormap)

        # Render recurrent connectivity matrix W with input weights U
        _, xpos, ypos = RNNService.plot_connectivity(W, U=U, neuron_identity_array=neuron_identity_array, grid_pop=grid_pop,
                                                     fig=fig, xpos=xpos, ypos=ypos, plot_size_matrix=plot_size_matrix,
                                                     padding=padding, value_lim=[-1, 1], cmap_pop=colormap)
        xpos += plot_size_matrix

    if show_connectivity_stats:
        print("Plotting connectivity stats")
        # Extract population connectivity statistics (sparsity and biological rediscovery rate)
        Wt = W.T
        dict_neurons = model.dict_neurons
        pop_idx_list = [dict_neurons["neurons"][side][cell]["idx_list"] \
                        for side in dict_neurons["neurons"].keys() for cell in dict_neurons["neurons"][side].keys()
                        if cell != "idx_list"]
        sparsity_W = RNNService.compute_sparsity(Wt, pop_idx_list)
        rediscovery_rate_pop = RNNService.compute_rediscovery_rate(Wt, pop_idx_list)[..., np.newaxis]
        pop_identity_array = np.concatenate((np.arange(4), np.arange(4)), dtype=float)[..., np.newaxis]
        pop_identity_array /= np.max(pop_identity_array)
        # Plot population connection probability and rediscovery rates
        _, xpos, ypos = RNNService.plot_connectivity(sparsity_W.T, U=rediscovery_rate_pop,
                                                     neuron_identity_array=pop_identity_array,
                                                     grid_pop=None,
                                                     fig=fig, xpos=xpos, ypos=ypos,
                                                     plot_size_matrix=plot_size_matrix,
                                                     padding=padding, value_lim=[0, 0.15], value_lim_U=[-1, 1],
                                                     cmap="Blues", cmap_U="PiYG", logscale=False,
                                                     plot_title=f"Connection probability",
                                                     plot_title_U="Rediscovery rate", show_text=True)
        xpos = xpos_start
        ypos -= plot_size_matrix + padding

if show_activity:
    print("Simulating networks and plotting activity")
    # Rebase coordinates for response plots
    plot_size_here = style.plot_size_big * 2/3
    padding_here = padding
    xpos_start_here = xpos_here = xpos_start
    ypos_start_here = ypos_here = ypos - padding_here

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
         "scale_target": [0.3, 1], # Scale the target signal found at path_traces and the input signal. Set to None for no scaling.
         "input_signal": input_signal_constant,
         "time_target_array": None,
         "dt_data": dt_data,
         "side_list": ("preferred", "null"),
         "flip_side": False,
         "cell_type_list": ("iMI", "cMI", "MON", "sMI"),
         "fix_offset_response": True},
        # {"label": "Train high BOTH",
        #  "duration_rest_start": 20,
        #  "duration_stimulus": 40,
        #  "duration_rest_end": 20,
        #  "path_traces": path_data,
        #  "path_noise": None,
        #  "filename_root": "avgresponses",
        #  "file_extension": "csv",
        #  "stimulus_name": "constant",
        #  # "filename": "avgresponses_*_constant.csv", # not used yet
        #  "combine_data": None,
        #  "scale_target": [1], # Scale the target signal found at path_traces and the input signal. Set to None for no scaling.
        #  "input_signal": input_signal_constant_bilateral,
        #  "time_target_array": None,
        #  "dt_data": dt_data,
        #  "side_list": ("preferred", "null"),
        #  "flip_side": False,
        #  "cell_type_list": ("iMI", "cMI", "MON", "sMI"),
        #  "fix_offset_response": True},
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

    # ----------------------------------------------------------------
    # Loop over all tests to show
    # ----------------------------------------------------------------
    fix_offset_response = None
    for i_test, test in enumerate(test_list):
        # ----------------------------------------------------------------
        # Load noise model
        # ----------------------------------------------------------------
        # If specified, instantiate Ornstein-Uhlenbeck (OU) stochastic noise process based on empirical fits
        if test["path_noise"] is not None:
            with open(test["path_noise"], 'rb') as f:
                p_noise = pickle.load(f)
            def noise_filter(x):
                return DSService.ou_noise(x, p_noise["tau"], p_noise["sigma"], dt_data, 3)
        else:
            def noise_filter(x): # no noise applied to augment the dataset
                return np.zeros_like(x)

        # ----------------------------------------------------------------
        # Load traces to use as target signals
        # ----------------------------------------------------------------
        # Read biological calcium traces across cell populations and sides (handling format variations)
        traces_dict = {ct: {s: None for s in test["side_list"]} for ct in test["cell_type_list"]}
        all_signals = []
        min_traces_all = 0
        for ct in test["cell_type_list"]:
            for s in test["side_list"]:
                try: # Jon's data naming convention
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
                        data_key = [k for k in data_dict_raw.keys() if f"{ct}_{test['stimulus_name']}_{s}" in k][0] # Extract the first key starting with identifier
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
                target_signal_list.append(traces_dict[test['cell_type_list'][i_ct]][test['side_list'][int(np.abs(i_s - flip_side))]])
                # Handle datasets grouping iMI and cMI into a single unseparated MI population
                if i_ct == 0 and len(test['cell_type_list']) == 3: # do it again if there is no contra/ipsi differentiation of MI cells
                    target_signal_list.append(traces_dict[test['cell_type_list'][i_ct]][test['side_list'][int(np.abs(i_s - flip_side))]])
                sanity_check_list.append((test['cell_type_list'][i_ct], test['side_list'][int(np.abs(i_s - flip_side))]))
        target_signal = np.stack(target_signal_list, axis=-1).copy()

        # Scale signal (sqrt is applied to scaling factor)
        scale_list = test['scale_target'] if test['scale_target'] is not None else [1]
        if not hasattr(scale_list, "__iter__"):
            scale_list = [scale_list]

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
            input_signal_ = test['input_signal'](test['duration_rest_start'], test['duration_stimulus'], test['duration_rest_end'], side="R" if test['flip_side'] else "L", scale=s)
            # Define target output signal with square-root amplitude scaling, noise, and baseline shift
            output_signal = target_signal * np.sqrt(s) # scaling
            output_signal += noise_filter(output_signal)
            output_signal += min_traces_all
            output_signal_list.append(output_signal)
            # Define initial value for simulation: invert softplus to obtain initial latent state voltages
            initial_value = inv_softplus(np.concatenate((np.array([output_signal[0, 0] for _ in range(n_units_LiMI)]), # + np.random.normal(0, np.abs(target_signal[0, 0]) / 5, n_units_LiMI),
                                                         np.array([output_signal[0, 1] for _ in range(n_units_LcMI)]), # + np.random.normal(0, np.abs(target_signal[0, 1]) / 5, n_units_LcMI),
                                                         np.array([output_signal[0, 2] for _ in range(n_units_LMON)]), # + np.random.normal(0, np.abs(target_signal[0, 2]) / 5, n_units_LMON),
                                                         np.array([output_signal[0, 3] for _ in range(n_units_LsMI)]), # + np.random.normal(0, np.abs(target_signal[0, 3]) / 5, n_units_LsMI),
                                                         np.array([output_signal[0, 4] for _ in range(n_units_RiMI)]), # + np.random.normal(0, np.abs(target_signal[0, 4]) / 5, n_units_RiMI),
                                                         np.array([output_signal[0, 5] for _ in range(n_units_RcMI)]), # + np.random.normal(0, np.abs(target_signal[0, 5]) / 5, n_units_RcMI),
                                                         np.array([output_signal[0, 6] for _ in range(n_units_RMON)]), # + np.random.normal(0, np.abs(target_signal[0, 6]) / 5, n_units_RMON),
                                                         np.array([output_signal[0, 7] for _ in range(n_units_RsMI)])))) # + np.random.normal(0, np.abs(target_signal[0, 7]) / 5, n_units_RsMI)))

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
        # Execute RNN forward simulation and render cell-by-cell response comparison traces
        res = RNNService.plot_response_by_cell(model_list, t_sim, input_signal, xpos_here, ypos_here,
                                               t_exp=time_target_array, output_signal_array=output_signal, x0=x0,
                                               fig=fig, show_xaxis=True, show_yaxis=True, compute_tau=False,
                                               plot_title_label=label, plot_size=plot_size_here,
                                               time_structure={"rest_start": test['duration_rest_start'], "stimulus": test['duration_stimulus'], "rest_end": test['duration_rest_end'], "duration": duration_simulation})

        # Advance horizontal position for the next condition panel
        xpos_here += plot_size_here * 2 + padding_here * 3

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
# Ensure destination folder exists and write the assembled figure to PDF
path_save.mkdir(parents=True, exist_ok=True)
fig.save(path_save / f"figure_main_train_test{'' if label_save is None else label_save}.pdf",
         open_file=False, tight=style.page_tight)