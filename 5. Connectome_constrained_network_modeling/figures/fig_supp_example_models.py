import pickle
import torch

import numpy as np
from pathlib import Path
from dotenv import dotenv_values

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "..")

from style import RNNDSStyle
from utils.services.ds_service import DSService
from utils.services.rnn_service import RNNService
from utils.math.operators import get_hist, inv_softplus
from utils.figure_helper import Figure
from utils.load_model import load_model

# ------------------------------------------------
# Env and paths
# ------------------------------------------------
env = dotenv_values()
path_dir = Path(env["PATH_DIR"])
path_traces = Path(env["PATH_DATA"])
path_noise_estimation = Path(env["PATH_NOISE_ESTIMATION"])
path_models = Path(env["PATH_MODELS"])   # directory containing avgresponses_X.csv
path_save = Path(env["PATH_SAVE"])

# ------------------------------------------------
# Configuration
# ------------------------------------------------
loop_over_trained_models = False

n_input_signal = 2
dt = 0.01
dt_data = 0.5
duration_rest_start = 20
duration_stimulus = 40
duration_rest_end = 20
duration_simulation = duration_rest_start + duration_stimulus + duration_rest_end

# ================================================================
# Plot configuration (layout, sizes, padding, etc.)
# ================================================================
style = RNNDSStyle()

xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

plot_height = style.plot_height
plot_height_small = plot_height / 2.5

plot_width = style.plot_width
plot_width_small = style.plot_width_small

plot_size = style.plot_size_small
plot_size_matrix = style.plot_size_big * 1.2

padding = style.padding / 2
padding_big = style.padding * 2
padding_vertical = style.padding

palette = style.palette["neurons_4"]

# ================================================================
# Initialize figure container
# ================================================================
fig = Figure()

# ================================================================
# Loop over all trained models
# ================================================================
n_model = 0
loss_list = []
path_model_list = []
for path_model in path_models.glob(f"model_*.pt"):

    # Load model instance
    model = load_model(path_model)
    # with open(path_model, 'rb') as f:
    #     model = pickle.load(f)
    model.eval()

    # Extract simulation configuration and architecture from the first model found
    if n_model == 0:
        dt = model.dt

        n_units_A = model.nA
        n_units_B = model.nB
        n_units_C = model.nC
        n_units_D = model.nD
        n_units_hemi = n_units_A + n_units_B + n_units_C + n_units_D
        n_units = n_units_hemi * 2

        if hasattr(model, "nX"):                     # This is the case for RNNFreePop models
            model_is_free_pop = True
            n_units_X = model.nX
            n_units += n_units_X
            cmap_here = style.cmap_list["neurons_5"]
        else:   # This is the case for RNNFreeNeurons models
            model_is_free_pop = False
            cmap_here = style.cmap_list["neurons_4"]

    try:
        loss = model.loss_mse
    except AttributeError:
        continue

    path_model_list.append(path_model)
    loss_list.append(loss)
    n_model += 1

# ================================================================
# Extract best, median and worst models
# ================================================================
top_indices = np.argsort(loss_list)
index_best_model = top_indices[0]
index_median_model = top_indices[int(n_model / 2)]
index_worst_model = top_indices[-1]
ref_model_list = [{"label": "Best",
                   "path": path_model_list[index_best_model]},
                  {"label": "Median",
                   "path": path_model_list[index_median_model]},
                  # {"label": "Worst",
                  #  "path": path_model_list[index_worst_model]}
                 ]

# ================================================================
# Plot histogram for models across sparsity masks
# ================================================================
max_value = 0.025
n_bins = 20
h, b = get_hist(loss_list, bins=n_bins, hist_range=(0, max_value), center_bin=True)
plot_loss_dist = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=plot_height,
                                 plot_width=plot_size_matrix,
                                 xmin=0, xmax=max_value, xticks=[0, max_value],
                                 ymin=0, ymax=50, yticks=[0, 25, 50],
                                 xl="Loss", yl="Model count",
                                 vlines=[loss_list[index_best_model], loss_list[index_median_model]], helper_lines_lc="r")
plot_loss_dist.draw_vertical_bars(b, h, vertical_bar_width=max_value/n_bins)

xpos = xpos_start
ypos -= plot_height + padding_big

# ================================================================
# Define input signals used in training and test configuration
# ================================================================
def input_signal_constant(duration_rest_start, duration_stimulus, duration_rest_end, side="L", scale=1):
    input_signal_ = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)), np.zeros(int(duration_rest_end / dt)))) * scale
    if side[0] in ["l", "L"]:
        input_signal = np.concatenate((np.repeat(input_signal_[..., np.newaxis], n_units_hemi, axis=1),
                                             np.zeros((len(input_signal_), n_units_hemi))), axis=1)
    elif side[0] in ["r", "R"]:
        input_signal = np.concatenate((np.zeros((len(input_signal_), n_units_hemi)),
                                             np.repeat(input_signal_[..., np.newaxis], n_units_hemi, axis=1)), axis=1)
    else:
        input_signal = None
    return input_signal

test_list = [
    {"label": "Train high L",
     "duration_rest_start": 20,
     "duration_stimulus": 40,
     "duration_rest_end": 20,
     "path_traces": path_traces,
     "path_noise": None,
     "filename_root": "avgresponses",
     "file_extension": "csv",
     "stimulus_name": "constant",
     # "filename": "avgresponses_*_constant.csv",  # not used yet
     "combine_data": None,
     "scale_target": [0.1, 1],  # None or 1: don't scale the target signal found at path_traces and the input signal
     "input_signal": input_signal_constant,
     "time_target_array": None,
     "dt_data": dt_data,
     "side_list": ("preferred", "null"),
     "flip_side": False,
     "cell_type_list": ("iMI", "cMI", "MON", "sMI"),
     "fix_offset_response": True},
]

# ================================================================
# Show model parameters and performance on training dataset
# ================================================================
for ref_model in ref_model_list:
    # Load model instance
    model = load_model(ref_model["path"])
    # with open(ref_model["path"], 'rb') as f:
    #     model = pickle.load(f)
    model.eval()

    W = model.W().detach().numpy().T
    U = model.U().detach().numpy()
    offset_hemisphere = model.nA + model.nB + model.nC + model.nD
    neuron_identity_array = np.concatenate(
        (np.zeros((model.nA, 1)), np.ones((model.nB, 1)), 2 * np.ones((model.nC, 1)), 3 * np.ones((model.nD, 1)),
         np.zeros((model.nA, 1)), np.ones((model.nB, 1)), 2 * np.ones((model.nC, 1)), 3 * np.ones((model.nD, 1)),
         4 * np.ones((model.nX, 1)))) / 4
    grip_pop = np.array([model.nA, model.nA + model.nB, model.nA + model.nB + model.nC,
                                          model.nA + model.nB + model.nC + model.nD,
                                          offset_hemisphere + model.nA, offset_hemisphere + model.nA + model.nB,
                                          offset_hemisphere + model.nA + model.nB + model.nC])

    _, xpos, ypos = RNNService.plot_connectivity(W, U=U, neuron_identity_array=neuron_identity_array, grid_pop=None,
                                                 fig=fig, xpos=xpos, ypos=ypos, plot_size_matrix=plot_size_matrix,
                                                 padding=padding, value_lim=[-1, 1])
    xpos += padding*2

    plot_size_here = style.plot_size_big * 2/3
    padding_here = padding
    fix_offset_response = None
    for i_test, test in enumerate(test_list):
        # ================================================================
        # Load noise model
        # ================================================================
        if test["path_noise"] is not None:
            with open(test["path_noise"], 'rb') as f:
                p_noise = pickle.load(f)


                def noise_filter(x):
                    return DSService.ou_noise(x, p_noise["tau"], p_noise["sigma"], dt_data, 3)
        else:
            def noise_filter(x):  # no noise applied to augment the dataset
                return np.zeros_like(x)

            # Load traces to use as target signals
            traces_dict = {ct: {s: None for s in test["side_list"]} for ct in test["cell_type_list"]}
            all_signals = []
            min_traces_all = 0
            for ct in test["cell_type_list"]:
                for s in test["side_list"]:
                    try:  # Jon's data
                        filename = f"{test['filename_root']}_{ct}_{s}_{test['stimulus_name']}.{test['file_extension']}"
                        data_raw = np.loadtxt(test['path_traces'] / filename, dtype=float, delimiter=",", skiprows=1)
                    except FileNotFoundError:
                        filename = f"{test['filename_root']}_{ct}_{test['stimulus_name']}_{s}.{test['file_extension']}"
                        try:
                            data_raw = np.loadtxt(test['path_traces'] / filename, dtype=float, delimiter=",",
                                                  skiprows=1)
                        except ValueError:
                            with open(test['path_traces'] / f"{test['filename_root']}.pkl", 'rb') as f:
                                data_dict_raw = pickle.load(f)
                                data_raw = list(data_dict_raw[f"{ct}_{test['stimulus_name']}_{s}"].values())[0]

                    if test['combine_data'] == "average":
                        data = np.mean(data_raw, axis=1).copy()
                    else:
                        data = data_raw.copy()
                    if test['time_target_array'] is None:
                        time_target_array = data[:, 0]
                        traces_dict[ct][s] = data[:, 1] / 100
                    else:
                        time_target_array = test['time_target_array']
                        traces_dict[ct][s] = data / 100
                    min_trace_here = np.min(traces_dict[ct][s])
                    if min_trace_here < min_traces_all:
                        min_traces_all = min_trace_here
            min_traces_all = np.abs(min_traces_all)

            # Define fixed offset based on one reference test, to keep all responses operating in the same range
            if fix_offset_response is None and test["fix_offset_response"]:
                fix_offset_response = min_traces_all
            if fix_offset_response is not None and not test["fix_offset_response"]:
                min_traces_all = fix_offset_response

            # Extract target signal keeping it strictly ordered
            sanity_check_list = []
            target_signal_list = []
            flip_side = 1 if test['flip_side'] else 0
            for i_s in range(len(test['side_list'])):
                for i_ct in range(len(test['cell_type_list'])):
                    target_signal_list.append(
                        traces_dict[test['cell_type_list'][i_ct]][test['side_list'][int(np.abs(i_s - flip_side))]])
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

            # Compute and plot responses to test signals
            duration_simulation = test['duration_rest_start'] + test['duration_stimulus'] + test['duration_rest_end']

            # apply scaling
            input_signal_list = []
            output_signal_list = []
            initial_value_list = []
            for s in scale_list:
                # Define input signal for simulation
                input_signal_ = test['input_signal'](test['duration_rest_start'], test['duration_stimulus'],
                                                     test['duration_rest_end'], side="R" if test['flip_side'] else "L",
                                                     scale=s)
                # Define target output signal
                output_signal = target_signal * np.sqrt(s)  # scaling
                output_signal += noise_filter(output_signal)
                output_signal += min_traces_all
                output_signal_list.append(output_signal)
                # Define initial value for simulation
                initial_value = inv_softplus(np.concatenate((np.array([output_signal[0, 0] for _ in range(n_units_A)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 0]) / 5, n_units_A),
                                                             np.array([output_signal[0, 1] for _ in range(n_units_B)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 1]) / 5, n_units_B),
                                                             np.array([output_signal[0, 2] for _ in range(n_units_C)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 2]) / 5, n_units_C),
                                                             np.array([output_signal[0, 3] for _ in range(n_units_D)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 3]) / 5, n_units_D),
                                                             np.array([output_signal[0, 4] for _ in range(n_units_A)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 4]) / 5, n_units_A),
                                                             np.array([output_signal[0, 5] for _ in range(n_units_B)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 5]) / 5, n_units_B),
                                                             np.array([output_signal[0, 6] for _ in range(n_units_C)]),
                                                             # + np.random.normal(0, np.abs(target_signal[0, 6]) / 5, n_units_C),
                                                             np.array([output_signal[0, 7] for _ in range(
                                                                 n_units_D)]))))  # + np.random.normal(0, np.abs(target_signal[0, 7]) / 5, n_units_D)))

                if model_is_free_pop:
                    input_signal_ = np.concatenate((input_signal_, np.zeros((len(input_signal_), n_units_X))), axis=1)
                    initial_value = np.concatenate((initial_value, np.array(
                        [np.mean(output_signal[0]) for _ in range(n_units_X)]) + np.random.normal(0, np.abs(
                        np.mean(output_signal[0])) / 5, n_units_X)))

                input_signal_list.append(input_signal_)
                initial_value_list.append(initial_value * np.sqrt(s))

            input_signal = [torch.tensor(signal, dtype=torch.float32) for signal in input_signal_list]
            input_signal = torch.stack(input_signal)
            output_signal = [torch.tensor(signal, dtype=torch.float32) for signal in output_signal_list]
            output_signal = torch.stack(output_signal)
            # Add offset to make the whole signal positive
            x0 = [torch.tensor(iv, dtype=torch.float32) for iv in initial_value_list]
            x0 = torch.stack(x0)

            label = test['label']
            t_sim = np.linspace(0, duration_simulation, input_signal.shape[1])
            res = RNNService.plot_response_by_cell(model, t_sim, input_signal, xpos, ypos+2*plot_size_here+padding_here,
                                                   t_exp=time_target_array, output_signal=output_signal, x0=x0,
                                                   fig=fig, show_xaxis=True, show_yaxis=i_test % 2 == 0,
                                                   plot_title_label=label, plot_size=plot_size_here, compute_tau=False,
                                                   time_structure={"rest_start": test['duration_rest_start'],
                                                                   "stimulus": test['duration_stimulus'],
                                                                   "rest_end": test['duration_rest_end'],
                                                                   "duration": duration_simulation})

            xpos = xpos_start
            ypos -= plot_size_here * 3 + padding_here * 3

# ================================================================
# Save final figure
# ================================================================
path_save.mkdir(parents=True, exist_ok=True)
fig.save(path_save / "figure_supp_example_models.pdf", open_file=False, tight=style.page_tight)
