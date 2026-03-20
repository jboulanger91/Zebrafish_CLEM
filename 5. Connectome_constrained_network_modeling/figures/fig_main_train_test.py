import pickle

import torch
import numpy as np
from pathlib import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable

from plot.style import RNNDSStyle
from utils.ds_service import DSService
from utils.operators import inv_softplus, get_hist
from utils.rnn_service import RNNService
from utils.figure_helper import Figure

# ------------------------------------------------
# Configuration
# ------------------------------------------------
# ---- Paths -------------------------------------
path_dir = Path(r"C:\Users\Roberto\Desktop\highlights\clem_rnns\data")   # directory containing model_X.pkl
path_noise_estimation = path_dir / "noise_estimation" / "contralateral_motion_integrator_preferred_noise_estimation.pkl"
path_models = path_dir / "results" / "freepop" / "mask_traces_freepop_16" / "RNNFreePop_neurons102_tau0.1_input2step_softplus" / "top_5"  # / "ablation" / "2026-02-19_14-57-21"  # RNNConstrainedMask_neurons86_tau0.2_input2step_elu"
path_save = Path(r"C:\Users\Roberto\Desktop\highlights\clem_rnns\figures")

# ---- Show -------------------------------------
show_matrices = True


# ---- Simulate ----------------------------------
dt_data = 0.5
duration_rest_start = 20
duration_stimulus = 40
duration_rest_end = 20
duration_simulation = duration_rest_start + duration_stimulus + duration_rest_end

# ================================================================
# Plot configuration (layout, sizes, padding, etc.)
# ================================================================
style = RNNDSStyle()

plot_height = style.plot_height
plot_height_small = plot_height / 2.5

plot_width = style.plot_width
plot_width_small = style.plot_width_small

plot_size_matrix = style.plot_size_big * 1.2

padding = style.padding / 2
padding_big = style.padding * 2
padding_vertical = style.padding

xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start - padding

palette = style.palette["neurons_4"]

# ================================================================
# Initialize figure container
# ================================================================
fig = Figure()


# ================================================================
# Plot histogram of loss functions
# ================================================================
models_across_mask = {"label": "Loss distribution\nacross masks",
                      "path": Path(r"C:\Users\Roberto\Desktop\highlights\clem_rnns\data\results\freepop\mask_traces_freepop_16\RNNFreePop_neurons102_tau0.1_input2step_softplus")}
models_same_mask = {"label": "Loss distribution\nfor best model's mask",
                    "path": Path(r"C:\Users\Roberto\Desktop\highlights\clem_rnns\data\results\freepop\mask_traces_freepop16_loadtop0\RNNFreePop_neurons102_tau0.1_input2step_softplus")}

# Plot histogram for models across sparsity masks
for i_m, models in enumerate([models_across_mask, models_same_mask]):
    loss_list = []
    for path_model in models["path"].glob(f"model_*.pkl"):
        with open(path_model, 'rb') as f:
            model = pickle.load(f)
        model.eval()
        loss_list.append(model.loss_mse)

    best_loss = loss_list[np.argsort(loss_list)[0]]
    max_value = 0.025
    n_bins = 20
    h, b = get_hist(loss_list, bins=n_bins, hist_range=(0, max_value), center_bin=True)
    plot_loss_dist = fig.create_plot(plot_title=models["label"],
                                     xpos=xpos, ypos=ypos, plot_height=plot_height,
                                     plot_width=plot_size_matrix,
                                     xmin=0, xmax=max_value, xticks=[0, max_value],
                                     ymin=0, ymax=50, yticks=[0, 25, 50],
                                     # vlines=[best_loss] if i_m == 0 else None, helper_lines_lc="r" if i_m == 0 else None
                                     )
    plot_loss_dist.draw_vertical_bars(b, h, vertical_bar_width=max_value/n_bins)

    xpos += plot_size_matrix + padding

xpos = xpos_start
ypos -= plot_height + padding_big



# ================================================================
# Load model instance
# ================================================================
i_model = 0
for path_model in path_models.glob(f"model_top0_*.pkl"):
    with open(path_model, 'rb') as f:
        model = pickle.load(f)
    model.eval()
    break

dt = model.dt

n_units_hemi = model.n_units_hemi
n_units_A = model.nA
n_units_B = model.nB
n_units_C = model.nC
n_units_D = model.nD
n_units = int(n_units_hemi * 2)

if hasattr(model, "nX"):  # This is the case for RNNFreePop models
    model_is_free_pop = True
    n_units_X = model.nX
    n_units += n_units_X
    palette = style.cmap_list["neurons_5"]
else:  # This is the case for RNNFreeNeurons models
    model_is_free_pop = False
    palette = style.cmap_list["neurons_4"]

offset_hemisphere = model.nA + model.nB + model.nC + model.nD
neuron_identity_array = np.concatenate((np.zeros((model.nA, 1)), np.ones((model.nB, 1)), 2*np.ones((model.nC, 1)), 3*np.ones((model.nD, 1)),
                                        np.zeros((model.nA, 1)), np.ones((model.nB, 1)), 2*np.ones((model.nC, 1)), 3*np.ones((model.nD, 1))))
if model_is_free_pop:
    neuron_identity_array = np.concatenate((neuron_identity_array, 4 * np.ones((model.nX, 1))))

U = model.U().detach().numpy()
mask_U = model.mask_U.detach().numpy()
W = model.W().detach().numpy().T
mask_W = (model.mask_W * model.signs).detach().numpy().T

# Plot U, W, and associated masks
if show_matrices:
    # Draw heatmap with the mask for W
    plot_size_vector = plot_size_matrix / n_units
    plot_mask_U = fig.create_plot(plot_title="Signed\ninput mask",
                                  xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                  plot_width=plot_size_vector,
                                  xmin=-0.5, xmax=0.5,  # xticklabels_rotation=90,
                                  # xticks=np.arange(n_units),
                                  ymin=-0.5, ymax=n_units - 0.5)

    xpos += plot_size_vector + padding
    im = plot_mask_U.draw_image(mask_U, (-0.5, 0.5, n_units - 0.5, -0.5),
                                colormap='binary', zmin=0, zmax=1, image_interpolation=None)

    # Draw heatmap with the mask for W
    plot_mask_W = fig.create_plot(plot_title="Signed\nconnectivity mask",
                                  xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                  plot_width=plot_size_matrix,
                                  xmin=-0.5, xmax=n_units - 0.5,  # xticklabels_rotation=90,
                                  # xticks=np.arange(n_units),
                                  ymin=-0.5, ymax=n_units - 0.5)

    # Draw neuron identity vectors around mask_W
    plot_ni_c = fig.create_plot(xpos=xpos - plot_size_vector, ypos=ypos, plot_height=plot_size_matrix,
                                plot_width=plot_size_vector,
                                xmin=-0.5, xmax=0.5,
                                ymin=-0.5, ymax=n_units - 0.5)
    im = plot_ni_c.draw_image(neuron_identity_array, (-0.5, 0.5, n_units - 0.5, -0.5),
                              colormap=style.cmap_list["neurons_4"], zmin=0, zmax=3, image_interpolation=None)

    plot_ni_r = fig.create_plot(xpos=xpos, ypos=ypos + plot_size_matrix, plot_height=plot_size_vector,
                                plot_width=plot_size_matrix,
                                xmin=-0.5, xmax=n_units - 0.5,
                                ymin=-0.5, ymax=0.5)
    im = plot_ni_r.draw_image(neuron_identity_array.T, (-0.5, n_units - 0.5, -0.5, 0.5),
                              colormap=style.cmap_list["neurons_4"], zmin=0, zmax=3, image_interpolation=None)

    x_ = np.arange(n_units)
    x = np.tile(x_, (n_units, 1))
    y = x.T
    im = plot_mask_W.draw_image(mask_W, (-0.5, n_units - 0.5, n_units - 0.5, -0.5),
                                colormap='PiYG', zmin=-1, zmax=1, image_interpolation=None)

    # Grid in mask_W
    position_line_between_pop = np.array([model.nA, model.nA + model.nB, model.nA + model.nB + model.nC,
                                          model.nA + model.nB + model.nC + model.nD,
                                          offset_hemisphere + model.nA, offset_hemisphere + model.nA + model.nB,
                                          offset_hemisphere + model.nA + model.nB + model.nC])
    plot_mask_W_grid = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=plot_size_matrix, plot_width=plot_size_matrix,
                                       xmin=-0.5, xmax=n_units - 0.5, ymin=-0.5, ymax=n_units - 0.5,
                                       helper_lines_lc="white",
                                       hlines=model.n_units - position_line_between_pop - 0.5,
                                       vlines=position_line_between_pop - 0.5)

    xpos += plot_size_matrix + padding

    # Draw input vector U after training
    plot_size_vector = plot_size_matrix / n_units
    plot_U = fig.create_plot(plot_title="U",
                             xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                             plot_width=plot_size_vector,
                             xmin=-0.5, xmax=0.5,  # xticklabels_rotation=90,
                             # xticks=np.arange(n_units),
                             ymin=-0.5, ymax=n_units - 0.5)

    xpos += plot_size_vector + padding
    im = plot_U.draw_image(U, (-0.5, 0.5, n_units - 0.5, -0.5),
                           colormap='PiYG', zmin=-1, zmax=1, image_interpolation=None)

    # Draw connectivity matrix W after training
    plot_W = fig.create_plot(plot_title="W",
                             xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                             plot_width=plot_size_matrix * 1.1,
                             xmin=-0.5, xmax=n_units - 0.5,  # xticklabels_rotation=90,
                             # xticks=np.arange(n_units),
                             ymin=-0.5, ymax=n_units - 0.5,
                             )
    # Draw neuron identity vectors around W
    plot_ni_c = fig.create_plot(xpos=xpos - plot_size_vector, ypos=ypos, plot_height=plot_size_matrix,
                                plot_width=plot_size_vector,
                                xmin=-0.5, xmax=0.5,
                                ymin=-0.5, ymax=n_units - 0.5)
    im = plot_ni_c.draw_image(neuron_identity_array, (-0.5, 0.5, n_units - 0.5, -0.5),
                              colormap=style.cmap_list["neurons_4"], zmin=0, zmax=3, image_interpolation=None)

    plot_ni_r = fig.create_plot(xpos=xpos, ypos=ypos + plot_size_matrix, plot_height=plot_size_vector,
                                plot_width=plot_size_matrix,
                                xmin=-0.5, xmax=n_units - 0.5,
                                ymin=-0.5, ymax=0.5)
    im = plot_ni_r.draw_image(neuron_identity_array.T, (-0.5, n_units - 0.5, -0.5, 0.5),
                              colormap=style.cmap_list["neurons_4"], zmin=0, zmax=3, image_interpolation=None)

    x_ = np.arange(n_units)
    x = np.tile(x_, (n_units, 1))
    y = x.T
    value_lim = 1  # np.max(np.abs(W))
    im = plot_W.draw_image(W, (-0.5, n_units - 0.5, n_units - 0.5, -0.5),
                           colormap='PiYG', zmin=-value_lim, zmax=value_lim, image_interpolation=None)

    plot_W_grid = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=plot_size_matrix, plot_width=plot_size_matrix,
                                  xmin=-0.5, xmax=n_units - 0.5, ymin=-0.5, ymax=n_units - 0.5,
                                  helper_lines_lc="white",
                                  hlines=model.n_units - position_line_between_pop - 0.5,
                                  vlines=position_line_between_pop - 0.5)
    divider = make_axes_locatable(plot_W.ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plot_W.figure.fig.colorbar(im, cax=cax, orientation='vertical',
                               ticks=[-value_lim, -value_lim / 2, 0, value_lim / 2, value_lim])
    xpos += plot_size_matrix + padding_big * 2 / 3

# Rebase coordinates for response plots
plot_size_here = style.plot_size_big * 2/3
padding_here = padding
xpos_start_here = xpos_here = xpos_start
ypos_start_here = ypos_here = ypos - plot_size_matrix - padding_vertical * 1.5

# Define input signals used in training
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

# Define input signal for bilateral test
def input_signal_constant_bilateral(duration_rest_start, duration_stimulus, duration_rest_end, side="L", scale=1, ratio_LR=1):
    input_signal_ = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)), np.zeros(int(duration_rest_end / dt)))) * scale
    input_signal = np.repeat(input_signal_[..., np.newaxis], n_units_hemi*2, axis=1)
    if side[0] in ["l", "L"]:
        input_signal[:, n_units_hemi:] *= ratio_LR
    elif side[0] in ["r", "R"]:
        input_signal[:, :n_units_hemi] *= ratio_LR
    return input_signal

# Define input signals used for testing (sine wave)
def input_signal_sine(duration_rest_start, duration_stimulus, duration_rest_end, side="L", scale=1):
    sine = lambda t: 0.5 * np.sin(t-np.pi/2) + 0.5
    input_signal_ = np.concatenate((np.zeros(int(duration_rest_start / dt)), sine(np.arange(0, duration_stimulus, dt)), np.zeros(int(duration_rest_end / dt)))) * scale
    if side[0] in ["l", "L"]:
        input_signal = np.concatenate((np.repeat(input_signal_[..., np.newaxis], n_units_hemi, axis=1),
                                             np.zeros((len(input_signal_), n_units_hemi))), axis=1)
    elif side[0] in ["r", "R"]:
        input_signal = np.concatenate((np.zeros((len(input_signal_), n_units_hemi)),
                                             np.repeat(input_signal_[..., np.newaxis], n_units_hemi, axis=1)), axis=1)
    else:
        input_signal = None
    return input_signal

# Define input signals used for testing (side switching)
def input_signal_switch(duration_rest_start, duration_stimulus, duration_rest_end, side="L", scale=1):
    duration_stimulus_side = duration_stimulus / 2
    input_signal_step_first  = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus_side / dt)), np.zeros(int((duration_stimulus_side + duration_rest_end) / dt)))) * scale
    input_signal_step_second = np.concatenate((np.zeros(int((duration_rest_start + duration_stimulus_side) / dt)), np.ones(int(duration_stimulus_side / dt)), np.zeros(int(duration_rest_end / dt)))) * scale

    if side[0] in ["l", "L"]:
        input_signal = np.concatenate((np.repeat(input_signal_step_first[..., np.newaxis], n_units_hemi, axis=1),
                                       np.repeat(input_signal_step_second[..., np.newaxis], n_units_hemi, axis=1)), axis=1)
    elif side[0] in ["r", "R"]:
        input_signal = np.concatenate((np.repeat(input_signal_step_second[..., np.newaxis], n_units_hemi, axis=1),
                                      np.repeat(input_signal_step_first[..., np.newaxis], n_units_hemi, axis=1)), axis=1)
    else:
        input_signal = None
    return input_signal



# ================================================================
# Define tests to show
# ================================================================
test_list = [
    {"label": "Train high L",
     "duration_rest_start": 20,
     "duration_stimulus": 40,
     "duration_rest_end": 20,
     "path_traces": path_dir,
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

    {"label": "Test sine L",
     "duration_rest_start": 32,
     "duration_stimulus": 32,
     "duration_rest_end": 32,
     "path_traces": path_dir / "single_traces" / "kim_experiments",
     "path_noise": None,
     "filename_root": "responses",
     "file_extension": "csv",
     "stimulus_name": "oscillating",
     # "filename": "responses_*_oscillating_*.csv",
     "combine_data": "average",
     "scale_target": None,  # None or 1: don't scale the target signal found at path_traces and the input signal
     "input_signal": input_signal_sine,
     "time_target_array": np.arange(0, 96, dt_data),
     "dt_data": dt_data,
     "side_list": ("left", "right"),
     "flip_side": False,
     "cell_type_list": ("MI", "MON", "SMI"),
     "fix_offset_response": False},
    {"label": "Test switch L",
     "duration_rest_start": 32,
     "duration_stimulus": 32,
     "duration_rest_end": 32,
     "path_traces": path_dir / "single_traces" / "kim_experiments",
     "path_noise": None,
     "filename_root": "responses",
     "file_extension": "csv",
     "stimulus_name": "switching",
     # "filename": "responses_*_switching_*.csv",
     "combine_data": "average",
     "scale_target": None,  # None or 1: don't scale the target signal found at path_traces and the input signal
     "input_signal": input_signal_switch,
     "time_target_array": np.arange(0, 96, dt_data),
     "dt_data": dt_data,
     "side_list": ("left", "right"),
     "flip_side": False,
     "cell_type_list": ("MI", "MON", "SMI"),
     "fix_offset_response": False},

    # {"label": "Test bilateral",
    #  "duration_rest_start": 32,
    #  "duration_stimulus": 32,
    #  "duration_rest_end": 32,
    #  "path_traces": path_dir / "single_traces" / "kim_experiments",
    #  "path_noise": None,
    #  "filename_root": "responses",
    #  "file_extension": "csv",
    #  "stimulus_name": "switching",
    #  # "filename": "responses_*_switching_*.csv",
    #  "combine_data": "average",
    #  "scale_target": None,  # None or 1: don't scale the target signal found at path_traces and the input signal
    #  "input_signal": input_signal_constant_bilateral,
    #  "time_target_array": np.arange(0, 96, dt_data),
    #  "dt_data": dt_data,
    #  "side_list": ("left", "right"),
    #  "flip_side": False,
    #  "cell_type_list": ("MI", "MON", "SMI"),
    #  "fix_offset_response": False},
]

# ================================================================
# Loop over all tests to show
# ================================================================
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

    # ================================================================
    # Load traces to use as target signals
    # ================================================================
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
                    data_raw = np.loadtxt(test['path_traces'] / filename, dtype=float, delimiter=",", skiprows=1)
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
            target_signal_list.append(traces_dict[test['cell_type_list'][i_ct]][test['side_list'][int(np.abs(i_s - flip_side))]])
            if i_ct == 0 and len(test['cell_type_list']) == 3:  # do it again if there is no contra/ipsi differentiation of MI cells
                target_signal_list.append(traces_dict[test['cell_type_list'][i_ct]][test['side_list'][int(np.abs(i_s - flip_side))]])
            sanity_check_list.append((test['cell_type_list'][i_ct], test['side_list'][int(np.abs(i_s - flip_side))]))
    target_signal = np.stack(target_signal_list, axis=-1).copy()

    # Scale signal (sqrt is applied to scaling factor)
    scale_list = test['scale_target'] if test['scale_target'] is not None else [1]
    if not hasattr(scale_list, "__iter__"):
        scale_list = [scale_list]
    # scale_target = np.sqrt(scale)
    # target_signal *= scale_target  # scaling
    # target_signal += noise_filter(target_signal)

    # ==============================================================
    # Compute and plot responses to test signals
    # ==============================================================
    duration_simulation = test['duration_rest_start'] + test['duration_stimulus'] + test['duration_rest_end']

    # apply scaling
    input_signal_list = []
    output_signal_list = []
    initial_value_list = []
    for s in scale_list:
        # Define input signal for simulation
        input_signal_ = test['input_signal'](test['duration_rest_start'], test['duration_stimulus'], test['duration_rest_end'], side="R" if test['flip_side'] else "L", scale=s)
        # Define target output signal
        output_signal = target_signal * np.sqrt(s)  # scaling
        output_signal += noise_filter(output_signal)
        output_signal += min_traces_all
        output_signal_list.append(output_signal)
        # Define initial value for simulation
        initial_value = inv_softplus(np.concatenate((np.array([output_signal[0, 0] for _ in range(n_units_A)]),  # + np.random.normal(0, np.abs(target_signal[0, 0]) / 5, n_units_A),
                                                     np.array([output_signal[0, 1] for _ in range(n_units_B)]),  # + np.random.normal(0, np.abs(target_signal[0, 1]) / 5, n_units_B),
                                                     np.array([output_signal[0, 2] for _ in range(n_units_C)]),  # + np.random.normal(0, np.abs(target_signal[0, 2]) / 5, n_units_C),
                                                     np.array([output_signal[0, 3] for _ in range(n_units_D)]),  # + np.random.normal(0, np.abs(target_signal[0, 3]) / 5, n_units_D),
                                                     np.array([output_signal[0, 4] for _ in range(n_units_A)]),  # + np.random.normal(0, np.abs(target_signal[0, 4]) / 5, n_units_A),
                                                     np.array([output_signal[0, 5] for _ in range(n_units_B)]),  # + np.random.normal(0, np.abs(target_signal[0, 5]) / 5, n_units_B),
                                                     np.array([output_signal[0, 6] for _ in range(n_units_C)]),  # + np.random.normal(0, np.abs(target_signal[0, 6]) / 5, n_units_C),
                                                     np.array([output_signal[0, 7] for _ in range(n_units_D)]))))  # + np.random.normal(0, np.abs(target_signal[0, 7]) / 5, n_units_D)))

        if model_is_free_pop:
            input_signal_ = np.concatenate((input_signal_, np.zeros((len(input_signal_), n_units_X))), axis=1)
            initial_value = np.concatenate((initial_value, np.array([np.mean(output_signal[0]) for _ in range(n_units_X)]) + np.random.normal(0, np.abs(np.mean(output_signal[0])) / 5, n_units_X)))

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
    res = RNNService.plot_response_by_cell(model, t_sim, input_signal, xpos_here, ypos_here,
                                   t_exp=time_target_array, output_signal=output_signal, x0=x0,
                                   fig=fig, show_xaxis=True, show_yaxis=i_test % 2 == 0,
                                   plot_title_label=label, plot_size=plot_size_here,
                                   time_structure={"rest_start": test['duration_rest_start'], "stimulus": test['duration_stimulus'], "rest_end": test['duration_rest_end'], "duration": duration_simulation})

    xpos_here = xpos_start_here
    ypos_here -= plot_size_here * 3 + padding_here * 3

xpos = xpos_start
ypos -= plot_size_matrix + padding_vertical * 1.5

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
path_save.mkdir(parents=True, exist_ok=True)
fig.save(path_save / "figure_main_train_test_new.pdf", open_file=False, tight=style.page_tight)
