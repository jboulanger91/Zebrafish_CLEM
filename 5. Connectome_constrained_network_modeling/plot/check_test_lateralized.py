import pickle

import torch
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter

from plot.style import RNNDSStyle
from utils.ds_service import DSService
from utils.operators import integrate, get_hist, pid
from utils.rnn_service import RNNService
from utils.train_batch import TrainSignal
from utils.figure_helper import Figure

# ------------------------------------------------
# Configuration
# ------------------------------------------------
path_dir = Path(r"C:\Users\Roberto\Desktop\highlights\clem_rnns\data")   # directory containing model_X.pkl
path_noise_estimation = path_dir / "noise_estimation" / "contralateral_motion_integrator_preferred_noise_estimation.pkl"
path_models = path_dir / "results" / "freepop" / "mask_traces_freepop16_loadtop0" / "RNNFreePop_neurons102_tau0.1_input2step_softplus" / "top_5" # / "ablation" / "2026-02-18_23-32-11"  # RNNConstrainedMask_neurons86_tau0.2_input2step_elu"
path_traces = path_dir
path_save = path_models / "results"
TOP_FRAC  = 0.05
TOP_N = 5  # 10  # 0.05
device = "cpu"
free_neurons_per_pop = 2
save_top_models = False

n_units_A = 15 + free_neurons_per_pop
n_units_B = 15 + free_neurons_per_pop
n_units_C = 2 + free_neurons_per_pop
n_units_D = 11 + free_neurons_per_pop
n_units_hemi = n_units_A + n_units_B + n_units_C + n_units_D
n_units = n_units_hemi * 2
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

plot_size_matrix = style.plot_size_big * 1.2

padding = style.padding / 2
padding_big = style.padding * 2
padding_vertical = style.padding

palette = style.palette["neurons_4"]

# ================================================================
# Initialize figure container
# ================================================================
fig = Figure()

# ------------------------------------------------
# Storage
# ------------------------------------------------
performance_list = []
loss_list = []
W_list = []
U_list = []
mask_W_list = []
mask_U_list = []
model_path_list = []

# Define input signals used in training
amplitude_input_signal_list = np.linspace(0.1, 1, n_input_signal)
input_signal = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)), np.zeros(int(duration_rest_end / dt))))
t_sim = np.linspace(0, duration_simulation, len(input_signal))

period_sine = 8  # seconds
sine = lambda t: 0.5 * np.sin(2*np.pi/period_sine*t-np.pi/2) + 0.5
input_signal_sine = np.concatenate((np.zeros(int(duration_rest_start / dt)), sine(np.arange(0, duration_stimulus, dt)), np.zeros(int(duration_rest_end / dt))))
input_signal_sine_L = np.concatenate((np.repeat(input_signal_sine[..., np.newaxis], n_units_hemi, axis=1),
                                             np.zeros((len(input_signal_sine), n_units_hemi))), axis=1)
input_signal_sine_R = np.concatenate((np.zeros((len(input_signal_sine), n_units_hemi)),
                                             np.repeat(input_signal_sine[..., np.newaxis], n_units_hemi, axis=1)), axis=1)

time_rest_reversal = 16
time_stimulus_reversal = time_rest_reversal * 2
time_reversal = np.arange(0, time_rest_reversal * 2 + time_stimulus_reversal, dt)
# time_downsample_reversal = np.arange(0, time_rest_reversal * 2 + time_stimulus_reversal, dt_data)
input_signal_step_first = np.concatenate((np.zeros(int(time_rest_reversal / dt)), np.ones(int(time_stimulus_reversal / dt)), np.zeros(int(time_rest_reversal / dt))))
input_signal_step_second = np.concatenate((np.zeros(int(time_rest_reversal / dt)), np.zeros(int(time_stimulus_reversal / dt)), np.ones(int(time_rest_reversal / dt))))
input_signal_step_reversal_LR = np.concatenate((np.repeat(input_signal_step_first[..., np.newaxis], n_units_hemi, axis=1),
                                               np.repeat(input_signal_step_second[..., np.newaxis], n_units_hemi, axis=1)), axis=1)
input_signal_step_reversal_RL = np.concatenate((np.repeat(input_signal_step_second[..., np.newaxis], n_units_hemi, axis=1),
                                               np.repeat(input_signal_step_first[..., np.newaxis], n_units_hemi, axis=1)), axis=1)

weight_dominant_side = 0.7
weight_other_side = 0.3
input_signal_twosided_LR = np.concatenate((np.repeat(input_signal_step_first[..., np.newaxis] * weight_dominant_side, n_units_hemi, axis=1),
                                               np.repeat(input_signal_step_first[..., np.newaxis] * weight_other_side, n_units_hemi, axis=1)), axis=1)
input_signal_twosided_RL = np.concatenate((np.repeat(input_signal_step_first[..., np.newaxis] * weight_other_side, n_units_hemi, axis=1),
                                               np.repeat(input_signal_step_first[..., np.newaxis] * weight_dominant_side, n_units_hemi, axis=1)), axis=1)


# output_signal_sine_differ = pid_(input_signal_sine)
# output_signal_sine_integrate_0 = integrate_0(input_signal_sine)
# output_signal_sine_integrate_1 = integrate_1(input_signal_sine)
# output_signal_sine = np.stack((output_signal_sine_integrate_0, output_signal_sine_integrate_0, output_signal_sine_differ, output_signal_sine_integrate_1,
#                                output_signal_sine_integrate_0, output_signal_sine_integrate_0, output_signal_sine_differ, output_signal_sine_integrate_1), axis=-1)

# test_inputs = [input_signal_ramp, input_signal_sine]
# test_targets = [output_signal_ramp, output_signal_sine]

# ------------------------------------------------
# Loop over all trained models
# ------------------------------------------------
i_model = 0
for path_model in path_models.glob(f"model_*.pkl"):
    print(f"Evaluating model {i_model}")
    i_model += 1

    model_path_list.append(path_model)
    # Load model instance
    with open(path_model, 'rb') as f:
        model = pickle.load(f)
    model.eval()

    # Evaluate on all test signals and compute average MSE
    mse_list = []

    x0 = torch.zeros(model.n_units)
    try:
        loss = model.loss_mse
    except AttributeError:
        continue

    loss_list.append(loss)

    # Extract connectivity matrix
    W_list.append(model.W().detach().cpu().numpy())
    U_list.append(model.U().detach().cpu().numpy())
    mask_W_list.append((model.mask_W * model.signs).detach().cpu().numpy())
    mask_U_list.append((model.mask_U).detach().cpu().numpy())

N_MODELS = len(loss_list)
performance_list = np.array(performance_list)
loss_list = np.array(loss_list)
W_list = np.stack(W_list, axis=0)   # shape: (N_MODELS, N, N)
U_list = np.stack(U_list, axis=0)   # shape: (N_MODELS, N, N)
mask_W_list = np.stack(mask_W_list, axis=0)   # shape: (N_MODELS, N, N)
mask_U_list = np.stack(mask_U_list, axis=0)   # shape: (N_MODELS, N)

# Plot distribution of performance
h, b = get_hist(loss_list, bins=20, hist_range=(0, 0.1), center_bin=True)
plot_loss_dist = fig.create_plot(
                             plot_title="Loss distribution",
                             xpos=xpos, ypos=ypos, plot_height=plot_height,
                             plot_width=plot_size_matrix,
                             xmin=0, xmax=np.max(b), xticks=[0, np.max(b)],
                             ymin=0, ymax=np.max(h), yticks=[0, np.max(h)])
plot_loss_dist.draw_line(b, h)

xpos = xpos_start
ypos -= plot_height + padding_big

# Select top-performant models
num_top = TOP_N  # int(TOP_FRAC * N_MODELS)
top_indices = np.argsort(loss_list)[:num_top]
W_top = W_list[top_indices]    # shape: (num_top, N, N)
mask_top = mask_W_list[top_indices]

top_label = 0
for i in top_indices:
    with open(model_path_list[i], 'rb') as f:
        model = pickle.load(f)
    model.eval()
    n_neurons = model.n_units

    W = W_list[i].T
    U = U_list[i]
    mask_W = mask_W_list[i].T
    mask_U = mask_U_list[i]

    # Save trained model
    if save_top_models:
        model_name_split = model_path_list[i].name.replace(".pkl", "").split('_')
        model_name_top = f"model_top{top_label}_{model_name_split[1]}_{model_name_split[2]}.pkl"
        path_save_top_model = path_models / f"top_{TOP_N}"
        path_save_top_model.mkdir(parents=True, exist_ok=True)
        with open(path_save_top_model / model_name_top, 'wb') as f:
            pickle.dump(model, f)
    top_label += 1

    offset_hemisphere = model.nA + model.nB + model.nC + model.nD
    neuron_identity_array = np.concatenate((np.zeros((model.nA, 1)), np.ones((model.nB, 1)), 2*np.ones((model.nC, 1)), 3*np.ones((model.nD, 1)),
                                            np.zeros((model.nA, 1)), np.ones((model.nB, 1)), 2*np.ones((model.nC, 1)), 3*np.ones((model.nD, 1))))

    # Draw heatmap with the mask for W
    plot_size_vector = plot_size_matrix / n_neurons
    plot_mask_U = fig.create_plot(plot_title="Signed\ninput mask",
                                  xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                  plot_width=plot_size_vector,
                                  xmin=-0.5, xmax=0.5,  # xticklabels_rotation=90,
                                  # xticks=np.arange(n_neurons),
                                  ymin=-0.5, ymax=n_neurons - 0.5)

    xpos += plot_size_vector + padding
    im = plot_mask_U.draw_image(mask_U, (-0.5, 0.5, n_neurons - 0.5, -0.5),
                                colormap='binary', zmin=0, zmax=1, image_interpolation=None)

    # Draw heatmap with the mask for W
    plot_mask_W = fig.create_plot(plot_title="Signed\nconnectivity mask",
                                xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                plot_width=plot_size_matrix,
                                xmin=-0.5, xmax=n_neurons - 0.5,  # xticklabels_rotation=90,
                                # xticks=np.arange(n_neurons),
                                ymin=-0.5, ymax=n_neurons - 0.5)

    # Draw neuron identity vectors around mask_W
    plot_ni_c = fig.create_plot(xpos=xpos - plot_size_vector, ypos=ypos, plot_height=plot_size_matrix,
                                plot_width=plot_size_vector,
                                xmin=-0.5, xmax=0.5,
                                ymin=-0.5, ymax=n_neurons - 0.5)
    im = plot_ni_c.draw_image(neuron_identity_array, (-0.5, 0.5, n_neurons - 0.5, -0.5),
                              colormap=style.cmap_list["neurons_4"], zmin=0, zmax=3, image_interpolation=None)

    plot_ni_r = fig.create_plot(xpos=xpos, ypos=ypos + plot_size_matrix, plot_height=plot_size_vector,
                                plot_width=plot_size_matrix,
                                xmin=-0.5, xmax=n_neurons - 0.5,
                                ymin=-0.5, ymax=0.5)
    im = plot_ni_r.draw_image(neuron_identity_array.T, (-0.5, n_neurons - 0.5, -0.5, 0.5),
                              colormap=style.cmap_list["neurons_4"], zmin=0, zmax=3, image_interpolation=None)

    x_ = np.arange(n_neurons)
    x = np.tile(x_, (n_neurons, 1))
    y = x.T
    im = plot_mask_W.draw_image(mask_W, (-0.5, n_neurons - 0.5, n_neurons - 0.5, -0.5),
                              colormap='PiYG', zmin=-1, zmax=1, image_interpolation=None)

    # Grid in mask_W
    position_line_between_pop = np.array([model.nA, model.nA + model.nB, model.nA + model.nB + model.nC,
                                          model.nA + model.nB + model.nC + model.nD,
                                          offset_hemisphere + model.nA, offset_hemisphere + model.nA + model.nB,
                                          offset_hemisphere + model.nA + model.nB + model.nC])
    plot_mask_W_grid = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=plot_size_matrix, plot_width=plot_size_matrix,
                                  xmin=-0.5, xmax=n_neurons - 0.5, ymin=-0.5, ymax=n_neurons - 0.5,
                                  helper_lines_lc="white",
                                  hlines=model.n_units - position_line_between_pop - 0.5,
                                  vlines=position_line_between_pop - 0.5)

    xpos += plot_size_matrix + padding

    # Draw input vector U after training
    plot_size_vector = plot_size_matrix / n_neurons
    plot_U = fig.create_plot(plot_title="U",
                                  xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                  plot_width=plot_size_vector,
                                  xmin=-0.5, xmax=0.5,  # xticklabels_rotation=90,
                                  # xticks=np.arange(n_neurons),
                                  ymin=-0.5, ymax=n_neurons - 0.5)

    xpos += plot_size_vector + padding
    im = plot_U.draw_image(U, (-0.5, 0.5, n_neurons - 0.5, -0.5),
                                colormap='PiYG', zmin=-1, zmax=1, image_interpolation=None)

    # Draw connectivity matrix W after training
    plot_W = fig.create_plot(plot_title="W",
                                xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                plot_width=plot_size_matrix * 1.1,
                                xmin=-0.5, xmax=n_neurons - 0.5,  # xticklabels_rotation=90,
                                # xticks=np.arange(n_neurons),
                                ymin=-0.5, ymax=n_neurons - 0.5,
                                )
    # Draw neuron identity vectors around W
    plot_ni_c = fig.create_plot(xpos=xpos-plot_size_vector, ypos=ypos, plot_height=plot_size_matrix,
                                plot_width=plot_size_vector,
                                xmin=-0.5, xmax=0.5,
                                ymin=-0.5, ymax=n_neurons - 0.5)
    im = plot_ni_c.draw_image(neuron_identity_array, (-0.5, 0.5, n_neurons - 0.5, -0.5),
                              colormap=style.cmap_list["neurons_4"], zmin=0, zmax=3, image_interpolation=None)

    plot_ni_r = fig.create_plot(xpos=xpos, ypos=ypos+plot_size_matrix, plot_height=plot_size_vector,
                                plot_width=plot_size_matrix,
                                xmin=-0.5, xmax=n_neurons - 0.5,
                                ymin=-0.5, ymax=0.5)
    im = plot_ni_r.draw_image(neuron_identity_array.T, (-0.5, n_neurons - 0.5, -0.5, 0.5),
                              colormap=style.cmap_list["neurons_4"], zmin=0, zmax=3, image_interpolation=None)


    x_ = np.arange(n_neurons)
    x = np.tile(x_, (n_neurons, 1))
    y = x.T
    value_lim = np.max(np.abs(W))
    im = plot_W.draw_image(W, (-0.5, n_neurons - 0.5, n_neurons - 0.5, -0.5),
                              colormap='PiYG', zmin=-value_lim, zmax=value_lim, image_interpolation=None)

    plot_W_grid = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=plot_size_matrix, plot_width=plot_size_matrix,
                                  xmin=-0.5, xmax=n_neurons - 0.5, ymin=-0.5, ymax=n_neurons - 0.5,
                                  helper_lines_lc="white",
                                  hlines=model.n_units - position_line_between_pop - 0.5,
                                  vlines=position_line_between_pop - 0.5)
    divider = make_axes_locatable(plot_W.ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plot_W.figure.fig.colorbar(im, cax=cax, orientation='vertical',
                               ticks=[-value_lim, -value_lim/2, 0, value_lim/2, value_lim])
    xpos += plot_size_matrix + padding_big * 2/3


    # ==============================================================
    # Plot lateral responses to training signals
    # ==============================================================
    plot_size_here = plot_size_matrix / 3
    padding_here = padding
    xpos_start_here = xpos_here = xpos
    ypos_start_here = ypos_here = ypos + plot_size_here + padding_here

    # Draw network response to sine wave (test only)
    RNNService.plot_response(model, t_sim, input_signal_sine_L, xpos_here, ypos_here, fig=fig, plot_title_label="Sine L", show_xaxis=False)
    ypos_here -= plot_size_here + padding_here
    RNNService.plot_response(model, t_sim, input_signal_sine_R, xpos_here, ypos_here, fig=fig, plot_title_label="Sine R")

    xpos_here += plot_size_here * 2 + padding_here * 3
    ypos_here = ypos_start_here

    # Draw network response to step-reversal (test only)
    RNNService.plot_response(model, time_reversal, input_signal_step_reversal_LR, xpos_here, ypos_here, plot_title_label="Step reversal LR",
                  fig=fig, show_xaxis=False)
    ypos_here -= plot_size_here + padding_here
    RNNService.plot_response(model, time_reversal, input_signal_step_reversal_RL, xpos_here, ypos_here, plot_title_label="Step reversal RL",
                  fig=fig)

    xpos_here += plot_size_here * 2 + padding_here * 3
    ypos_here = ypos_start_here

    # Draw network response to two-sided weighted stimulus (test only)
    RNNService.plot_response(model, time_reversal, input_signal_twosided_LR, xpos_here, ypos_here, fig=fig,
                             plot_title_label="Two sides dominant L", show_xaxis=False)
    ypos_here -= plot_size_here + padding_here
    RNNService.plot_response(model, time_reversal, input_signal_twosided_RL, xpos_here, ypos_here, fig=fig,
                             plot_title_label="Two sides dominant R")

    xpos = xpos_start
    ypos -= plot_size_matrix + padding_vertical * 1.5

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
path_save.mkdir(parents=True, exist_ok=True)
fig.save(path_save / "check_test.pdf", open_file=False, tight=style.page_tight)
