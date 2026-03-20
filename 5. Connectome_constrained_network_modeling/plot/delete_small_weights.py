import pickle

import torch
import numpy as np
from pathlib import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable

from plot.style import RNNDSStyle
from utils.configuration_rnn import ConfigurationRNN
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
path_models = path_dir / "results" / "mask_traces_freeneurons_1_attempt2" / "RNNFreeNeurons_neurons102_tau0.1_input2step_softplus" / "top_5"  # RNNConstrainedMask_neurons86_tau0.2_input2step_elu"
path_traces = path_dir
path_save = path_models / "results"
TOP_FRAC  = 0.05
TOP_N = 5  # 10  # 0.05
device = "cpu"
free_neurons_per_pop = 2
save_top_models = False
clamp_value = 1e-2

n_units_A = 15 + free_neurons_per_pop
n_units_B = 15 + free_neurons_per_pop
n_units_C = 2 + free_neurons_per_pop
n_units_D = 11 + free_neurons_per_pop
n_units_hemi = n_units_A + n_units_B + n_units_C + n_units_D
n_units = n_units_hemi * 2
n_input_signal = 2
tau_integrator_0 = 5
tau_integrator_1 = 10
tau_differ = 3
dt = 0.01
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
input_signal_list = []
for i in range(n_input_signal):
    input_signal_list.append(input_signal * amplitude_input_signal_list[i])

# ================================================================
# Load traces to use as target signals
# ================================================================
cell_types_list = ["ipsilateral_motion_integrator", "contralateral_motion_integrator", "motion_onset", "slow_motion_integrator"]
side_list = ["preferred", "null"]
traces_dict = {ct: {s: None for s in side_list} for ct in cell_types_list}
all_signals = []
min_traces_all = 0
for ct in cell_types_list:
    for s in side_list:
        filename = f"{ct}_{s}_activity_traces.csv"
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
        return DSService.ou_noise(x, p_noise["tau"], p_noise["sigma"], 0.5, 3)

train_list = []
for i, amplitude in enumerate(amplitude_input_signal_list):
    input_signal = input_signal_list[i]
    target_signal_L = np.stack((traces_dict["ipsilateral_motion_integrator"]["preferred"],
                                traces_dict["contralateral_motion_integrator"]["preferred"],
                                traces_dict["motion_onset"]["preferred"],
                                traces_dict["slow_motion_integrator"]["preferred"],
                                traces_dict["ipsilateral_motion_integrator"]["null"],
                                traces_dict["contralateral_motion_integrator"]["null"],
                                traces_dict["motion_onset"]["null"],
                                traces_dict["slow_motion_integrator"]["null"]
                                ),
                               axis=-1)
    # scale_signal_original = np.max(np.abs(target_signal_L))
    # target_signal_L /= scale_signal_original
    target_signal_L = target_signal_L * np.sqrt(amplitude)  # scaling
    if amplitude != 1:
        target_signal_L += noise_filter(target_signal_L)

    target_signal_L += min_traces_all
    # target_signal_L *= scale_signal_original
    input_signal_neurons_L = np.concatenate((np.repeat(input_signal[..., np.newaxis], n_units_hemi, axis=1),
                                             np.zeros((len(input_signal), n_units_hemi))), axis=1)
    initial_value_L = np.concatenate((np.array([target_signal_L[0, 0] for _ in range(n_units_A)]) + np.random.normal(0, np.abs(target_signal_L[0, 0])/5, n_units_A),
                                      np.array([target_signal_L[0, 1] for _ in range(n_units_B)]) + np.random.normal(0, np.abs(target_signal_L[0, 1])/5, n_units_B),
                                      np.array([target_signal_L[0, 2] for _ in range(n_units_C)]) + np.random.normal(0, np.abs(target_signal_L[0, 2])/5, n_units_C),
                                      np.array([target_signal_L[0, 3] for _ in range(n_units_D)]) + np.random.normal(0, np.abs(target_signal_L[0, 3])/5, n_units_D),
                                      np.array([target_signal_L[0, 4] for _ in range(n_units_A)]) + np.random.normal(0, np.abs(target_signal_L[0, 4])/5, n_units_A),
                                      np.array([target_signal_L[0, 5] for _ in range(n_units_B)]) + np.random.normal(0, np.abs(target_signal_L[0, 5])/5, n_units_B),
                                      np.array([target_signal_L[0, 6] for _ in range(n_units_C)]) + np.random.normal(0, np.abs(target_signal_L[0, 6])/5, n_units_C),
                                      np.array([target_signal_L[0, 7] for _ in range(n_units_D)]) + np.random.normal(0, np.abs(target_signal_L[0, 7])/5, n_units_D)))
    train_list.append(TrainSignal(input_signal_neurons_L, target_signal_L, initial_value_L, label=f"Train {'High' if amplitude == 1 else 'Low'} L"))

    target_signal_R = np.stack((traces_dict["ipsilateral_motion_integrator"]["null"],
                                traces_dict["contralateral_motion_integrator"]["null"],
                                traces_dict["motion_onset"]["null"],
                                traces_dict["slow_motion_integrator"]["null"],
                                traces_dict["ipsilateral_motion_integrator"]["preferred"],
                                traces_dict["contralateral_motion_integrator"]["preferred"],
                                traces_dict["motion_onset"]["preferred"],
                                traces_dict["slow_motion_integrator"]["preferred"]
                                ),
                               axis=-1)
    target_signal_R += noise_filter(target_signal_R)
    target_signal_R = target_signal_R * np.sqrt(amplitude)
    target_signal_R += min_traces_all

    # target_signal_R *= scale_signal_original
    input_signal_neurons_R = np.concatenate((np.zeros((len(input_signal), n_units_hemi)),
                                             np.repeat(input_signal[..., np.newaxis], n_units_hemi, axis=1)), axis=1)
    initial_value_R = np.concatenate((np.array([target_signal_R[0, 0] for _ in range(n_units_A)]) + np.random.normal(0, np.abs(target_signal_R[0, 0])/5, n_units_A),
                                      np.array([target_signal_R[0, 1] for _ in range(n_units_B)]) + np.random.normal(0, np.abs(target_signal_R[0, 1])/5, n_units_B),
                                      np.array([target_signal_R[0, 2] for _ in range(n_units_C)]) + np.random.normal(0, np.abs(target_signal_R[0, 2])/5, n_units_C),
                                      np.array([target_signal_R[0, 3] for _ in range(n_units_D)]) + np.random.normal(0, np.abs(target_signal_R[0, 3])/5, n_units_D),
                                      np.array([target_signal_R[0, 4] for _ in range(n_units_A)]) + np.random.normal(0, np.abs(target_signal_R[0, 4])/5, n_units_A),
                                      np.array([target_signal_R[0, 5] for _ in range(n_units_B)]) + np.random.normal(0, np.abs(target_signal_R[0, 5])/5, n_units_B),
                                      np.array([target_signal_R[0, 6] for _ in range(n_units_C)]) + np.random.normal(0, np.abs(target_signal_R[0, 6])/5, n_units_C),
                                      np.array([target_signal_R[0, 7] for _ in range(n_units_D)]) + np.random.normal(0, np.abs(target_signal_R[0, 7])/5, n_units_D)))
    train_list.append(TrainSignal(input_signal_neurons_R, target_signal_R, initial_value_R, label=f"Train {'High' if amplitude == 1 else 'Low'} R"))

sine = lambda t: 0.5 * np.sin(t-np.pi/2) + 0.5
input_signal_sine = np.concatenate((np.zeros(int(duration_rest_start / dt)), sine(np.arange(0, duration_stimulus, dt)), np.zeros(int(duration_rest_end / dt))))
input_signal_sine_L = np.concatenate((np.repeat(input_signal_sine[..., np.newaxis], n_units_hemi, axis=1),
                                             np.zeros((len(input_signal_sine), n_units_hemi))), axis=1)
input_signal_sine_R = np.concatenate((np.zeros((len(input_signal_sine), n_units_hemi)),
                                             np.repeat(input_signal_sine[..., np.newaxis], n_units_hemi, axis=1)), axis=1)

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
        # loss = model.loss.detach().numpy()
        loss = model.loss_mse
        # loss = model.raise_error.detach().numpy()
    except AttributeError:
        continue
        # with torch.no_grad():
        #     loss_array = np.zeros(len(train_list))
        #     for i_t, train_signals in enumerate(train_list):
        #         input_signal = torch.tensor(train_signals.input_signal, dtype=torch.float32)
        #         output_signal = train_signals.output_signal
        #         x0 = torch.tensor(train_signals.initial_value, dtype=torch.float32)
        #         _, y_pred = model.forward(x0, input_signal)
        #         y_pred = y_pred.detach().numpy()
        #         y_pred = DSService.downsample_signal(y_pred, model.dt, downsample_time_list)
        #         loss_array[i_t] = np.mean((output_signal - y_pred) ** 2)
        #     loss = np.mean(loss_array)


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

for i in top_indices:
    with open(model_path_list[i], 'rb') as f:
        model = pickle.load(f)
    model.eval()
    n_neurons = model.n_units

    W = W_list[i].T
    U = U_list[i]
    mask_W = mask_W_list[i].T
    mask_U = mask_U_list[i]

    # Remove clamping from the matrix W to show
    i_clamp_weight = np.argwhere(np.logical_and(np.abs(W) <= clamp_value, mask_W > 0))
    n_clamp_weight = len(i_clamp_weight)
    for ij in i_clamp_weight:
        W[ij] = 0

    # Remove clamping from model
    model.clamp_weights_min = 0


    # Save trained model
    if save_top_models:
        label_model = model_path_list[i].name
        path_save_top_model = path_models / f"top_{TOP_N}"
        path_save_top_model.mkdir(parents=True, exist_ok=True)
        with open(path_save_top_model / label_model, 'wb') as f:
            pickle.dump(model, f)

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

    xpos += plot_size_matrix + padding

    x_ = np.arange(n_neurons)
    x = np.tile(x_, (n_neurons, 1))
    y = x.T
    im = plot_mask_W.draw_image(mask_W, (-0.5, n_neurons - 0.5, n_neurons - 0.5, -0.5),
                              colormap='PiYG', zmin=-1, zmax=1, image_interpolation=None)

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
    plot_W = fig.create_plot(plot_title=f"W with {n_clamp_weight}\nzeroed weights",
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

    position_line_between_pop = np.array([model.nA, model.nA + model.nB, model.nA + model.nB + model.nC,
     model.nA + model.nB + model.nC + model.nD,
     offset_hemisphere + model.nA, offset_hemisphere + model.nA + model.nB,
     offset_hemisphere + model.nA + model.nB + model.nC])

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

    # def plot_response(model, t, input_signal, xpos, ypos, output_signal=None, x0=None, plot_title_label="", show_xaxis=True, show_yaxis=True):
    #     # Draw network response to low step function (used in training)
    #     x0 = torch.zeros(model.n_units) if x0 is None else x0
    #     with torch.no_grad():
    #         inputs = torch.tensor(input_signal, dtype=torch.float32)
    #         xs, y_pred = model.forward(x0, inputs)
    #
    #     xs = np.squeeze(xs.detach().numpy())
    #     y_pred = np.squeeze(y_pred.detach().numpy())
    #     # t_sim = np.linspace(np.min(t), np.max(t), y_pred.shape[0])
    #
    #     for side in range(2):
    #         plot_title = f"{plot_title_label}" if side == 0 else ""
    #         plot_title = plot_title + "\nActivity L" if side == 0 else plot_title + "\nActivity R"
    #         offset_index = side * 4
    #         offset_hemisphere = side * (model.nA + model.nB + model.nC + model.nD)
    #         plot_response = fig.create_plot(
    #             plot_title=plot_title,
    #             xpos=xpos, ypos=ypos, plot_height=plot_size_here,
    #             plot_width=plot_size_here,
    #             xmin=0, xmax=duration_simulation, xl="Time (s)" if show_xaxis else None,
    #             xticks=[duration_rest_start, duration_rest_start + duration_stimulus] if show_xaxis else None,
    #             ymin=0, ymax=2, yticks=[0, 1, 2] if show_yaxis and side == 0 else None)
    #         plot_response.draw_line(t_sim, input_signal[:, offset_hemisphere], lc="k", alpha=0.5)
    #         if output_signal is not None:
    #             plot_response.draw_line(t, output_signal[:, 0+offset_index], lc=palette[0], line_dashes=(1, 2))
    #             plot_response.draw_line(t, output_signal[:, 1+offset_index], lc=palette[1], line_dashes=(1, 2))
    #             plot_response.draw_line(t, output_signal[:, 2+offset_index], lc=palette[2], line_dashes=(1, 2))
    #             plot_response.draw_line(t, output_signal[:, 3+offset_index], lc=palette[3], line_dashes=(1, 2))
    #         plot_response.draw_line(t_sim, xs[:, offset_hemisphere:offset_hemisphere+model.nA], lc=palette[0], lw=0.1)
    #         plot_response.draw_line(t_sim, xs[:, offset_hemisphere+model.nA:offset_hemisphere+model.nA + model.nB], lc=palette[1], lw=0.1)
    #         plot_response.draw_line(t_sim, xs[:, offset_hemisphere+model.nA + model.nB:offset_hemisphere+model.nA + model.nB + model.nC], lc=palette[2], lw=0.1)
    #         plot_response.draw_line(t_sim, xs[:, offset_hemisphere+model.nA + model.nB + model.nC:offset_hemisphere+model.nA + model.nB + model.nC + model.nD],
    #                                       lc=palette[3], lw=0.1)
    #         plot_response.draw_line(t_sim, y_pred[:, 0+offset_index], lc=palette[0])
    #         plot_response.draw_line(t_sim, y_pred[:, 1+offset_index], lc=palette[1])
    #         plot_response.draw_line(t_sim, y_pred[:, 2+offset_index], lc=palette[2])
    #         plot_response.draw_line(t_sim, y_pred[:, 3+offset_index], lc=palette[3])
    #         xpos += plot_size_here + padding_here

    for i_t, train_signals in enumerate(train_list):
        input_signal = torch.tensor(train_signals.input_signal, dtype=torch.float32)
        output_signal = train_signals.output_signal
        x0 = torch.tensor(train_signals.initial_value, dtype=torch.float32)
        label = train_signals.label
        RNNService.plot_response(model, t_sim, input_signal, xpos_here, ypos_here, t_exp=downsample_time_list, fig=fig,
                                 output_signal=output_signal, x0=x0, show_xaxis=i_t in [2, 3], show_yaxis=i_t % 2 == 0,
                                 plot_title_label=label, time_structure=ConfigurationRNN.time_structure_simulation_train)

        if i_t % 2 == 0:
            xpos_here += plot_size_here * 2 + padding_here * 2
        else:
            xpos_here = xpos_start_here
            ypos_here -= plot_size_here + padding_here

    # xpos_here += plot_size_here * 4 + padding_here * 4
    # ypos_here = ypos_start_here
    #
    # # Draw network response to sine wave (test only)
    # RNNService.plot_response(model, t_sim, input_signal_sine_L, xpos_here, ypos_here, plot_title_label="Test L", fig=fig,
    #                          show_xaxis=False, time_structure=ConfigurationRNN.time_structure_simulation_train)
    # ypos_here -= plot_size_here + padding_here
    # RNNService.plot_response(model, t_sim, input_signal_sine_R, xpos_here, ypos_here, plot_title_label="Test R", fig=fig,
    #                          show_yaxis=False, time_structure=ConfigurationRNN.time_structure_simulation_train)

    xpos = xpos_start
    ypos -= plot_size_matrix + padding_vertical * 1.5

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
path_save.mkdir(parents=True, exist_ok=True)
fig.save(path_save / "check_train_cut_clamp.pdf", open_file=False, tight=style.page_tight)