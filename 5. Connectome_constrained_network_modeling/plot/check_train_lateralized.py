import pickle

import torch
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter

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
path_models = path_dir / "results" / "freepop" / "mask_traces_freepop_16" / "RNNFreePop_neurons102_tau0.1_input2step_softplus"  # / "top_5" / "ablation" / "2026-02-19_14-57-21"  # RNNConstrainedMask_neurons86_tau0.2_input2step_elu"
path_traces = path_dir
path_save = path_models / "results"
TOP_N = 5  # 10  # 0.05
save_top_models = True

n_input_signal = 2
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

# ================================================================
# Load traces to use as target signals
# ================================================================
cell_types_list = ["iMI", "cMI", "MON", "sMI"]
side_list = ["preferred", "null"]
traces_dict = {ct: {s: None for s in side_list} for ct in cell_types_list}
all_signals = []
min_traces_all = 0
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

    # Extract siulation configuration and architecture from the first model found
    if i_model == 1:
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

# Select top-performant models
num_top = TOP_N  # int(TOP_FRAC * N_MODELS)
top_indices = np.argsort(loss_list)[:num_top]
W_top = W_list[top_indices]    # shape: (num_top, N, N)
mask_top = mask_W_list[top_indices]


# ------------------------------------------------
# Define model input and output used in training
# ------------------------------------------------
# Define input signals used in training
amplitude_input_signal_list = np.linspace(0.1, 1, n_input_signal)
input_signal = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)), np.zeros(int(duration_rest_end / dt))))
t_sim = np.linspace(0, duration_simulation, len(input_signal))
input_signal_list = []
for i in range(n_input_signal):
    input_signal_list.append(input_signal * amplitude_input_signal_list[i])

# Define noise generation function
with open(path_noise_estimation, 'rb') as f:
    p_noise = pickle.load(f)
    def noise_filter(x):
        return DSService.ou_noise(x, p_noise["tau"], p_noise["sigma"], dt_data, 3)

# Define TrainSignal corresponding to each training condition
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
    # scale_signal_original = np.max(np.abs(target_signal_L))
    # target_signal_L /= scale_signal_original
    target_signal_L = target_signal_L * np.sqrt(amplitude)  # scaling
    if amplitude != 1:
        target_signal_L += noise_filter(target_signal_L)

    # # ##### DEBUG START
    # if amplitude == 1:
    #     plot_traces(target_signal_L)
    # # ##### DEBUG END
    target_signal_L += min_traces_all
    # # ##### DEBUG START
    # if amplitude == 1:
    #     plot_traces(target_signal_L)
    # # ##### DEBUG END
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
    if model_is_free_pop:
        input_signal_neurons_L = np.concatenate((input_signal_neurons_L, np.zeros((len(input_signal), n_units_X))), axis=1)
        initial_value_L = np.concatenate((initial_value_L, np.array([np.mean(target_signal_L[0]) for _ in range(n_units_X)]) + np.random.normal(0, np.abs(np.mean(target_signal_L[0])) / 5, n_units_X)))

    train_list.append(TrainSignal(input_signal_neurons_L, target_signal_L, initial_value_L, label=f"Train {'High' if amplitude == 1 else 'Low'} L"))

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
    # # ##### DEBUG START
    # if amplitude == 1:
    #     plot_traces(target_signal_R)
    # # ##### DEBUG END
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
    if model_is_free_pop:
        input_signal_neurons_R = np.concatenate((input_signal_neurons_R, np.zeros((len(input_signal), n_units_X))), axis=1)
        initial_value_R = np.concatenate((initial_value_R, np.array([np.mean(target_signal_L[0]) for _ in range(n_units_X)]) + np.random.normal(0, np.abs(np.mean(target_signal_L[0])) / 5, n_units_X)))

    train_list.append(TrainSignal(input_signal_neurons_R, target_signal_R, initial_value_R, label=f"Train {'High' if amplitude == 1 else 'Low'} R"))


# ------------------------------------------------
# Plot distribution of performance
# ------------------------------------------------
h, b = get_hist(loss_list, bins=20, hist_range=(0, 0.1), center_bin=True)
plot_loss_dist = fig.create_plot(plot_title="Loss distribution",
                                 xpos=xpos, ypos=ypos, plot_height=plot_height,
                                 plot_width=plot_size_matrix,
                                 xmin=0, xmax=np.max(b), xticks=[0, np.max(b)],
                                 ymin=0, ymax=np.max(h), yticks=[0, np.max(h)],
                                 vlines=[loss_list[top_indices[0]]], helper_lines_lc="r")
plot_loss_dist.draw_line(b, h)

xpos = xpos_start
ypos -= plot_height + padding_big


# ------------------------------------------------
# Plot parameters and simulations on train input for top models
# ------------------------------------------------
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

    # tau = 0.2
    # W_eff = RNNService.compute_effective_jacobian(W, h)
    # eigvals, mu, timescales, eigvecs = RNNService.eigen_timescales(W_eff, dt=model.dt, tau=tau)
    # alignments = RNNService.slow_mode_alignment(eigvecs, timescales, torch.ones(model.n_units)/model.n_units, k=5)

    offset_hemisphere = model.nA + model.nB + model.nC + model.nD
    neuron_identity_array = np.concatenate((np.zeros((model.nA, 1)), np.ones((model.nB, 1)), 2*np.ones((model.nC, 1)), 3*np.ones((model.nD, 1)),
                                            np.zeros((model.nA, 1)), np.ones((model.nB, 1)), 2*np.ones((model.nC, 1)), 3*np.ones((model.nD, 1))))
    if model_is_free_pop:
        neuron_identity_array = np.concatenate((neuron_identity_array, 4*np.ones((model.nX, 1))))

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
                              colormap=cmap_here, zmin=0, zmax=3, image_interpolation=None)

    plot_ni_r = fig.create_plot(xpos=xpos, ypos=ypos + plot_size_matrix, plot_height=plot_size_vector,
                                plot_width=plot_size_matrix,
                                xmin=-0.5, xmax=n_neurons - 0.5,
                                ymin=-0.5, ymax=0.5)
    im = plot_ni_r.draw_image(neuron_identity_array.T, (-0.5, n_neurons - 0.5, -0.5, 0.5),
                              colormap=cmap_here, zmin=0, zmax=3, image_interpolation=None)

    x_ = np.arange(n_neurons)
    x = np.tile(x_, (n_neurons, 1))
    y = x.T
    im = plot_mask_W.draw_image(mask_W, (-0.5, n_neurons - 0.5, n_neurons - 0.5, -0.5),
                              colormap='PiYG', zmin=-1, zmax=1, image_interpolation=None)

    # Grid in mask_W
    position_line_between_pop = np.array([model.nA, model.nA + model.nB, model.nA + model.nB + model.nC,
                                          model.nA + model.nB + model.nC + model.nD,
                                          offset_hemisphere + model.nA, offset_hemisphere + model.nA + model.nB,
                                          offset_hemisphere + model.nA + model.nB + model.nC,
                                          offset_hemisphere + model.nA + model.nB + model.nC + model.nD])
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
                              colormap=cmap_here, zmin=0, zmax=3, image_interpolation=None)

    plot_ni_r = fig.create_plot(xpos=xpos, ypos=ypos+plot_size_matrix, plot_height=plot_size_vector,
                                plot_width=plot_size_matrix,
                                xmin=-0.5, xmax=n_neurons - 0.5,
                                ymin=-0.5, ymax=0.5)
    im = plot_ni_r.draw_image(neuron_identity_array.T, (-0.5, n_neurons - 0.5, -0.5, 0.5),
                              colormap=cmap_here, zmin=0, zmax=3, image_interpolation=None)


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

    for i_t, train_signals in enumerate(train_list):
        input_signal = torch.tensor(train_signals.input_signal, dtype=torch.float32)
        output_signal = train_signals.output_signal
        x0 = torch.tensor(train_signals.initial_value, dtype=torch.float32)
        label = train_signals.label
        res = RNNService.plot_response(model, t_sim, input_signal, xpos_here, ypos_here,
                                       t_exp=downsample_time_list, output_signal=output_signal, x0=x0,
                                       fig=fig, show_xaxis=i_t in [2, 3], show_yaxis=i_t % 2 == 0, plot_title_label=label,
                                       time_structure=ConfigurationRNN.time_structure_simulation_train)

        if i_t % 2 == 0:
            xpos_here += plot_size_here * 2 + padding_here * 2
        else:
            xpos_here = xpos_start_here
            ypos_here -= plot_size_here + padding_here

    xpos = xpos_start
    ypos -= plot_size_matrix + padding_vertical * 1.5

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
path_save.mkdir(parents=True, exist_ok=True)
fig.save(path_save / "check_train.pdf", open_file=False, tight=style.page_tight)
