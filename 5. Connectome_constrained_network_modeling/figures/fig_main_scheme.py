import pickle

import numpy as np
from pathlib import Path

import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model.RNNFreeNeurons import RNNFreeNeurons
from model.RNNFreePop import RNNFreePop
from plot.style import RNNDSStyle
from utils.figure_helper import Figure

# ------------------------------------------------
# Configuration
# ------------------------------------------------
path_dir = Path(r"C:\Users\Roberto\Desktop\highlights\clem_rnns")   # directory containing model_X.pkl
path_traces = path_dir / "data"
path_noise_estimation = path_traces / "noise_estimation" / "contralateral_motion_integrator_preferred_noise_estimation.pkl"
path_models = path_traces / "results" / "freeneurons" / "mask_traces_freeneurons_1_attempt2" / "RNNFreeNeurons_neurons102_tau0.1_input2step_softplus" / "top_5"
path_save = path_dir / "figures"
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
# Load traces to use as target signals
# ================================================================
# Define input signals used in training
input_signal = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)), np.zeros(int(duration_rest_end / dt))))
t_sim = np.linspace(0, duration_simulation, len(input_signal))
amplitude_input_signal_list = np.linspace(0.3, 1, n_input_signal)

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

target_signal_L = np.stack((traces_dict["iMI"]["preferred"], traces_dict["cMI"]["preferred"], traces_dict["MON"]["preferred"], traces_dict["sMI"]["preferred"],
                            traces_dict["iMI"]["null"], traces_dict["cMI"]["null"], traces_dict["MON"]["null"], traces_dict["sMI"]["null"]), axis=-1)
target_signal_L += min_traces_all
input_signal_neurons_L = np.concatenate((np.repeat(input_signal[..., np.newaxis], 4, axis=1),
                                         np.zeros((len(input_signal), 4))), axis=1)

# ------------------------------------------------
# Plot input and output in small panels
# ------------------------------------------------
ymin = 0; ymax = 2
duration_t_short = 10
t_short = downsample_time_list[-int(duration_t_short/dt_data):]
plot_size_here = plot_size
for i_side, side in enumerate(side_list):
    for i_cell, cell in enumerate(cell_types_list):
        plot_input = fig.create_plot(
            xpos=xpos, ypos=ypos + plot_height + padding, plot_height=plot_height/2, plot_width=plot_width,
            xmin=0, xmax=duration_simulation,  # xl="Time (s)" if i_cell == 0 else None,
            ymin=0, ymax=1, yl="Stimulus strength" if i_cell == 0 and i_side == 0 else None,
            yticks=[0, 1] if i_cell == 0 and i_side == 0 else None,
            hlines=[0]
        )
        plot_trace_cell = fig.create_plot(
            xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
            xmin=0, xmax=duration_simulation,  # xl="Time (s)" if i_cell == 0 else None,
            ymin=ymin, ymax=ymax,  # yticks=[ymin, ymax] if show_yaxis and side == 0 else None
            hlines=[min_traces_all]
        )
        for i_amp, amp in enumerate(amplitude_input_signal_list):
            plot_input.draw_line(t_sim, input_signal_neurons_L[:, int(i_cell+(i_side*4))] * amp, lc="k", alpha=0.5+(i_amp/len(amplitude_input_signal_list*0.5)))
            plot_trace_cell.draw_line(downsample_time_list, traces_dict[cell][side] * np.sqrt(amp) + min_traces_all, lc=palette[i_cell], alpha=0.5+(i_amp/len(amplitude_input_signal_list*0.5)))
        if i_cell == len(cell_types_list)-1 and i_side == len(side_list)-1:
            plot_trace_cell.draw_line(t_short, np.ones(len(t_short)), lc="k")
        # plot_trace_cell.draw_text(t_short, np.ones(len(t_short)), f"{duration_t_short} s")
        xpos += plot_size_here + padding

xpos = xpos_start
ypos -= plot_height * 2 + padding * 3

# ------------------------------------------------
# Plot target traces all in one panel
# ------------------------------------------------
plot_size_here = plot_height * 1.5
for i_amp, amp in enumerate(amplitude_input_signal_list):
    for i_side, side in enumerate(side_list):
        label_side = "L" if i_side == 0 else "R"
        label = f"Experimental data\nStimulus {label_side}" if (i_side == 0 and amp == 1) else f"Processed data\nStimulus {label_side}"
        plot_traces = fig.create_plot(
                    plot_title=label,
                    xpos=xpos, ypos=ypos, plot_height=plot_size_here, plot_width=plot_size_here,
                    xmin=0, xmax=duration_simulation,  # xl="Time (s)" if i_cell == 0 else None,
                    ymin=ymin, ymax=ymax,  # yticks=[ymin, ymax] if show_yaxis and side == 0 else None
                    hlines=[min_traces_all]
                )
        xpos += plot_width + padding
        for i_cell, cell in enumerate(cell_types_list):
            plot_traces.draw_line(downsample_time_list, traces_dict[cell][side_list[i_side]] * amp + min_traces_all, lc=palette[i_cell],
                                  label="Left-side cells" if i_cell == len(cell_types_list)-1 and i_amp == len(amplitude_input_signal_list)-1 and i_side == len(side_list)-1 else None)
            plot_traces.draw_line(downsample_time_list, traces_dict[cell][side_list[int(np.abs(i_side-1))]] * amp + min_traces_all, lc=palette[i_cell], line_dashes=(1, 2),
                                  label="Right-side cells" if i_cell == len(cell_types_list)-1 and i_amp == len(amplitude_input_signal_list)-1 and i_side == len(side_list)-1 else None)
            if i_cell == len(cell_types_list)-1 and i_amp == len(amplitude_input_signal_list)-1 and i_side == len(side_list)-1:
                plot_traces.draw_line(t_short, np.ones(len(t_short)), lc="k")
                plot_traces.draw_line(t_short[0]*np.ones(2), [min_traces_all, min_traces_all+0.2], lc="k")

xpos = xpos_start
ypos -= plot_height * 2 + padding * 3

# ------------------------------------------------
# Define function to show the model parameters
# ------------------------------------------------
value_lim = 1
def plot_model_matrices(model, fig, xpos, ypos, value_lim=1):
    n_neurons = model.n_units
    plot_size_vector = plot_size_matrix / n_neurons
    U = model.U().detach().cpu().numpy()
    W = model.W().detach().cpu().numpy().T
    mask_W = (model.mask_W * model.signs).detach().cpu().numpy().T
    W_clamp = torch.abs(
        torch.clamp(torch.abs(model.W_raw), model.clamp_weights_min, model.clamp_weights_max)).detach().numpy().T

    offset_hemisphere = model.nA + model.nB + model.nC + model.nD
    neuron_identity_array = np.concatenate(
        (np.zeros((model.nA, 1)), np.ones((model.nB, 1)), 2 * np.ones((model.nC, 1)), 3 * np.ones((model.nD, 1)),
         np.zeros((model.nA, 1)), np.ones((model.nB, 1)), 2 * np.ones((model.nC, 1)), 3 * np.ones((model.nD, 1))))

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
                              colormap=style.cmap_list["neurons_5"], zmin=0, zmax=3, image_interpolation=None)

    plot_ni_r = fig.create_plot(xpos=xpos, ypos=ypos + plot_size_matrix, plot_height=plot_size_vector,
                                plot_width=plot_size_matrix,
                                xmin=-0.5, xmax=n_neurons - 0.5,
                                ymin=-0.5, ymax=0.5)
    im = plot_ni_r.draw_image(neuron_identity_array.T, (-0.5, n_neurons - 0.5, -0.5, 0.5),
                              colormap=style.cmap_list["neurons_5"], zmin=0, zmax=3, image_interpolation=None)

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
    plot_ni_c = fig.create_plot(xpos=xpos - plot_size_vector, ypos=ypos, plot_height=plot_size_matrix,
                                plot_width=plot_size_vector,
                                xmin=-0.5, xmax=0.5,
                                ymin=-0.5, ymax=n_neurons - 0.5)
    im = plot_ni_c.draw_image(neuron_identity_array, (-0.5, 0.5, n_neurons - 0.5, -0.5),
                              colormap=style.cmap_list["neurons_5"], zmin=0, zmax=3, image_interpolation=None)

    plot_ni_r = fig.create_plot(xpos=xpos, ypos=ypos + plot_size_matrix, plot_height=plot_size_vector,
                                plot_width=plot_size_matrix,
                                xmin=-0.5, xmax=n_neurons - 0.5,
                                ymin=-0.5, ymax=0.5)
    im = plot_ni_r.draw_image(neuron_identity_array.T, (-0.5, n_neurons - 0.5, -0.5, 0.5),
                              colormap=style.cmap_list["neurons_5"], zmin=0, zmax=3, image_interpolation=None)

    x_ = np.arange(n_neurons)
    x = np.tile(x_, (n_neurons, 1))
    y = x.T
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
                               ticks=[-value_lim, -value_lim / 2, 0, value_lim / 2, value_lim])
    xpos += plot_size_matrix + padding * 1.5

    # Draw connectivity matrix W_clamp after training, before masking
    plot_W = fig.create_plot(plot_title="W fast",
                             xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                             plot_width=plot_size_matrix,
                             xmin=-0.5, xmax=n_neurons - 0.5,  # xticklabels_rotation=90,
                             # xticks=np.arange(n_neurons),
                             ymin=-0.5, ymax=n_neurons - 0.5,
                             )
    plot_ni_c = fig.create_plot(xpos=xpos - plot_size_vector, ypos=ypos, plot_height=plot_size_matrix,
                                plot_width=plot_size_vector,
                                xmin=-0.5, xmax=0.5,
                                ymin=-0.5, ymax=n_neurons - 0.5)
    im = plot_ni_c.draw_image(neuron_identity_array, (-0.5, 0.5, n_neurons - 0.5, -0.5),
                              colormap=style.cmap_list["neurons_5"], zmin=0, zmax=3, image_interpolation=None)

    plot_ni_r = fig.create_plot(xpos=xpos, ypos=ypos + plot_size_matrix, plot_height=plot_size_vector,
                                plot_width=plot_size_matrix,
                                xmin=-0.5, xmax=n_neurons - 0.5,
                                ymin=-0.5, ymax=0.5)
    im = plot_ni_r.draw_image(neuron_identity_array.T, (-0.5, n_neurons - 0.5, -0.5, 0.5),
                              colormap=style.cmap_list["neurons_5"], zmin=0, zmax=3, image_interpolation=None)

    x_ = np.arange(n_neurons)
    x = np.tile(x_, (n_neurons, 1))
    y = x.T
    im = plot_W.draw_image(W_clamp, (-0.5, n_neurons - 0.5, n_neurons - 0.5, -0.5),
                           colormap='Greys', zmin=0, zmax=value_lim, image_interpolation=None)

    plot_W_grid = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=plot_size_matrix, plot_width=plot_size_matrix,
                                  xmin=-0.5, xmax=n_neurons - 0.5, ymin=-0.5, ymax=n_neurons - 0.5,
                                  helper_lines_lc="white",
                                  hlines=model.n_units - position_line_between_pop - 0.5,
                                  vlines=position_line_between_pop - 0.5)
    # divider = make_axes_locatable(plot_W.ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # plot_W.figure.fig.colorbar(im, cax=cax, orientation='vertical',
    #                            ticks=[0, value_lim / 2, value_lim])

    xpos += plot_size_matrix + padding * 1.5

    # W_slow_pop_n = torch.abs(model.W_slow_module.gammas()[i] * torch.outer(model.W_slow_module.v_slow[i], model.W_slow_module.u_slow[i])).T
    W_slow_pop_n = torch.abs(model.W_slow_module(model.device)).T
    value_lim_slow = value_lim  # torch.max(W_slow_pop_n)

    # Draw connectivity matrix W_clamp after training, before masking
    plot_W = fig.create_plot(plot_title=f"W slow",
                             xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                             plot_width=plot_size_matrix * 1.1,
                             xmin=-0.5, xmax=n_neurons - 0.5,  # xticklabels_rotation=90,
                             # xticks=np.arange(n_neurons),
                             ymin=-0.5, ymax=n_neurons - 0.5,
                             )
    plot_ni_c = fig.create_plot(xpos=xpos - plot_size_vector, ypos=ypos, plot_height=plot_size_matrix,
                                plot_width=plot_size_vector,
                                xmin=-0.5, xmax=0.5,
                                ymin=-0.5, ymax=n_neurons - 0.5)
    im = plot_ni_c.draw_image(neuron_identity_array, (-0.5, 0.5, n_neurons - 0.5, -0.5),
                              colormap=style.cmap_list["neurons_5"], zmin=0, zmax=3, image_interpolation=None)

    plot_ni_r = fig.create_plot(xpos=xpos, ypos=ypos + plot_size_matrix, plot_height=plot_size_vector,
                                plot_width=plot_size_matrix,
                                xmin=-0.5, xmax=n_neurons - 0.5,
                                ymin=-0.5, ymax=0.5)
    im = plot_ni_r.draw_image(neuron_identity_array.T, (-0.5, n_neurons - 0.5, -0.5, 0.5),
                              colormap=style.cmap_list["neurons_5"], zmin=0, zmax=3, image_interpolation=None)

    x_ = np.arange(n_neurons)
    x = np.tile(x_, (n_neurons, 1))
    y = x.T
    im = plot_W.draw_image(W_slow_pop_n.detach().numpy(), (-0.5, n_neurons - 0.5, n_neurons - 0.5, -0.5),
                           colormap='Greys', zmin=0, zmax=value_lim_slow, image_interpolation=None)

    divider = make_axes_locatable(plot_W.ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plot_W.figure.fig.colorbar(im, cax=cax, orientation='vertical',
                               ticks=[0, value_lim / 2, value_lim])

    return xpos, ypos

# ------------------------------------------------
# Showcase a random untrained model
# ------------------------------------------------
tau_neuron = 0.1
n_units_A = 15
n_units_B = 15
n_units_C = 2
n_units_D = 11
n_units_hemi = n_units_A + n_units_B + n_units_C + n_units_D
n_units = n_units_hemi * 2
n_free_neurons = 16
n_slow_pops = 8
model = RNNFreePop(nA=n_units_A, nB=n_units_B, nC=n_units_C, nD=n_units_D, nX=n_free_neurons,
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
                     sparsity_XA=0.07, sparsity_XB=0.1, sparsity_XC=0.1, sparsity_XD=0.02, sparsity_XX=0.074,
                     sparsity_U=1,
                     tau=tau_neuron, dt=dt, clamp_weights_min=1e-2, n_slow_pops=n_slow_pops)
model.eval()
xpos, ypos = plot_model_matrices(model, fig, xpos, ypos, value_lim)


# ------------------------------------------------
# Loop over trained models
# ------------------------------------------------
if loop_over_trained_models:
    i_model = 0
    for path_model in path_models.glob(f"model_*.pkl"):
        print(f"Evaluating model {i_model}")
        i_model += 1

        # Load model instance
        with open(path_model, 'rb') as f:
            model = pickle.load(f)
        model.eval()

        xpos, ypos = plot_model_matrices(model, fig, xpos, ypos, value_lim)

        xpos = xpos_start
        ypos -= plot_size_matrix + padding_vertical * 1.5

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
path_save.mkdir(parents=True, exist_ok=True)
fig.save(path_save / "figure_main_scheme.pdf", open_file=False, tight=style.page_tight)
