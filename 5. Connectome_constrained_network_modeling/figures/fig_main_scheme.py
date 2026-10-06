import pickle

import numpy as np
from pathlib import Path

import torch
from dotenv import dotenv_values
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "..")

from model.core.RNNConnectome import RNNConnectome
from style import RNNDSStyle
from utils.figure_helper import Figure
from utils.services.rnn_service import RNNService
from utils.load_model import load_model



# ------------------------------------------------
# Env and paths
# ------------------------------------------------
env = dotenv_values()
path_dir = Path(env["PATH_DIR"])
path_traces = path_dir / "data"   # directory containing avgresponses_X.csv
path_noise_estimation = path_dir / "data" / "noise_estimation"
path_models = path_dir / "models"   # directory containing model_X.pt
path_save = path_dir / "results"
path_model = path_dir / "data" / "connectome.csv"


# ------------------------------------------------
# Configuration
# ------------------------------------------------
loop_over_trained_models = True

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
colormap = style.cmap_list["neurons_4"]

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
def plot_model_matrices(model, fig, xpos, ypos, value_lim=1):
    n_units_hemi = model.n_units_hemi
    n_units_LiMI = len(model.idx_LiMI)
    n_units_LcMI = len(model.idx_LcMI)
    n_units_LMON = len(model.idx_LMON)
    n_units_LsMI = len(model.idx_LsMI)
    n_units_RiMI = len(model.idx_RiMI)
    n_units_RcMI = len(model.idx_RcMI)
    n_units_RMON = len(model.idx_RMON)
    n_units_RsMI = len(model.idx_RsMI)
    neuron_identity_array = np.concatenate(
        (np.zeros((n_units_LiMI, 1)), np.ones((n_units_LcMI, 1)), 2 * np.ones((n_units_LMON, 1)),
         3 * np.ones((n_units_LsMI, 1)),
         np.zeros((n_units_RiMI, 1)), np.ones((n_units_RcMI, 1)), 2 * np.ones((n_units_RMON, 1)),
         3 * np.ones((n_units_RsMI, 1))))
    # Normalize
    neuron_identity_array /= 3

    mask_W = (model.mask_W).detach().numpy().T
    W = model.W().detach().numpy().T
    U = model.U().detach().numpy()

    grid_pop = np.array([n_units_LiMI, n_units_LiMI + n_units_LcMI, n_units_LiMI + n_units_LcMI + n_units_LMON,
                         n_units_LiMI + n_units_LcMI + n_units_LMON + n_units_LsMI,
                         n_units_hemi + n_units_RiMI, n_units_hemi + n_units_RiMI + n_units_RcMI,
                         n_units_hemi + n_units_RiMI + n_units_RcMI + n_units_RMON,
                         n_units_hemi + n_units_RiMI + n_units_RcMI + n_units_RMON + n_units_RsMI])

    _, xpos, ypos = RNNService.plot_connectivity(mask_W, U=None, neuron_identity_array=neuron_identity_array,
                                                 grid_pop=grid_pop,
                                                 fig=fig, xpos=xpos, ypos=ypos, plot_size_matrix=plot_size_matrix,
                                                 padding=padding, value_lim=[-value_lim, value_lim], plot_title=f"Mask W",
                                                 cmap_pop=colormap, show_colorbar=False)
    _, xpos, ypos = RNNService.plot_connectivity(W, U=U, neuron_identity_array=neuron_identity_array, grid_pop=grid_pop,
                                                 fig=fig, xpos=xpos, ypos=ypos, plot_size_matrix=plot_size_matrix,
                                                 padding=padding, value_lim=[-value_lim, value_lim], cmap_pop=colormap)
    return xpos, ypos


# ------------------------------------------------
# Showcase the best trained model
# ------------------------------------------------
model = load_model(path_model)
model.eval()

# xpos += plot_size_matrix
xpos, ypos = plot_model_matrices(model, fig, xpos, ypos)

xpos = xpos_start
ypos -= plot_size_matrix + padding_vertical * 1.5


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

        xpos, ypos = plot_model_matrices(model, fig, xpos, ypos)
        xpos = xpos_start
        ypos -= plot_size_matrix + padding_vertical * 1.5

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
path_save.mkdir(parents=True, exist_ok=True)
fig.save(path_save / "figure_main_scheme.pdf", open_file=False, tight=style.page_tight)
