from pathlib import Path
from dotenv import dotenv_values

import torch
import numpy as np

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "..")

from style import RNNDSStyle
from utils.services.rnn_service import RNNService
from utils.figure_helper import Figure
from utils.load_model import load_model
from utils.config import ConfigurationRNN
from utils.math.operators import inv_softplus


# ================================================================
# Configurations
# ================================================================
mode = "lda"  # alternatives: "connectome", "lda". It produces the related panels in supplement
show_activity_single_neurons = True
i_trial = 0
n_show_models = 3

# ================================================================
# Env and paths
# ================================================================
env = dotenv_values()
path_dir = Path(env["PATH_DIR"])
path_data = path_dir / "data"
if mode == "connectome":
    path_models = path_dir / "models" / mode / "top_5"
elif mode == "lda":
    path_models = path_dir / "models" / "connectome_noLDA_target" / "top_5"
else:
    raise NotImplementedError
path_save = path_dir / "results"


# ================================================================
# Plot configuration (layout, sizes, padding, etc.)
# ================================================================
style = RNNDSStyle()

xpos_start = style.xpos_start
ypos_start = style.ypos_start - 1
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
colormap = style.cmap_list["neurons_4"]

# ================================================================
# Initialize figure container
# ================================================================
fig = Figure()

loss_list = []
model_list = []
# Fetch models
for path_model in path_models.glob("model_*.pt"):
    model = load_model(path_model)
    model_list.append(model)
    loss_list.append(model.loss_mse)

# Sort them by performance in training
top_indices = np.argsort(loss_list)[:n_show_models]

for i_m, i_m_sorted in enumerate(top_indices):
    model = model_list[i_m_sorted]
    model.eval()
    # ----- Basic checks -----
    if not hasattr(model, "xs"):
        raise ValueError("model.xs not found. Run a forward pass before plotting.")
    if not hasattr(model, "population_indices"):
        raise ValueError("model.population_indices not found.")


    if show_activity_single_neurons:
        # Populate model.xs by simulating once a step on the left
        duration_rest_start = ConfigurationRNN.time_structure_simulation_train["rest_start"]
        duration_stimulus = ConfigurationRNN.time_structure_simulation_train["stimulus"]
        duration_rest_end = ConfigurationRNN.time_structure_simulation_train["rest_end"]
        dt = ConfigurationRNN.dt_simulation

        dict_neurons = model.dict_neurons
        n_units_LiMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["iMI"]["n_neurons"]
        n_units_LcMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["cMI"]["n_neurons"]
        n_units_LMON = dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["MON"]["n_neurons"]
        n_units_LsMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["sMI"]["n_neurons"]
        n_units_RiMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["iMI"]["n_neurons"]
        n_units_RcMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["cMI"]["n_neurons"]
        n_units_RMON = dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["MON"]["n_neurons"]
        n_units_RsMI = dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["sMI"]["n_neurons"]
        n_units = model.n_units

        # Define input/output signals for training
        input_signal = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)),
                                       np.zeros(int(duration_rest_end / dt))))

        # Load traces to use as target signals
        cell_types_list = ["iMI", "cMI", "MON", "sMI"]
        side_list = ["preferred", "null"]
        traces_dict = {ct: {s: None for s in side_list} for ct in cell_types_list}
        min_traces_all = 0  # initialize offset
        for ct in cell_types_list:
            for s in side_list:
                filename = f"avgresponses_{ct}_{s}_constant.csv"
                data = np.loadtxt(path_data / filename, dtype=float, delimiter=",", skiprows=1)
                downsample_time_list = data[:, 0]
                traces_dict[ct][s] = data[:, 1] / 100
                min_trace_here = np.min(data[:, 1] / 100)
                if min_trace_here < min_traces_all:
                    min_traces_all = min_trace_here
        min_traces_all = np.abs(min_traces_all)

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
        target_signal_L += min_traces_all

        input_signal_neurons_L = np.concatenate((np.array([input_signal for _ in range(n_units_LiMI)]),
                                                 np.array([input_signal for _ in range(n_units_LcMI)]),
                                                 np.array([input_signal for _ in range(n_units_LMON)]),
                                                 np.array([input_signal for _ in range(n_units_LsMI)]),
                                                 np.array(
                                                     [np.zeros_like(input_signal) for _ in range(n_units_RiMI)]),
                                                 np.array(
                                                     [np.zeros_like(input_signal) for _ in range(n_units_RcMI)]),
                                                 np.array(
                                                     [np.zeros_like(input_signal) for _ in range(n_units_RMON)]),
                                                 np.array([np.zeros_like(input_signal) for _ in
                                                           range(n_units_RsMI)]),)).T
        initial_value_L = np.concatenate(
            (np.array([target_signal_L[0, 0] for _ in range(n_units_LiMI)]) + np.random.normal(0, np.abs(
                target_signal_L[0, 0]) / 5, n_units_LiMI),
             np.array([target_signal_L[0, 1] for _ in range(n_units_LcMI)]) + np.random.normal(0, np.abs(
                 target_signal_L[0, 1]) / 5, n_units_LcMI),
             np.array([target_signal_L[0, 2] for _ in range(n_units_LMON)]) + np.random.normal(0, np.abs(
                 target_signal_L[0, 2]) / 5, n_units_LMON),
             np.array([target_signal_L[0, 3] for _ in range(n_units_LsMI)]) + np.random.normal(0, np.abs(
                 target_signal_L[0, 3]) / 5, n_units_LsMI),
             np.array([target_signal_L[0, 4] for _ in range(n_units_RiMI)]) + np.random.normal(0, np.abs(
                 target_signal_L[0, 4]) / 5, n_units_RiMI),
             np.array([target_signal_L[0, 5] for _ in range(n_units_RcMI)]) + np.random.normal(0, np.abs(
                 target_signal_L[0, 5]) / 5, n_units_RcMI),
             np.array([target_signal_L[0, 6] for _ in range(n_units_RMON)]) + np.random.normal(0, np.abs(
                 target_signal_L[0, 6]) / 5, n_units_RMON),
             np.array([target_signal_L[0, 7] for _ in range(n_units_RsMI)]) + np.random.normal(0, np.abs(
                 target_signal_L[0, 7]) / 5, n_units_RsMI),))

        initial_value_L = torch.tensor(inv_softplus(initial_value_L), dtype=torch.float32)
        input_signal = torch.tensor(input_signal_neurons_L, dtype=torch.float32)
        model.forward(initial_value_L, input_signal, filter_xs=True)

        # Activity vector
        xs = model.xs  # (N, T, n_units)
        if not torch.is_tensor(xs):
            xs = torch.tensor(xs, dtype=torch.float32)
        N, T, _ = xs.shape
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

    if show_activity_single_neurons:
        # Time vector
        time = np.arange(T) * model.dt

    # ================================================================
    # Show matrices
    # ================================================================
    neuron_identity_array = np.concatenate(
        (np.zeros((n_units_LiMI, 1)), np.ones((n_units_LcMI, 1)), 2 * np.ones((n_units_LMON, 1)),
         3 * np.ones((n_units_LsMI, 1)),
         np.zeros((n_units_RiMI, 1)), np.ones((n_units_RcMI, 1)), 2 * np.ones((n_units_RMON, 1)),
         3 * np.ones((n_units_RsMI, 1))))
    # Normalize
    neuron_identity_array /= 3

    U = model.U().detach().numpy()
    mask_U = model.mask_U.detach().numpy()
    W = model.W().detach().numpy().T
    try:
        mask_W = (model.mask_W * model.signs).detach().numpy().T
    except AttributeError:
        mask_W = None

    grid_pop = np.array([n_units_LiMI, n_units_LiMI + n_units_LcMI, n_units_LiMI + n_units_LcMI + n_units_LMON,
                         n_units_LiMI + n_units_LcMI + n_units_LMON + n_units_LsMI,
                         n_units_hemi + n_units_RiMI, n_units_hemi + n_units_RiMI + n_units_RcMI,
                         n_units_hemi + n_units_RiMI + n_units_RcMI + n_units_RMON,
                         n_units_hemi + n_units_RiMI + n_units_RcMI + n_units_RMON + n_units_RsMI])

    mask_U = model.mask_U
    if mask_W is not None:
        _, _, ypos = RNNService.plot_connectivity(mask_W, U=mask_U, neuron_identity_array=neuron_identity_array,
                                                     grid_pop=grid_pop,
                                                     fig=fig, xpos=xpos, ypos=ypos, plot_size_matrix=plot_size_matrix,
                                                     padding=padding, value_lim=[-1, 1],
                                                     plot_title="Mask W",
                                                     cmap_pop=colormap, show_colorbar=i_m == len(top_indices)-1)

    ypos -= plot_size_matrix + padding * 1.5
    _, _, ypos = RNNService.plot_connectivity(W, U=U, neuron_identity_array=neuron_identity_array,
                                                 grid_pop=grid_pop,
                                                 fig=fig, xpos=xpos, ypos=ypos, plot_size_matrix=plot_size_matrix,
                                                 padding=padding, value_lim=[-1, 1], cmap_pop=colormap, show_colorbar=i_m == len(top_indices)-1)
    # xpos += plot_size_matrix + padding
    ypos -= plot_size_matrix + padding*1.5

    # Store reference position after matrix
    xpos_start_here = xpos
    ypos_start_here = ypos

    if show_activity_single_neurons:
        # Select the trial (the example shown in training)
        x_trial = xs[0].detach().cpu().numpy()  # (T, n_units)

        # store info about populations
        population_indices = model.population_indices_all
        anchor_by_pop = model.population_indices  # model.anchor_indices_by_pop
        n_pops = len(population_indices)
        if len(population_indices) == 0:
            raise ValueError("No valid populations to plot were provided.")

        # ================================================================
        # Show single-neuron activity
        # ================================================================
        # ----- Create figure -----
        offset_side = 0
        for i_pop, i_neurons_in_pop in enumerate(population_indices):
            if i_pop == 4:
                offset_side = 1
                xpos = xpos_start_here + plot_width + padding
                ypos = ypos_start_here

            # Find anchor neuron traces
            anchor_indices = np.array(anchor_by_pop[i_pop], dtype=int)
            if mode == "lda":
                free_indices = list(set(population_indices[i_pop]) - set(anchor_indices))

            # Find free neuron traces
            anchor_traces = x_trial[:, anchor_indices]  # (T, n_anchor)
            if mode == "lda":
                free_traces = x_trial[:, free_indices]  # (T, n_free)

            plot_pop_n = fig.create_plot(plot_title=f"{RNNDSStyle.population_name_list[i_pop]}" if i_m == 0 else None,
                                         xpos=xpos, ypos=ypos, plot_width=plot_width*1.2, plot_height=plot_height*1.2,
                                         xmin=0, xmax=np.max(time),  # xl="Time (s)", xticks=[0, 20, 60, 80],
                                         ymin=-1, ymax=10, yticks=(-1, 0, 10),  # yl=f"Model {i_m}\nActivity" if i_pop == 0 else None, yticks=[0, 4, 8] if i_pop == 0 else None,
                                         vspans=[[20, 60, "k", 0.1]])
            if i_pop == 0:
                plot_pop_n.draw_line(np.ones(2) * 25, [6, 7], lc="k")
                plot_pop_n.draw_text(0, 5, r"$\Delta$F/F" + "\n1")
            elif i_pop == len(population_indices) - 2:
                plot_pop_n.draw_line([1, 6], np.ones(2)*6, lc="k")
                plot_pop_n.draw_text(1, 4, f"5\nTime (s)")
            ypos -= plot_height + padding*1.5

            color = palette[-1] if i_pop == len(population_indices)-1 else palette[i_pop % 4]
            plot_pop_n.draw_line(time, anchor_traces, lc=color)
            if mode == "lda":
                plot_pop_n.draw_line(time, free_traces, lc="k", alpha=0.2)

    xpos += plot_size_matrix * 4/5
    ypos = ypos_start

# ================================================================
# Save final figure
# ================================================================
path_save.mkdir(parents=True, exist_ok=True)
fig.save(path_save / f"figure_supp_neurons_{mode}.pdf", open_file=False, tight=style.page_tight)
