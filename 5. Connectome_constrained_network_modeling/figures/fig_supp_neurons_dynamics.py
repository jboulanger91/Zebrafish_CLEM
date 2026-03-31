import pickle
from pathlib import Path

import torch
import numpy as np

# Manually add root path for imports to improve interoperability
import sys;

from dotenv import dotenv_values

sys.path.insert(0, "..")

from style import RNNDSStyle
from utils.figure_helper import Figure

# ================================================================
# Env and paths
# ================================================================
env =  dotenv_values()
path_top_model = Path(env["PATH_MODELS"])
path_save = Path(env["PATH_SAVE"])

# ================================================================
# Signal configuration
# ================================================================
i_trial = 2

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

palette = style.palette["neurons_5"]

# ================================================================
# Initialize figure container
# ================================================================
fig = Figure()

loss_list = []
model_list = []
# Fetch models
for path_model in path_top_model.glob("model_*.pkl"):
    model = pickle.load(open(path_model, "rb"))
    model_list.append(model)
    loss_list.append(model.loss_mse)

# Sort them by performance in training
top_indices = np.argsort(loss_list)  # [:1]

for i_m, i_m_sorted in enumerate(top_indices):
    model = model_list[i_m_sorted]
    # ----- Basic checks -----
    if not hasattr(model, "xs"):
        raise ValueError("model.xs not found. Run a forward pass before plotting.")
    if not hasattr(model, "population_indices"):
        raise ValueError("model.population_indices not found.")
    # if not hasattr(model, "free_indices_by_pop") or not hasattr(model, "anchor_indices_by_pop"):
    #     raise ValueError("model.free_indices_by_pop / anchor_indices_by_pop not found.")

    # Activity vector
    xs = model.xs  # (N, T, n_units)
    if not torch.is_tensor(xs):
        xs = torch.tensor(xs, dtype=torch.float32)
    N, T, n_units = xs.shape

    # Time vector
    time = np.arange(T) * model.dt

    # Select the trial (the example shown in training)
    x_trial = xs[i_trial].detach().cpu().numpy()  # (T, n_units)

    # store info about populations
    population_indices = model.population_indices
    free_by_pop = [model.idx_X]  # model.free_indices_by_pop
    free_indices = model.idx_X  # model.free_indices_by_pop
    anchor_by_pop = population_indices  # model.anchor_indices_by_pop
    n_pops = len(population_indices)
    if len(population_indices) == 0:
        raise ValueError("No valid populations to plot were provided.")

    # ----- Create figure -----
    for i_pop, i_neurons_in_pop in enumerate(population_indices):
        # Find free neuron traces
        # free_indices = np.array(free_by_pop[i_pop], dtype=int)
        free_traces = x_trial[:, free_indices]  # (T, n_free)

        # Find anchor neuron traces
        anchor_indices = np.array(anchor_by_pop[i_pop], dtype=int)
        anchor_traces = x_trial[:, anchor_indices]  # (T, n_anchor)

        plot_pop_n = fig.create_plot(plot_title=f"{RNNDSStyle.population_name_list[i_pop]}" if i_m == 0 else None,
                                     xpos=xpos, ypos=ypos, plot_width=plot_width, plot_height=plot_height,
                                     xmin=0, xmax=np.max(time), xl="Time (s)", xticks=[0, 20, 60, 80],
                                     ymin=0, ymax=8, yl=f"Model {i_m}\nActivity" if i_pop == 0 else None, yticks=[0, 4, 8] if i_pop == 0 else None,
                                     vspans=[[20, 60, "k", 0.1]])
        xpos += plot_width + padding

        color = palette[-1] if i_pop == len(population_indices)-1 else palette[i_pop % 4]
        plot_pop_n.draw_line(time, anchor_traces, lc=color)
        # plot_pop_n.draw_line(time, free_traces, lc="k", label="Free neurons" if i_m == 0 and i_pop==len(population_indices)-1 else None)

    xpos = xpos_start
    ypos -= plot_height + padding * 1.5

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
path_save.mkdir(parents=True, exist_ok=True)
fig.save(path_save / f"figure_supp_neurons_dynamics.pdf", open_file=False, tight=style.page_tight)
