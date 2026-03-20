import pickle
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from plot.style import RNNDSStyle
from utils.figure_helper import Figure

# ================================================================
# Paths
# ================================================================
path_top_model = Path(r"C:\Users\Roberto\Desktop\highlights\clem_rnns\data\results\freepop\mask_traces_freepop_16_slow9\RNNFreePop_neurons102_tau0.1_input2step_softplus\top_5")
path_save = path_top_model / "results"

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

palette = style.palette["neurons_4"]

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
top_indices = np.argsort(loss_list)

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
    population_indices = model.population_indices[:-1]
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
                                     ymin=0, ymax=10, yl="Activity" if i_pop == 0 else None, yticks=[0, 5, 10] if i_pop == 0 else None,
                                     vspans=[[20, 60, "k", 0.3]])
        xpos += plot_width + padding

        plot_pop_n.draw_line(time, anchor_traces, lc=palette[i_pop % 4])
        plot_pop_n.draw_line(time, free_traces, lc="k", label="Free neurons" if i_m == 0 and i_pop==len(population_indices)-1 else None)

    xpos = xpos_start
    ypos -= plot_height + padding

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
fig.save(path_save / f"neurons_dynamics_trial{i_trial}.pdf", open_file=False, tight=style.page_tight)
