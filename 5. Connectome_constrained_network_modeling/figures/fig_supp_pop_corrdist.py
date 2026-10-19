"""
Overview
--------
This script generates the following figure panels in Boulanger-Weill et al. 2026:
- Extended Data Fig. 8c

This script evaluates the fidelity of trained recurrent neural network (RNN) models
against empirical calcium imaging target traces across discrete anatomical neural
populations (iMI, cMI, MON, sMI across both hemispheres). It quantifies reconstruction
accuracy using Pearson correlation and Root Mean Square Error (RMSE) at both the population
readout level and the individual single-neuron level, visualizing the resulting metric
distributions across subpopulations as violin plots.

Before running it, please make sure you have set up your .env with all the necessary variables:
- PATH_DIR="/path/to/project_root"  # Root project directory containing data/, models/, and results/

Core Pipeline & Workflow:
1. Environment & Path Resolution:
   - Reads `PATH_DIR` from the .env configuration to construct paths for empirical target
     traces (`data/`), top model checkpoints (`models/connectome/top_6/`), and output PDF exports (`results/`).
   - Configures canvas layout, margins, and panel dimensions via `RNNDSStyle` and `Figure`.

2. Simulation Conditions & Empirical Target Preparation:
   - Formulates step driving currents across resting, stimulus integration, and recovery phases,
     supporting either full-duration or truncated window simulations (`reduced_simulation`).
   - Ingests empirical average response curves (`avgresponses_{cell}_{side}_constant.csv`) for
     preferred and null orientations, shifts baselines to non-negative ranges, and stacks them into
     a bilateral reference target matrix.

3. Model Forward Simulation & Temporal Alignment:
   - Iterates through top model checkpoints (`model_*.pt`), extracts population sizes and
     subpopulation indices from metadata dictionaries, and samples noisy initial baseline states.
   - Inverts the softplus activation (`inv_softplus`) to initialize latent state vectors and executes
     the model forward pass under left-hemisphere driving current.
   - Slices activity over the full trial duration or restricts evaluation to an initial transient
     integration window (`compute_in_window`), interpolating model dynamics and empirical recordings
     onto a shared temporal grid via `DSService.interpolate_to_common_grid`.

4. Population Readout & Single-Neuron Metric Evaluation:
   - Evaluates population-averaged readout trajectories against target signals to compute Pearson
     correlation and RMSE.
   - Extracts individual anchor neuron trajectories per subpopulation, filters out quiescent/sub-threshold
     units using dynamic range quantiles, and computes single-neuron correlation and RMSE metrics against
     the target curve.
   - Compiles bilateral population distributions as well as side-collapsed (4-class) cell distributions.

5. Distribution Visualization & Figure Export:
   - Generates categorical violin plots comparing correlation and RMSE distributions across neural
     populations, superimposing median and mean markers.
   - Exports the assembled multi-panel diagnostic vector graphic as a PDF to the results directory.
"""

import torch
import numpy as np

from dotenv import dotenv_values
from pathlib import Path

# Manually add root path for imports to improve interoperability across parent modules
import sys; sys.path.insert(0, "..")

from style import RNNDSStyle
from utils.services.rnn_service import RNNService
from utils.services.ds_service import DSService
from utils.figure_helper import Figure
from utils.load_model import load_model
from utils.config import ConfigurationRNN, ConfigurationNeural
from utils.math.operators import inv_softplus


# ------------------------------------------------
# Configuration
# ------------------------------------------------
reduced_simulation = False  # simulate only between start and 10s after simulus start
compute_in_window = False  # compute corr and rmse only in the 10s window after stimulus start

# Toggle visualization of the loss comparison violin plot
show_corr = False
show_neuron_corr = True
show_neuron_corr_noside = True
show_rmse = False
show_neuron_rmse = True

# Load directory paths from environment configuration (.env)
label_dir = "connectome"
env = dotenv_values()
path_dir = Path(env["PATH_DIR"])
path_data = path_dir / "data"
path_models = path_dir / "models" / label_dir / "top_6"
path_save = path_dir / "results"


# ----------------------------------------------------------------
# Plot configuration (layout, sizes, padding, etc.)
# ----------------------------------------------------------------
# Initialize custom canvas styling, plot dimensions, and layout coordinates
style = RNNDSStyle()

plot_height = style.plot_size_big
plot_height_small = plot_height / 2.5

plot_width_small = style.plot_width_small

plot_width = style.plot_size_big * 3

padding = style.padding / 2
padding_big = style.padding * 2
padding_vertical = style.padding

xpos_start = style.xpos_start
ypos_start = style.ypos_start - plot_height/2
xpos = xpos_start
ypos = ypos_start - padding


# ----------------------------------------------------------------
# Initialize figure container
# ----------------------------------------------------------------
# Instantiate figure canvas container
fig = Figure()


# ----------------------------------------------------------------
# Define simulation conditions
# ----------------------------------------------------------------
# time structure
duration_rest_start = ConfigurationRNN.time_structure_simulation_train["rest_start"]
duration_stimulus = ConfigurationRNN.time_structure_simulation_train["stimulus"]
duration_rest_end = ConfigurationRNN.time_structure_simulation_train["rest_end"]
dt = ConfigurationRNN.dt_simulation

# Define input/output signals for training: rectangular pulse step waveform
if reduced_simulation:
    t_sim = np.arange(0, duration_rest_start + 10, dt)
    input_signal = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(10 / dt))))
else:
    t_sim = np.arange(0, duration_rest_start + duration_stimulus + duration_rest_end, dt)
    input_signal = np.concatenate(
        (np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)),
         np.zeros(int(duration_rest_end / dt))))

if compute_in_window:
    label_save = "_window"
else:
    label_save = ""

# Load traces to use as target signals across cell types and preferred/null directions
cell_types_list = ["iMI", "cMI", "MON", "sMI"]
side_list = ["preferred", "null"]
traces_dict = {ct: {s: None for s in side_list} for ct in cell_types_list}
min_traces_all = 0  # initialize offset
for ct in cell_types_list:
    for s in side_list:
        filename = f"avgresponses_{ct}_{s}_constant.csv"
        data = np.loadtxt(path_data / filename, dtype=float, delimiter=",", skiprows=1)
        t_exp = data[:, 0]
        traces_dict[ct][s] = data[:, 1] / 100
        min_trace_here = np.min(data[:, 1] / 100)
        if min_trace_here < min_traces_all:
            min_traces_all = min_trace_here
        if reduced_simulation:
            i_stop = np.argwhere(t_exp >= duration_rest_start+10)[0][0]
            t_exp = t_exp[:i_stop]
            traces_dict[ct][s] = traces_dict[ct][s][:i_stop]
# Compute baseline shift to ensure strictly positive target values
min_traces_all = np.abs(min_traces_all)

# Assemble composite bilateral target signal matrix
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

threshold_range_pop = (np.max(target_signal_L, axis=0) - np.min(target_signal_L, axis=0)) / 2
threshold_std_pop = np.std(target_signal_L, axis=0) / 2

# ----------------------------------------------------------------
# Loop over all tests
# ----------------------------------------------------------------
# Preallocate containers for correlation and RMSE by population
pop_list = ConfigurationNeural.POP_META["name"]
corr_pop_list = [[] for c in pop_list]  # Initialize list by population of correlations in the current model condition
corr_pop_neuron_list = [[] for c in pop_list]  # Initialize list by population of single-neuron correlations in the current model condition
corr_pop_neuron_list_ = [[] for c in pop_list[:4]]  # Initialize list by population of single-neuron correlations in the current model condition
rmse_pop_list = [[] for c in pop_list]  # Initialize list by population of rmse in the current model condition
rmse_pop_neuron_list = [[] for c in pop_list]  # Initialize list by population of single-neuron rmse in the current model condition
i_model = 0
# ------------------------------------------------
# Loop over all trained models
# ------------------------------------------------
for path_model in path_models.glob("model_*.pt"):
    print(f"Evaluating model {i_model}")
    i_model += 1
    model = load_model(path_model)
    model.eval()
    # ----- Basic checks -----
    # Validate required internal attributes on the loaded PyTorch module
    if not hasattr(model, "xs"):
        raise ValueError("model.xs not found. Run a forward simulation before plotting.")
    if not hasattr(model, "population_indices"):
        raise ValueError("model.population_indices not found.")

    # --------------------------------------------------------
    # Populate model.xs by simulating once a step on the left
    # --------------------------------------------------------

    # Extract unit counts across bilateral cell subpopulations from model metadata
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

    # Formulate input signal across all individual units, driving only left-hemisphere neurons
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
    # Sample noisy initial firing rates centered on target baseline activity
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

    # Convert initial states and drive signals to tensors and execute model forward pass
    initial_value_L = torch.tensor(inv_softplus(initial_value_L), dtype=torch.float32)
    input_signal_neurons_L = torch.tensor(input_signal_neurons_L, dtype=torch.float32)
    xs, ys = model.forward(initial_value_L, input_signal_neurons_L, filter_xs=True)

    # Retrieve simulated hidden unit trajectory tensor: shape (batch_size, time_steps, n_units)
    if not torch.is_tensor(xs):
        xs = torch.tensor(xs, dtype=torch.float32)
    N, T, _ = xs.shape

    # Retrieve population indices and identify target anchor neurons versus free neurons
    population_indices = model.population_indices_all
    anchor_by_pop = model.population_indices  # model.anchor_indices_by_pop
    n_pops = len(population_indices)
    if len(population_indices) == 0:
        raise ValueError(f"No valid populations to plot were found for model {i_model}.")

    # ----------------------------------------------------------------
    # Analyze correlation of readout activity with target
    # ----------------------------------------------------------------
    for i_pop in range(len(population_indices)):
        # Compute amplitude-independent performance for integration window
        if compute_in_window:
            t_start, t_end = duration_rest_start, duration_rest_start + 10
        else:
            t_start, t_end = 0, duration_rest_start + duration_stimulus + duration_rest_end
        # Create boolean masks for both time grids
        mask_exp = (t_exp >= t_start) & (t_exp <= t_end)
        mask_sim = (t_sim >= t_start) & (t_sim <= t_end)
        # Ensure masks are PyTorch tensors matching each tensor's device
        if torch.is_tensor(target_signal_L) and not torch.is_tensor(mask_exp):
            mask_exp = torch.as_tensor(mask_exp, device=target_signal_L.device)
        if torch.is_tensor(ys) and not torch.is_tensor(mask_sim):
            mask_sim = torch.as_tensor(mask_sim, device=ys.device)
        # Slice time arrays
        t_exp_window = t_exp[mask_exp]
        t_sim_window = t_sim[mask_sim]
        # Slice signal arrays along the time axis
        # Works whether arrays are NumPy arrays or PyTorch tensors
        output_signal_window = target_signal_L[mask_exp, :]
        y_pred_window = ys.detach().numpy()[0, mask_sim, :]
        s1, s2, _ = DSService.interpolate_to_common_grid(output_signal_window[:, i_pop], t_exp_window,
                                                         y_pred_window[:, i_pop], t_sim_window,
                                                         num_points=None)
        rmse = np.sqrt(np.mean((s1 - s2)**2))
        rmse_pop_list[i_pop].append(rmse)
        corr = np.abs(DSService.pearson_correlation(output_signal_window[:, i_pop], t_exp_window,
                                                    y_pred_window[:, i_pop], t_sim_window))[0]
        corr_pop_list[i_pop].append(corr)

    # ----------------------------------------------------------------
    # Analyze correlation of single-neuron activity with target
    # ----------------------------------------------------------------
    for i_trial in range(xs.shape[0]):
        # Extract trial information
        x_trial = xs[i_trial].detach().cpu().numpy()
        for i_pop, i_neurons_in_pop in enumerate(population_indices):

            # Find anchor neuron traces (units constrained directly by target trajectories)
            anchor_indices = np.array(anchor_by_pop[i_pop], dtype=int)

            # Extract temporal traces for anchor and free subpopulations
            anchor_traces = x_trial[:, anchor_indices]  # (T, n_anchor)

            target_trace = target_signal_L[np.newaxis, ...]

            threshold_range_pop_ = np.quantile((np.max(anchor_traces[mask_sim], axis=0) - np.min(anchor_traces[mask_sim], axis=0)), [0.75])[0]
            for i_c in range(anchor_traces.shape[1]):
                # if np.max(anchor_traces[mask_sim, i_c]) - np.min(anchor_traces[mask_sim, i_c]) < threshold_range_pop[i_pop]:
                if np.max(anchor_traces[mask_sim, i_c]) - np.min(anchor_traces[mask_sim, i_c]) < threshold_range_pop_:
                        continue
                s1, s2, _ = DSService.interpolate_to_common_grid(target_signal_L[mask_exp, i_pop], t_exp_window,
                                                           anchor_traces[mask_sim, i_c], t_sim_window, num_points=None)
                rmse = np.sqrt(np.mean((s1 - s2)**2))
                rmse_pop_neuron_list[i_pop].append(rmse)

                corr = np.abs(DSService.pearson_correlation(target_signal_L[mask_exp, i_pop], t_exp_window, 
                                                            anchor_traces[mask_sim, i_c], t_sim_window))[0]
                corr_pop_neuron_list[i_pop].append(corr)
                corr_pop_neuron_list_[int(i_pop%4)].append(corr)

# Remove nans
for i_corr in range(len(corr_pop_list)):
    corr_pop_list[i_corr] = np.array(corr_pop_list[i_corr])[~np.isnan(corr_pop_list[i_corr])]
for i_corr in range(len(corr_pop_neuron_list)):
    corr_pop_neuron_list[i_corr] = np.array(corr_pop_neuron_list[i_corr])[~np.isnan(corr_pop_neuron_list[i_corr])]

for i_rmse in range(len(rmse_pop_list)):
    rmse_pop_list[i_rmse] = np.array(rmse_pop_list[i_rmse])[~np.isnan(rmse_pop_list[i_rmse])]
for i_rmse in range(len(rmse_pop_neuron_list)):
    rmse_pop_neuron_list[i_rmse] = np.array(rmse_pop_neuron_list[i_rmse])[~np.isnan(rmse_pop_neuron_list[i_rmse])]


if show_corr:
    # Set y-axis upper boundary for corr display
    max_value = 1
    # Create categorical plot panel for corr distributions
    plot_corr = fig.create_plot(plot_title="Correlation of readout activity with target across populations",
                                     xpos=xpos, ypos=ypos, plot_height=plot_height,
                                     plot_width=plot_width,
                                     xmin=0.5, xmax=ConfigurationNeural.N_POPS+0.5, xticks=np.arange(ConfigurationNeural.N_POPS)+1,
                                     xticklabels=pop_list,
                                     xticklabels_rotation=45,
                                     ymin=0, ymax=max_value, yticks=[0, max_value/2, max_value])

    # Render violin distribution plots for each cohort
    plot_corr.draw_violin(corr_pop_list, facecolor="k")

    # Advance horizontal coordinate for subsequent panels
    xpos += plot_width + padding*2

if show_rmse:
    # Set y-axis upper boundary for rmse display
    max_value = 1
    # Create categorical plot panel for rmse distributions
    plot_rmse = fig.create_plot(plot_title="RMSE of readout activity with target across populations",
                                     xpos=xpos, ypos=ypos, plot_height=plot_height,
                                     plot_width=plot_width,
                                     xmin=0.5, xmax=ConfigurationNeural.N_POPS+0.5, xticks=np.arange(ConfigurationNeural.N_POPS)+1,
                                     xticklabels=pop_list,
                                     xticklabels_rotation=45,
                                     ymin=0, ymax=max_value, yticks=[0, max_value/2, max_value])

    # Render violin distribution plots for each cohort
    plot_rmse.draw_violin(rmse_pop_list, facecolor="k")

    # Advance horizontal coordinate for subsequent panels
    xpos = xpos_start
    ypos -= plot_height + padding*5


if show_neuron_corr:
    # Set y-axis upper boundary for corr display
    max_value = 1
    # Create categorical plot panel for corr distributions
    plot_neuron_corr = fig.create_plot(plot_title="Correlation of cell activity with target across populations",
                                     xpos=xpos, ypos=ypos, plot_height=plot_height,
                                     plot_width=plot_width,
                                     xmin=0.5, xmax=ConfigurationNeural.N_POPS+0.5, xticks=np.arange(ConfigurationNeural.N_POPS)+1,
                                     xticklabels=pop_list,
                                     xticklabels_rotation=45,
                                     ymin=0, ymax=max_value, yticks=[0, max_value/2, max_value])

    # Render violin distribution plots for each cohort
    plot_neuron_corr.draw_violin(corr_pop_neuron_list, facecolor="k")
    for i_corr, corr_neuron in enumerate(corr_pop_neuron_list):
        median_corr = np.median(corr_neuron)
        plot_neuron_corr.draw_line([i_corr+0.75, i_corr+1.25], np.ones(2)*median_corr, lc="k")

    # Advance horizontal coordinate for subsequent panels
    xpos += plot_width + padding*2

if show_neuron_rmse:
    # Set y-axis upper boundary for rmse display
    max_value = 5
    # Create categorical plot panel for rmse distributions
    plot_neuron_rmse = fig.create_plot(plot_title="RMSE of cell activity with target across populations",
                                     xpos=xpos, ypos=ypos, plot_height=plot_height,
                                     plot_width=plot_width,
                                     xmin=0.5, xmax=ConfigurationNeural.N_POPS+0.5, xticks=np.arange(ConfigurationNeural.N_POPS)+1,
                                     xticklabels=pop_list,
                                     xticklabels_rotation=45,
                                     ymin=0, ymax=max_value, yticks=[0, max_value/2, max_value])

    # Render violin distribution plots for each cohort
    plot_neuron_rmse.draw_violin(rmse_pop_neuron_list, facecolor="k")
    for i_rmse, rmse_neuron in enumerate(rmse_pop_neuron_list):
        plot_neuron_rmse.draw_line([i_rmse+0.75, i_rmse+1.25], np.ones(2)*np.median(rmse_neuron), lc="k")

    # Advance horizontal coordinate for subsequent panels
    xpos = xpos_start
    ypos -= plot_height + padding*5


if show_neuron_corr_noside:
    # Set y-axis upper boundary for corr display
    max_value = 1
    plot_width_here = plot_width / 2
    # Create categorical plot panel for corr distributions
    plot_neuron_corr = fig.create_plot(plot_title="Correlation of cell activity with target across populations",
                                     xpos=xpos, ypos=ypos, plot_height=plot_height,
                                     plot_width=plot_width_here,
                                     xmin=0.5, xmax=len(ConfigurationRNN.cell_list)+0.5, xticks=np.arange(len(ConfigurationRNN.cell_list))+1,
                                     xticklabels=ConfigurationRNN.cell_list,
                                     xticklabels_rotation=45,
                                     ymin=0, ymax=max_value, yticks=[0, max_value/2, max_value])

    # Render violin distribution plots for each cohort
    plot_neuron_corr.draw_violin(corr_pop_neuron_list_, facecolor="k")
    for i_corr, corr_neuron in enumerate(corr_pop_neuron_list_):
        median_corr = np.median(corr_neuron)
        quantiles_corr = np.quantile(corr_neuron, q=[0.25, 0.75])
        # print(f"Correlation for population {ConfigurationRNN.cell_list[i_corr]}: {np.mean(corr_neuron)} \pm {np.std(corr_neuron)}")
        print(f"Median correlation for population {ConfigurationRNN.cell_list[i_corr]}: {median_corr}")
        print(f"Interquartile range correlation for population {ConfigurationRNN.cell_list[i_corr]}: {quantiles_corr[1]-quantiles_corr[0]}")
        plot_neuron_corr.draw_line([i_corr+0.75, i_corr+1.25], np.ones(2)*np.mean(corr_neuron), lc="k")

    # Advance horizontal coordinate for subsequent panels
    xpos += plot_width_here + padding*2


# ----------------------------------------------------------------
# Save final figure
# ----------------------------------------------------------------
# Ensure output directory exists and export multi-panel vector plot to PDF
path_save.mkdir(parents=True, exist_ok=True)
fig.save(path_save / f"figure_supp_neurons_popcorrdist_{label_dir}{label_save}_top6.pdf", open_file=False, tight=style.page_tight)
