"""
Overview
--------
This script generates the following figure panels in Boulanger-Weiss et al. 2026:
- Supp. Fig. 10...

This script analyzes and compares across-neuron population variability over time
between empirical calcium imaging recordings and trained recurrent neural network (RNN)
model simulations across distinct cell types (MI, MON, SMI). It quantifies the temporal
correlation of population variability curves (e.g., standard deviation across units vs. time)
and evaluates statistical significance against a time-shuffled null control distribution.

Before running it, please make sure you have set up your .env with all the necessary variables:
# No environment variables are directly read by this script; filesystem paths are configured
# via the hardcoded `path_dir` variable at the top of the configuration section.

Core Pipeline & Workflow:
1. Environment & Architecture Configuration:
   - Sets root directory paths for model checkpoints (`top_5/`), test datasets (`cyto8s/`),
     noise estimates, and results export destinations.
   - Loads a reference model instance to extract structural unit counts, integration time step
     (`dt`), subpopulation indices, and simulation phase durations.
   - Configures canvas layouts, padding, and categorical color palettes via `RNNDSStyle` and `Figure`.

2. Target Data Loading & Signal Preparation:
   - Formulates unilateral step input driving current arrays.
   - Loads empirical multi-neuron response traces for MI, MON, and SMI populations under constant
     leftward stimulation, scales delta F/F, and calculates non-negative baseline offsets.
   - Synthesizes initial population firing rate vectors perturbed with Gaussian noise to initialize
     continuous network states.

3. Simulation & Temporal Variability Computation:
   - Iterates through top model checkpoints (`path_models_top`), running forward simulations to
     generate hidden unit trajectory tensors (`model_xs`).
   - Computes across-neuron population variability over time (`pop_variability_vs_time`) using
     the specified metric (standard deviation, variance, or coefficient of variation) for both
     empirical targets (`V_t`) and simulated model units (`V_m`).
   - Downsamples simulated variability curves to match experimental temporal sampling grids.

4. Statistical Shuffling & Correlation Analysis:
   - Evaluates Pearson correlation coefficients between empirical and model variability profiles.
   - Performs Monte Carlo time-shuffle permutation testing (`compare_to_time_shuffle_controls`,
     n=10,000) to construct empirical null correlation distributions and compute p-values.

5. Distribution Visualization & Export:
   - Assembles correlation distributions comparing intact model-target alignment against
     time-shuffled controls.
   - Renders summary violin plots and exports the resulting figure as a vector PDF.
"""

import torch
import numpy as np
from pathlib import Path

# Manually add root path for imports to improve interoperability across nested research subdirectories
import sys; sys.path.insert(0, "..")

from figures.style import RNNDSStyle
from utils.load_model import load_model
from utils.figure_helper import Figure

# ------------------------------------------------
# Configuration
# ------------------------------------------------
path_dir = Path(r"C:\Users\Roberto\Desktop\highlights\clem_rnns\package")
path_noise_estimation = path_dir / "data" / "noise_estimation" / "contralateral_motion_integrator_preferred_noise_estimation.pkl"
path_models = path_dir / "models" / "connectome"   # directory containing model_X.pkl
path_models_top = path_models / "top_5"
path_model_example = path_models_top / "model_top0_822826_2026-10-04.pt"
path_traces = path_dir / "data" / "test_dataset" / "cyto8s"
path_save = path_dir / "results"


# Extract core architecture parameters, time resolution, and population unit sizes from the last active model
model = load_model(path_model_example)
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
n_input_signal = 2
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

padding = style.padding * 2 / 3
padding_big = style.padding * 2
padding_vertical = style.padding

palette = style.palette["neurons_3"]

# ================================================================
# Initialize figure container
# ================================================================
fig = Figure()

# ------------------------------------------------
# Define input signals used in training
# ------------------------------------------------
input_signal = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)), np.zeros(int(duration_rest_end / dt))))
t_sim = np.linspace(0, duration_simulation, len(input_signal))
input_signal_neurons_L = np.concatenate((np.repeat(input_signal[..., np.newaxis], n_units_hemi, axis=1),
                                           np.zeros((len(input_signal), n_units-n_units_hemi))), axis=1)

# ================================================================
# Load traces to use as target signals
# ================================================================
cell_types_list = ["MI", "MON", "SMI"]
traces_dict = {ct: None for ct in cell_types_list}
all_signals = []
min_traces_all = 0
for ct in cell_types_list:
    filename = f"responses_{ct}_constant_left.csv"
    data = np.loadtxt(path_traces / filename, dtype=float, delimiter=",", skiprows=1)
    downsample_time_list = data[:, 0]
    traces_dict[ct] = data[:, 1:] / 100
    min_trace_here = np.min(data[:, 1:] / 100)
    if min_trace_here < min_traces_all:
        min_traces_all = min_trace_here
target_signal_L = np.stack((np.mean(traces_dict["MI"], axis=1),
                            np.mean(traces_dict["MI"], axis=1),
                            np.mean(traces_dict["MON"], axis=1),
                            np.mean(traces_dict["SMI"], axis=1),
                            # repeat on the other side. Quick and dirty solution, acceptable only because
                            # this trace is only useful to determine the pop-specific initial values for the
                            # simulations, and preferred and null side may share the same initial value.
                            np.mean(traces_dict["MI"], axis=1),
                            np.mean(traces_dict["MI"], axis=1),
                            np.mean(traces_dict["MON"], axis=1),
                            np.mean(traces_dict["SMI"], axis=1),
                            ),
                           axis=-1)
target_signal_L -= min_traces_all
initial_value_signal_L = np.concatenate((np.array([target_signal_L[0, 0] for _ in range(n_units_LiMI)]) + np.random.normal(0, np.abs(target_signal_L[0, 0])/5, n_units_LiMI),
                                         np.array([target_signal_L[0, 1] for _ in range(n_units_LcMI)]) + np.random.normal(0, np.abs(target_signal_L[0, 1])/5, n_units_LcMI),
                                         np.array([target_signal_L[0, 2] for _ in range(n_units_LMON)]) + np.random.normal(0, np.abs(target_signal_L[0, 2])/5, n_units_LMON),
                                         np.array([target_signal_L[0, 3] for _ in range(n_units_LsMI)]) + np.random.normal(0, np.abs(target_signal_L[0, 3])/5, n_units_LsMI),
                                         np.array([target_signal_L[0, 4] for _ in range(n_units_RiMI)]) + np.random.normal(0, np.abs(target_signal_L[0, 4])/5, n_units_RiMI),
                                         np.array([target_signal_L[0, 5] for _ in range(n_units_RcMI)]) + np.random.normal(0, np.abs(target_signal_L[0, 5])/5, n_units_RcMI),
                                         np.array([target_signal_L[0, 6] for _ in range(n_units_RMON)]) + np.random.normal(0, np.abs(target_signal_L[0, 6])/5, n_units_RMON),
                                         np.array([target_signal_L[0, 7] for _ in range(n_units_RsMI)]) + np.random.normal(0, np.abs(target_signal_L[0, 7])/5, n_units_RsMI)))

# min_traces_all = np.abs(min_traces_all)
t_exp = np.linspace(0, duration_simulation, len(traces_dict["MI"]))

def pop_variability_vs_time(
    X,                     # (T, U) or (N, T, U)
    metric="std",          # "std" or "var" or "cv"
    reduction_trials="mean",  # how to collapse N if X is (N,T,U): "mean" or "median" or None
    eps=1e-8,
):
    """
    Returns:
      V: (n_pops, T) numpy array with variability across neurons at each time.
         If reduction_trials=None and X is (N,T,U), returns (N, n_pops, T).
    """
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    if X.ndim == 2:
        X = X.unsqueeze(0)  # -> (1,T,U)
    assert X.ndim == 3, f"Expected (N,T,U) or (T,U), got {tuple(X.shape)}"

    N, T, U = X.shape

    out = torch.zeros(N, T, dtype=X.dtype, device=X.device)

    if metric == "std":
        out[:, :] = X.std(dim=2, correction=0)
    elif metric == "var":
        out[:, :] = X.var(dim=2, correction=0)
    elif metric == "cv":
        mu = X.mean(dim=2)
        sd = X.std(dim=2, correction=0)
        out[:, :] = sd / (mu.abs() + eps)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if reduction_trials is None:
        return out.detach().cpu().numpy()  # (N, n_pops, T)

    if reduction_trials == "mean":
        V = out.mean(dim=0)  # (n_pops, T)
    elif reduction_trials == "median":
        V = out.median(dim=0).values
    else:
        raise ValueError("reduction_trials must be 'mean', 'median', or None")

    return V.detach().cpu().numpy()

def pearsonr(a, b, eps=1e-12):
    a = np.asarray(a); b = np.asarray(b)
    a0 = a - a.mean(); b0 = b - b.mean()
    return float((a0 @ b0.T) / (np.linalg.norm(a0) * np.linalg.norm(b0) + eps))

def downsample_signal(raw_signal, time_sample_list, dt=dt):
    # raw_signal: (N,T,dim), time_sample_list: (T_ds,)
    if not torch.is_tensor(time_sample_list):
        time_sample_list = torch.tensor(time_sample_list, dtype=torch.float32, device=raw_signal.device)
    t_raw = torch.arange(0, raw_signal.shape[1], device=raw_signal.device, dtype=torch.float32) * dt
    dt = torch.abs(t_raw[:, None] - time_sample_list[None, :])  # (T, T_ds)
    idx_sample = torch.argmin(dt, dim=0)                         # (T_ds,)
    return raw_signal[:, idx_sample]

def time_shuffle_control(x, rng):
    """
    Returns a time-shuffled copy of x (1D array), destroying temporal structure
    but preserving the set of values. Uses numpy.random.permutation. [web:241]
    """
    x = np.asarray(x).squeeze()
    return x[rng.permutation(len(x))]

def compare_to_time_shuffle_controls(V_target, V_model, n_shuffles=10000, seed=0):
    """
    V_target, V_model: (n_pops, T) variability curves (e.g., across-neuron std vs time)

    Returns: list of dicts per population with:
      - corr_target_model
      - shuffle distributions (mean) and p-values vs shuffle control
    """
    V_target = np.asarray(V_target)
    V_model = np.asarray(V_model)
    assert V_target.shape == V_model.shape

    corr_tm = pearsonr(V_target, V_model)

    rng = np.random.default_rng(seed)
    corr_sh = np.empty(n_shuffles, dtype=float)
    for k in range(n_shuffles):
        s = time_shuffle_control(V_target, rng)
        corr_sh[k] = pearsonr(V_target, s)

    # p-values: how often shuffle does as well/better than model
    p_corr = float(np.mean(np.abs(corr_sh) >= np.abs(corr_tm)))     # larger corr is better

    results = {
        "corr_target_model": corr_tm,
        "corr_shuffle_mean": float(corr_sh.mean()),
        "pvalue_corr": p_corr,
    }

    return results

metric = "std"  # "cv"  # "var"  #
corr_original_list = []
corr_shuffle_list = []
input_signal_neurons_L = torch.tensor(input_signal_neurons_L, dtype=torch.float32)
for i_model, path_model in enumerate(path_models_top.glob("model_*")):
    model = load_model(path_model)

    print(f"Evaluating model {i_model}")

    model_xs, _ = model(initial_value_signal_L, input_signal_neurons_L)
    model_xs = model_xs.squeeze()

    for i_ct, ct in enumerate(cell_types_list):
        target_xs = torch.tensor(traces_dict[ct] - min_traces_all, dtype=torch.float32)
        V_t = pop_variability_vs_time(target_xs, metric=metric, reduction_trials=None)

        if i_ct == 0:
            index_pop = np.concatenate((model.population_indices[0], model.population_indices[1]))
        elif i_ct == 1:
            index_pop = model.population_indices[2]
        elif i_ct == 2:
            index_pop = model.population_indices[3]
        else:
            raise ValueError(f"Unknown i_ct {i_ct}")
        model_xs_pop = model_xs[:, index_pop]

        # # Enable this block to remove flat neurons from the population, useful for DEBUG purposes
        # index_neurons_behaving = torch.squeeze(torch.argwhere(torch.std(model_xs_pop, dim=0) > 5e-2))
        # model_xs_pop = model_xs_pop[:, index_neurons_behaving]

        V_m = pop_variability_vs_time(model_xs_pop, metric=metric, reduction_trials=None)

        # ylabel = f"Model {i_model+1}\nActivity" if i_ct == 0 else "Activity"
        # plot_m_pop_xs = fig.create_plot(plot_title=f"Target\n{ct}" if i_model == 0 else None,
        #                              xpos=xpos, ypos=ypos, plot_width=plot_width, plot_height=plot_height,
        #                              xl="Time (s)" if i_model == 4 else None, yl=ylabel,
        #                              xmin=0, xmax=duration_simulation, ymin=0, ymax=5,
        #                              xticks=[0, duration_rest_start, duration_rest_start + duration_stimulus,
        #                                      duration_simulation],
        #                              yticks=[0, 2.5, 5])
        # xpos += plot_width + padding / 2
        # for i_neuron in range(target_xs.shape[1]):
        #     plot_m_pop_xs.draw_line(t_exp, target_xs[:, i_neuron], lc=palette[i_ct], lw=0.1)
        # plot_m_pop_xs = fig.create_plot(plot_title=f"Model\n{ct}" if i_model == 0 else None,
        #                                 xpos=xpos, ypos=ypos, plot_width=plot_width, plot_height=plot_height,
        #                                 xl="Time (s)" if i_model == 4 else None,
        #                                 xmin=0, xmax=duration_simulation, ymin=0, ymax=5,
        #                                 xticks=[0, duration_rest_start, duration_rest_start + duration_stimulus,
        #                                         duration_simulation],
        #                                 yticks=None)
        # xpos += plot_width + padding
        # for i_neuron in range(model_xs_pop.shape[1]):
        #     plot_m_pop_xs.draw_line(t_sim, model_xs_pop[ :, i_neuron].detach().numpy(), lc=palette[i_ct], lw=0.1)

        # plot_m_pop_var = fig.create_plot(plot_title=ct if i_model == 0 else None,
        #                              xpos=xpos, ypos=ypos, plot_width=plot_width, plot_height=plot_height,
        #                              xl="Time (s)" if i_model == 4 else None, yl="Activity variability",
        #                              xmin=0, xmax=duration_simulation, ymin=0, ymax=5,
        #                              xticks=[0, duration_rest_start, duration_rest_start + duration_stimulus, duration_simulation],
        #                              yticks=[0, 2.5, 5])
        # xpos += plot_width + padding * 2
        # plot_m_pop_var.draw_line(t_exp, V_t.squeeze(), lc=palette[i_ct], label="Target" if i_ct==len(cell_types_list)-1 else None)
        # plot_m_pop_var.draw_line(t_sim, V_m.squeeze(), lc=palette[i_ct], line_dashes=(1, 2), label="Model" if i_ct==len(cell_types_list)-1 else None)

        results_p = compare_to_time_shuffle_controls(V_t.squeeze(), downsample_signal(V_m, t_exp).squeeze())
        corr_original_list.append(results_p["corr_target_model"])
        corr_shuffle_list.append(results_p["corr_shuffle_mean"])
        # print(f"Results for {ct}")
        # print(results_p)

    xpos = xpos_start
    ypos -= plot_height + padding

corr_original_array = np.array(corr_original_list)
corr_shuffle_array = np.array(corr_shuffle_list)
corr_list = np.stack((corr_original_array, corr_shuffle_array))

xpos = xpos_start
ypos = ypos_start
max_value = 1
# Create categorical plot panel for loss distributions
plot_corr = fig.create_plot(plot_title="Std correlation across conditions",
                            xpos=xpos, ypos=ypos, plot_height=plot_height,
                            plot_width=plot_width,
                            xmin=0.5, xmax=2+0.5, xticks=np.arange(2)+1,
                            xticklabels=["Original", "Shuffle"],
                            xticklabels_rotation=45,
                            ymin=-1, ymax=1, yticks=[-1, 1])

# Render violin distribution plots for each cohort
plot_corr.draw_violin(corr_list, facecolor="k")

# # -----------------------------------------------------------------------------
# # Save final figure
# # -----------------------------------------------------------------------------
path_save.mkdir(parents=True, exist_ok=True)
fig.save(path_save / f"figure_supp_neurons_variance_{metric}_dist.pdf", open_file=False, tight=style.page_tight)
