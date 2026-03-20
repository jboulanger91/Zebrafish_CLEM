import pickle

import torch
import numpy as np
from pathlib import Path

from plot.style import RNNDSStyle
from utils.figure_helper import Figure

# ------------------------------------------------
# Configuration
# ------------------------------------------------
path_dir = Path(r"C:\Users\Roberto\Desktop\highlights\clem_rnns\data")   # directory containing model_X.pkl
path_noise_estimation = path_dir / "noise_estimation" / "contralateral_motion_integrator_preferred_noise_estimation.pkl"
path_models = path_dir / "results" / "freepop" / "mask_traces_freepop_16" / "RNNFreePop_neurons102_tau0.1_input2step_softplus"  # RNNConstrainedMask_neurons86_tau0.2_input2step_elu"
path_model_top = path_models / "top_5"
path_traces = path_dir / "single_traces" / "jon_experiments"
path_save = path_models / "results"
TOP_FRAC  = 0.05
TOP_N = 5  # 10  # 0.05
device = "cpu"
free_neurons = 16
model_is_free_pop = True
save_top_models = True


n_units_A = 15
n_units_B = 15
n_units_C = 2
n_units_D = 11
n_units_X = free_neurons
n_units_hemi = n_units_A + n_units_B + n_units_C + n_units_D
n_units = n_units_hemi * 2 + free_neurons
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
input_signal = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)), np.zeros(int(duration_rest_end / dt))))
t_sim = np.linspace(0, duration_simulation, len(input_signal))
input_signal_neurons_L = torch.tensor(np.concatenate((np.repeat(input_signal[..., np.newaxis], n_units_hemi, axis=1),
                                             np.zeros((len(input_signal), n_units_hemi))), axis=1), dtype=torch.float32)

# ================================================================
# Load traces to use as target signals
# ================================================================
cell_types_list = ["motion_integrator", "motion_onset", "slow_motion_integrator"]
traces_dict = {ct: {} for ct in cell_types_list}
all_signals = []
min_traces_all = 0
for ct in cell_types_list:
    filename = f"{ct}_hindbrain_preferred_raw_individual_traces.csv"
    data = np.loadtxt(path_traces / filename, dtype=float, delimiter=",", skiprows=1)
    downsample_time_list = data[:, 0]
    traces_dict[ct]["signal"] = data[:, 1:] / 100
    min_trace_here = np.min(data[:, 1:] / 100)
    if min_trace_here < min_traces_all:
        min_traces_all = min_trace_here
target_signal_L = np.stack((traces_dict["motion_integrator"]["preferred"],
                                traces_dict["motion_integrator"]["preferred"],
                                traces_dict["motion_onset"]["preferred"],
                                traces_dict["slow_motion_integrator"]["preferred"],
                                traces_dict["motion_integrator"]["null"],
                                traces_dict["motion_integrator"]["null"],
                                traces_dict["motion_onset"]["null"],
                                traces_dict["slow_motion_integrator"]["null"]
                                ),
                               axis=-1)
initial_value_signal_L = np.concatenate((np.array([traces_dict[ct]["signal"][0, 0] for _ in range(n_units_A)]) + np.random.normal(0, np.abs(traces_dict[ct]["signal"][0, 0])/5, n_units_A),
                                      np.array([traces_dict[ct]["signal"][0, 1] for _ in range(n_units_B)]) + np.random.normal(0, np.abs(traces_dict[ct]["signal"][0, 1])/5, n_units_B),
                                      np.array([traces_dict[ct]["signal"][0, 2] for _ in range(n_units_C)]) + np.random.normal(0, np.abs(traces_dict[ct]["signal"][0, 2])/5, n_units_C),
                                      np.array([traces_dict[ct]["signal"][0, 3] for _ in range(n_units_D)]) + np.random.normal(0, np.abs(traces_dict[ct]["signal"][0, 3])/5, n_units_D),
                                      np.array([traces_dict[ct]["signal"][0, 4] for _ in range(n_units_A)]) + np.random.normal(0, np.abs(traces_dict[ct]["signal"][0, 4])/5, n_units_A),
                                      np.array([traces_dict[ct]["signal"][0, 5] for _ in range(n_units_B)]) + np.random.normal(0, np.abs(traces_dict[ct]["signal"][0, 5])/5, n_units_B),
                                      np.array([traces_dict[ct]["signal"][0, 6] for _ in range(n_units_C)]) + np.random.normal(0, np.abs(traces_dict[ct]["signal"][0, 6])/5, n_units_C),
                                      np.array([traces_dict[ct]["signal"][0, 7] for _ in range(n_units_D)]) + np.random.normal(0, np.abs(traces_dict[ct]["signal"][0, 7])/5, n_units_D)))
if model_is_free_pop:
    input_signal_neurons_L = np.concatenate((input_signal_neurons_L, np.zeros((len(input_signal), n_units_X))), axis=1)
    initial_value_signal_L = np.concatenate((initial_value_signal_L, np.array([np.mean(target_signal_L[0]) for _ in range(n_units_X)]) + np.random.normal(0, np.abs(
        np.mean(target_signal_L[0])) / 5, n_units_X)))
min_traces_all = np.abs(min_traces_all)
t_exp = np.linspace(0, duration_simulation, len(traces_dict[ct]["signal"]))

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

import numpy as np

def nrmse(a, b, eps=1e-8):
    a = np.asarray(a); b = np.asarray(b)
    rmse = np.sqrt(np.mean((a - b)**2))
    return rmse / (np.std(a) + eps)

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
      - corr_target_model, nrmse_target_model
      - shuffle distributions (mean) and p-values vs shuffle control
    """
    V_target = np.asarray(V_target)
    V_model = np.asarray(V_model)
    assert V_target.shape == V_model.shape

    rng = np.random.default_rng(seed)

    corr_tm = pearsonr(V_target, V_model)
    nrmse_tm = nrmse(V_target, V_model)

    corr_sh = np.empty(n_shuffles, dtype=float)
    nrmse_sh = np.empty(n_shuffles, dtype=float)

    for k in range(n_shuffles):
        s = time_shuffle_control(V_target, rng)
        corr_sh[k] = pearsonr(V_target, s)
        nrmse_sh[k] = nrmse(V_target, s)

    # p-values: how often shuffle does as well/better than model
    p_corr = float(np.mean(np.abs(corr_sh) >= np.abs(corr_tm)))     # larger corr is better
    p_nrmse = float(np.mean(np.abs(nrmse_sh) <= np.abs(nrmse_tm)))  # smaller nrmse is better

    results = {
        "corr_target_model": corr_tm,
        "nrmse_target_model": nrmse_tm,
        "corr_shuffle_mean": float(corr_sh.mean()),
        "nrmse_shuffle_mean": float(nrmse_sh.mean()),
        "pvalue_corr": p_corr,
        "pvalue_nrmse": p_nrmse,
    }

    return results

metric = "std"
for i_model, path_model in enumerate(path_model_top.glob("model_*.pkl")):
    with open(path_model, 'rb') as f:
        model = pickle.load(f)

    print(f"Evaluating model {i_model}")

    model_xs, _ = model(initial_value_signal_L, input_signal_neurons_L)
    model_xs = model_xs.squeeze()

    for i_ct, ct in enumerate(cell_types_list):
        target_xs = torch.tensor(traces_dict[ct]["signal"], dtype=torch.float32)
        V_t = pop_variability_vs_time(target_xs, metric=metric, reduction_trials=None)

        if i_ct == 0:
            index_pop = np.concatenate((model.anchor_indices_by_pop[0], model.anchor_indices_by_pop[1]))
        elif i_ct == 1:
            index_pop = model.anchor_indices_by_pop[2]
        elif i_ct == 2:
            index_pop = model.anchor_indices_by_pop[3]
        else:
            raise ValueError(f"Unknown i_ct {i_ct}")
        model_xs_pop = model_xs[:, index_pop]

        V_m = pop_variability_vs_time(model_xs_pop, metric=metric, reduction_trials=None)

        ylabel = f"Model {i_model+1}\nActivity" if i_ct == 0 else "Activity"
        plot_m_pop_xs = fig.create_plot(plot_title=f"Target\n{ct}" if i_model == 0 else None,
                                     xpos=xpos, ypos=ypos, plot_width=plot_width, plot_height=plot_height,
                                     xl="Time (s)" if i_model == 4 else None, yl=ylabel,
                                     xmin=0, xmax=duration_simulation, ymin=0, ymax=3,
                                     xticks=[0, duration_rest_start, duration_rest_start + duration_stimulus,
                                             duration_simulation],
                                     yticks=[0, 1.5, 3])
        xpos += plot_width + padding / 2
        for i_neuron in range(target_xs.shape[1]):
            plot_m_pop_xs.draw_line(t_exp, target_xs[:, i_neuron], lc=palette[i_ct], lw=0.1)
        plot_m_pop_xs = fig.create_plot(plot_title=f"Model\n{ct}" if i_model == 0 else None,
                                        xpos=xpos, ypos=ypos, plot_width=plot_width, plot_height=plot_height,
                                        xl="Time (s)" if i_model == 4 else None,
                                        xmin=0, xmax=duration_simulation, ymin=0, ymax=3,
                                        xticks=[0, duration_rest_start, duration_rest_start + duration_stimulus,
                                                duration_simulation],
                                        yticks=None)
        xpos += plot_width + padding
        for i_neuron in range(model_xs_pop.shape[1]):
            plot_m_pop_xs.draw_line(t_sim, model_xs_pop[ :, i_neuron].detach().numpy(), lc=palette[i_ct], lw=0.1)

        plot_m_pop_var = fig.create_plot(plot_title=ct if i_model == 0 else None,
                                     xpos=xpos, ypos=ypos, plot_width=plot_width, plot_height=plot_height,
                                     xl="Time (s)" if i_model == 4 else None, yl="Activity variability",
                                     xmin=0, xmax=duration_simulation, ymin=0, ymax=3,
                                     xticks=[0, duration_rest_start, duration_rest_start + duration_stimulus, duration_simulation],
                                     yticks=[0, 1.5, 3])
        xpos += plot_width + padding * 2
        plot_m_pop_var.draw_line(t_exp, V_t.squeeze(), lc=palette[i_ct], label="Target" if i_ct==len(cell_types_list)-1 else None)
        plot_m_pop_var.draw_line(t_sim, V_m.squeeze(), lc=palette[i_ct], line_dashes=(1, 2), label="Model" if i_ct==len(cell_types_list)-1 else None)

        results_p = compare_to_time_shuffle_controls(V_t, downsample_signal(V_m, t_exp))
        print(f"Results for {ct}")
        print(results_p)

    xpos = xpos_start
    ypos -= plot_height + padding

# # -----------------------------------------------------------------------------
# # Save final figure
# # -----------------------------------------------------------------------------
# fig.save(path_save / f"check_neurons_variance_{metric}.pdf", open_file=False, tight=style.page_tight)
