# compare_population_variance.py

import numpy as np
import torch

from scipy.stats import wasserstein_distance

try:
      # optional, nicer scalar distance
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def _as_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)

@torch.no_grad()
def run_model_get_xs(model, train_list, x0=None, use_raw_if_available=True):
    """
    Returns:
        xs: (N, T, n_units) torch.Tensor
    """
    device = next(model.parameters()).device

    inputs = torch.stack([_as_tensor(t.input_signal, device) for t in train_list])  # (N,T,input_dim) or (N,T)
    if inputs.ndim == 2:
        inputs = inputs[..., None]  # (N,T,1)

    N = inputs.shape[0]

    if x0 is None:
        x0_list = []
        for t in train_list:
            iv = getattr(t, "initial_value", None)
            if iv is None:
                x0_list.append(torch.zeros(model.n_units, device=device))
            else:
                x0_list.append(_as_tensor(iv, device).view(-1))
        x0 = torch.stack(x0_list, dim=0)  # (N, n_units)
    else:
        x0 = _as_tensor(x0, device)
        if x0.ndim == 1:
            x0 = x0[None, :].repeat(N, 1)

    # forward
    model.eval()
    _ = model.forward(x0, inputs)

    # choose xs
    if use_raw_if_available and hasattr(model, "xs_raw") and model.xs_raw is not None:
        xs = model.xs_raw
    else:
        xs = model.xs

    return xs

def per_neuron_variance(xs, mode="time", correction=0):
    """
    xs: (N,T,U)
    mode:
      - "time": for each neuron/trial, var over time -> (N,U), then average over trials -> (U,)
      - "time_and_trials": flatten N and T and var -> (U,)
      - "trials": var across trials of time-mean -> (U,)
    correction: torch.var correction (0 => population variance)
    """
    if mode == "time":
        # var over time within each trial -> (N,U), then mean over trials -> (U,)
        v = xs.var(dim=1, correction=correction)      # (N,U)
        return v.mean(dim=0)                          # (U,)
    elif mode == "time_and_trials":
        # var over flattened NT -> (U,)
        x = xs.reshape(-1, xs.shape[-1])              # (NT,U)
        return x.var(dim=0, correction=correction)    # (U,)
    elif mode == "trials":
        # variance across trials of trial-averaged activity
        m = xs.mean(dim=1)                            # (N,U)
        return m.var(dim=0, correction=correction)    # (U,)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _summarize_distribution(x):
    """
    x: 1D numpy array
    """
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)) if x.size else np.nan,
        "median": float(np.median(x)) if x.size else np.nan,
        "p10": float(np.percentile(x, 10)) if x.size else np.nan,
        "p90": float(np.percentile(x, 90)) if x.size else np.nan,
    }

def _compare_variance_by_population(
    model_var, target_var, population_indices, pop_names=None
):
    """
    model_var, target_var: (U,) numpy arrays (per-neuron variance)
    population_indices: list[list[int]] length n_pops
    Returns list[dict] per population
    """
    results = []
    n_pops = len(population_indices)
    if pop_names is None:
        pop_names = [f"pop_{i}" for i in range(n_pops)]

    for i, idx in enumerate(population_indices):
        idx = np.array(idx, dtype=int)
        mv = model_var[idx]
        tv = target_var[idx]

        out = {
            "pop": pop_names[i],
            **{f"model_{k}": v for k, v in _summarize_distribution(mv).items()},
            **{f"target_{k}": v for k, v in _summarize_distribution(tv).items()},
            "mean_ratio_model_over_target": float((np.mean(mv) + 1e-12) / (np.mean(tv) + 1e-12)),
        }

        # Wasserstein distance between distributions
        out["wasserstein_1d"] = float(wasserstein_distance(mv, tv))

        results.append(out)

    return results

def check_neurons_variance(
    model,
    train_list,
    target_xs,                       # (N,T,U) or (T,U) or (N,T,U)
    x0=None,
    variance_mode="time",
    correction=0,
    use_raw_if_available=True,
    pop_names=None,
    print_table=True,
):
    """
    target_xs should be the dataset neuron-level activity you want to compare against.
    If you only have population averages in the dataset, you can't compare per-neuron variance
    unless you have neuron-level traces or you define a proxy.
    """
    device = next(model.parameters()).device

    # Run model
    xs_model = run_model_get_xs(model, train_list, x0=x0, use_raw_if_available=use_raw_if_available)  # (N,T,U)

    # Load target xs
    xs_target = _as_tensor(target_xs, device)
    if xs_target.ndim == 2:
        xs_target = xs_target[None, :, :].repeat(xs_model.shape[0], 1, 1)  # broadcast to (N,T,U)
    if xs_target.shape != xs_model.shape:
        raise ValueError(f"Shape mismatch: model xs {tuple(xs_model.shape)} vs target xs {tuple(xs_target.shape)}")

    # Per-neuron variance vectors
    v_model = per_neuron_variance(xs_model, mode=variance_mode, correction=correction).detach().cpu().numpy()
    v_target = per_neuron_variance(xs_target, mode=variance_mode, correction=correction).detach().cpu().numpy()

    # Compare by population
    results = _compare_variance_by_population(
        v_model, v_target, model.population_indices, pop_names=pop_names
    )

    if print_table:
        # simple print
        cols = [
            "pop",
            "model_mean", "target_mean", "mean_ratio_model_over_target",
            "model_median", "target_median",
            "model_p10", "model_p90", "target_p10", "target_p90",
            "wasserstein_1d",
        ]
        print("\t".join(cols))
        for r in results:
            print("\t".join(str(r.get(c, "")) for c in cols))

    return results, v_model, v_target

