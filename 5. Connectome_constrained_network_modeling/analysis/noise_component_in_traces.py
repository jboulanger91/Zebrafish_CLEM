"""
Overview
--------
This script estimates stochastic noise characteristics from empirical calcium imaging
traces (specifically a contralateral motion integrator neuron population, cMI) by decomposing
the signal into a deterministic leaky integration component and a stochastic residual process,
subsequently parameterized as an Ornstein-Uhlenbeck (OU) continuous-time Markov process.

Before running it, please make sure you have set up your .env with all the necessary variables:
- PATH_DIR="/path/to/project_root"  # Root project directory containing data/ and noise_estimation/

Core Pipeline & Workflow:
1. Data Ingestion & Window Cropping:
   - Loads empirical average response curves (`avgresponses_cMI_preferred_constant.csv`)
     and converts percentage delta F/F to fractional activity.
   - Restricts analysis to the active stimulus integration window (e.g., 20 s to 60 s)
     and shifts timestamps to start at t = 0.

2. Deterministic Integration Fitting:
   - Fits an exponential saturating function x(t) = 1 - exp(-t / tau_int) via non-linear
     least squares (`curve_fit`) to capture the primary sensory/motor integration timescale.
   - Subtracts the deterministic model from the normalized trace to isolate transient fluctuations.

3. Timescale & Stochastic Residual Extraction:
   - Evaluates continuous-time derivative approximations dx/dt and fits an underlying linear
     drift decay timescale (tau_hat) by minimizing sum-of-squared derivative residuals.
   - Computes empirical noise residuals: eps_hat = dx + x / tau_hat.

4. Ornstein-Uhlenbeck Parameter Fitting:
   - Parameterizes residual dynamics as an AR(1) discrete-time realization of an OU process:
     pred = alpha * r, with alpha = exp(-dt / tau_n) and variance var = sigma^2 * (1 - alpha^2).
   - Minimizes Gaussian negative log-likelihood via `scipy.optimize.minimize` to recover the
     noise correlation time (tau_n) and diffusion magnitude (sigma).

5. Model Validation & Diagnostic Plotting:
   - Compares empirical versus simulated OU residuals across autocorrelation functions (ACF)
     and Welch power spectral density (PSD) estimates.
   - Synthesizes modeled noise realizations and compares reconstructed traces with raw recordings.
   - Optionally serializes estimated OU hyperparameters (tau, sigma, scale) into a `.pkl` file.
"""

import pickle
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from scipy import optimize, signal
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ------------------------------------------------------------
# Env and paths
# ------------------------------------------------------------
# Load root filesystem directory from environment configuration (.env)
env = dotenv_values()
path_dir = Path(env["PATH_DIR"])
# Target empirical trace file for noise estimation
path_trace = path_dir / "data" / "avgresponses_cMI_preferred_constant.csv"
# Output directory to store serialized noise parameter dictionaries
path_save = path_dir / "noise_estimation"

# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------
# Toggle interactive matplotlib diagnostic visualizations
display = True
# Toggle serialization of fitted parameters to disk (.pkl)
save_estimation = False
# Time bounds (seconds) defining the active constant stimulus epoch
time_window_integration = (20, 60)  # (s)

# ------------------------------------------------------------
# 1. Load time series
# ------------------------------------------------------------
# Parse condition metadata and cell identifiers from target trace filename
trace_id = path_trace.name.replace("_activity_traces.csv", "")
side = trace_id.split("_")[-1]
cell_type = trace_id.replace(f"_{side}", "")

# Read raw CSV recording: column 0 is time (s), column 1 is dF/F percentage
data = np.loadtxt(path_trace, dtype=float, delimiter=",", skiprows=1)
t_raw = data[:, 0]
x_raw = data[:, 1] / 100
# Isolate temporal segment corresponding to the stimulus presentation window
idx_in_window = np.argwhere(np.logical_and(t_raw>time_window_integration[0], t_raw<time_window_integration[1]))
t = np.squeeze(t_raw[idx_in_window]) - np.min(t_raw[idx_in_window])  # shift to zero
scale_raw = 1
scale_hat = scale_raw * 4  # double because the associated contribution has to prevail in a signal that already has one such contribution
x_int = np.squeeze(x_raw[idx_in_window]) / scale_raw  # normalize x so that only tau remains
# Fit deterministic exponential saturating integration model: x(t) = 1 - exp(-t / tau)
int_exp = lambda t, tau: 1 - np.exp(-t/tau)
res = curve_fit(int_exp, t, x_int)
tau_int = res[0]
# Isolate fluctuations by subtracting deterministic saturating curve
x = np.squeeze(x_int - int_exp(t, tau_int))
dt = t[1] - t[0]

rng = np.random.default_rng(0)

# Render initial trace decomposition: raw recording, deterministic fit, and isolated noise
if display:
    plt.figure()
    plt.plot(t, x_int, label="Original trace")
    plt.plot(t, np.squeeze(int_exp(t, tau_int)), label="Integration only")
    plt.plot(t, x, label="Noise only")
    plt.legend()
    plt.show()

# ------------------------------------------------------------
# 2. Fit leaky integrator timescale tau
# ------------------------------------------------------------
# Compute numerical first-order forward differences and midpoint activity levels
dx = np.diff(x) / dt
x_mid = x[:-1]

# Negative log-likelihood assuming residual driving derivative conforms to Gaussian white drift
def neg_log_likelihood(tau):
    if tau <= 0:
        return np.inf
    resid = dx + x_mid / tau
    return 0.5 * np.sum(resid**2)

# Minimize bounded scalar objective to extract optimal leak timescale tau
res = optimize.minimize_scalar(neg_log_likelihood, bounds=(1e-3, 100), method="bounded")
tau_hat = res.x
print(f"Estimated tau = {tau_hat:.3f}")

# ------------------------------------------------------------
# 3. Extract noise residuals
# ------------------------------------------------------------
# Compute residual driving noise term driving the differential equation: dx/dt = -x/tau + eps
eps_hat = dx + x_mid / tau_hat

# ------------------------------------------------------------
# 4. Fit OU process to residuals
# ------------------------------------------------------------
# Maximum likelihood estimation of correlation time (tau_n) and diffusion (sigma) for an Ornstein-Uhlenbeck process
def fit_ou(residuals, dt):
    r = residuals[:-1]
    r_next = residuals[1:]

    def nll(params):
        tau_n, sigma = params
        if tau_n <= 0 or sigma <= 0:
            return np.inf
        # Exact discrete-time transition operator for continuous OU dynamics: r_{t+dt} ~ N(alpha * r_t, var)
        alpha = np.exp(-dt / tau_n)
        var = sigma**2 * (1 - alpha**2)
        pred = alpha * r
        return 0.5 * np.sum((r_next - pred)**2 / var + np.log(var))

    # Initial parameter guess based on sampling interval and sample standard deviation
    x0 = [dt * 10, np.std(residuals)]
    res = optimize.minimize(nll, x0, bounds=[(1e-4, 100), (1e-6, None)])
    return res.x

tau_n_hat, sigma_hat = fit_ou(eps_hat, dt)
print(f"Estimated OU tau_n = {tau_n_hat:.3f}")
print(f"Estimated OU sigma = {sigma_hat:.3f}")

# ------------------------------------------------------------
# 5. Validate noise model
# ------------------------------------------------------------
if display:
    # Synthesize forward realization of fitted OU noise process
    eps_sim = np.zeros_like(eps_hat)
    alpha = np.exp(-dt / tau_n_hat)
    for i in range(1, len(eps_sim)):
        eps_sim[i] = alpha * eps_sim[i-1] + sigma_hat * np.sqrt(1 - alpha**2) * rng.normal()

    # Empirical Autocorrelation Function (ACF) computation via cross-correlation
    def acf(x, nlags=200):
        x = x - x.mean()
        return np.correlate(x, x, mode="full")[len(x)-1:len(x)+nlags] / np.var(x) / len(x)

    lags = np.arange(len(eps_hat)) * dt

    # Plot ACF comparison between empirical residuals and simulated OU process
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(lags, acf(eps_hat), label="Residuals")
    plt.plot(lags, acf(eps_sim), label="OU model", linestyle="--")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.legend()

    # Welch Power Spectral Density (PSD) estimation comparison
    f1, P1 = signal.welch(eps_hat, fs=1/dt, nperseg=2048)
    f2, P2 = signal.welch(eps_sim, fs=1/dt, nperseg=2048)

    plt.subplot(1, 2, 2)
    plt.loglog(f1, P1, label="Residuals")
    plt.loglog(f2, P2, label="OU model", linestyle="--")
    plt.xlabel("Frequency")
    plt.ylabel("PSD")
    plt.legend()

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 7. Generate modeled noise realization
# ------------------------------------------------------------
# Generator producing synthetic OU colored noise trajectory matched to target length
def noise_cMI(x, tau, sigma, dt, scale=1, seed=None):
    rng = np.random.default_rng(seed)

    eps_model = np.zeros_like(x)
    alpha = np.exp(-dt / tau)
    noise_scale = sigma * np.sqrt(1 - alpha**2)

    for i in range(1, len(eps_model)):
        eps_model[i] = alpha * eps_model[i-1] + noise_scale * rng.normal()
    return eps_model * scale

# Synthesize noise realization over the cropped integration segment
eps_model = noise_cMI(x_int, tau_n_hat, sigma_hat, dt, scale_hat, 0)
x_model = x_int + eps_model

# ----------------------------------------------------------------
# 7. Plot comparison for integration part only in normalized space
# ----------------------------------------------------------------
# Visualize empirical cropped segment alongside the synthetic noise augmented model
if display:
    plt.figure(figsize=(10, 5))
    plt.plot(t, x_int, label="Original signal", linewidth=2)
    plt.plot(t, x_model, label="Modeled noise signal", linestyle="--")
    # plt.plot(t, x_det, label="Deterministic component", linestyle=":")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 8. Generate modeled noise realization
# ------------------------------------------------------------
# Generate synthetic noise realization across the full uncropped time-series
eps_model = noise_cMI(x_raw, tau_n_hat, sigma_hat, dt, scale_hat)
x_model = x_raw + eps_model

# ----------------------------------------------------------------
# 9. Plot comparison for original signal
# ----------------------------------------------------------------
# Render full time-series comparison: original data, synthetic composite, and noise trajectory
if display:
    plt.figure(figsize=(10, 5))
    plt.plot(t_raw, x_raw, label="Original signal", linewidth=2)
    plt.plot(t_raw, x_model, label="Synthetic signal", linestyle="--")
    plt.plot(t_raw, eps_model, label="Noise component", linestyle=":")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------
# 10. Save estimation
# ----------------------------------------------------------------
# Export fitted Ornstein-Uhlenbeck hyperparameters to pickle file for downstream model training
if save_estimation:
    noise_estimation = {"label": "ou",
                        "tau": tau_n_hat,
                        "sigma": sigma_hat,
                        "scale": scale_hat}
    with open(path_save / f"{trace_id}_noise_estimation.pkl", 'wb') as f:
        pickle.dump(noise_estimation, f)