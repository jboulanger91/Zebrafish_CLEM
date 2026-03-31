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
env = dotenv_values()
path_trace = Path(env["PATH_DATA_NOISE"])
path_save = Path("../")

# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------
display = True
save_estimation = False
time_window_integration = (20, 60)  # (s)

# ------------------------------------------------------------
# 1. Load time series
# ------------------------------------------------------------
# Load traces to use as target signals
trace_id = path_trace.name.replace("_activity_traces.csv", "")
side = trace_id.split("_")[-1]
cell_type = trace_id.replace(f"_{side}", "")

data = np.loadtxt(path_trace, dtype=float, delimiter=",", skiprows=1)
t_raw = data[:, 0]
x_raw = data[:, 1] / 100
idx_in_window = np.argwhere(np.logical_and(t_raw>time_window_integration[0], t_raw<time_window_integration[1]))
t = np.squeeze(t_raw[idx_in_window]) - np.min(t_raw[idx_in_window])  # shift to zero
scale_raw = 1
scale_hat = scale_raw * 4  # double because the associated contribution has to prevail in a signal that already has one such contribution
x_int = np.squeeze(x_raw[idx_in_window]) / scale_raw  # normalize x so that only tau remains
int_exp = lambda t, tau: 1 - np.exp(-t/tau)
res = curve_fit(int_exp, t, x_int)
tau_int = res[0]
x = np.squeeze(x_int - int_exp(t, tau_int))
dt = t[1] - t[0]

rng = np.random.default_rng(0)

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
dx = np.diff(x) / dt
x_mid = x[:-1]

def neg_log_likelihood(tau):
    if tau <= 0:
        return np.inf
    resid = dx + x_mid / tau
    return 0.5 * np.sum(resid**2)

res = optimize.minimize_scalar(neg_log_likelihood, bounds=(1e-3, 100), method="bounded")
tau_hat = res.x
print(f"Estimated tau = {tau_hat:.3f}")

# ------------------------------------------------------------
# 3. Extract noise residuals
# ------------------------------------------------------------
eps_hat = dx + x_mid / tau_hat

# ------------------------------------------------------------
# 4. Fit OU process to residuals
# ------------------------------------------------------------
def fit_ou(residuals, dt):
    r = residuals[:-1]
    r_next = residuals[1:]

    def nll(params):
        tau_n, sigma = params
        if tau_n <= 0 or sigma <= 0:
            return np.inf
        alpha = np.exp(-dt / tau_n)
        var = sigma**2 * (1 - alpha**2)
        pred = alpha * r
        return 0.5 * np.sum((r_next - pred)**2 / var + np.log(var))

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
    # Simulate OU noise
    eps_sim = np.zeros_like(eps_hat)
    alpha = np.exp(-dt / tau_n_hat)
    for i in range(1, len(eps_sim)):
        eps_sim[i] = alpha * eps_sim[i-1] + sigma_hat * np.sqrt(1 - alpha**2) * rng.normal()

    # ACF comparison
    def acf(x, nlags=200):
        x = x - x.mean()
        return np.correlate(x, x, mode="full")[len(x)-1:len(x)+nlags] / np.var(x) / len(x)

    lags = np.arange(len(eps_hat)) * dt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(lags, acf(eps_hat), label="Residuals")
    plt.plot(lags, acf(eps_sim), label="OU model", linestyle="--")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.legend()

    # PSD comparison
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
def noise_cMI(x, tau, sigma, dt, scale=1, seed=None):
    rng = np.random.default_rng(seed)

    eps_model = np.zeros_like(x)
    alpha = np.exp(-dt / tau)
    noise_scale = sigma * np.sqrt(1 - alpha**2)

    for i in range(1, len(eps_model)):
        eps_model[i] = alpha * eps_model[i-1] + noise_scale * rng.normal()
    return eps_model * scale

eps_model = noise_cMI(x_int, tau_n_hat, sigma_hat, dt, scale_hat, 0)
x_model = x_int + eps_model

# ----------------------------------------------------------------
# 7. Plot comparison for integration part only in normalized space
# ----------------------------------------------------------------
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
eps_model = noise_cMI(x_raw, tau_n_hat, sigma_hat, dt, scale_hat)
x_model = x_raw + eps_model

# ----------------------------------------------------------------
# 9. Plot comparison for original signal
# ----------------------------------------------------------------
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
if save_estimation:
    noise_estimation = {"label": "ou",
                        "tau": tau_n_hat,
                        "sigma": sigma_hat,
                        "scale": scale_hat}
    with open(path_save / f"{trace_id}_noise_estimation.pkl", 'wb') as f:
        pickle.dump(noise_estimation, f)
