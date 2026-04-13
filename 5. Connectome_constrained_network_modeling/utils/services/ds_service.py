import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.signal import welch

from sklearn.decomposition import PCA
from torch import nn


class DSService():
    def __init__(self):
        pass

    @staticmethod
    def bifurcation_diagram(rnn, input_value_array, n_steps=1000, n_initializations=20, initial_state_rule="zero", pca=None):
        input_array = np.zeros(n_initializations * len(input_value_array))
        h_array = np.zeros((n_initializations * len(input_value_array), rnn.network_size))

        index_in_array = 0
        for _ in range(n_initializations):
            if initial_state_rule is None: initial_state = rnn.h  # custom input passed to the function or already generated
            elif initial_state_rule == "zero": initial_state = np.zeros(rnn.network_size)
            elif initial_state_rule == "random": initial_state = np.random.rand(rnn.network_size)
            else: raise NotImplementedError

            for i_input, input_value in enumerate(input_value_array):
                h, converged, _ = rnn.simulate_until_convergence(initial_state, input_signal_in_time=input_value, max_steps=n_steps, tol=1e-8)
                if converged:
                    input_array[index_in_array] = input_value
                    h_array[index_in_array] = h

                index_in_array += 1

        if index_in_array < len(input_array):
            input_array = input_array[:index_in_array]
            h_array = h_array[:index_in_array]

        if pca is None:
            pca = PCA(n_components=2)
            pca.fit(h_array)
        proj_state = np.squeeze(np.array([pca.transform(state.reshape(1, -1)) for state in h_array]))

        plt.figure(figsize=(10, 7))
        plt.scatter(input_array, proj_state[:, 0], c='k', alpha=0.8)
        plt.title('Bifurcation diagram')
        plt.xlabel('Input value')
        plt.ylabel('State')
        plt.xlim(np.min(input_value_array), np.max(input_value_array))
        plt.show()

    @staticmethod
    def gcamp_kernel(tau_rise, tau_decay, dt, duration=10.0):
        """
        Create a normalized double-exponential kernel for GCaMP calcium dynamics.

        tau_rise:   rise time constant (seconds)
        tau_decay:  decay time constant (seconds)
        fs:         sampling rate (Hz)
        duration:   total kernel length (seconds)
        """
        t = np.arange(0, duration, dt)
        kernel = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
        kernel[kernel < 0] = 0  # ensure non-negativity
        kernel /= np.sum(kernel)  # normalize to peak=1
        return kernel

    @staticmethod
    def gcamp_kernel_torch(tau_rise, tau_decay, dt, duration=10.0):
        """
        Create a normalized double-exponential kernel for GCaMP calcium dynamics.

        tau_rise:   rise time constant (seconds)
        tau_decay:  decay time constant (seconds)
        fs:         sampling rate (Hz)
        duration:   total kernel length (seconds)
        """
        t = torch.arange(0, duration, dt)
        kernel = torch.exp(-t / tau_decay) - torch.exp(-t / tau_rise)
        kernel[kernel < 0] = 0  # ensure non-negativity
        kernel /= kernel.sum()  # normalize to peak=1
        return kernel

    @classmethod
    def apply_gcamp_kernel(cls, signal, tau_rise, tau_decay, dt, amplitude_scale=1.0, use_np=False):
        """
        Convolve an input neural signal with a GCaMP kernel.

        signal:           input signal (e.g., spikes)
        tau_rise:         rise time constant (s)
        tau_decay:        decay time constant (s)
        fs:               sampling rate (Hz)
        amplitude_scale:  scale factor to simulate ΔF/F differences
        """
        if use_np:
            kernel = cls.gcamp_kernel(tau_rise, tau_decay, dt)
            convolved = np.convolve(signal.detach().numpy(), kernel, mode='full')[:len(signal)]
        else:
            B, T, C = signal.shape
            kernel = cls.gcamp_kernel_torch(tau_rise, tau_decay, dt)
            K = kernel.numel()
            signal_reshaped = signal.permute(0, 2, 1).reshape(B * C, 1, T)
            kernel = kernel.flip(0).view(1, 1, K)
            signal_padded = nn.functional.pad(signal_reshaped, (K - 1, 0), mode="replicate")
            convolved_reshaped = nn.functional.conv1d(signal_padded, kernel)
            convolved = convolved_reshaped[:, :, :T].reshape(B, C, T).permute(0, 2, 1)
        return amplitude_scale * convolved

    @staticmethod
    def ou_noise(x, tau, sigma, dt, scale=1, seed=None):
        rng = np.random.default_rng(seed)

        eps_model = np.zeros_like(x)
        alpha = np.exp(-dt / tau)
        noise_scale = sigma * np.sqrt(1 - alpha ** 2)

        for i in range(1, len(eps_model)):
            eps_model[i] = alpha * eps_model[i - 1] + noise_scale * rng.normal()
        return eps_model * scale

    @staticmethod
    def downsample_signal(raw_signal, dt_raw, time_sample_list):
        if not torch.is_tensor(time_sample_list):
            time_sample_list = torch.tensor(time_sample_list, dtype=torch.float32)
        t_raw = torch.arange(0, len(raw_signal), dt_raw)
        dt = torch.abs(t_raw.unsqueeze(1) - time_sample_list)
        idx_sample = torch.argmin(dt, dim=0)
        sampled_signal = raw_signal[:, idx_sample, :]
        return sampled_signal

    @staticmethod
    def estimate_tau_rise_decay_windowed_step(
            y,
            dt,
            t_on=None,  # optional: stimulus onset time (s)
            t_off=None,  # optional: stimulus offset time (s)
            smooth=0,  # optional moving-average window (samples), 0 = none
            eps=1e-12,
    ):
        """
        Estimate tau_rise and tau_decay from a windowed step-like response.

        Uses 63.2% rule:
          - Rising: time from onset to reach baseline + 0.632*(plateau-baseline)
          - Decay:  time from offset to reach final + 0.632*(plateau-final)  (i.e., 36.8% remaining)

        Returns:
          tau_rise_s, tau_decay_s, info_dict
        """
        y = np.asarray(y, dtype=float).copy()
        T = len(y)

        if smooth and smooth > 1:
            k = int(smooth)
            kernel = np.ones(k) / k
            y = np.convolve(y, kernel, mode="same")

        # If onset/offset are not provided, infer them from the largest positive/negative derivative peaks
        dy = np.diff(y, prepend=y[0])
        if t_on is None:
            i_on = int(np.argmax(dy))
        else:
            i_on = int(np.clip(round(t_on / dt), 0, T - 1))

        if t_off is None:
            i_off = int(np.argmin(dy))
        else:
            i_off = int(np.clip(round(t_off / dt), 0, T - 1))

        # Ensure ordering
        if i_off <= i_on:
            # fallback: put offset after onset by searching negative slope after onset
            j = np.argmin(dy[i_on + 1:]) if i_on < T - 2 else 0
            i_off = i_on + 1 + int(j)

        # Baseline: average before onset
        pre0 = max(0, i_on - int(0.1 * T))  # rough window
        baseline = np.mean(y[pre0:i_on]) if i_on > pre0 else y[i_on]

        # Plateau: average in middle of stimulation window (avoid edges)
        mid_a = i_on + int(0.2 * max(1, i_off - i_on))
        mid_b = i_on + int(0.8 * max(1, i_off - i_on))
        plateau = np.abs(y[i_on:i_off]).max()

        # Final: average after offset
        post_b = min(T, i_off + int(0.2 * T))
        final = baseline

        # Determine whether response goes up or down during stim
        amp_rise = plateau - baseline
        if abs(amp_rise) < eps:
            return np.nan, np.nan, {"reason": "flat_rise", "i_on": i_on, "i_off": i_off}

        # ----- Rise tau -----
        target_rise = baseline + 0.6321205588 * amp_rise  # 1 - 1/e [web:122]
        # Find first crossing after onset in the correct direction
        if amp_rise > 0:
            idx = np.where(y[i_on:] >= target_rise)[0]
        else:
            idx = np.where(y[i_on:] <= target_rise)[0]

        if len(idx) == 0:
            tau_rise = np.nan
            i_rise = None
        else:
            i_rise = i_on + int(idx[0])
            tau_rise = (i_rise - i_on) * dt

        # ----- Decay tau -----
        amp_decay = plateau - final
        if abs(amp_decay) < eps:
            tau_decay = np.nan
            i_decay = None
        else:
            target_decay = final + 0.3678794412 * amp_decay  # 1/e remaining [web:122]
            # after offset, should move from plateau toward final
            if amp_decay > 0:
                # plateau > final, decay downward: cross below target
                idx2 = np.where(y[i_off:] <= target_decay)[0]
            else:
                # plateau < final, decay upward: cross above target
                idx2 = np.where(y[i_off:] >= target_decay)[0]

            if len(idx2) == 0:
                tau_decay = np.nan
                i_decay = None
            else:
                i_decay = i_off + int(idx2[0])
                tau_decay = (i_decay - i_off) * dt

        info = {
            "i_on": i_on,
            "i_off": i_off,
            "baseline": float(baseline),
            "plateau": float(plateau),
            "final": float(final),
            "i_rise": i_rise,
            "i_decay": i_decay,
        }
        return float(tau_rise), float(tau_decay), info

    import numpy as np
    from scipy.optimize import curve_fit  # [web:266]

    @staticmethod
    def _exp_rise(t, y0, A, tau):
        # y(t) = y0 + A*(1 - exp(-t/tau))
        return y0 + A * (1.0 - np.exp(-t / tau))

    @staticmethod
    def _exp_decay(t, y_inf, B, tau):
        # y(t) = y_inf + B*exp(-t/tau)
        return y_inf + B * np.exp(-t / tau)

    @classmethod
    def fit_tau_rise_decay(
            cls,
            y,
            dt,
            t_on_s,
            t_off_s,
            fit_pad_rise_s=0.0,  # optionally skip initial transient after onset
            fit_pad_decay_s=0.0,  # optionally skip initial transient after offset
            max_tau_s=30.0,
            smooth=0,
    ):
        """
        Fit exponential rise and decay time constants from a windowed step-like signal.

        Args:
          y: 1D array length T
          dt: seconds per sample
          t_on_s, t_off_s: stimulus onset and offset in seconds (recommended to supply)
          smooth: optional moving average window length (samples), 0 = none

        Returns:
          tau_rise_s, tau_decay_s, fit_info dict
        """

        y = np.asarray(y, dtype=float)
        T = len(y)
        t = np.arange(T) * dt

        if smooth and smooth > 1:
            k = int(smooth)
            kernel = np.ones(k) / k
            y = np.convolve(y, kernel, mode="same")

        i_on = int(np.clip(round(t_on_s / dt), 0, T - 1))
        i_off = int(np.clip(round(t_off_s / dt), 0, T - 1))
        if i_off <= i_on + 3:
            raise ValueError("t_off must be > t_on by a few samples for fitting.")

        # Select fit windows
        i_on_fit = int(np.clip(i_on + round(fit_pad_rise_s / dt), 0, T - 1))
        i_off_fit_rise = i_off

        i_off_fit = int(np.clip(i_off + round(fit_pad_decay_s / dt), 0, T - 1))
        i_end = T

        # ----- Rise fit window -----
        tr = t[i_on_fit:i_off_fit_rise] - t[i_on_fit]
        yr = y[i_on_fit:i_off_fit_rise]

        # Initial guesses (important for curve_fit) [web:266]
        y0_guess = float(np.median(y[max(0, i_on - 10):i_on + 1]))
        y_plateau_guess = float(np.median(y[max(i_on, i_off - 10):i_off]))
        A_guess = y_plateau_guess - y0_guess
        tau_guess_rise = max(dt, 0.2 * (t_off_s - t_on_s))  # rough

        # bounds: tau must be positive [web:266]
        bounds_rise = ([-np.inf, -np.inf, dt], [np.inf, np.inf, max_tau_s])

        popt_rise, _ = curve_fit(cls._exp_rise, tr, yr, p0=[y0_guess, A_guess, tau_guess_rise], bounds=bounds_rise)
        tau_rise = float(popt_rise[2])

        # ----- Decay fit window -----
        td = t[i_off_fit:i_end] - t[i_off_fit]
        yd = y[i_off_fit:i_end]

        y_inf_guess = float(np.median(y[i_off:min(T, i_off + 10)]))
        y_start_decay = float(np.median(y[max(0, i_off - 10):i_off + 1]))
        B_guess = y_start_decay - y_inf_guess
        tau_guess_decay = max(dt, 0.2 * (t[-1] - t_off_s))

        bounds_decay = ([-np.inf, -np.inf, dt], [np.inf, np.inf, max_tau_s])


        popt_decay, _ = curve_fit(cls._exp_decay, td, yd, p0=[y_inf_guess, B_guess, tau_guess_decay], bounds=bounds_decay)
        tau_decay = float(popt_decay[2])


        info = {
            "i_on": i_on,
            "i_off": i_off,
            "i_on_fit": i_on_fit,
            "i_off_fit": i_off_fit,
            "popt_rise": popt_rise,
            "popt_decay": popt_decay,
        }
        return tau_rise, tau_decay, info

    @staticmethod
    def compute_time_rise(time, signal, percentage_threshold=0.9, show_plot=False):
        threshold = percentage_threshold * torch.max(signal)
        time_rise = time[torch.argwhere(signal >= threshold)[0][0]]
        if show_plot:
            plt.figure()
            plt.plot(time, signal)
            plt.ylim(0, 2)
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.hlines(threshold, np.min(time), np.max(time), color="red", linestyle="--")
            plt.vlines(time_rise, 0, 2, color="red", linestyle="--")
            plt.show()

        return time_rise - time[0]

    @classmethod
    def interpolate_to_common_grid(cls, signal1, times1, signal2, times2, num_points=None):
        """
        Interpolate both signals onto a shared time grid covering their overlapping interval.
        Resolution defaults to the finer of the two original samplings.
        """
        t_start = max(times1[0], times2[0])
        t_end = min(times1[-1], times2[-1])
        if t_end <= t_start:
            raise ValueError("Signals have no overlapping time interval.")

        if num_points is None:
            dt = min(np.mean(np.diff(times1)), np.mean(np.diff(times2)))
            num_points = int((t_end - t_start) / dt) + 1

        t_common = np.linspace(t_start, t_end, num_points)
        s1 = interp1d(times1, signal1, kind='linear', bounds_error=True)(t_common)
        s2 = interp1d(times2, signal2, kind='linear', bounds_error=True)(t_common)
        return s1, s2, t_common

    @classmethod
    def pearson_correlation(cls, signal1, times1, signal2, times2, num_points=None):
        """
        Pearson correlation between two signals with (potentially) different time samplings.

        Both signals are linearly interpolated onto the overlapping time interval at the
        resolution of the finer sampling before computing the correlation.

        Parameters
        ----------
        signal1, signal2 : array-like  — signal values
        times1,  times2  : array-like  — corresponding timestamps (same units)
        num_points       : int, optional — override the number of points on the common grid

        Returns
        -------
        r : float   — Pearson correlation coefficient in [-1, 1]
        p : float   — two-tailed p-value
        """
        s1, s2, _ = cls.interpolate_to_common_grid(signal1, times1, signal2, times2, num_points)
        r, p = pearsonr(s1, s2)
        return r, p

    @classmethod
    def compute_acf(cls, x, max_lag):
        """Normalized (biased) ACF via direct correlation, lags 0 … max_lag. ACF[0] == 1."""
        x = x - x.mean()
        n = len(x)
        full = np.correlate(x, x, mode='full')  # length 2n-1
        acf = full[n - 1: n + max_lag + 1]  # lags 0 … max_lag
        acf = acf / (n * x.var())  # normalize: acf[0] = 1
        return acf

    @classmethod
    def acf_distance(cls, signal1, times1, signal2, times2,
                     num_points=None, max_lag=None, metric='rmse'):
        """
        Distance between the autocorrelation functions (ACFs) of two signals.

        ACFs are normalized to [-1, 1] and encode only temporal structure (not amplitude),
        making this metric amplitude-independent by construction.

        Parameters
        ----------
        signal1, signal2 : array-like
        times1,  times2  : array-like
        num_points       : int, optional — grid resolution override
        max_lag          : int, optional — max lag index (default: half signal length)
        metric           : 'rmse'    → RMSE between ACFs  (lower = more similar)
                           'pearson' → 1 − r between ACFs (lower = more similar)

        Returns
        -------
        distance      : float       — 0 means identical ACFs / timescales
        acf1, acf2    : np.ndarray  — the two ACF arrays (useful for plotting / fitting)
        """
        s1, s2, _ = cls.interpolate_to_common_grid(signal1, times1, signal2, times2, num_points)
        n = len(s1)

        if max_lag is None:
            max_lag = n // 2

        acf1 = cls.compute_acf(s1, max_lag)
        acf2 = cls.compute_acf(s2, max_lag)

        if metric == 'rmse':
            dist = float(np.sqrt(np.mean((acf1 - acf2) ** 2)))
        elif metric == 'pearson':
            r, _ = pearsonr(acf1, acf2)
            dist = float(1.0 - r)
        else:
            raise ValueError("metric must be 'rmse' or 'pearson'.")

        return dist, acf1, acf2

    @classmethod
    def jsd_psd(cls, signal1, times1, signal2, times2, num_points=None, nperseg=None):
        """
        Jensen-Shannon Divergence (JSD) between the normalized Power Spectral Densities.

        Normalizing each PSD to sum to 1 turns it into a probability distribution over
        frequency, making the metric fully amplitude-independent.

        JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M),   M = 0.5*(P + Q)

        Parameters
        ----------
        signal1, signal2 : array-like
        times1,  times2  : array-like
        num_points       : int, optional — grid resolution override
        nperseg          : int, optional — Welch segment length (default: min(256, N//4))

        Returns
        -------
        jsd           : float in [0, 1] — 0 = identical spectra, 1 = maximally different
        freqs         : np.ndarray — frequency axis
        psd1_norm     : np.ndarray — normalized PSD of signal1
        psd2_norm     : np.ndarray — normalized PSD of signal2
        """
        s1, s2, t_common = cls.interpolate_to_common_grid(signal1, times1, signal2, times2, num_points)
        fs = 1.0 / (t_common[1] - t_common[0])

        if nperseg is None:
            nperseg = min(256, len(s1) // 4)

        freqs, psd1 = welch(s1, fs=fs, nperseg=nperseg)
        _, psd2 = welch(s2, fs=fs, nperseg=nperseg)

        p = psd1 / psd1.sum()
        q = psd2 / psd2.sum()
        m = 0.5 * (p + q)

        def _kl(a, b):
            mask = (a > 0) & (b > 0)
            return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

        jsd = (0.5 * _kl(p, m) + 0.5 * _kl(q, m)) / np.log(2)  # normalize to [0, 1]
        return float(np.clip(jsd, 0.0, 1.0)), freqs, p, q