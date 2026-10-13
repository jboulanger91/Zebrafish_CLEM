import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, wasserstein_distance
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
        kernel /= kernel.sum()  # NB: this is sum-normalisation (unit DC gain),
                                # not peak=1 as the comment used to say
        return kernel

    # Kernels depend only on (tau_rise, tau_decay, dt, duration, device, dtype),
    # none of which change during a fit, but apply_gcamp_kernel used to rebuild
    # one on every call: arange + two exps + a boolean mask + a sum + a flip,
    # twice per training epoch. Cached here instead.
    #
    # The cached tensor is SHARED -- treat it as read-only. gcamp_kernel_torch
    # mutates its kernel in place (`kernel[kernel < 0] = 0`, `kernel /= ...`), so
    # anything that wants to modify a kernel must clone it first.
    _gcamp_kernel_cache = {}

    @classmethod
    def gcamp_kernel_flipped(cls, tau_rise, tau_decay, dt, duration=10.0,
                             device=None, dtype=torch.float32):
        """Cached, time-reversed kernel, ready to hand to conv1d as a weight."""
        key = (float(tau_rise), float(tau_decay), float(dt), float(duration),
               str(device), str(dtype))
        kernel = cls._gcamp_kernel_cache.get(key)
        if kernel is None:
            kernel = cls.gcamp_kernel_torch(tau_rise, tau_decay, dt, duration)
            kernel = kernel.flip(0).to(device=device, dtype=dtype).contiguous()
            cls._gcamp_kernel_cache[key] = kernel
        return kernel

    @classmethod
    def apply_gcamp_kernel(cls, signal, tau_rise, tau_decay, dt, amplitude_scale=1.0,
                           use_np=False, method="conv"):
        """
        Convolve an input neural signal with a GCaMP kernel.

        signal:           input signal (B, T, C)
        tau_rise:         rise time constant (s)
        tau_decay:        decay time constant (s)
        dt:               timestep (s)
        amplitude_scale:  scale factor to simulate dF/F differences
        method:           "conv" -> depthwise conv1d (default; bit-identical to
                          the previous implementation)
                          "fft"  -> circular convolution via rfft/irfft. Same
                          result to ~1e-6 relative, and ~3x faster again on the
                          shapes fit() uses. Worth it if the filter shows up in
                          your profile; not the default because it is not
                          bit-exact.

        Performance note
        ----------------
        The previous version did `permute(0, 2, 1).reshape(B * C, 1, T)` and then
        a conv1d with batch B*C and ONE channel. That shape is the worst case for
        the CPU conv kernels -- each of the B*C rows goes through its own tiny
        im2col -- and the reshape of a permuted tensor forces a full copy first.
        Running it instead as a depthwise convolution (batch B, C channels,
        groups=C) is the same arithmetic on the same numbers, bit for bit, and on
        (4, 8000, 8) with K=1000 it is ~11x faster fwd+bwd (0.49 s -> 0.04 s).
        """
        if use_np:
            kernel = cls.gcamp_kernel(tau_rise, tau_decay, dt)
            convolved = np.convolve(signal.detach().numpy(), kernel, mode='full')[:len(signal)]
            return amplitude_scale * convolved

        B, T, C = signal.shape
        kernel = cls.gcamp_kernel_flipped(tau_rise, tau_decay, dt,
                                          device=signal.device, dtype=signal.dtype)
        K = kernel.numel()

        # (B, C, T), no copy of a permuted tensor, and replicate-pad the causal
        # history exactly as before.
        signal_padded = nn.functional.pad(signal.permute(0, 2, 1), (K - 1, 0),
                                          mode="replicate")

        if method == "fft":
            n = 1 << (signal_padded.shape[-1] + K - 1).bit_length()
            spec = torch.fft.rfft(signal_padded, n=n) * torch.fft.rfft(kernel.flip(0), n=n)
            convolved = torch.fft.irfft(spec, n=n)[..., K - 1:K - 1 + T]
        elif method == "conv":
            weight = kernel.view(1, 1, K).expand(C, 1, K)
            convolved = nn.functional.conv1d(signal_padded, weight, groups=C)
        else:
            raise ValueError(f"Unknown method: {method!r} (use 'conv' or 'fft')")

        return amplitude_scale * convolved.permute(0, 2, 1)

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

    @classmethod
    def compute_normalized_psd(
            cls,
            trace: np.ndarray,
            fs: float,
            window_sec: float = None,
            freq_range: tuple = (0.0, 1.0),
            detrend: str = "constant",
    ):
        """Computes PSD, truncates it to freq_range, and re-normalizes strictly within the band."""
        if torch.is_tensor(trace):
            trace = trace.detach().cpu().numpy()
        trace = np.asarray(trace, dtype=np.float64)

        n_samples = len(trace)
        if window_sec is not None:
            nperseg = int(round(window_sec * fs))
            nperseg = max(16, min(n_samples, nperseg))
        else:
            nperseg = n_samples

        freqs, psd = welch(
            trace, fs=fs, nperseg=nperseg, detrend=detrend
        )

        # 1. Truncate strictly to the window of interest [f_min, f_max]
        f_min, f_max = freq_range
        band_mask = (freqs >= f_min) & (freqs <= f_max)

        if not np.any(band_mask):
            # Fallback in case of degenerate range selection
            band_mask = freqs <= f_max

        freqs_band = freqs[band_mask]
        psd_band = psd[band_mask]

        # 2. Re-normalize strictly within the band so sum equals 1.0
        psd_sum = np.sum(psd_band)
        if psd_sum <= 0 or np.isnan(psd_sum):
            psd_norm = np.ones_like(psd_band) / len(psd_band)
        else:
            psd_norm = psd_band / psd_sum

        return freqs_band, psd_norm

    @classmethod
    def evaluate_population_timescales(
            cls,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            dt_pred: float = None,
            dt_true: float = None,
            duration: float = None,
            freq_range: tuple = (0.0, 1.0),
            spectral_scale: float = None,
            plot_spectra: bool = False,
            pop_names: list = None,
    ) -> dict:
        """Evaluates timescale and trajectory alignment across populations.

        Parameters:
        -----------
        y_pred : np.ndarray, shape (T_pred, N_pops)
            RNN model predictions (e.g., shape 8000 x 8).
        y_true : np.ndarray, shape (T_true, N_pops)
            Experimental 2P target traces (e.g., shape 160 x 8).
        dt_pred, dt_true : float, optional
            Sampling intervals in seconds (e.g., dt_pred=0.01, dt_true=0.5).
        duration : float, optional
            Total signal duration in seconds if dt is not explicitly supplied.
        freq_range : tuple of (float, float), default=(0.0, 1.0)
            Physical frequency window (in Hz) to isolate and normalize.
        spectral_scale : float, optional
            Sensitivity scale for exp(-emd / spectral_scale). If None, defaults
            to 10% of the frequency band width (e.g., 0.10 Hz for a 0-1 Hz
            band),
            sharply separating good from bad fits.
        plot_spectra : bool, optional
            Whether to generate an overlaid multi-panel spectra comparison plot.
        pop_names : list of str, optional
            Custom names for the populations (length N_pops).
        max_freq : float, optional
            Upper x-axis limit in Hz for the plots (defaults to target Nyquist).
        """
        t_steps_pred, n_pops = y_pred.shape
        t_steps_true = y_true.shape[0]

        # Resolve physical time axes and sampling rates
        if dt_pred is not None and dt_true is not None:
            t_pred = np.arange(t_steps_pred) * dt_pred
            t_true = np.arange(t_steps_true) * dt_true
            fs_pred = 1.0 / dt_pred
            fs_true = 1.0 / dt_true
            matched_window_sec = min(t_pred[-1], t_true[-1])
        elif duration is not None:
            t_pred = np.linspace(0, duration, t_steps_pred)
            t_true = np.linspace(0, duration, t_steps_true)
            fs_pred = (t_steps_pred - 1) / duration
            fs_true = (t_steps_true - 1) / duration
            matched_window_sec = duration
        else:
            # Default to normalized duration = 1.0 if no timing is supplied
            t_pred = np.linspace(0, 1.0, t_steps_pred)
            t_true = np.linspace(0, 1.0, t_steps_true)
            fs_pred = float(t_steps_pred)
            fs_true = float(t_steps_true)
            matched_window_sec = 1.0

        # Calibrate sensitivity to the active window width
        band_width = max(freq_range[1] - freq_range[0], 1e-4)
        if spectral_scale is None:
            spectral_scale = 0.08 * band_width

        # 1. Resample RNN predictions onto experimental 2P frame timestamps
        resampler = interp1d(
            t_pred, y_pred, axis=0, kind="linear", fill_value="extrapolate"
        )
        y_pred_resampled = resampler(t_true)

        pearson_corrs, spectral_sims, spectral_emds, spectral_data = [], [], [], []
        centroids_pred, centroids_true = [], []
        for i in range(n_pops):
            pred_resampled_i = y_pred_resampled[:, i]
            true_i = y_true[:, i]
            pred_raw_i = y_pred[:, i]

            # 1. Pearson correlation on time-aligned grids
            r = np.corrcoef(pred_resampled_i, true_i)[0, 1]
            r_clipped = max(0.0, float(r)) if not np.isnan(r) else 0.0
            pearson_corrs.append(r_clipped)

            # # Spectral analysis using native sampling grids in Hz
            # freqs_pred, psd_pred = cls.compute_normalized_psd(pred_raw_i, fs=fs_pred)
            # freqs_true, psd_true = cls.compute_normalized_psd(true_i, fs=fs_true)
            #
            # # Wasserstein-1 distance across physical frequency axes
            # emd_hz = wasserstein_distance(freqs_pred, freqs_true, psd_pred, psd_true)
            # spectral_emds.append(float(emd_hz))
            #
            # # Convert frequency mismatch in Hz to similarity in [0, 1]
            # sim = float(np.exp(-emd_hz / spectral_scale))
            # spectral_sims.append(sim)

            # 2. Band-isolated power spectra
            fp, pp = cls.compute_normalized_psd(
                pred_raw_i,
                fs=fs_pred,
                window_sec=matched_window_sec,
                freq_range=freq_range,
                detrend="constant",
            )
            ft, pt = cls.compute_normalized_psd(
                true_i,
                fs=fs_true,
                window_sec=matched_window_sec,
                freq_range=freq_range,
                detrend="constant",
            )

            emd_hz = float(wasserstein_distance(fp, ft, pp, pt))
            spectral_emds.append(emd_hz)
            spectral_sims.append(float(np.exp(-emd_hz / spectral_scale)))

            # Spectral centroids (center of mass in Hz)
            c_pred = float(np.sum(fp * pp))
            c_true = float(np.sum(ft * pt))
            centroids_pred.append(c_pred)
            centroids_true.append(c_true)

            if plot_spectra:
                spectral_data.append((fp, pp, ft, pt))

        pearson_corrs = np.array(pearson_corrs)
        spectral_sims = np.array(spectral_sims)
        spectral_emds = np.array(spectral_emds)
        centroids_pred = np.array(centroids_pred)
        centroids_true = np.array(centroids_true)

        # Composite score per population balancing trajectory and timescale
        pop_scores = np.sqrt(pearson_corrs * spectral_sims)

        # Harmonic mean across populations to prevent compensatory tradeoffs
        eps = 1e-6
        harmonic_composite = len(pop_scores) / np.sum(1.0 / (pop_scores + eps))
        harmonic_spectral = len(spectral_sims) / np.sum(1.0 / (spectral_sims + eps))

        # Optional plotting of spectra
        fig = None
        if plot_spectra:
            n_cols = 4 if n_pops >= 4 else n_pops
            n_rows = int(np.ceil(n_pops / n_cols))
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(4.0 * n_cols, 3.2 * n_rows),
                sharex=True,
                sharey=False,
            )
            axes = np.array(axes).flatten()

            for i in range(n_pops):
                ax = axes[i]
                fp, pp, ft, pt = spectral_data[i]
                name = (
                    pop_names[i]
                    if pop_names and i < len(pop_names)
                    else f"Pop {i + 1}"
                )

                ax.plot(ft, pt, color="black", lw=2.0, label="2P Data")
                ax.fill_between(ft, pt, color="gray", alpha=0.25)
                ax.plot(
                    fp,
                    pp,
                    color="#1f77b4",
                    lw=1.8,
                    linestyle="--",
                    label="Model",
                )
                ax.fill_between(fp, pp, color="#1f77b4", alpha=0.15)

                ax.set_xlim(freq_range[0], freq_range[1])
                ax.set_title(
                    f"{name}\nEMD: {spectral_emds[i]:.4f} Hz | Sim: {spectral_sims[i]:.3f}\n"
                    f"Centroid (M/D): {centroids_pred[i]:.3f} / {centroids_true[i]:.3f} Hz",
                    fontsize=8.5,
                )
                ax.grid(True, linestyle=":", alpha=0.6)

                if i >= (n_rows - 1) * n_cols:
                    ax.set_xlabel("Frequency (Hz)", fontsize=8.5)
                if i % n_cols == 0:
                    ax.set_ylabel("Norm. PSD", fontsize=8.5)
                if i == 0:
                    ax.legend(frameon=False, fontsize=8)

            for j in range(n_pops, len(axes)):
                axes[j].set_visible(False)

            plt.suptitle(
                f"Physiological Band Spectrum [{freq_range[0]} - {freq_range[1]} Hz]",
                fontsize=11,
                y=0.99,
            )
            plt.tight_layout()
            plt.show()

        return {
            "pop_pearson": pearson_corrs,
            "pop_spectral_emd_hz": spectral_emds,
            "pop_spectral_similarity": spectral_sims,
            "pop_composite": pop_scores,
            "harmonic_composite_score": float(harmonic_composite),
            "harmonic_spectral_score": float(harmonic_spectral),
            "min_composite_score": float(np.min(pop_scores)),
            "figure": fig,
        }
