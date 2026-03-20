import numpy as np


def integrate(input_signal, output_signal_start=None, tau=10, dt=1):
    output_signal = np.zeros(len(input_signal))
    if output_signal_start is not None:
        output_signal[0] = output_signal_start

    for i in range(1, len(input_signal)):
        output_signal[i] = output_signal[i-1] + (input_signal[i-1] - output_signal[i-1]) * dt / tau

    return output_signal

def pseudo_integrate(input_signal, dt, order_integration=4/7):
    # Implement a function that behaves similar to exponetial integrator, but it is not
    # The order 4/7 allows for a slower rise time than the more natural value 1/2
    output_signal = np.zeros_like(input_signal)
    t_integration = 0
    for i in range(len(input_signal)):
        output_signal[i] = (dt * t_integration) ** order_integration * input_signal[i]
        if input_signal[i] == 0:
            t_integration = 0
        else:
            t_integration += dt
    return output_signal

def integrate_not(input_signal, output_signal_start=None, tau=10, dt=1):
    return 1 - integrate(input_signal, output_signal_start=output_signal_start, tau=tau, dt=dt)

def differ(input_signal):
    return np.gradient(input_signal)

def pid(input_signal, scale=1, tau=1, dt=0.01, abs=False, smooth_window=None):
    intermediate_signal = integrate(input_signal, tau=tau, dt=dt)
    output_signal = differ(intermediate_signal) * scale
    if abs:
        output_signal = np.abs(output_signal)
    if smooth_window is not None and smooth_window > 0:
        smooth_array = np.zeros((smooth_window, len(intermediate_signal)-smooth_window))
        for i in range(smooth_window):
            smooth_array[i] = output_signal[i:len(output_signal)-smooth_window+i]
        output_signal[smooth_window:] = np.mean(smooth_array, axis=0)
    return output_signal

def get_hist(value_list, bins=10, hist_range=None, duration=None, allow_zero=True, center_bin=False, density=False):
    # value_list can be any iterable supported by numba, i.e. not pandas Series
    if len(value_list) == 0:
        return np.zeros(1), np.zeros(1)
    binned_value_list, bin_list = np.histogram(np.array(value_list), bins, hist_range)
    if duration is not None:
        binned_value_list = binned_value_list / duration
    if density:
        binned_value_list = binned_value_list / np.sum(binned_value_list)
    if not allow_zero:
        binned_value_list[binned_value_list == 0] = np.finfo(float).eps
    if center_bin:
        bin_list = (bin_list[:-1] + bin_list[1:]) / 2
    return binned_value_list, bin_list

def inv_softplus(x, beta=1):
    return np.log(-1 + np.exp(x * beta)) / beta