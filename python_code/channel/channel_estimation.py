import numpy as np

COST_LENGTH = 300


def estimate_channel(memory_length: int, gamma: float, channel_coefficients: str, noisy_est_var: float = 0,
                     fading: bool = False, index: int = 0, fading_taps_type: int = 1):
    """
    Returns the coefficients vector estimated from channel
    :param memory_length: memory length of channel
    :param gamma: coefficient
    :param channel_coefficients: coefficients type
    :param noisy_est_var: variance for noisy estimation of coefficients 2nd,3rd,...
    :param fading: fading flag - if true, apply fading.
    :param index: time index for the fading functionality
    :return: the channel coefficients [1,memory_length] numpy array
    """
    if channel_coefficients == 'time_decay':
        h = np.reshape(np.exp(-gamma * np.arange(memory_length)), [1, memory_length])
    else:
        raise ValueError('No such channel_coefficients value!!!')

    # noise in estimation of h taps
    if noisy_est_var > 0:
        h[:, 1:] += np.random.normal(0, noisy_est_var ** 0.5, [1, memory_length - 1])

    # fading in channel taps
    if fading and channel_coefficients == 'time_decay':
        if fading_taps_type == 1:
            fading_taps = np.array([51, 39, 33, 21])
            h *= (0.8 + 0.2 * np.cos(2 * np.pi * index / fading_taps)).reshape(1, memory_length)
        elif fading_taps_type == 2:
            fading_taps = 5 * np.array([51, 39, 33, 21])
            fading_taps = np.maximum(fading_taps - 1.5 * index, 10 * np.ones(4)) - 1e-5
            h *= (0.8 + 0.2 * np.cos(np.pi * index / fading_taps)).reshape(1, memory_length)
        else:
            raise ValueError("No such fading tap type!!!")
    return h
