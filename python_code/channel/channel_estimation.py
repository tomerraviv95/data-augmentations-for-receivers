from python_code.utils.config_singleton import Config
from dir_definitions import COST2100_DIR
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os

COST_LENGTH = 300

conf = Config()


def estimate_channel(memory_length: int, gamma: float, phase: str, fading: bool = False, index: int = 0):
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
    if conf.channel_coefficients == 'time_decay' or phase == 'train':
        h = np.reshape(np.exp(-gamma * np.arange(memory_length)), [1, memory_length])
    elif conf.channel_coefficients == 'cost2100':
        total_h = np.empty([COST_LENGTH, memory_length])
        for i in range(memory_length):
            total_h[:, i] = scipy.io.loadmat(os.path.join(COST2100_DIR, f'h_{i}'))[
                'h_channel_response_mag'].reshape(-1)
        h = np.reshape(total_h[index], [1, memory_length])
    else:
        raise ValueError('No such channel_coefficients value!!!')

    # fading in channel taps
    if fading and conf.channel_coefficients == 'time_decay':
        fading_taps = np.array([51, 39, 33, 21])
        h *= (0.8 + 0.2 * np.cos(2 * np.pi * index / fading_taps)).reshape(1, memory_length)
    else:
        h *= 0.8

    return h


if __name__ == '__main__':
    memory_length = 4
    gamma = 0.2
    noisy_est_var = 0
    channel_coefficients = 'cost2100'  # 'time_decay','cost2100'
    fading_taps_type = 1
    fading = False
    channel_length = COST_LENGTH
    phase = 'val'

    total_h = np.empty([channel_length, memory_length])
    for index in range(channel_length):
        total_h[index] = estimate_channel(memory_length, gamma, phase, fading=fading, index=index)
    for i in range(memory_length):
        plt.plot(total_h[:, i], label=f'Tap {i}')
    plt.xlabel('Block Index')
    plt.ylabel('Magnitude')
    plt.legend(loc='upper left')
    plt.show()
