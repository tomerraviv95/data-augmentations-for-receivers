from python_code.utils.config_singleton import Config
from numpy.random import default_rng
import numpy as np
import torch

conf = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ISIAWGNChannel:
    @staticmethod
    def transmit(s: np.ndarray, snr: float, h: np.ndarray, memory_length: int) -> np.ndarray:
        """
        The AWGN Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel function
        :param memory_length: length of channel memory
        :return: received word
        """
        snr_value = 10 ** (snr / 10)

        blockwise_s = np.concatenate([s[:, i:-memory_length + i] for i in range(memory_length)], axis=0)

        conv = np.dot(h[:, ::-1], blockwise_s)

        [row, col] = conv.shape

        noise_generator = default_rng(seed=conf.seed)

        w = (snr_value ** (-0.5)) * noise_generator.standard_normal((row, col))

        y = conv + w

        return y
