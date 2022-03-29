from python_code.channel.channel_estimation import estimate_channel
from python_code.channel.modulator import BPSKModulator
from python_code.channel.channel import ISIAWGNChannel
from python_code.channel.sed_channel import SEDChannel
from python_code.utils.config_singleton import Config
from torch.utils.data import Dataset
from numpy.random import default_rng
from typing import Tuple, List
import concurrent.futures
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()

DETECTOR_TYPE: str = 'deepsic'


def bpsk_modulate(b: np.ndarray) -> np.ndarray:
    return (-1) ** b


def calculate_sigma_from_snr(snr: int) -> float:
    """
    converts the Desired SNR into the noise power (noise variance)
    :param snr: signal-to-noise ratio
    :return: noise's sigma
    """
    return 10 ** (-0.1 * snr)


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words and transmitted
    """

    def __init__(self, block_length: int, transmission_length: int, words: int, n_ant: int):

        self.block_length = block_length
        self.transmission_length = transmission_length
        self.words = words
        self.bits_generator = default_rng(seed=conf.seed)
        self.n_ant = n_ant

    def get_snr_data(self, snr: float, gamma: float, database: list):
        if database is None:
            database = []
        b_full = np.empty((0, self.block_length))
        y_full = np.empty((0, self.transmission_length))
        h_full = np.empty((0, conf.memory_length))
        index = 0

        # accumulate words until reaches desired number
        while y_full.shape[0] < self.n_ant * self.words:
            if conf.detector_type == 'viterbi':
                b = self.bits_generator.integers(0, 2, size=(1, self.block_length)).reshape(1, -1)
                # add zero bits
                padded_b = np.concatenate([b, np.zeros([b.shape[0], conf.memory_length])], axis=1)
                # transmit
                h = estimate_channel(conf.memory_length, gamma,
                                     fading=conf.fading_in_channel,
                                     index=index)
                y = self.transmit(padded_b, h, snr)
            elif conf.detector_type == 'deepsic':
                # get channel
                h = SEDChannel.calculate_channel(conf.n_ant, conf.n_user, index, conf.fading_in_channel)
                b = self.bits_generator.integers(0, 2, size=(conf.n_user, self.block_length))
                # modulation
                x = bpsk_modulate(b)
                # pass through channel
                sigma = calculate_sigma_from_snr(snr)
                y = np.matmul(h, x) + np.sqrt(sigma) * np.random.randn(conf.n_ant, x.shape[1])
            else:
                raise ValueError("No such detector type!!!")
            # accumulate
            b_full = np.concatenate((b_full, b), axis=0)
            y_full = np.concatenate((y_full, y), axis=0)
            h_full = np.concatenate((h_full, h), axis=0)
            index += 1

        database.append((b_full, y_full, h_full))

    def transmit(self, c: np.ndarray, h: np.ndarray, snr: float):
        if conf.channel_type == 'ISI_AWGN':
            # modulation
            s = BPSKModulator.modulate(c)
            # transmit through noisy channel
            y = ISIAWGNChannel.transmit(s=s, h=h, snr=snr, memory_length=conf.memory_length)
        else:
            raise Exception('No such channel defined!!!')
        return y

    def __getitem__(self, snr_list: List[float], gamma: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        database = []
        # do not change max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            [executor.submit(self.get_snr_data, snr, gamma, database) for snr in snr_list]
        b, y, h = (np.concatenate(arrays) for arrays in zip(*database))
        b, y, h = torch.Tensor(b).to(device=device), torch.Tensor(y).to(device=device), torch.Tensor(h).to(
            device=device)
        return b, y, h

    def __len__(self):
        return self.transmission_length
