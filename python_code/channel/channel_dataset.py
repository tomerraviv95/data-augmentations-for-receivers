from python_code.channel.channel_estimation import estimate_channel
from python_code.channel.modulator import BPSKModulator
from python_code.channel.channel import ISIAWGNChannel
from python_code.utils.config_singleton import Config
from torch.utils.data import Dataset
from numpy.random import default_rng
from typing import Tuple, List
import concurrent.futures
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words and transmitted
    """

    def __init__(self, block_length: int, transmission_length: int, words: int):

        self.block_length = block_length
        self.transmission_length = transmission_length
        self.words = words
        self.bits_generator = default_rng(seed=conf.seed)

    def get_snr_data(self, snr: float, gamma: float, database: list):
        if database is None:
            database = []
        b_full = np.empty((0, self.block_length))
        y_full = np.empty((0, self.transmission_length))
        h_full = np.empty((0, conf.memory_length))
        index = 0

        # accumulate words until reaches desired number
        while y_full.shape[0] < self.words:
            b = self.bits_generator.integers(0, 2, size=(1, self.block_length)).reshape(1, -1)
            # add zero bits
            padded_b = np.concatenate([b, np.zeros([b.shape[0], conf.memory_length])], axis=1)
            # transmit
            h = estimate_channel(conf.memory_length, gamma,
                                 fading=conf.fading_in_channel,
                                 index=index)
            y = self.transmit(padded_b, h, snr)
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
