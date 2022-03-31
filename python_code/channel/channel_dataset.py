from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_ANT, N_USER
from python_code.channel.modulator import BPSKModulator
from python_code.channel.isi_awgn_channel import ISIAWGNChannel
from python_code.channel.sed_channel import SEDChannel
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
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
        if conf.channel_type == ChannelModes.SISO.name:
            h_full = np.empty((0, MEMORY_LENGTH))
            total_words = self.words
        elif conf.channel_type == ChannelModes.MIMO.name:
            h_full = np.empty((0, N_ANT))
            total_words = N_ANT * self.words

        index = 0
        # accumulate words until reaches desired number
        while y_full.shape[0] < total_words:
            if conf.channel_type == ChannelModes.SISO.name:
                b, h, y = self.siso_transmission(gamma, index, snr)
            elif conf.channel_type == ChannelModes.MIMO.name:
                b, h, y = self.mimo_transmission(index, snr)
            else:
                raise ValueError("No such channel type!!!")
            # accumulate
            b_full = np.concatenate((b_full, b), axis=0)
            y_full = np.concatenate((y_full, y), axis=0)
            h_full = np.concatenate((h_full, h), axis=0)
            index += 1

        database.append((b_full, y_full, h_full))

    def mimo_transmission(self, index, snr):
        b = self.bits_generator.integers(0, 2, size=(N_USER, self.block_length))
        # get channel values
        h = SEDChannel.calculate_channel(N_ANT, N_USER, index, conf.fading_in_channel)
        # modulation
        s = BPSKModulator.modulate(b)
        # pass through channel
        y = SEDChannel.transmit(s, h, snr)
        return b, h, y

    def siso_transmission(self, gamma, index, snr):
        b = self.bits_generator.integers(0, 2, size=(1, self.block_length)).reshape(1, -1)
        # add zero bits
        padded_b = np.concatenate([b, np.zeros([b.shape[0], MEMORY_LENGTH])], axis=1)
        # get channel values
        h = ISIAWGNChannel.calculate_channel(MEMORY_LENGTH, gamma, fading=conf.fading_in_channel, index=index)
        # modulation
        s = BPSKModulator.modulate(padded_b)
        # transmit through noisy channel
        y = ISIAWGNChannel.transmit(s=s, h=h, snr=snr, memory_length=conf.memory_length)
        return b, h, y

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
