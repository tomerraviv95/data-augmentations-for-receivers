import concurrent.futures
from typing import Tuple, List

import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Dataset

from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_ANT, N_USER
from python_code.channel.isi_awgn_channel import ISIAWGNChannel
from python_code.channel.modulator import BPSKModulator
from python_code.channel.sed_channel import SEDChannel
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
from python_code.utils.trellis_utils import calculate_mimo_states, calculate_siso_states, \
    break_transmitted_siso_word_to_symbols, break_received_siso_word_to_symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class SISOChannel:
    def __init__(self, block_length, pilots_length):
        self.b_length = MEMORY_LENGTH
        self.h_shape = [1, MEMORY_LENGTH]
        self.y_length = 1
        self.block_length = block_length
        self.pilots_length = pilots_length
        self.bits_generator = default_rng(seed=conf.seed)

    def transmit(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        b_pilots = self.generate_all_classes_pilots()
        b_data = self.bits_generator.integers(0, 2, size=(1, self.block_length - self.pilots_length))
        b = np.concatenate([b_pilots, b_data], axis=1).reshape(1, -1)
        # add zero bits
        padded_b = np.concatenate(
            [np.zeros([b.shape[0], MEMORY_LENGTH - 1]), b, np.zeros([b.shape[0], MEMORY_LENGTH])], axis=1)
        # modulation
        s = BPSKModulator.modulate(padded_b)
        # transmit through noisy channel
        y = ISIAWGNChannel.transmit(s=s, h=h, snr=snr, memory_length=MEMORY_LENGTH)
        symbols, y = break_transmitted_siso_word_to_symbols(MEMORY_LENGTH, b), y.T
        return symbols[:-MEMORY_LENGTH + 1], y[:-MEMORY_LENGTH + 1]

    def get_values(self, snr, index):
        # get channel values
        h = ISIAWGNChannel.calculate_channel(MEMORY_LENGTH, fading=conf.fading_in_channel, index=index)
        b, y = self.transmit(h, snr)
        return b, h, y

    def generate_all_classes_pilots(self):
        b_pilots = self.bits_generator.integers(0, 2, size=(1, self.pilots_length)).reshape(1, -1)
        b_pilots_by_symbols = break_transmitted_siso_word_to_symbols(MEMORY_LENGTH, b_pilots)
        states = calculate_siso_states(MEMORY_LENGTH,
                                       torch.Tensor(b_pilots_by_symbols[:-MEMORY_LENGTH + 1]).to(device)).cpu().numpy()
        n_unique = 2 ** MEMORY_LENGTH
        if len(np.unique(states)) < n_unique:
            return self.generate_all_classes_pilots()
        return b_pilots


class MIMOChannel:
    def __init__(self, block_length, pilots_length):
        self.b_length = N_USER
        self.h_shape = [N_ANT, N_USER]
        self.y_length = N_ANT
        self.block_length = block_length
        self.pilots_length = pilots_length
        self.bits_generator = default_rng(seed=conf.seed)

    def transmit(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        b_pilots = self.generate_all_classes_pilots()
        b_data = self.bits_generator.integers(0, 2, size=(N_USER, self.block_length - self.pilots_length))
        b = np.concatenate([b_pilots, b_data], axis=1)
        # modulation
        s = BPSKModulator.modulate(b)
        # pass through channel
        y = SEDChannel.transmit(s=s, h=h, snr=snr)
        b, y = b.T, y.T
        return b, y

    def get_values(self, snr, index):
        # get channel values
        h = SEDChannel.calculate_channel(N_ANT, N_USER, index, conf.fading_in_channel)
        b, y = self.transmit(h, snr)
        return b, h, y

    def generate_all_classes_pilots(self):
        b_pilots = self.bits_generator.integers(0, 2, size=(N_USER, self.pilots_length))
        states = calculate_mimo_states(N_USER, torch.Tensor(b_pilots).T.to(device)).cpu().numpy()
        n_unique = 2 ** N_USER
        if len(np.unique(states)) < n_unique:
            return self.generate_all_classes_pilots()
        return b_pilots


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words and transmitted
    """

    def __init__(self, block_length: int, pilots_length: int, blocks_num: int):
        self.blocks_num = blocks_num  # if conf.channel_type == ChannelModes.SISO.name else N_ANT * blocks_num
        self.block_length = block_length
        if conf.channel_type == ChannelModes.SISO.name:
            self.channel_model = SISOChannel(block_length, pilots_length)
        elif conf.channel_type == ChannelModes.MIMO.name:
            self.channel_model = MIMOChannel(block_length, pilots_length)
        else:
            raise ValueError("No such channel value!")

    def get_snr_data(self, snr: float, database: list):
        if database is None:
            database = []
        b_full = np.empty((self.blocks_num, self.block_length, self.channel_model.b_length))
        h_full = np.empty((self.blocks_num, *self.channel_model.h_shape))
        y_full = np.empty((self.blocks_num, self.block_length, self.channel_model.y_length))
        # accumulate words until reaches desired number
        for index in range(self.blocks_num):
            b, h, y = self.channel_model.get_values(snr, index)
            # accumulate
            b_full[index] = b
            y_full[index] = y
            h_full[index] = h

        database.append((b_full, y_full, h_full))

    def __getitem__(self, snr_list: List[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        database = []
        # do not change max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            [executor.submit(self.get_snr_data, snr, database) for snr in snr_list]
        b, y, h = (np.concatenate(arrays) for arrays in zip(*database))
        b, y, h = torch.Tensor(b).to(device=device), torch.Tensor(y).to(device=device), torch.Tensor(h).to(
            device=device)
        return b, y, h

    def __len__(self):
        return self.block_length
