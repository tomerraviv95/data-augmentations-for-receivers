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
    break_transmitted_siso_word_to_symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words and transmitted
    """

    def __init__(self, block_length: int, pilots_length: int, words: int, seed: int):
        self.block_length = block_length
        self.pilots_length = pilots_length
        self.words = words if conf.channel_type == ChannelModes.SISO.name else N_ANT * words
        self.bits_generator = default_rng(seed=seed)
        self.h_length = MEMORY_LENGTH if conf.channel_type == ChannelModes.SISO.name else N_ANT

    def get_snr_data(self, snr: float, database: list):
        if database is None:
            database = []
        b_full = np.empty((0, self.block_length))
        y_full = np.empty((0, self.block_length))
        h_full = np.empty((0, self.h_length))
        index = 0
        # accumulate words until reaches desired number
        while y_full.shape[0] < self.words:
            if conf.channel_type == ChannelModes.SISO.name:
                # get channel values
                h = ISIAWGNChannel.calculate_channel(MEMORY_LENGTH, fading=conf.fading_in_channel, index=index)
                b, y = self.siso_transmission(h, snr)
            elif conf.channel_type == ChannelModes.MIMO.name:
                # get channel values
                h = SEDChannel.calculate_channel(N_ANT, N_USER, index, conf.fading_in_channel)
                b, y = self.mimo_transmission(h, snr)
            else:
                raise ValueError("No such channel type!!!")
            # accumulate
            b_full = np.concatenate((b_full, b), axis=0)
            y_full = np.concatenate((y_full, y), axis=0)
            h_full = np.concatenate((h_full, h), axis=0)
            index += 1

        database.append((b_full, y_full, h_full))

    def generate_all_classes_pilots(self):
        if conf.channel_type == ChannelModes.SISO.name:
            b_pilots = self.bits_generator.integers(0, 2, size=(1, self.pilots_length)).reshape(1, -1)
            b_pilots_by_symbols = break_transmitted_siso_word_to_symbols(MEMORY_LENGTH,
                                                                         torch.Tensor(b_pilots).to(device))
            states = calculate_siso_states(MEMORY_LENGTH, b_pilots_by_symbols).cpu().numpy()
            n_unique = 2 ** MEMORY_LENGTH
        elif conf.channel_type == ChannelModes.MIMO.name:
            b_pilots = self.bits_generator.integers(0, 2, size=(N_USER, self.pilots_length))
            states = calculate_mimo_states(N_USER, torch.Tensor(b_pilots).T.to(device)).cpu().numpy()
            n_unique = 2 ** N_USER
        else:
            raise ValueError("No such channel type!!!")
        if len(np.unique(states)) < n_unique:
            return self.generate_all_classes_pilots()
        return b_pilots

    def siso_transmission(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        b_pilots = self.generate_all_classes_pilots()
        b_data = self.bits_generator.integers(0, 2, size=(1, self.block_length - self.pilots_length))
        b = np.concatenate([b_pilots, b_data], axis=1).reshape(1, -1)
        # add zero bits
        padded_b = np.concatenate([b, np.zeros([b.shape[0], MEMORY_LENGTH])], axis=1)
        # modulation
        s = BPSKModulator.modulate(padded_b)
        # transmit through noisy channel
        y = ISIAWGNChannel.transmit(s=s, h=h, snr=snr, memory_length=MEMORY_LENGTH)
        return b, y

    def mimo_transmission(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        b_pilots = self.generate_all_classes_pilots()
        b_data = self.bits_generator.integers(0, 2, size=(N_USER, self.block_length - self.pilots_length))
        b = np.concatenate([b_pilots, b_data], axis=1)
        # modulation
        s = BPSKModulator.modulate(b)
        # pass through channel
        y = SEDChannel.transmit(s=s, h=h, snr=snr)
        return b, y

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
