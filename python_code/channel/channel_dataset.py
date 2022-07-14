import concurrent.futures
from typing import Tuple, List

import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Dataset

from python_code import DEVICE
from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_ANT, N_USER
from python_code.channel.cost_mimo_channel import Cost2100MIMOChannel
from python_code.channel.cost_siso_channel import Cost2100SISOChannel
from python_code.channel.isi_awgn_channel import ISIAWGNChannel
from python_code.channel.modulator import MODULATION_DICT
from python_code.channel.sed_channel import SEDChannel
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes, ChannelModels, ModulationType
from python_code.utils.python_utils import normalize_for_modulation
from python_code.utils.trellis_utils import calculate_mimo_states, calculate_siso_states, \
    break_transmitted_siso_word_to_symbols

conf = Config()

DEBUG = True


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
        if conf.modulation_type == ModulationType.QPSK.name:
            raise ValueError("Did not implement the QPSK constellation for the SISO case, only MIMO!")
        # modulation
        s = MODULATION_DICT[conf.modulation_type].modulate(padded_b)
        # transmit through noisy channel
        if conf.channel_model == ChannelModels.Synthetic.name:
            y = ISIAWGNChannel.transmit(s=s, h=h, snr=snr, memory_length=MEMORY_LENGTH)
        elif conf.channel_model == ChannelModels.Cost2100.name:
            y = Cost2100SISOChannel.transmit(s=s, h=h, snr=snr, memory_length=MEMORY_LENGTH)
        else:
            raise ValueError("No such channel model!!!")
        symbols, y = break_transmitted_siso_word_to_symbols(MEMORY_LENGTH, b), y.T
        return symbols[:-MEMORY_LENGTH + 1], y[:-MEMORY_LENGTH + 1]

    def get_values(self, snr, index):
        # get channel values
        # transmit through noisy channel
        if conf.channel_model == ChannelModels.Synthetic.name:
            h = ISIAWGNChannel.calculate_channel(MEMORY_LENGTH, fading=conf.fading_in_channel, index=index)
        elif conf.channel_model == ChannelModels.Cost2100.name:
            h = Cost2100SISOChannel.calculate_channel(MEMORY_LENGTH, fading=conf.fading_in_channel, index=index)
        else:
            raise ValueError("No such channel model!!!")
        b, y = self.transmit(h, snr)
        return b, h, y

    def generate_all_classes_pilots(self):
        b_pilots = self.bits_generator.integers(0, 2, size=(1, self.pilots_length)).reshape(1, -1)
        b_pilots_by_symbols = break_transmitted_siso_word_to_symbols(MEMORY_LENGTH, b_pilots)
        states = calculate_siso_states(MEMORY_LENGTH,
                                       torch.Tensor(b_pilots_by_symbols[:-MEMORY_LENGTH + 1]).to(DEVICE)).cpu().numpy()
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
        s = MODULATION_DICT[conf.modulation_type].modulate(b)
        # pass through channel
        if conf.channel_model == ChannelModels.Synthetic.name:
            y = SEDChannel.transmit(s=s, h=h, snr=snr)
        elif conf.channel_model == ChannelModels.Cost2100.name:
            y = Cost2100MIMOChannel.transmit(s=s, h=h, snr=snr)
        else:
            raise ValueError("No such channel model!!!")
        b, y = b.T, y.T
        return b, y

    def get_values(self, snr, index):
        # get channel values
        if conf.channel_model == ChannelModels.Synthetic.name:
            h = SEDChannel.calculate_channel(N_ANT, N_USER, index, conf.fading_in_channel)
        elif conf.channel_model == ChannelModels.Cost2100.name:
            h = Cost2100MIMOChannel.calculate_channel(N_ANT, N_USER, index, conf.fading_in_channel)
        else:
            raise ValueError("No such channel model!!!")
        b, y = self.transmit(h, snr)
        return b, h, y

    def generate_all_classes_pilots(self):
        b_pilots = self.bits_generator.integers(0, 2, size=(N_USER, self.pilots_length))
        states = calculate_mimo_states(N_USER, torch.Tensor(b_pilots).T.to(DEVICE)).cpu().numpy()
        n_unique = 2 ** N_USER
        if not DEBUG and len(np.unique(states)) < n_unique:
            return self.generate_all_classes_pilots()
        return b_pilots


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words and transmitted
    """

    def __init__(self, block_length: int, pilots_length: int, blocks_num: int):
        self.blocks_num = blocks_num
        self.block_length = block_length
        if conf.channel_type == ChannelModes.SISO.name:
            self.channel_type = SISOChannel(block_length, pilots_length)
        elif conf.channel_type == ChannelModes.MIMO.name:
            self.channel_type = MIMOChannel(block_length, pilots_length)
        else:
            raise ValueError("No such channel value!")

    def get_snr_data(self, snr: float, database: list):
        if database is None:
            database = []
        b_full = np.empty((self.blocks_num, self.block_length, self.channel_type.b_length))
        h_full = np.empty((self.blocks_num, *self.channel_type.h_shape))
        y_full = np.empty((self.blocks_num, normalize_for_modulation(self.block_length), self.channel_type.y_length),
                          dtype=complex if conf.modulation_type == ModulationType.QPSK.name else float)
        # accumulate words until reaches desired number
        for index in range(self.blocks_num):
            b, h, y = self.channel_type.get_values(snr, index)
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
        b, y, h = torch.Tensor(b).to(device=DEVICE), torch.from_numpy(y).to(device=DEVICE), torch.Tensor(
            h).to(device=DEVICE)
        return b, y, h

    def __len__(self):
        return self.block_length
