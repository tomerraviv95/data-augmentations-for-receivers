from python_code.channel.channel_estimation import estimate_channel
from python_code.channel.modulator import BPSKModulator
from python_code.channel.channel import ISIAWGNChannel
from python_code.utils.config_singleton import Config
from python_code.ecc.rs_main import encode
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from numpy.random import mtrand
from typing import Tuple, List
import concurrent.futures
import numpy as np
import itertools
import torch

from python_code.utils.trellis_utils import calculate_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words and transmitted
    """

    def __init__(self, block_length: int, transmission_length: int, words: int,
                 use_ecc: bool, phase: str):

        self.phase = phase
        self.block_length = block_length
        self.transmission_length = transmission_length
        self.words = words
        if use_ecc and self.phase == 'val':
            self.encoding = lambda b: encode(b, conf.n_symbols)
        else:
            self.encoding = lambda b: b

    def get_snr_data(self, snr: float, gamma: float, database: list):
        if database is None:
            database = []
        b_full = np.empty((0, self.block_length))
        y_full = np.empty((0, self.transmission_length))
        h_full = np.empty((0, conf.memory_length))
        if self.phase == 'val':
            index = 0
        else:
            index = 0  # random.randint(0, 1e6)

        # if in training, and in augmentations mode, generate a pilot word that has all states
        if conf.augmentations != 'reg':
            b = np.random.randint(0, 2, size=(1, self.block_length))
        # if self.phase == 'train' and conf.augmentations != 'reg':
        #     b = self.draw_until_pilot_has_all_states()
        # accumulate words until reaches desired number
        while y_full.shape[0] < self.words:
            # if conf.augmentations == 'reg' or self.phase == 'val':
            #     b = np.random.randint(0, 2, size=(1, self.block_length))
            if conf.augmentations == 'reg':
                b = np.random.randint(0, 2, size=(1, self.block_length))
            # encoding - errors correction code
            c = self.encoding(b).reshape(1, -1)
            # add zero bits
            padded_c = np.concatenate([c, np.zeros([c.shape[0], conf.memory_length])], axis=1)
            # transmit
            h = estimate_channel(conf.memory_length, gamma,
                                 fading=conf.fading_in_channel if self.phase == 'val' else conf.fading_in_decoder,
                                 index=index)
            y = self.transmit(padded_c, h, snr)
            # accumulate
            b_full = np.concatenate((b_full, b), axis=0)
            y_full = np.concatenate((y_full, y), axis=0)
            h_full = np.concatenate((h_full, h), axis=0)
            index += 1

        database.append((b_full, y_full, h_full))

    def draw_until_pilot_has_all_states(self):
        while True:
            b = np.random.randint(0, 2, size=(1, self.block_length))
            gt_states = calculate_states(conf.memory_length, torch.Tensor(b).to(device))
            if len(torch.unique(gt_states)) == 2 ** conf.memory_length:
                return b

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


if __name__ == '__main__':

    phase = 'val'  # 'train','val'

    frames_per_phase = {'train': conf.train_frames, 'val': conf.val_frames}
    block_lengths = {'train': conf.train_block_length, 'val': conf.val_block_length}
    channel_coefficients = {'train': 'time_decay', 'val': conf.channel_coefficients}
    transmission_lengths = {
        'train': conf.train_block_length,
        'val': conf.val_block_length if not conf.use_ecc else conf.val_block_length + 8 * conf.n_symbols}
    channel_dataset_dict = {
        phase: ChannelModelDataset(
            block_length=block_lengths[phase],
            transmission_length=transmission_lengths[phase],
            words=frames_per_phase[phase],
            use_ecc=conf.use_ecc,
            phase=phase,
        )
        for phase in ['train', 'val']}

    channel_dataset = channel_dataset_dict[phase]
    _, _, hs = channel_dataset.__getitem__(snr_list=[conf.train_SNR_start], gamma=conf.gamma)
    for i in range(conf.memory_length):
        plt.plot(hs[:, i].cpu().numpy(), label=f'Tap {i}')
    plt.xlabel('Block Index')
    plt.ylabel('Magnitude')
    plt.legend(loc='upper left')
    plt.show()
