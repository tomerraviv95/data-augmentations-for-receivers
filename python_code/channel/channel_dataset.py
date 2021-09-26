from python_code.channel.channel_estimation import estimate_channel
from python_code.channel.modulator import BPSKModulator
from python_code.channel.channel import ISIAWGNChannel
from python_code.ecc.rs_main import encode
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from numpy.random import mtrand
from typing import Tuple, List
import concurrent.futures
import numpy as np
import itertools
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words and transmitted
    """

    def __init__(self, channel_type: str,
                 block_length: int,
                 transmission_length: int,
                 words: int,
                 memory_length: int,
                 channel_coefficients: str,
                 random: mtrand.RandomState,
                 word_rand_gen: mtrand.RandomState,
                 noisy_est_var: float,
                 fading_taps_type: int,
                 use_ecc: bool,
                 n_symbols: int,
                 fading_in_channel: bool,
                 fading_in_decoder: bool,
                 phase: str,
                 augmentations: str):

        self.block_length = block_length
        self.transmission_length = transmission_length
        self.word_rand_gen = word_rand_gen if word_rand_gen else np.random.RandomState()
        self.random = random if random else np.random.RandomState()
        self.channel_type = channel_type
        self.words = words
        self.memory_length = memory_length
        self.channel_coefficients = channel_coefficients
        self.noisy_est_var = noisy_est_var
        self.fading_taps_type = fading_taps_type
        self.fading_in_channel = fading_in_channel
        self.fading_in_decoder = fading_in_decoder
        self.n_symbols = n_symbols
        self.phase = phase
        self.augmentations = augmentations
        if use_ecc and self.phase == 'val':
            self.encoding = lambda b: encode(b, self.n_symbols)
        else:
            self.encoding = lambda b: b

    def get_snr_data(self, snr: float, gamma: float, database: list):
        if database is None:
            database = []
        b_full = np.empty((0, self.block_length))
        y_full = np.empty((0, self.transmission_length))
        h_full = np.empty((0, self.memory_length))
        if self.phase == 'val':
            index = 0
        else:
            index = 0  # random.randint(0, 1e6)
        # accumulate words until reaches desired number
        # generate word
        if self.augmentations != 'reg':
            b = self.word_rand_gen.randint(0, 2, size=(1, self.block_length))
        while y_full.shape[0] < self.words:
            if self.augmentations == 'reg':
                b = self.word_rand_gen.randint(0, 2, size=(1, self.block_length))
            # encoding - errors correction code
            c = self.encoding(b).reshape(1, -1)
            # add zero bits
            padded_c = np.concatenate([c, np.zeros([c.shape[0], self.memory_length])], axis=1)
            # transmit
            h = estimate_channel(self.memory_length, gamma,
                                 channel_coefficients=self.channel_coefficients,
                                 noisy_est_var=self.noisy_est_var,
                                 fading=self.fading_in_channel if self.phase == 'val' else self.fading_in_decoder,
                                 index=index,
                                 fading_taps_type=self.fading_taps_type)
            y = self.transmit(padded_c, h, snr)
            # accumulate
            b_full = np.concatenate((b_full, b), axis=0)
            y_full = np.concatenate((y_full, y), axis=0)
            h_full = np.concatenate((h_full, h), axis=0)
            index += 1

        database.append((b_full, y_full, h_full))

    def transmit(self, c: np.ndarray, h: np.ndarray, snr: float):
        if self.channel_type == 'ISI_AWGN':
            # modulation
            s = BPSKModulator.modulate(c)
            # transmit through noisy channel
            y = ISIAWGNChannel.transmit(s=s, random=self.random, h=h, snr=snr, memory_length=self.memory_length)
        else:
            raise Exception('No such channel defined!!!')
        return y

    def create_class_mapping(self, h):
        if self.channel_type == 'ISI_AWGN':
            c = np.array(list(itertools.product(range(2), repeat=self.memory_length))).T
            s = BPSKModulator.modulate(c)
            flipped_s = np.fliplr(s)
            classes_centers = ISIAWGNChannel.create_class_mapping(s=flipped_s, h=h.cpu().numpy())
            classes_centers.sort()
        else:
            raise Exception('No such channel defined!!!')
        return torch.Tensor(classes_centers).to(device)

    def map_bits_to_class(self, word, h):
        ## needs doc
        centers = self.create_class_mapping(h)
        decision_boundaries = centers.diff() / 2 + centers[:-1]
        tiled_word = torch.repeat_interleave(word.unsqueeze(-1), repeats=15, dim=-1)
        classes = torch.sum((decision_boundaries <= tiled_word), dim=-1)
        return classes

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
    memory_length = 4
    gamma = 0.5
    noisy_est_var = 0
    channel_coefficients = 'time_decay'  # 'time_decay','cost2100','from_pkl'
    fading_taps_type = 1
    fading = False
    channel_length = 2
    channel_dataset = ChannelModelDataset('ISI_AWGN',
                                          1784,
                                          1784,
                                          channel_length,
                                          memory_length,
                                          channel_coefficients,
                                          np.random.RandomState(10),
                                          np.random.RandomState(10),
                                          noisy_est_var,
                                          fading_taps_type,
                                          False,
                                          memory_length,
                                          fading,
                                          False,
                                          'val',
                                          'reg')

    total_h = np.empty([channel_length, memory_length])
    total_centers = np.empty([channel_length, 2 ** memory_length])
    for index in range(channel_length):
        total_h[index] = estimate_channel(memory_length, gamma, channel_coefficients, noisy_est_var,
                                          fading, index, fading_taps_type)
        h = torch.Tensor(total_h[index].reshape(1, -1))
        total_centers[index] = channel_dataset.create_class_mapping(h).cpu().numpy()

    for i in range(memory_length):
        plt.plot(total_h[:, i], label=f'Tap {i}')
    plt.xlabel('Block Index')
    plt.ylabel('Magnitude')
    plt.legend(loc='upper left')
    plt.show()
