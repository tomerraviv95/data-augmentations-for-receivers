from python_code.channel.channels_hyperparams import MEMORY_LENGTH
from python_code.channel.modulator import BPSKModulator
from python_code.utils.config_singleton import Config
from typing import Tuple
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()

HALF = 0.5


class PartialKnowledgeAugmenter:
    """
    Partial-knowledge augmentations scheme. Assumes the receiver knows the h coefficients but not the snr.
    This means it is able to generate new binary words and pass them through the convolution using the h,
    but with the noise estimated from the received and transmitted pairs.
    """

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, h: torch.Tensor, snr: float,
                update_hyper_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        h = h.cpu().numpy()
        #### first calculate estimated noise pattern
        c = transmitted_word.cpu().numpy()
        # add zero bits
        padded_c = np.concatenate([c, np.zeros([c.shape[0], MEMORY_LENGTH])], axis=1)
        # from channel dataset
        s = BPSKModulator.modulate(padded_c)
        blockwise_s = np.concatenate([s[:, i:-MEMORY_LENGTH + i] for i in range(conf.memory_length)],
                                     axis=0)
        trans_conv = np.dot(h[:, ::-1], blockwise_s)
        w_est = received_word.cpu().numpy() - trans_conv

        ### use the noise and add it to a new word
        binary_mask = torch.rand_like(transmitted_word) >= HALF
        new_transmitted_word = (transmitted_word + binary_mask) % 2
        # encoding - errors correction code
        c = new_transmitted_word.cpu().numpy()
        # add zero bits
        padded_c = np.concatenate([c, np.zeros([c.shape[0], MEMORY_LENGTH])], axis=1)
        # from channel dataset
        s = BPSKModulator.modulate(padded_c)
        blockwise_s = np.concatenate([s[:, i:-MEMORY_LENGTH + i] for i in range(conf.memory_length)],
                                     axis=0)
        new_trans_conv = np.dot(h[:, ::-1], blockwise_s)
        new_received_word = new_trans_conv + w_est
        return torch.Tensor(new_received_word).to(device), new_transmitted_word
