import pickle as pkl
from random import randint
from typing import Tuple

import numpy as np
import torch

from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes

conf = Config()


def sample_random_mimo_word(new_received_word: torch.Tensor, new_transmitted_word: torch.Tensor,
                            received_word: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if conf.channel_type == ChannelModes.MIMO.name:
        random_sample_ind = randint(a=0, b=received_word.shape[0] - 1)
        new_transmitted_word = new_transmitted_word[random_sample_ind]
        new_received_word = new_received_word[random_sample_ind]
    return new_received_word, new_transmitted_word


def save_pkl(pkls_path: str, array: np.ndarray):
    output = open(pkls_path, 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str):
    output = open(pkls_path, 'rb')
    return pkl.load(output)
