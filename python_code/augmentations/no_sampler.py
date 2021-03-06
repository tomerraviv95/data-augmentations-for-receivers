from random import randint
from typing import Tuple

import torch

from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes

conf = Config()


class NoSampler:
    """
    No sampling approach. Return the sample by index / randomly.
    """

    def __init__(self, received_words: torch.Tensor, transmitted_words: torch.Tensor):
        super().__init__()
        self._received_words = received_words
        self._transmitted_words = transmitted_words

    def sample(self, i: int, h: torch.Tensor, snr: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if conf.channel_type == ChannelModes.SISO.name:
            ind = i % self._received_words.shape[0]
        elif conf.channel_type == ChannelModes.MIMO.name:
            ind = randint(a=0, b=self._received_words.shape[0] - 1)
        else:
            raise ValueError("No such channel type!!!")
        return self._received_words[ind], self._transmitted_words[ind]
