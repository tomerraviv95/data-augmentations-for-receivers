import random
from typing import Tuple

import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_USER
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
from python_code.utils.trellis_utils import calculate_siso_states, calculate_mimo_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class TranslationAugmenter:
    """
    The proposed augmentations scheme. Calculates centers and variances for each class as specified in the paper,
    then smooths the estimate via a window running mean with alpha = 0.3
    """

    def __init__(self, centers: torch.Tensor, n_states: int):
        super().__init__()
        self._centers = centers
        self._n_states = n_states

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, to_augment_state: int) -> Tuple[
        torch.Tensor, torch.Tensor]:
        diff_from_centers = received_word - self._centers[to_augment_state]
        random_state = random.choice(
            list(range(0, to_augment_state)) + list(range(to_augment_state + 1, self._n_states)))
        state = -1
        while state != random_state:
            transmitted_word = (torch.rand([1, int(self._n_states ** 0.5)]).to(device) >= 0.5).int()
            # calculate states of transmitted, and copy to variable that will hold the new states for the new transmitted
            if conf.channel_type == ChannelModes.SISO.name:
                state = calculate_siso_states(MEMORY_LENGTH, transmitted_word)
            elif conf.channel_type == ChannelModes.MIMO.name:
                state = calculate_mimo_states(N_USER, transmitted_word)
            else:
                raise ValueError("No such channel type!!!")
        received_word = self._centers[random_state] + diff_from_centers
        return received_word, transmitted_word

    @property
    def centers(self) -> torch.Tensor:
        return self._centers
