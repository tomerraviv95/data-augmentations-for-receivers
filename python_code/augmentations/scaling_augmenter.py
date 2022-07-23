from collections import defaultdict
from typing import Tuple

import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_USER
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
from python_code.utils.trellis_utils import calculate_siso_states, calculate_mimo_states

conf = Config()


class ScalingAugmenter:
    """
    The proposed augmentations scheme. Calculates centers and variances for each class as specified in the paper,
    then smooths the estimate via a window running mean with alpha = 0.3
    """

    def __init__(self, centers: torch.Tensor, n_states: int, received_words: torch.Tensor,
                 transmitted_words: torch.Tensor):
        super().__init__()
        self._centers = centers
        self._n_states = n_states
        if conf.channel_type == ChannelModes.SISO.name:
            received_word_states = calculate_siso_states(MEMORY_LENGTH, transmitted_words)
        elif conf.channel_type == ChannelModes.MIMO.name:
            received_word_states = calculate_mimo_states(N_USER, transmitted_words)
        else:
            raise ValueError("No such channel type!!!")
        self.populate_diff_list(received_word_states, received_words)
        self.alpha = 0.8

    def populate_diff_list(self, received_word_states, received_words):
        self._diffs_dict = defaultdict(list)
        for i in range(len(received_word_states)):
            current_state = received_word_states[i]
            current_diff = received_words[i] - self._centers[current_state]
            self._diffs_dict[current_state.item()].append(current_diff)

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if conf.channel_type == ChannelModes.SISO.name:
            received_word_state = calculate_siso_states(MEMORY_LENGTH, transmitted_word)[0]
        elif conf.channel_type == ChannelModes.MIMO.name:
            received_word_state = calculate_mimo_states(N_USER, transmitted_word.reshape(1, -1))[0]
        else:
            raise ValueError("No such channel type!!!")
        current_diff = received_word - self._centers[received_word_state]
        new_received_word = self._centers[received_word_state] + self.alpha * current_diff
        return new_received_word, transmitted_word

    @property
    def centers(self) -> torch.Tensor:
        return self._centers
