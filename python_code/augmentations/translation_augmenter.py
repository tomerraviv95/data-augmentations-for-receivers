import random
from typing import Tuple

import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_USER
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
from python_code.utils.trellis_utils import calculate_siso_states, calculate_mimo_states, generate_symbols_by_state
from random import randint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class TranslationAugmenter:
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

    def populate_diff_list(self, received_word_states, received_words):
        self._diffs_list = []
        for i in range(len(received_word_states)):
            current_state = received_word_states[i]
            current_diff = received_words[i] - self._centers[current_state]
            self._diffs_list.append(current_diff)

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, to_augment_state: int) -> Tuple[
        torch.Tensor, torch.Tensor]:
        random_ind = randint(a=0, b=len(self._diffs_list) - 1)
        diff = self._diffs_list[random_ind]
        random_state = random.choice(
            list(range(0, to_augment_state)) + list(range(to_augment_state + 1, self._n_states)))
        if conf.channel_type == ChannelModes.SISO.name:
            transmitted_word = generate_symbols_by_state(random_state, MEMORY_LENGTH)
        elif conf.channel_type == ChannelModes.MIMO.name:
            transmitted_word = generate_symbols_by_state(random_state, N_USER)
        else:
            raise ValueError("No such channel type!!!")
        received_word = self._centers[random_state] + diff
        return received_word, transmitted_word

    @property
    def centers(self) -> torch.Tensor:
        return self._centers
