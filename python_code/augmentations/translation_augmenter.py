import math
from collections import defaultdict
from random import randint, uniform
from typing import Tuple

import torch

from python_code import DEVICE
from python_code.augmentations.rotation_augmenter import DEG_IN_CIRCLE
from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_USER, MODULATION_NUM_MAPPING
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes, ModulationType
from python_code.utils.trellis_utils import calculate_siso_states, calculate_mimo_states

conf = Config()

MAPPING_DICT = {
    ModulationType.BPSK.name:
        {0: 1,
         1: 0},
    ModulationType.QPSK.name:
        {0: 1,
         1: 3,
         3: 2,
         2: 0}
}

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
        self.alpha = 0.8

        deg_list = list(range(0, DEG_IN_CIRCLE, DEG_IN_CIRCLE // MODULATION_NUM_MAPPING[conf.modulation_type]))
        rad_list = [math.radians(degree) for degree in deg_list]
        self.degrees = torch.Tensor(rad_list).to(DEVICE)

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

        random_ind = randint(a=1, b=len(self.degrees) - 1)
        new_tx = transmitted_word
        map = MAPPING_DICT[conf.modulation_type]
        for i in range(random_ind):
            new_tx = torch.tensor([map[x.item()] for x in new_tx])

        current_diff = received_word - self._centers[received_word_state]
        # if received_word_state.item() in self._diffs_dict.keys():
        #     diffs_for_state = self._diffs_dict[received_word_state.item()]
        # else:
        #     diffs_for_state = torch.zeros_like(current_diff).to(DEVICE)
        # random_ind = randint(a=0, b=len(diffs_for_state) - 1)
        # sampled_diff = diffs_for_state[random_ind]
        new_received_word = self._centers[
                                received_word_state] + self.alpha * current_diff  # + (1 - self.alpha) * sampled_diff
        return new_received_word, transmitted_word

    @property
    def centers(self) -> torch.Tensor:
        return self._centers
