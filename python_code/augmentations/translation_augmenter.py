import math
from collections import defaultdict
from random import randint
from typing import Tuple

import torch

from python_code import DEVICE
from python_code.augmentations.rotation_augmenter import DEG_IN_CIRCLE
from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_USER, MODULATION_NUM_MAPPING
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes, ModulationType
from python_code.utils.trellis_utils import calculate_siso_states, calculate_mimo_states

conf = Config()

TX_MAPPING_DICT = {
    ModulationType.BPSK.name:
        {0: 1,
         1: 0},
    ModulationType.QPSK.name:
        {0: 1,
         1: 3,
         3: 2,
         2: 0}
}

RX_MAPPING_DICT = {
    ModulationType.BPSK.name:
        {0: (-1),
         1: (-1)},
    ModulationType.QPSK.name:
        {0: [-1, 1],
         1: [1, -1],
         3: [-1, 1],
         2: [1, -1]}
}


class TranslationAugmenter:
    """
    The proposed augmentations scheme. Calculates centers and variances for each class as specified in the paper,
    then smooths the estimate via a window running mean with alpha = 0.3
    """

    def __init__(self, centers: torch.Tensor):
        super().__init__()
        self._centers = centers
        self.alpha = 0.5
        self.degrees = list(range(0, DEG_IN_CIRCLE, DEG_IN_CIRCLE // MODULATION_NUM_MAPPING[conf.modulation_type]))

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if conf.channel_type == ChannelModes.SISO.name:
            received_word_state = calculate_siso_states(MEMORY_LENGTH, transmitted_word)[0]
        elif conf.channel_type == ChannelModes.MIMO.name:
            received_word_state = calculate_mimo_states(N_USER, transmitted_word.reshape(1, -1))[0]
        else:
            raise ValueError("No such channel type!!!")

        random_ind = randint(a=1, b=len(self.degrees) - 1)
        new_tx = transmitted_word
        tx_map = TX_MAPPING_DICT[conf.modulation_type]
        rx_map = RX_MAPPING_DICT[conf.modulation_type]
        rx_transformation = torch.ones(received_word.shape).to(DEVICE)
        for i in range(random_ind):
            rx_transformation *= torch.tensor([rx_map[x.item()] for x in new_tx]).to(DEVICE)
            new_tx = torch.tensor([tx_map[x.item()] for x in new_tx]).to(DEVICE)

        if conf.channel_type == ChannelModes.SISO.name:
            new_state = calculate_siso_states(MEMORY_LENGTH, new_tx)[0]
        elif conf.channel_type == ChannelModes.MIMO.name:
            new_state = calculate_mimo_states(N_USER, new_tx.reshape(1, -1))[0]
        else:
            raise ValueError("No such channel type!!!")

        transformed_received = rx_transformation * received_word
        delta = self._centers[new_state.item()] + self._centers[received_word_state.item()]
        new_received_word = self.alpha * delta + transformed_received

        return new_received_word, new_tx

    @property
    def centers(self) -> torch.Tensor:
        return self._centers
