from typing import Tuple

import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_USER
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
from python_code.utils.trellis_utils import calculate_siso_states, calculate_mimo_states, generate_symbols_by_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class GeometricSampler:
    """
    The proposed augmentations scheme. Calculates centers and variances for each class as specified in the paper,
    then smooths the estimate via a window running mean with alpha = 0.3
    """

    def __init__(self, centers: torch.Tensor, stds: torch.Tensor, n_states: int, state_size: int):
        super().__init__()
        self._centers = centers
        self._stds = stds
        self._n_states = n_states
        self._state_size = state_size

    def sample(self, rx, tx, i, h, snr) -> Tuple[torch.Tensor, torch.Tensor]:

        if conf.channel_type == ChannelModes.SISO.name:
            to_augment_state = calculate_siso_states(MEMORY_LENGTH, tx)
            transmitted_word = generate_symbols_by_state(to_augment_state, MEMORY_LENGTH)
        elif conf.channel_type == ChannelModes.MIMO.name:
            to_augment_state = calculate_mimo_states(N_USER, tx)
            transmitted_word = generate_symbols_by_state(to_augment_state, N_USER)
        else:
            raise ValueError("No such channel type!!!")

        received_word = self._centers[to_augment_state] + self._stds[to_augment_state] * torch.randn(
            [1, self._state_size]).to(device)
        return received_word, transmitted_word

    @property
    def centers(self) -> torch.Tensor:
        return self._centers

    @property
    def stds(self) -> torch.Tensor:
        return self._stds
