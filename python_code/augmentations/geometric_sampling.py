from typing import Tuple

import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_USER
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
from python_code.utils.trellis_utils import generate_symbols_by_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class GeometricSampler:
    """
    The proposed augmentations scheme. Calculates centers and variances for each class as specified in the paper,
    then smooths the estimate via a window running mean with alpha = 0.3
    """

    def __init__(self, centers: torch.Tensor, stds: torch.Tensor, n_states: int, state_size: int,
                 gt_states: torch.Tensor):
        super().__init__()
        self._centers = centers
        self._stds = stds
        self._n_states = n_states
        self._state_size = state_size
        self._gt_states = gt_states

    def sample(self, i: int, h: torch.Tensor, snr: float) -> Tuple[torch.Tensor, torch.Tensor]:

        to_augment_state = self._gt_states[i % self._gt_states.shape[0]]

        if conf.channel_type == ChannelModes.SISO.name:
            transmitted_word = generate_symbols_by_state(to_augment_state, MEMORY_LENGTH)
        elif conf.channel_type == ChannelModes.MIMO.name:
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
