from typing import Tuple

import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_USER
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
from python_code.utils.trellis_utils import calculate_siso_states, calculate_mimo_states

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

    def sample(self, to_augment_state: int, h: torch.Tensor, snr: float) -> Tuple[torch.Tensor, torch.Tensor]:
        state = -1
        while state != to_augment_state:
            transmitted_word = (torch.rand([1, int(self._n_states ** 0.5)]).to(device) >= 0.5).int()
            # calculate states of transmitted, and copy to variable that will hold the new states for the new transmitted
            if conf.channel_type == ChannelModes.SISO.name:
                state = calculate_siso_states(MEMORY_LENGTH, transmitted_word)
            elif conf.channel_type == ChannelModes.MIMO.name:
                state = calculate_mimo_states(N_USER, transmitted_word)
            else:
                raise ValueError("No such channel type!!!")
        received_word = self._centers[state] + self._stds[state] * torch.randn([1, self._state_size]).to(device)
        return received_word, transmitted_word

    @property
    def centers(self) -> torch.Tensor:
        return self._centers

    @property
    def stds(self) -> torch.Tensor:
        return self._stds
