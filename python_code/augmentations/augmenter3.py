from python_code.utils.trellis_utils import calculate_states
from python_code.utils.config_singleton import Config
from typing import Tuple
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class Augmenter3:
    def __init__(self):
        super().__init__()
        self._centers = None
        self._stds = None

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, h: torch.Tensor, snr: float) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        # first calculate estimated noise pattern
        gt_states = calculate_states(conf.memory_length, transmitted_word)
        self._centers = torch.empty(2 ** conf.memory_length).to(device)
        self._stds = torch.empty(2 ** conf.memory_length).to(device)
        for state in torch.unique(gt_states):
            state_ind = (gt_states == state)
            state_received = received_word[0, state_ind]
            self._centers[state] = torch.mean(state_received)
            self._stds[state] = torch.std(state_received)
        self._stds[torch.isnan(self._stds)] = torch.mean(self._stds[~torch.isnan(self._stds)])
        new_transmitted_word = torch.rand_like(transmitted_word) >= 0.5
        new_gt_states = calculate_states(conf.memory_length, new_transmitted_word)
        new_received_word = torch.empty_like(received_word)
        for state in torch.unique(new_gt_states):
            state_ind = (new_gt_states == state)
            new_received_word[0, state_ind] = self._centers[state] + self._stds[state] * \
                                              torch.randn_like(transmitted_word)[
                                                  0, state_ind]
        return new_received_word, new_transmitted_word

    @property
    def centers(self) -> torch.Tensor:
        return self._centers

    @property
    def stds(self) -> torch.Tensor:
        return self._stds
