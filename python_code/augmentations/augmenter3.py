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
        self._alpha = 0.3

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, h: torch.Tensor, snr: float,
                update_hyper_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if update_hyper_params:
            # first calculate estimated noise pattern
            cur_centers, cur_stds = self.estimate_cur_params(received_word, transmitted_word)
            # average the current centers & stds estimates with previous estimates to reduce noise
            self.update_centers_stds(cur_centers, cur_stds)

        new_transmitted_word = torch.rand_like(transmitted_word) >= 0.5
        new_gt_states = calculate_states(conf.memory_length, new_transmitted_word)
        new_received_word = torch.empty_like(received_word)
        for state in torch.unique(new_gt_states):
            state_ind = (new_gt_states == state)
            new_received_word[0, state_ind] = self._centers[state] + self._stds[state] * \
                                              torch.randn_like(transmitted_word)[
                                                  0, state_ind]
        return new_received_word, new_transmitted_word

    def update_centers_stds(self, cur_centers, cur_stds):
        self._centers = cur_centers
        if self._stds is not None:
            self._stds = self._alpha * cur_stds + (1 - self._alpha) * self._stds
        else:
            self._stds = cur_stds

    def estimate_cur_params(self, received_word, transmitted_word):
        gt_states = calculate_states(conf.memory_length, transmitted_word)
        centers = torch.empty(2 ** conf.memory_length).to(device)
        stds = torch.empty(2 ** conf.memory_length).to(device)
        for state in torch.unique(gt_states):
            state_ind = (gt_states == state)
            state_received = received_word[0, state_ind]
            centers[state] = torch.mean(state_received)
            stds[state] = torch.std(state_received)
        stds[torch.isnan(stds)] = torch.mean(stds[~torch.isnan(stds)])
        return centers, stds

    @property
    def centers(self) -> torch.Tensor:
        return self._centers

    @property
    def stds(self) -> torch.Tensor:
        return self._stds
