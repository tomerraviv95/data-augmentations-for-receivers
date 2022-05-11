from typing import Tuple

import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_USER, N_ANT
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
from python_code.utils.trellis_utils import calculate_siso_states, calculate_mimo_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()

DEBUG = True


class TranslationAugmenter:
    """
    The proposed augmentations scheme. Calculates centers and variances for each class as specified in the paper,
    then smooths the estimate via a window running mean with alpha = 0.3
    """

    def __init__(self, received_word: torch.Tensor, transmitted_word: torch.Tensor):
        super().__init__()
        self._centers = self.estimate_params(received_word, transmitted_word)

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, h: torch.Tensor, snr: float) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        new_transmitted_word = (torch.rand(transmitted_word.shape).to(device) >= 0.5).int()
        # calculate states of transmitted, and copy to variable that will hold the new states for the new transmitted
        if conf.channel_type == ChannelModes.SISO.name:
            gt_states = calculate_siso_states(MEMORY_LENGTH, transmitted_word)
            new_gt_states = calculate_siso_states(MEMORY_LENGTH, new_transmitted_word)
        elif conf.channel_type == ChannelModes.MIMO.name:
            gt_states = calculate_mimo_states(N_USER, transmitted_word)
            new_gt_states = calculate_mimo_states(N_USER, new_transmitted_word)
        else:
            raise ValueError("No such channel type!!!")
        # calculate diffs from centers for the original word
        diffs_from_centers = torch.empty_like(received_word)
        for state in torch.unique(gt_states):
            state_ind = (gt_states == state)
            diffs_from_centers[state_ind] = received_word[state_ind] - self._centers[state]

        new_received_word = torch.empty_like(received_word)
        # generate new words using the diffs centers
        for state in torch.unique(new_gt_states):
            state_ind = (new_gt_states == state)
            new_received_word[state_ind] = self._centers[state] + diffs_from_centers[state_ind]

        return new_received_word, new_transmitted_word

    def estimate_params(self, received_word: torch.Tensor, transmitted_word: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Estimate parameters of centers and stds in the jth step based on the known states of the pilot word.
        :param received_word: float words of channel values
        :param transmitted_word: binary word
        :return: updated centers and stds values
        """
        if conf.channel_type == ChannelModes.SISO.name:
            gt_states = calculate_siso_states(MEMORY_LENGTH, transmitted_word)
            n_states = 2 ** MEMORY_LENGTH
            state_size = 1
        elif conf.channel_type == ChannelModes.MIMO.name:
            gt_states = calculate_mimo_states(N_USER, transmitted_word)
            n_states = 2 ** N_USER
            state_size = N_ANT
        else:
            raise ValueError("No such channel type!!!")

        centers = torch.empty([n_states, state_size]).to(device)
        stds = torch.empty([n_states, state_size]).to(device)
        for state in range(n_states):
            state_ind = (gt_states == state)
            state_received = received_word[state_ind]
            stds[state] = torch.std(state_received, dim=0)
            if state_received.shape[0] > 0:
                centers[state] = torch.mean(state_received, dim=0)
            else:
                centers[state] = 0
        return centers

    @property
    def centers(self) -> torch.Tensor:
        return self._centers
