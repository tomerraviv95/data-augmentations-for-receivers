from typing import Tuple

import numpy as np
import smote_variants as sv
import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH
from python_code.utils.config_singleton import Config
from python_code.utils.trellis_utils import calculate_siso_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class SMOTEAugmenter:
    """
    No augmentations class, return the received and transmitted pairs. Implemented for completeness.
    """

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, h: torch.Tensor, snr: float,
                update_hyper_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:

        # generate new word, and populate the received bits by plugging in values from the samples above
        new_transmitted_word = transmitted_word.clone()
        new_received_word = received_word.clone()
        new_gt_states = calculate_siso_states(MEMORY_LENGTH, new_transmitted_word)

        # filter single samples states and received values
        states = calculate_siso_states(MEMORY_LENGTH, transmitted_word)
        unique_states, counts = torch.unique(states, return_counts=True)
        single_sample_states = unique_states[counts == 1]
        for single_sample_state in single_sample_states:
            received_word = received_word[0, states != single_sample_state].reshape(1, -1)
            states = states[states != single_sample_state]

        # oversample with SMOTE
        ros = sv.ANS(random_state=conf.seed)
        received_resampled, states_resampled = ros.fit_resample(received_word.cpu().numpy().T, states.cpu().numpy())
        states_resampled = torch.Tensor(states_resampled).to(device)
        received_resampled = torch.Tensor(received_resampled).to(device)

        # generate new words using the smote interpolated points
        for state in torch.unique(states):
            state_ind = (new_gt_states == state)
            all_state_indices = torch.where(states_resampled == state)
            random_indices = np.random.randint(low=0, high=len(all_state_indices[0]), size=torch.sum(state_ind).item())
            indices = all_state_indices[0][random_indices]
            new_received_word[0, state_ind] = received_resampled[indices].reshape(-1)
        return new_received_word, new_transmitted_word
