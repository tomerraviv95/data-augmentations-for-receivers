from typing import Tuple

import numpy as np
import torch
from imblearn.over_sampling import BorderlineSMOTE

from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_USER
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
from python_code.utils.python_utils import sample_random_mimo_word
from python_code.utils.trellis_utils import calculate_states, calculate_mimo_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class BorderSMOTEAugmenter:
    """
    No augmentations class, return the received and transmitted pairs. Implemented for completeness.
    """

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, h: torch.Tensor, snr: float,
                update_hyper_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:

        # generate new word, and populate the received bits by plugging in values from the samples above
        new_transmitted_word = transmitted_word.clone()
        new_received_word = received_word.clone()

        # calculate states of transmitted, and copy to variable that will hold the new states for the new transmitted
        if conf.channel_type == ChannelModes.SISO.name:
            states = calculate_states(MEMORY_LENGTH, transmitted_word)
            new_gt_states = states.clone()
        elif conf.channel_type == ChannelModes.MIMO.name:
            states = calculate_mimo_states(N_USER, transmitted_word)
            new_gt_states = states.clone()
        else:
            raise ValueError("No such channel type!!!")

        # filter single samples states and received values
        unique_states, counts = torch.unique(states, return_counts=True)
        single_sample_states = unique_states[counts == 1]
        for single_sample_state in single_sample_states:
            if conf.channel_type == ChannelModes.SISO.name:
                received_word = received_word[0, states != single_sample_state].reshape(1, -1)
            elif conf.channel_type == ChannelModes.MIMO.name:
                received_word = received_word[states != single_sample_state]
            else:
                raise ValueError("No such channel type!!!")
            states = states[states != single_sample_state]

        if conf.channel_type == ChannelModes.SISO.name:
            received_word = received_word.T

        # oversample with SMOTE
        ros = BorderlineSMOTE(random_state=conf.seed, k_neighbors=1)
        received_resampled, states_resampled = ros.fit_resample(received_word.cpu().numpy(), states.cpu().numpy())
        states_resampled = torch.Tensor(states_resampled).to(device)
        received_resampled = torch.Tensor(received_resampled).to(device)

        # generate new words using the smote interpolated points
        for state in torch.unique(states):
            state_ind = (new_gt_states == state)
            all_state_indices = torch.where(states_resampled == state)
            random_indices = np.random.randint(low=0, high=len(all_state_indices[0]), size=torch.sum(state_ind).item())
            indices = all_state_indices[0][random_indices]
            if conf.channel_type == ChannelModes.SISO.name:
                new_received_word[0, state_ind] = received_resampled[indices].reshape(-1)
            elif conf.channel_type == ChannelModes.MIMO.name:
                new_received_word[state_ind] = received_resampled[indices]
            else:
                raise ValueError("No such channel type!!!")

        new_received_word, new_transmitted_word = sample_random_mimo_word(new_received_word,
                                                                          new_transmitted_word,
                                                                          received_word)
        return new_received_word, new_transmitted_word
