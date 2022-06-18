from typing import Tuple, List

import torch

from python_code.augmentations.full_knowledge_sampler import FullKnowledgeSampler
from python_code.augmentations.geometric_sampling import GeometricSampler
from python_code.augmentations.negation_augmenter import NegationAugmenter
from python_code.augmentations.random_sampling import RandomSampler
from python_code.augmentations.translation_augmenter import TranslationAugmenter
from python_code.channel.channels_hyperparams import MEMORY_LENGTH, N_USER, N_ANT
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
from python_code.utils.trellis_utils import calculate_siso_states, calculate_mimo_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


def estimate_params(received_words: torch.Tensor, transmitted_words: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    Estimate parameters of centers and stds in the jth step based on the known states of the pilot word.
    :param received_words: float words of channel values
    :param transmitted_words: binary word
    :return: updated centers and stds values
    """
    if conf.channel_type == ChannelModes.SISO.name:
        gt_states = calculate_siso_states(MEMORY_LENGTH, transmitted_words)
        n_states = 2 ** MEMORY_LENGTH
        state_size = 1
    elif conf.channel_type == ChannelModes.MIMO.name:
        gt_states = calculate_mimo_states(N_USER, transmitted_words)
        n_states = 2 ** N_USER
        state_size = N_ANT
    else:
        raise ValueError("No such channel type!!!")

    centers = torch.empty([n_states, state_size]).to(device)
    stds = torch.empty([n_states, state_size]).to(device)
    for state in range(n_states):
        state_ind = (gt_states == state)
        state_received = received_words[state_ind]
        stds[state] = torch.std(state_received, dim=0)
        if state_received.shape[0] > 0:
            centers[state] = torch.mean(state_received, dim=0)
        else:
            centers[state] = 0
    stds[torch.isnan(stds)] = torch.mean(stds[~torch.isnan(stds)])
    return centers, stds, gt_states, n_states, state_size


ALPHA1 = 0.3
ALPHA2 = 0.3


class AugmenterWrapper:

    def __init__(self, augmentations: List[str], fading_in_channel: bool):
        self._augmentations = augmentations
        self._fading_in_channel = fading_in_channel
        self._centers = None
        self._stds = None

    def update_hyperparams(self, received_words: torch.Tensor, transmitted_words: torch.Tensor):
        centers, stds, gt_states, n_states, state_size = estimate_params(received_words, transmitted_words)
        if self._fading_in_channel:
            self._centers, self._stds = self.smooth_parameters(centers, stds)
        else:
            self._centers, self._stds = centers, stds

        self._samplers_dict = {
            'geometric_sampler': GeometricSampler(self._centers, self._stds, n_states, state_size),
            'random_sampler': RandomSampler(received_words, transmitted_words, gt_states),
            'full_knowledge_sampler': FullKnowledgeSampler(),
        }
        self._augmenters_dict = {
            'negation_augmenter': NegationAugmenter(),
            'translation_augmenter': TranslationAugmenter(self._centers, n_states, received_words,
                                                          transmitted_words),
        }
        self._n_states = n_states

    def smooth_parameters(self, cur_centers: torch.Tensor, cur_stds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the parameters via temporal smoothing over a window with parameter alpha
        :param cur_centers: jth step estimated centers
        :param cur_stds:  jth step estimated stds
        :return: smoothed centers and stds vectors
        """

        # self._centers = cur_centers
        if self._centers is not None:
            centers = ALPHA1 * cur_centers + (1 - ALPHA1) * self._centers
        else:
            centers = cur_centers

        if self._stds is not None:
            stds = ALPHA2 * cur_stds + (1 - ALPHA2) * self._stds
        else:
            stds = cur_stds

        return centers, stds

    @property
    def n_states(self) -> int:
        return self._n_states

    def augment_single(self, to_augment_state: int, h: torch.Tensor, snr: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augment the received word using one of the given augmentations methods.
        :param received_word: Tensor of float values
        :param transmitted_word: Ground truth transmitted word
        :param h: float function
        :param snr: signal to noise ratio value
        :return: the augmented received and transmitted pairs
        """
        aug_rx, aug_tx = self._samplers_dict[conf.sampler_type].sample(to_augment_state, h, snr)
        # run through the desired augmentations
        for augmentation_name in self._augmentations:
            augmenter = self._augmenters_dict[augmentation_name]
            aug_rx, aug_tx = augmenter.augment(aug_rx, aug_tx, to_augment_state)
        return aug_rx, aug_tx

    def augment_batch(self, h: torch.Tensor, received_words: torch.Tensor, transmitted_words: torch.Tensor):
        """
        The main augmentation function, used to augment each pilot in the evaluation phase.
        :param h: channel coefficients
        :param received_words: float channel values
        :param transmitted_words: binary transmitted word
        :param total_size: total number of examples to augment
        :param n_repeats: the number of repeats per augmentation
        :param phase: validation phase
        :return: the received and transmitted words
        """
        aug_tx = torch.empty([conf.online_repeats_n, transmitted_words.shape[1]]).to(device)
        aug_rx = torch.empty([conf.online_repeats_n, received_words.shape[1]]).to(device)
        debug = True
        for i in range(aug_tx.shape[0]):
            if i < transmitted_words.shape[0]:
                aug_rx[i], aug_tx[i] = received_words[i], transmitted_words[i]
            else:
                if debug:
                    to_augment_state = calculate_siso_states(MEMORY_LENGTH,
                                                             transmitted_words[i % transmitted_words.shape[0]]).item()
                else:
                    to_augment_state = i % self.n_states
                aug_rx[i], aug_tx[i] = self.augment_single(to_augment_state, h, conf.val_snr)
        return aug_rx, aug_tx
