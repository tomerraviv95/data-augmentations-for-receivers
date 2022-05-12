from typing import Tuple, List

import torch

from python_code.augmentations.full_knowledge_sampler import FullKnowledgeSampler
from python_code.augmentations.geometric_sampling import GeometricSampler
from python_code.augmentations.negation_augmenter import NegationAugmenter
from python_code.augmentations.no_augmenter import NoAugmenter
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


class AugmenterWrapper:

    def __init__(self, augmentations: List[str], received_words: torch.Tensor,
                 transmitted_words: torch.Tensor):
        centers, stds, gt_states, n_states, state_size = estimate_params(received_words, transmitted_words)
        self._samplers_dict = {
            'geometric_sampler': GeometricSampler(centers, stds, n_states, state_size),
            'random_sampler': RandomSampler(received_words, transmitted_words, gt_states),
            'full_knowledge_sampler': FullKnowledgeSampler(),
        }
        self._augmenters_dict = {
            'negation_augmenter': NegationAugmenter(),
            'translation_augmenter': TranslationAugmenter(centers),
            'no_aug': NoAugmenter()
        }
        self._augmentations = augmentations
        self._n_states = n_states

    @property
    def n_states(self) -> int:
        return self._n_states

    def augment(self, to_augment_state: int, h: torch.Tensor, snr: float) -> Tuple[torch.Tensor, torch.Tensor]:
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
            if augmentation_name == 'geometric_sampler':
                continue
            augmenter = self._augmenters_dict[augmentation_name]
            aug_rx, aug_tx = augmenter.augment(aug_rx, aug_tx, h, snr)
        return aug_rx, aug_tx
