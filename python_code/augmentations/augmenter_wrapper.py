from typing import Tuple, List

import torch

from python_code.augmentations.full_knowledge_augmenter import FullKnowledgeAugmenter
from python_code.augmentations.geometric_augmenter import GeometricAugmenter
from python_code.augmentations.negation_augmenter import NegationAugmenter
from python_code.augmentations.no_augmenter import NoAugmenter
from python_code.augmentations.partial_knowledge_augmenter import PartialKnowledgeAugmenter
from python_code.augmentations.translation_augmenter import TranslationAugmenter
from python_code.utils.python_utils import sample_random_mimo_word


class AugmenterWrapper:

    def __init__(self, augmentations: List[str], received_word, transmitted_word):
        self._augmenters_dict = {'full_knowledge_augmenter': FullKnowledgeAugmenter(),
                                 'partial_knowledge_augmenter': PartialKnowledgeAugmenter(),
                                 'geometric_augmenter': GeometricAugmenter(received_word, transmitted_word),
                                 'negation_augmenter': NegationAugmenter(),
                                 'translation_augmenter': TranslationAugmenter(received_word, transmitted_word),
                                 'no_aug': NoAugmenter()}
        self._augmentations = augmentations

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor,
                h: torch.Tensor, snr: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augment the received word using one of the given augmentations methods.
        :param received_word: Tensor of float values
        :param transmitted_word: Ground truth transmitted word
        :param h: float function
        :param snr: signal to noise ratio value
        :return: the augmented received and transmitted pairs
        """
        x, y = received_word, transmitted_word
        for augmentation_name in self._augmentations:
            augmenter = self._augmenters_dict[augmentation_name]
            x, y = augmenter.augment(x, y, h, snr)
        new_received_word, new_transmitted_word = sample_random_mimo_word(x, y, x)
        return new_received_word, new_transmitted_word
