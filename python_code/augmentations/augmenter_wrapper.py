from python_code.augmentations.flipping_augmenter import FlippingAugmenter
from python_code.augmentations.full_knowledge_augmenter import FullKnowledgeAugmenter
from python_code.augmentations.partial_knowledge_augmenter import PartialKnowledgeAugmenter
from python_code.augmentations.adaptive_augmenter import AdaptiveAugmenter
from python_code.augmentations.no_augmenter import NoAugmenter
from typing import Tuple
import torch


class AugmenterWrapper:

    def __init__(self, augmentations: str):
        self._augmentations_dict = {'full_knowledge_augmenter': FullKnowledgeAugmenter,
                                    'partial_knowledge_augmenter': PartialKnowledgeAugmenter,
                                    'adaptive_augmenter': AdaptiveAugmenter,
                                    'flipping_augmenter': FlippingAugmenter,
                                    'no_aug': NoAugmenter}
        self._augmenter = self._augmentations_dict[augmentations]()

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor,
                h: torch.Tensor, snr: float, update_hyper_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augment the received word using one of the given augmentations methods.
        :param received_word: Tensor of float values
        :param transmitted_word: Ground truth transmitted word
        :param h: float function
        :param snr: signal to noise ratio value
        :param update_hyper_params: whether to update the hyper parameters of an augmentation scheme
        :return: the augmented received and transmitted pairs
        """
        x, y = self._augmenter.augment(received_word, transmitted_word.reshape(1, -1), h, snr, update_hyper_params)
        return x, y
