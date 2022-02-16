from python_code.augmentations.full_knowledge_augmenter import FullKnowledgeAugmenter
from python_code.augmentations.partial_knowledge_augmenter import PartialKnowledgeAugmenter
from python_code.augmentations.self_supervised_augmenter import SelfSupervisedAugmenter
from python_code.augmentations.no_augmenter import NoAugmenter
from typing import Tuple
import torch


class AugmenterWrapper:

    def __init__(self, augmentations: str):
        self._augmentations_dict = {'full_knowledge_augmenter': FullKnowledgeAugmenter,
                                    'partial_knowledge_augmenter': PartialKnowledgeAugmenter,
                                    'self_supervised_augmenter': SelfSupervisedAugmenter,
                                    'no_aug': NoAugmenter}
        self._augmenter = self._augmentations_dict[augmentations]()

    def augment(self, received_word: torch.Tensor, current_transmitted: torch.Tensor,
                h: torch.Tensor, snr: float, update_hyper_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self._augmenter.augment(received_word, current_transmitted.reshape(1, -1), h, snr, update_hyper_params)
        return x, y
