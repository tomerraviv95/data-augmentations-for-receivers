from python_code.augmentations.augmenter1 import Augmenter1
from python_code.augmentations.augmenter2 import Augmenter2
from python_code.augmentations.augmenter3 import Augmenter3
from python_code.augmentations.reg_augmenter import RegAugmenter
from typing import Tuple
import torch


class AugmenterWrapper:

    def __init__(self, augmentations: str):
        self._augmentations_dict = {'aug1': Augmenter1,
                                    'aug2': Augmenter2,
                                    'aug3': Augmenter3,
                                    'reg': RegAugmenter}
        self._augmenter = self._augmentations_dict[augmentations]()

    def augment(self, received_word: torch.Tensor, current_transmitted: torch.Tensor,
                h: torch.Tensor, snr: float, update_hyper_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self._augmenter.augment(received_word, current_transmitted.reshape(1, -1), h, snr, update_hyper_params)
        return x, y
