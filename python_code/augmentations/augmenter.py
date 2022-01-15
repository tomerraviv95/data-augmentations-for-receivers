from python_code.augmentations.augmenter1 import Augmenter1
from python_code.augmentations.augmenter2 import Augmenter2
from python_code.augmentations.augmenter3 import Augmenter3
from python_code.augmentations.reg_augmenter import RegAugmenter
from typing import Tuple, Dict
import torch


class Augmenter:

    def __init__(self):
        self._centers = None
        self._stds = None
        self._augmentations_dict = {'aug1': Augmenter1,
                                    'aug2': Augmenter2,
                                    'aug3': Augmenter3,
                                    'reg': RegAugmenter}

    def augment(self, augmentations: str, received_word: torch.Tensor, current_transmitted: torch.Tensor,
                h: torch.Tensor,
                snr: float) -> Tuple[torch.Tensor, torch.Tensor]:
        augmenter = self._augmentations_dict[augmentations]()
        x, y = augmenter.augment(received_word, current_transmitted.reshape(1, -1), h, snr)
        return x, y

    @property
    def centers(self) -> torch.Tensor:
        return self._centers

    @property
    def stds(self) -> torch.Tensor:
        return self._stds
