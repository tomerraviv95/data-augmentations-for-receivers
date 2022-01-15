from python_code.augmentations.augmenter1 import Augmenter1
from python_code.augmentations.augmenter2 import Augmenter2
from python_code.augmentations.augmenter3 import Augmenter3
from typing import Tuple
import torch

names_to_methods_aug_dict = {'aug1': Augmenter1(),
                             'aug2': Augmenter2(),
                             'aug3': Augmenter3()}


class Augmenter:
    def __init__(self):
        self._centers = None

    def augment(self, current_received: torch.Tensor, current_transmitted: torch.Tensor, type: str, h: torch.Tensor,
                snr: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if type == 'reg':
            x, y = current_received, current_transmitted.reshape(1, -1)
        elif type == 'aug1':
            augmenter = names_to_methods_aug_dict[type]
            x, y = augmenter.augment(current_transmitted.reshape(1, -1), h, snr)
        elif type == 'aug2':
            augmenter = names_to_methods_aug_dict[type]
            x, y = augmenter.augment(current_received, current_transmitted.reshape(1, -1), h)
        elif type == 'aug3':
            augmenter = names_to_methods_aug_dict[type]
            x, y = augmenter.augment(current_received, current_transmitted)
        else:
            raise ValueError("No sucn augmentation method!!!")

        return x, y
