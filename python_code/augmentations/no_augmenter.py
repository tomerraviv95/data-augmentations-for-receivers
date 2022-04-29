from typing import Tuple

import torch

from python_code.utils.config_singleton import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class NoAugmenter:
    """
    No augmentations class, return the received and transmitted pairs. Implemented for completeness.
    """

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, h: torch.Tensor, snr: float) -> \
    Tuple[torch.Tensor, torch.Tensor]:
        new_received_word, new_transmitted_word = received_word, transmitted_word
        return new_received_word, new_transmitted_word
