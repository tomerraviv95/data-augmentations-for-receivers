from typing import Tuple

import torch

from python_code.utils.config_singleton import Config

conf = Config()


class NegationAugmenter:
    """
    No augmentations class, return the received and transmitted pairs. Implemented for completeness.
    """

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        return (-1) * received_word, (1 - transmitted_word)
