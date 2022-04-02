from typing import Tuple

import torch

from python_code.utils.config_singleton import Config
from python_code.utils.python_utils import sample_random_mimo_word

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class NoAugmenter:
    """
    No augmentations class, return the received and transmitted pairs. Implemented for completeness.
    """

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, h: torch.Tensor, snr: float,
                update_hyper_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        new_received_word, new_transmitted_word = received_word, transmitted_word
        new_received_word, new_transmitted_word = sample_random_mimo_word(new_received_word,
                                                                          new_transmitted_word,
                                                                          received_word)
        return new_received_word, new_transmitted_word
