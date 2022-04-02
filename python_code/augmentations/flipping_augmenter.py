from python_code.utils.config_singleton import Config
from random import randint
from typing import Tuple
import torch

from python_code.utils.python_utils import sample_random_mimo_word

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class FlippingAugmenter:
    """
    No augmentations class, return the received and transmitted pairs. Implemented for completeness.
    """

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, h: torch.Tensor, snr: float,
                update_hyper_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        random_ind = randint(a=0, b=1)
        new_transmitted_word = (1 - random_ind) * transmitted_word + random_ind * (1 - transmitted_word)
        new_received_word = (-1) ** random_ind * received_word
        new_received_word, new_transmitted_word = sample_random_mimo_word(new_received_word,
                                                                          new_transmitted_word,
                                                                          received_word)
        return new_received_word, new_transmitted_word
