from typing import Tuple

import torch

from python_code.utils.config_singleton import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class NoAugSampler:
    """
    The proposed augmentations scheme. Calculates centers and variances for each class as specified in the paper,
    then smooths the estimate via a window running mean with alpha = 0.3
    """

    def __init__(self):
        super().__init__()

    def sample(self, rx, tx, i, h, snr) -> Tuple[torch.Tensor, torch.Tensor]:
        return rx, tx
