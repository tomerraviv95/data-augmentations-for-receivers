import math
from random import randint
from typing import Tuple

import torch

from python_code import DEVICE
from python_code.channel.channels_hyperparams import MODULATION_NUM_MAPPING
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ModulationType

conf = Config()

DEG_IN_CIRCLE = 360

MAPPING_DICT = {
    ModulationType.BPSK.name:
        {0: 1,
         1: 0},
    ModulationType.QPSK.name:
        {0: 1,
         1: 3,
         3: 2,
         2: 0}
}


class RotationAugmenter:
    """
    Rotations Augmentations.
    """

    def __init__(self):
        ## creating the rotation-preserving transformations
        deg_list = list(range(0, DEG_IN_CIRCLE, DEG_IN_CIRCLE // MODULATION_NUM_MAPPING[conf.modulation_type]))
        rad_list = [math.radians(degree) for degree in deg_list]
        self.degrees = torch.Tensor(rad_list).to(DEVICE)

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        random_ind = randint(a=1, b=len(self.degrees) - 1)
        chosen_transformation = self.degrees[random_ind]
        if conf.modulation_type == ModulationType.BPSK.name:
            received_word = torch.cat([received_word.unsqueeze(-1), torch.zeros_like(received_word.unsqueeze(-1))],
                                      dim=1)
        new_angle = torch.view_as_complex(received_word).angle() + chosen_transformation
        new_complex_rx = torch.view_as_complex(received_word).abs() * (torch.cos(new_angle) + 1j * torch.sin(new_angle))
        new_rx = torch.view_as_real(new_complex_rx)

        new_tx = transmitted_word
        map = MAPPING_DICT[conf.modulation_type]
        for i in range(random_ind):
            new_tx = torch.tensor([map[x.item()] for x in new_tx])

        if conf.modulation_type == ModulationType.BPSK.name:
            new_rx = new_rx[:, 0]
        return new_rx, new_tx.to(DEVICE)
