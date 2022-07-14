from typing import Tuple

import torch

from python_code import DEVICE
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ModulationType

conf = Config()


def calculate_error_rates(prediction: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
    """
    Returns the ber,fer and error indices
    """
    prediction = prediction.long()
    target = target.long()
    if conf.modulation_type == ModulationType.QPSK.name:
        first_bit = target % 2
        second_bit = target // 2
        target = torch.cat([first_bit.unsqueeze(-1), second_bit.unsqueeze(-1)], dim=2).transpose(1, 2).reshape(
            2 * first_bit.shape[0], -1)
    bits_acc = torch.mean(torch.eq(prediction, target).float()).item()
    all_bits_sum_vector = torch.sum(torch.abs(prediction - target), 1).long()
    frames_acc = torch.eq(all_bits_sum_vector, torch.LongTensor(1).fill_(0).to(device=DEVICE)).float().mean().item()
    return max([1 - bits_acc, 0.0]), max([1 - frames_acc, 0.0]), torch.nonzero(all_bits_sum_vector,
                                                                               as_tuple=False).reshape(-1)
