from random import randint
from typing import Tuple

import torch

from python_code import DEVICE
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.utils.config_singleton import Config

conf = Config()
FK_KNOWLEDGE_BUFFER = int(1e5)


class FullKnowledgeSampler:
    """
    Full-knowledge augmentations scheme. Assumes the receiver knows the snr and the h coefficients, thus is able to
    generate new random words from this channel
    """

    def sample(self, i: int, h: torch.Tensor, snr: float) -> Tuple[torch.Tensor, torch.Tensor]:
        channel_dataset = ChannelModelDataset(block_length=FK_KNOWLEDGE_BUFFER,
                                              pilots_length=FK_KNOWLEDGE_BUFFER,
                                              blocks_num=1)
        transmitted_word, received_word = channel_dataset.channel_type._transmit(h.cpu().numpy(), snr)
        random_ind = randint(a=0, b=transmitted_word.shape[0] - 1)
        return torch.Tensor(received_word).to(DEVICE)[random_ind], torch.Tensor(transmitted_word).to(DEVICE)[random_ind]
