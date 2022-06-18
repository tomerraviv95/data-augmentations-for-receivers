from random import randint
from typing import Tuple

import torch

from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.utils.config_singleton import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        transmitted_word, received_word = channel_dataset.channel_model.transmit(h.cpu().numpy(), snr)
        random_ind = randint(a=0, b=transmitted_word.shape[0] - 1)
        return torch.Tensor(received_word).to(device)[random_ind], torch.Tensor(transmitted_word).to(device)[random_ind]
