import random
from typing import Tuple

import torch

from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()
HALF = 0.5


class FullKnowledgeAugmenter:
    """
    Full-knowledge augmentations scheme. Assumes the receiver knows the snr and the h coefficients, thus is able to
    generate new random words from this channel
    """

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, h: torch.Tensor, snr: float,
                update_hyper_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        channel_dataset = ChannelModelDataset(block_length=conf.pilot_size, words=1, seed=random.randint(0, 1e8))
        if conf.channel_type == ChannelModes.SISO.name:
            new_transmitted_word, new_received_word = channel_dataset.siso_transmission(h.cpu().numpy(), snr)
        elif conf.channel_type == ChannelModes.MIMO.name:
            new_transmitted_word, new_received_word = channel_dataset.mimo_transmission(h.cpu().numpy(), snr)
        else:
            raise ValueError("No such channel type!!!")
        return torch.Tensor(new_received_word).to(device), torch.Tensor(new_transmitted_word).to(device)
