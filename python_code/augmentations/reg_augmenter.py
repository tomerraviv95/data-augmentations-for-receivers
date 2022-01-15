from python_code.channel.channel import ISIAWGNChannel
from python_code.channel.modulator import BPSKModulator
from python_code.utils.config_singleton import Config
from typing import Tuple
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class RegAugmenter:
    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, h: torch.Tensor, snr: float,
                update_hyper_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        return received_word, transmitted_word
