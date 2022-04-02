from python_code.channel.channels_hyperparams import MEMORY_LENGTH
from python_code.channel.isi_awgn_channel import ISIAWGNChannel
from python_code.channel.modulator import BPSKModulator
from python_code.channel.sed_channel import SEDChannel
from python_code.utils.config_singleton import Config
from typing import Tuple
import numpy as np
import torch

from python_code.utils.constants import ChannelModes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()

HALF = 0.5


class PartialKnowledgeAugmenter:
    """
    Partial-knowledge augmentations scheme. Assumes the receiver knows the h coefficients but not the snr.
    This means it is able to generate new binary words and pass them through the convolution using the h,
    but with the noise estimated from the received and transmitted pairs.
    """

    def augment(self, received_word: torch.Tensor, transmitted_word: torch.Tensor, h: torch.Tensor, snr: float,
                update_hyper_params: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if conf.channel_type == ChannelModes.SISO.name:
            # add zero bits
            padded_b = np.concatenate(
                [transmitted_word.cpu().numpy(), np.zeros([transmitted_word.shape[0], MEMORY_LENGTH])], axis=1)
            # modulation
            s = BPSKModulator.modulate(padded_b)
            # compute convolution
            conv = ISIAWGNChannel.compute_channel_signal_convolution(h.cpu().numpy(), MEMORY_LENGTH, s)
            # estimate noise as difference between received and transmitted symbols words
            w_est = received_word.cpu().numpy() - conv

            # generate a random transmitted word
            new_transmitted_word = torch.rand_like(transmitted_word) >= HALF
            # add zero bits
            new_padded_b = np.concatenate(
                [new_transmitted_word.cpu().numpy(), np.zeros([new_transmitted_word.shape[0], MEMORY_LENGTH])], axis=1)
            # modulation
            new_s = BPSKModulator.modulate(new_padded_b)
            # compute convolution
            new_conv = ISIAWGNChannel.compute_channel_signal_convolution(h.cpu().numpy(), MEMORY_LENGTH, new_s)
            # estimate new received word using the above noise
            new_received_word = new_conv + w_est
        elif conf.channel_type == ChannelModes.MIMO.name:
            # modulation
            s = BPSKModulator.modulate(transmitted_word.cpu().numpy().T)
            # compute convolution
            conv = SEDChannel.compute_channel_signal_convolution(h.cpu().numpy(), s).T
            # estimate noise as difference between received and transmitted symbols words
            w_est = np.mean(received_word.cpu().numpy() - conv, axis=0).reshape(1, -1)

            # generate a random transmitted word
            new_transmitted_word = torch.rand([1,transmitted_word.shape[1]]) >= HALF
            # modulation
            new_s = BPSKModulator.modulate(new_transmitted_word.cpu().numpy().T)
            # compute convolution
            new_conv = SEDChannel.compute_channel_signal_convolution(h.cpu().numpy(), new_s).T
            # estimate new received word using the above noise
            new_received_word = new_conv + w_est
        else:
            raise ValueError("No such channel type!!!")
        return torch.Tensor(new_received_word).to(device), new_transmitted_word.int()
