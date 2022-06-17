import torch
import torch.nn as nn
from math import log2

from python_code.utils.trellis_utils import calculate_symbols_from_siso_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES_NUM = 16
NUM_LAYERS = 2
INPUT_SIZE = 1
HIDDEN_SIZE = 8
MEMORY_LENGTH = 4


class RNNDetector(nn.Module):
    """
    This class implements an LSTM detector
    """

    def __init__(self):
        super(RNNDetector, self).__init__()
        self.output_size = CLASSES_NUM
        self.initialize_dnn()

    def initialize_dnn(self):
        layers = [nn.Linear(1, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, self.output_size)]
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, y: torch.Tensor, phase: str, snr: float = None, gamma: float = None,
                count: int = None) -> torch.Tensor:
        """
        The forward pass of the LSTM detector
        :param y: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :param snr: channel snr
        :param gamma: channel coefficient
        :return: if in 'train' - the estimated bitwise prob [batch_size,transmission_length,N_CLASSES]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        out = self.net(y)
        if phase == 'val':
            # Decode the output
            estimated_states = torch.argmax(out, dim=1)
            estimated_words = calculate_symbols_from_siso_states(MEMORY_LENGTH, estimated_states)
            return estimated_words[:, -1].reshape(-1, 1).long()
        else:
            return out
