import torch
import torch.nn as nn

from python_code.channel.channels_hyperparams import MEMORY_LENGTH
from python_code.utils.trellis_utils import calculate_symbols_from_siso_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 1
NUM_LAYERS = 2
HIDDEN_SIZE = 64


class RNNDetector(nn.Module):
    """
    This class implements an LSTM detector
    """

    def __init__(self):
        super(RNNDetector, self).__init__()
        self.output_size = 2 ** MEMORY_LENGTH
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
        self.linear = nn.Linear(HIDDEN_SIZE, self.output_size).to(device)

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

        # Set initial states
        h_n = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
        c_n = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)

        # Forward propagate LSTM - lstm_out: tensor of shape (seq_length, batch_size, input_size)
        lstm_out, _ = self.lstm(y.unsqueeze(1), (h_n.contiguous(), c_n.contiguous()))

        # Linear layer output
        out = self.linear(lstm_out.squeeze(1))
        if phase == 'val':
            # Decode the output
            estimated_states = torch.argmax(out, dim=1)
            estimated_words = calculate_symbols_from_siso_states(self.output_size, estimated_states)
            return estimated_words[:, 0].reshape(-1, 1).long()
        else:
            return out
