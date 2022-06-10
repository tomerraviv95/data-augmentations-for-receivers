import torch
import torch.nn as nn
from math import log2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 4
NUM_LAYERS = 2
START_VALUE_PADDING = -100


class RNNDetector(nn.Module):
    """
    This class implements an LSTM detector
    """

    def __init__(self, n_states):
        super(RNNDetector, self).__init__()
        self.input_size = int(log2(n_states))
        self.output_size = n_states
        self.lstm = nn.LSTM(self.input_size, self.output_size, NUM_LAYERS).to(
            device)

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
        transmission_length = y.size(0)

        # Set initial states
        h_n = torch.zeros(NUM_LAYERS, 1, self.output_size).to(device)
        c_n = torch.zeros(NUM_LAYERS, 1, self.output_size).to(device)

        # pad and reshape y to the proper shape - (batch_size,seq_length,input_size)
        padded_y = torch.cat([torch.ones([self.input_size - 1, 1]).to(device), y])
        sequence_y = torch.cat([padded_y[i:-self.input_size + i] for i in range(self.input_size)], dim=1)

        # Forward propagate LSTM - lstm_out: tensor of shape (batch_size, seq_length, hidden_size*2)
        lstm_out = torch.zeros(batch_size, transmission_length, HIDDEN_SIZE).to(device)
        for i in range(batch_size):
            lstm_out[i], _ = self.lstm(sequence_y[i].unsqueeze(0),
                                       (h_n[:, i].unsqueeze(1).contiguous(), c_n[:, i].unsqueeze(1).contiguous()))

        # out: tensor of shape (batch_size, seq_length, N_CLASSES)
        out = self.fc(lstm_out.reshape(-1, HIDDEN_SIZE)).reshape(transmission_length, self.output_size)

        if phase == 'val':
            # Decode the output
            return torch.argmax(out, dim=1).float()
        else:
            return out
