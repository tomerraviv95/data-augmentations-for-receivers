import torch
import torch.nn as nn

from python_code import DEVICE
from python_code.utils.trellis_utils import create_transition_table, acs_block

HIDDEN1_SIZE = 75
HIDDEN2_SIZE = 16


class VNETDetector(nn.Module):
    """
    This implements the VA decoder by an NN on each stage
    """

    def __init__(self, n_states: int):

        super(VNETDetector, self).__init__()
        self.n_states = n_states
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(DEVICE)
        self.initialize_dnn()

    def initialize_dnn(self):
        layers = [nn.Linear(1, HIDDEN1_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN1_SIZE, self.n_states)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, y: torch.Tensor, phase: str) -> torch.Tensor:
        """
        The forward pass of the ViterbiNet algorithm
        :param y: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :param snr: channel snr
        :param gamma: channel coefficient
        :returns if in 'train' - the estimated priors [batch_size,transmission_length,n_states]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        # initialize input probabilities
        in_prob = torch.zeros([1, self.n_states]).to(DEVICE)
        priors = self.net(y)

        if phase == 'val':
            detected_word = torch.zeros(y.shape).to(DEVICE)
            for i in range(y.shape[0]):
                # get the lsb of the state
                detected_word[i] = torch.argmin(in_prob, dim=1) % 2
                # run one Viterbi stage
                out_prob = acs_block(in_prob, -priors[i], self.transition_table, self.n_states)
                # update in-probabilities for next layer
                in_prob = out_prob

            return detected_word
        else:
            return priors
