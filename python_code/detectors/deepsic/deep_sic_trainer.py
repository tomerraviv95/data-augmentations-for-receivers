from python_code.channel.channels_hyperparams import N_ANT, N_USER
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase, HALF
from typing import List
from torch import nn
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()
ITERATIONS = 5


def symbol_to_prob(s: torch.Tensor) -> torch.Tensor:
    """
    symbol_to_prob(x:PyTorch/Numpy Tensor/Array)
    Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'
    :param s: symbols vector
    :return: probabilities vector
    """
    return HALF * (s + 1)


def prob_to_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    prob_to_symbol(x:PyTorch/Numpy Tensor/Array)
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,0] -> '+1'
    :param p: probabilities vector
    :return: symbols vector
    """
    return torch.sign(p - HALF)


class DeepSICTrainer(Trainer):
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.memory_length = 1
        self.n_user = N_USER
        self.n_ant = N_ANT
        super().__init__()

    def __str__(self):
        return 'DeepSIC Trainer'

    def initialize_detector(self):
        self.detector = [[self.initialize_single_detector() for _ in range(ITERATIONS)] for _ in
                         range(self.n_user)]  # 2D list for Storing the DeepSIC Networks

    def train_models(self, model: nn.Module, i: int, b_train_all: torch.Tensor, y_train_all: torch.Tensor,
                     max_epochs: int):
        for user in range(self.n_user):
            self.train_model(model[user][i], b_train_all[user], y_train_all[user], max_epochs)

    def init_priors(self):
        self.probs_vec = HALF * torch.ones(N_ANT, conf.val_block_length - conf.pilot_size).to(device)

    def online_training(self, model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int):
        pass

    def train_loop(self, model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int):
        initial_probs = b_train.clone()
        b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self.train_models(model, 0, b_train_all, y_train_all, max_epochs)
        # Initializing the probabilities
        probs_vec = HALF * torch.ones(b_train.shape).to(device)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, ITERATIONS):
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(model, i, probs_vec, y_train)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.train_models(model, i, b_train_all, y_train_all, max_epochs)

    def forward(self, model: nn.Module, y: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        for i in range(ITERATIONS):
            probs_vec = self.calculate_posteriors(model, i + 1, probs_vec, y)
        detected_word = symbol_to_prob(prob_to_symbol(probs_vec.float()))
        return detected_word

    def prepare_data_for_training(self, b_train: torch.Tensor, y_train: torch.Tensor, probs_vec: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the DeepSIC Networks for Each User for the Iterations>1

        Parameters
        ----------
        data : dict
            The Data Dictionary Generated from DataGenerator class.

        Returns
        -------
        nets_list
            A list of length s_nK (number of users) containing instances of DeepSICNet for each user.
        v_cNet_m_fYtrain
            A list of data dictionaries with the prepard training data for each user
            [list_idx][dictionary_key]:
            [i]['b_train'] --> Training Labels (Symbol probabilities) for the i-th user.
            [i]['y_train'] --> Output of the Channel and the Predicted Symbol Probs. of the j-th users, where for j != i
        """
        b_train_all = []
        y_train_all = []
        for k in range(self.n_user):
            idx = [i for i in range(self.n_user) if i != k]
            current_y_train = torch.cat((y_train, probs_vec[:, idx]), dim=1)
            b_train_all.append(b_train[:, k])
            y_train_all.append(current_y_train)
        return b_train_all, y_train_all

    def calculate_posteriors(self, model: nn.Module, i: int, probs_vec: torch.Tensor,
                             y_train: torch.Tensor, para_list_detector=None) -> torch.Tensor:
        next_probs_vec = torch.zeros(probs_vec.shape).to(device)
        for user in range(self.n_user):
            idx = [i for i in range(self.n_user) if i != user]
            input = torch.cat((y_train, probs_vec[:, idx]), dim=1)
            with torch.no_grad():
                output = self.softmax(model[user][i - 1](input))
            next_probs_vec[:, user] = output[:, 1]
        return next_probs_vec
