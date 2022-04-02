from python_code.channel.channels_hyperparams import N_ANT, N_USER
from python_code.detectors.deepsic.deep_sic_detector import DeepSICDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase, HALF
from typing import List
from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()
ITERATIONS = 5
EPOCHS = 250

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
        return 'DeepSIC'

    def init_priors(self):
        self.probs_vec = HALF * torch.ones(N_ANT, conf.val_block_length - conf.pilot_size).to(device)

    def initialize_detector(self):
        self.detector = [[DeepSICDetector().to(device) for _ in range(ITERATIONS)] for _ in
                         range(self.n_user)]  # 2D list for Storing the DeepSIC Networks

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        return self.criterion(input=soft_estimation, target=transmitted_words.squeeze(-1).long())

    def train_model(self, single_model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(single_model.parameters(), lr=conf.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(device)
        loss = 0
        for _ in range(EPOCHS):
            soft_estimation = single_model(y_train)
            current_loss = self.run_train_loop(soft_estimation, b_train)
            loss += current_loss

    def train_models(self, model: List[List[DeepSICDetector]], i: int, b_train_all: List[torch.Tensor],
                     y_train_all: List[torch.Tensor]):
        for user in range(self.n_user):
            self.train_model(model[user][i], b_train_all[user], y_train_all[user])

    def online_training(self, b_train: torch.Tensor, y_train: torch.Tensor, h: torch.Tensor):
        if conf.from_scratch_flag:
            self.initialize_detector()
        b_train, y_train = b_train.T, y_train.T
        y_train, b_train = self.augment_words_wrapper(h, y_train, b_train, conf.online_total_words,
                                                      conf.online_repeats_n)
        initial_probs = b_train.clone()
        b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self.train_models(self.detector, 0, b_train_all, y_train_all)
        # Initializing the probabilities
        probs_vec = HALF * torch.ones(b_train.shape).to(device)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, ITERATIONS):
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(self.detector, i, probs_vec, y_train)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.train_models(self.detector, i, b_train_all, y_train_all)

    def forward(self, y: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        # fit dimensions
        y, probs_vec = y.T, probs_vec.T
        # detect and decode
        for i in range(ITERATIONS):
            probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, y)
        detected_word = symbol_to_prob(prob_to_symbol(probs_vec.float()))
        return detected_word.T

    def prepare_data_for_training(self, b_train: torch.Tensor, y_train: torch.Tensor, probs_vec: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the DeepSIC Networks for Each User for the Iterations>1
        """
        b_train_all = []
        y_train_all = []
        for k in range(self.n_user):
            idx = [i for i in range(self.n_user) if i != k]
            current_y_train = torch.cat((y_train, probs_vec[:, idx]), dim=1)
            b_train_all.append(b_train[:, k])
            y_train_all.append(current_y_train)
        return b_train_all, y_train_all

    def calculate_posteriors(self, model: List[List[nn.Module]], i: int, probs_vec: torch.Tensor,
                             y_train: torch.Tensor) -> torch.Tensor:
        next_probs_vec = torch.zeros(probs_vec.shape).to(device)
        for user in range(self.n_user):
            idx = [i for i in range(self.n_user) if i != user]
            input = torch.cat((y_train, probs_vec[:, idx]), dim=1)
            with torch.no_grad():
                output = self.softmax(model[user][i - 1](input))
            next_probs_vec[:, user] = output[:, 1]
        return next_probs_vec
