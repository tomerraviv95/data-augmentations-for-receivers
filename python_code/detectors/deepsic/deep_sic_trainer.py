from typing import List

import torch
from torch import nn

from python_code.channel.channels_hyperparams import N_ANT, N_USER, MODULATION_NUM_MAPPING
from python_code.channel.modulator import BPSKModulator, QPSKModulator
from python_code.detectors.deepsic.deep_sic_detector import DeepSICDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import HALF, ModulationType, QUARTER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()
ITERATIONS = 5
EPOCHS = 250


def prob_to_BPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    prob_to_symbol(x:PyTorch/Numpy Tensor/Array)
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,0] -> '+1'
    :param p: probabilities vector
    :return: symbols vector
    """
    return torch.sign(p - HALF)


def prob_to_QPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    prob_to_symbol(x:PyTorch/Numpy Tensor/Array)
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,0] -> '+1'
    :param p: probabilities vector
    :return: symbols vector
    """
    p_real_neg = p[:, :, 0] + p[:, :, 2]
    first_symbol = torch.sign(p_real_neg - HALF)
    p_img_neg = p[:, :, 1] + p[:, :, 2]
    second_symbol = torch.sign(p_img_neg - HALF)
    s = torch.cat([first_symbol.unsqueeze(-1), second_symbol.unsqueeze(-1)], dim=-1)
    return torch.view_as_complex(s)


class DeepSICTrainer(Trainer):
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.memory_length = 1
        self.n_user = N_USER
        self.n_ant = N_ANT
        self.lr = 1e-3
        super().__init__()

    def __str__(self):
        return 'DeepSIC'

    def init_priors(self):
        val_size = 2 * (conf.val_block_length - conf.pilot_size) // MODULATION_NUM_MAPPING[conf.modulation_type]
        if conf.modulation_type == ModulationType.BPSK.name:
            self.probs_vec = HALF * torch.ones(val_size, N_ANT).to(device).float()
        elif conf.modulation_type == ModulationType.QPSK.name:
            self.probs_vec = QUARTER * torch.ones(val_size, N_ANT).to(device).unsqueeze(-1).repeat([1, 1, 3]).float()
        else:
            raise ValueError("No such constellation!")

    def initialize_detector(self):
        self.detector = [[DeepSICDetector().to(device) for _ in range(ITERATIONS)] for _ in
                         range(self.n_user)]  # 2D list for Storing the DeepSIC Networks

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        return self.criterion(input=soft_estimation, target=transmitted_words.squeeze(-1).long())

    @staticmethod
    def feed_to_model(single_model, y):
        if conf.modulation_type == ModulationType.BPSK.name:
            soft_estimation = single_model(y.float())
        elif conf.modulation_type == ModulationType.QPSK.name:
            y_input = torch.view_as_real(y[:, :N_ANT]).float().reshape(y.shape[0], -1)
            y_total = torch.cat([y_input, y[:, N_ANT:].float()], dim=1)
            soft_estimation = single_model(y_total)
        return soft_estimation

    def train_model(self, single_model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(single_model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(device)
        loss = 0
        for _ in range(EPOCHS):
            soft_estimation = self.feed_to_model(single_model, y_train)
            current_loss = self.run_train_loop(soft_estimation, b_train)
            loss += current_loss

    def train_models(self, model: List[List[DeepSICDetector]], i: int, b_train_all: List[torch.Tensor],
                     y_train_all: List[torch.Tensor]):
        for user in range(self.n_user):
            self.train_model(model[user][i], b_train_all[user], y_train_all[user])

    def online_training(self, b_train: torch.Tensor, y_train: torch.Tensor, h: torch.Tensor):
        if not conf.fading_in_channel:
            self.initialize_detector()
        if conf.modulation_type == ModulationType.QPSK.name:
            b_train = b_train[::2] + 2 * b_train[1::2]

        if conf.modulation_type == ModulationType.BPSK.name:
            initial_probs = b_train.clone()
        elif conf.modulation_type == ModulationType.QPSK.name:
            initial_probs = torch.zeros(b_train.shape).to(device).unsqueeze(-1).repeat([1, 1, 3])
            relevant_inds = []
            for i in range(MODULATION_NUM_MAPPING[conf.modulation_type] - 1):
                relevant_ind = (b_train == i + 1)
                relevant_inds.append(relevant_ind.unsqueeze(-1))
            relevant_inds = torch.cat(relevant_inds, dim=2)
            initial_probs[relevant_inds] = 1
        else:
            raise ValueError("No such constellation!")
        b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self.train_models(self.detector, 0, b_train_all, y_train_all)
        # Initializing the probabilities
        if conf.modulation_type == ModulationType.BPSK.name:
            probs_vec = HALF * torch.ones(b_train.shape).to(device)
        elif conf.modulation_type == ModulationType.QPSK.name:
            probs_vec = QUARTER * torch.ones(b_train.shape).to(device).unsqueeze(-1).repeat([1, 1, 3])
        else:
            raise ValueError("No such constellation!")
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, ITERATIONS):
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(self.detector, i, probs_vec, y_train)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.train_models(self.detector, i, b_train_all, y_train_all)

    def forward(self, y: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        for i in range(ITERATIONS):
            probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, y)
        if conf.modulation_type == ModulationType.BPSK.name:
            detected_word = BPSKModulator.demodulate(prob_to_BPSK_symbol(probs_vec.float()))
        elif conf.modulation_type == ModulationType.QPSK.name:
            detected_word = QPSKModulator.demodulate(prob_to_QPSK_symbol(probs_vec.float()))
        else:
            raise ValueError("No such constellation!")
        return detected_word

    def prepare_data_for_training(self, b_train: torch.Tensor, y_train: torch.Tensor, probs_vec: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the DeepSIC Networks for Each User for the Iterations>1
        """
        b_train_all = []
        y_train_all = []
        for k in range(self.n_user):
            idx = [i for i in range(self.n_user) if i != k]
            current_y_train = torch.cat((y_train, probs_vec[:, idx].reshape(y_train.shape[0], -1)), dim=1)
            b_train_all.append(b_train[:, k])
            y_train_all.append(current_y_train)
        return b_train_all, y_train_all

    def calculate_posteriors(self, model: List[List[nn.Module]], i: int, probs_vec: torch.Tensor,
                             y_train: torch.Tensor) -> torch.Tensor:
        next_probs_vec = torch.zeros(probs_vec.shape).to(device)
        for user in range(self.n_user):
            idx = [i for i in range(self.n_user) if i != user]
            input = torch.cat((y_train, probs_vec[:, idx].reshape(y_train.shape[0], -1)), dim=1)
            with torch.no_grad():
                output = self.softmax(self.feed_to_model(model[user][i - 1], input))
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        return next_probs_vec
