from random import randint

import torch

from python_code.channel.channels_hyperparams import N_ANT, N_USER
from python_code.channel.modulator import BPSKModulator, QPSKModulator
from python_code.detectors.dnn.dnn_detector import DNNDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ModulationType
from python_code.utils.trellis_utils import calculate_mimo_states, prob_to_BPSK_symbol, prob_to_QPSK_symbol, \
    get_bits_from_qpsk_symbols

conf = Config()

EPOCHS = 500
BATCH_SIZE = 32


class DNNTrainer(Trainer):
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.memory_length = 1
        self.n_user = N_USER
        self.n_ant = N_ANT
        self.probs_vec = None
        self.lr = 1e-2
        super().__init__()

    def __str__(self):
        return 'DNN Detector'

    def initialize_detector(self):
        """
            Loads the DNN detector
        """
        self.detector = DNNDetector(self.n_user, self.n_ant)

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param soft_estimation: [1,transmission_length,n_states], each element is a probability
        :param transmitted_words: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_mimo_states(self.n_ant, transmitted_words)
        loss = self.criterion(input=soft_estimation, target=gt_states)
        return loss

    def forward(self, y: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:

        if conf.modulation_type == ModulationType.BPSK.name:
            y = y.float()
        elif conf.modulation_type == ModulationType.QPSK.name:
            y = torch.view_as_real(y).float().reshape(y.shape[0], -1)
        detected_word = self.detector(y, phase='val')

        if conf.modulation_type == ModulationType.QPSK.name:
            detected_word = get_bits_from_qpsk_symbols(detected_word)

        return detected_word

    def online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the saved meta-trained weights.
        :param tx: transmitted word
        :param rx: received word
        """
        if conf.from_scratch:
            self.initialize_detector()
        self.deep_learning_setup()

        if conf.modulation_type == ModulationType.QPSK.name:
            rx = torch.view_as_real(rx).float().reshape(rx.shape[0], -1)
        #     raise ValueError("Not implemented DNN for this case!!")

        # run training loops
        loss = 0
        for i in range(EPOCHS):
            ind = randint(a=0, b=tx.shape[0] - BATCH_SIZE)
            # pass through detector
            soft_estimation = self.detector(rx[ind: ind + BATCH_SIZE].float(), phase='train')
            current_loss = self.run_train_loop(soft_estimation=soft_estimation,
                                               transmitted_words=tx[ind:ind + BATCH_SIZE])
            loss += current_loss
