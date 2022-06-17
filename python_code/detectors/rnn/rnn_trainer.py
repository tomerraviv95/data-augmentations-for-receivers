import torch

from python_code.channel.channels_hyperparams import MEMORY_LENGTH
from python_code.detectors.trainer import Trainer
from python_code.detectors.rnn.rnn_detector import RNNDetector
from python_code.utils.config_singleton import Config
from python_code.utils.trellis_utils import calculate_siso_states, calculate_symbols_from_siso_states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()
EPOCHS = 200
BATCH_SIZE = 128


class RNNTrainer(Trainer):
    """
    Trainer for the RNNTrainer model.
    """

    def __init__(self):
        self.memory_length = MEMORY_LENGTH
        self.n_states = 2 ** self.memory_length
        self.n_user = 1
        self.n_ant = 1
        self.lr = 1e-2
        self.probs_vec = None
        super().__init__()

    def __str__(self):
        return 'RNN Detector'

    def initialize_detector(self):
        """
        Loads the RNN detector
        """
        self.detector = RNNDetector()

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param soft_estimation: [1,transmission_length,n_states], each element is a probability
        :param transmitted_words: [1, transmission_length]
        :return: loss value
        """
        # labels = transmitted_words[:, -1].long()
        gt_states = calculate_siso_states(self.memory_length, transmitted_words)
        loss = self.criterion(input=soft_estimation, target=gt_states)
        # equal_ind = torch.where(torch.argmax(soft_estimation, dim=1) != gt_states)
        # not_equal_gt_values = torch.unique(gt_states[equal_ind],return_counts=True)
        # not_equal_values = torch.unique(self.cur_rx[equal_ind],return_counts=True)
        # # print(torch.sum(torch.argmax(soft_estimation, dim=1) == gt_states))
        # print(not_equal_gt_values,not_equal_values)
        return loss

    def forward(self, y: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        detected_word = self.detector(y, phase='val')
        return detected_word

    def online_training(self, tx: torch.Tensor, rx: torch.Tensor, h: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the saved meta-trained weights.
        :param tx: transmitted word
        :param rx: received word
        :param h: channel coefficients
        """
        if conf.from_scratch:
            self.initialize_detector()
        self.deep_learning_setup()

        # run training loops
        loss = 0
        for i in range(EPOCHS):
            random_ind = 0 # torch.randperm(rx.size(0) - BATCH_SIZE)[:1]
            cur_rx, cur_tx = rx[random_ind:random_ind + BATCH_SIZE], tx[random_ind:random_ind + BATCH_SIZE]
            self.cur_rx = cur_rx
            # pass through detector
            soft_estimation = self.detector(cur_rx, phase='train')
            current_loss = self.run_train_loop(soft_estimation=soft_estimation, transmitted_words=cur_tx)
            loss += current_loss
