from python_code.utils.config_singleton import Config
from python_code.vnet.vnet_detector import VNETDetector
from python_code.utils.trellis_utils import calculate_states
from python_code.vnet.trainer import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class VNETTrainer(Trainer):
    """
    Trainer for the ViterbiNet model.
    """

    def __init__(self):
        super().__init__()

    def __name__(self):
        if not conf.self_supervised:
            training = ', untrained'
        else:
            training = ''

        return 'ViterbiNet' + conf.channel_state + training

    def initialize_detector(self):
        """
        Loads the ViterbiNet detector
        """
        self.detector = VNETDetector(n_states=self.n_states)

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param soft_estimation: [1,transmission_length,n_states], each element is a probability
        :param transmitted_words: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_states(conf.memory_length, transmitted_words)
        gt_states_batch, input_batch = self.select_batch(gt_states, soft_estimation.reshape(-1, self.n_states))
        loss = self.criterion(input=input_batch, target=gt_states_batch)
        return loss

    def online_training(self, tx: torch.Tensor, rx: torch.Tensor, h: torch.Tensor, snr: float):
        """
        Online training module - trains on the detected word.
        Start from the saved meta-trained weights.
        :param tx: transmitted word
        :param rx: received word
        :param h: channel coefficients
        :param snr: float signal to noise value
        """
        # augment received words by the number of desired repeats
        aug_rx, aug_tx = self.augment_words_wrapper(h, rx, tx, conf.online_total_words, conf.online_repeats_n,
                                                    phase='val')

        if conf.from_scratch_flag:
            self.initialize_detector()
            self.deep_learning_setup()

        # run training loops
        loss = 0
        for i in range(conf.online_total_words):
            cur_rx = aug_rx[i].reshape(1, -1)
            cur_tx = aug_tx[i].reshape(1, -1)
            # pass through detector
            soft_estimation = self.detector(cur_rx, phase='train')
            current_loss = self.run_train_loop(soft_estimation=soft_estimation, transmitted_words=cur_tx)
            loss += current_loss
