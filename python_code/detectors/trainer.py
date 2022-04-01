from python_code.augmentations.augmenter_wrapper import AugmenterWrapper
from python_code.utils.config_singleton import Config
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.utils.metrics import calculate_error_rates
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from python_code.utils.constants import ChannelModes
from torch.optim import RMSprop, Adam, SGD
from typing import Tuple, Union
from torch import nn
import numpy as np
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)

PRINT_FREQ = 10
HALF = 0.5


class Trainer(object):
    def __init__(self):
        # initialize matrices, datasets and detector
        self.initialize_dataloaders()
        self.initialize_detector()
        self.augmenter = AugmenterWrapper(conf.aug_type)
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference

    def get_name(self):
        return self.__name__()

    def initialize_detector(self):
        """
        Every trainer must have some base detector model
        """
        self.detector = None

    # calculate train loss
    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.Tensor) -> torch.Tensor:
        """
         Every trainer must have some loss calculation
        """
        pass

    # setup the optimization algorithm
    def deep_learning_setup(self):
        """
        Sets up the optimizer and loss criterion
        """
        if conf.optimizer_type == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                  lr=conf.lr)
        elif conf.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                     lr=conf.lr)
        elif conf.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                 lr=conf.lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")
        if conf.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(device)
        elif conf.loss_type == 'MSE':
            self.criterion = MSELoss().to(device)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    def initialize_dataloaders(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.channel_dataset = ChannelModelDataset(block_length=conf.val_block_length,
                                                   transmission_length=conf.val_block_length,
                                                   words=conf.val_frames)
        self.dataloaders = torch.utils.data.DataLoader(self.channel_dataset)

    def online_training(self, tx: torch.Tensor, rx: torch.Tensor, h: torch.Tensor):
        pass

    def forward(self, y: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        pass

    def init_priors(self):
        pass

    def evaluate(self) -> Union[float, np.ndarray]:
        """
        The online evaluation run. Main function for running the experiments of sequential transmission of pilots and
        data blocks for the paper.
        :return: np.ndarray
        """
        total_ser = 0
        # draw words of given gamma for all snrs
        transmitted_words, received_words, hs = self.channel_dataset.__getitem__(snr_list=[conf.val_snr])
        self.init_priors()
        ser_by_word = np.zeros(transmitted_words.shape[0])
        for frame in range(conf.val_frames - 1):
            # get current word and channel
            start_ind = frame * self.n_ant
            end_ind = (frame + 1) * self.n_ant
            transmitted_word = transmitted_words[start_ind:end_ind]
            received_word = received_words[start_ind:end_ind]
            h = hs[start_ind:end_ind]
            # split words into data and pilot part
            x_pilot, x_data = transmitted_word[:, :conf.pilot_size], transmitted_word[:, conf.pilot_size:]
            y_pilot, y_data = received_word[:, :conf.pilot_size], received_word[:, conf.pilot_size:]
            # if online training flag is on - train using pilots part
            if conf.is_online_training:
                self.online_training(x_pilot, y_pilot, h)
            # detect data part
            detected_word = self.forward(y_data, self.probs_vec)
            # calculate accuracy
            ser, fer, err_indices = calculate_error_rates(detected_word, x_data)
            print('*' * 20)
            print(f'current: {frame, ser}')
            total_ser += ser
            ser_by_word[frame] = ser
            # print progress
            if (frame + 1) % PRINT_FREQ == 0:
                print(f'Self-supervised: {frame + 1}/{transmitted_words.shape[0]}, SER {total_ser / (frame + 1)}')
            self.init_priors()

        total_ser /= transmitted_words.shape[0]
        print(f'Final ser: {total_ser}')
        return ser_by_word

    def augment_words_wrapper(self, h: torch.Tensor, received_words: torch.Tensor, transmitted_words: torch.Tensor,
                              total_size: int, n_repeats: int, phase: str):
        """
        The main augmentation function, used to augment each pilot in the evaluation phase.
        :param h: channel coefficients
        :param received_words: float channel values
        :param transmitted_words: binary transmitted word
        :param total_size: total number of examples to augment
        :param n_repeats: the number of repeats per augmentation
        :param phase: validation phase
        :return: the received and transmitted words
        """
        transmitted_words = transmitted_words.repeat(total_size, 1)
        received_words = received_words.repeat(total_size, 1)
        for i in range(total_size):
            update_hyper_params_flag = (i == 0)
            upd_idx = i % n_repeats
            current_received = received_words[upd_idx].reshape(1, -1)
            current_transmitted = transmitted_words[upd_idx].reshape(1, -1)
            if i < n_repeats:
                received_words[i], transmitted_words[i] = self.augmenter.augment(current_received,
                                                                                 current_transmitted,
                                                                                 h, conf.val_snr,
                                                                                 update_hyper_params=
                                                                                 update_hyper_params_flag)
            else:
                received_words[i], transmitted_words[i] = current_received, current_transmitted
        return received_words, transmitted_words

    def run_train_loop(self, soft_estimation: torch.Tensor, transmitted_words: torch.Tensor) -> float:
        # calculate loss
        loss = self.calc_loss(soft_estimation=soft_estimation, transmitted_words=transmitted_words)
        # if loss is Nan inform the user
        if torch.sum(torch.isnan(loss)):
            print('Nan value')
            return np.nan
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss

    def select_batch(self, gt_examples: torch.LongTensor, soft_estimation: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Select a batch from the input and gt labels
        :param gt_examples: training labels
        :param soft_estimation: the soft approximation, distribution over states (per word)
        :return: selected batch from the entire "epoch", contains both labels and the NN soft approximation
        """
        rand_ind = torch.multinomial(torch.arange(gt_examples.shape[0]).float(),
                                     conf.train_minibatch_size).long().to(device)
        return gt_examples[rand_ind], soft_estimation[rand_ind]
