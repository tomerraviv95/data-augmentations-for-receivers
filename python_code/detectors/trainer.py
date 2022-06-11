import random
from typing import Union

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import RMSprop, Adam, SGD

from python_code.augmentations.augmenter_wrapper import AugmenterWrapper
from python_code.augmentations.plotting_utils import online_plotting
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.utils.config_singleton import Config
from python_code.utils.metrics import calculate_error_rates

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)


class Trainer(object):
    def __init__(self):
        # initialize matrices, datasets and detector
        self.initialize_dataloader()
        self.initialize_detector()
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
                                  lr=self.lr)
        elif conf.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                     lr=self.lr)
        elif conf.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                 lr=self.lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")
        if conf.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(device)
        elif conf.loss_type == 'MSE':
            self.criterion = MSELoss().to(device)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    def initialize_dataloader(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.channel_dataset = ChannelModelDataset(block_length=conf.val_block_length,
                                                   pilots_length=conf.pilot_size,
                                                   blocks_num=conf.blocks_num)
        self.dataloader = torch.utils.data.DataLoader(self.channel_dataset)

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
        print(conf.sampler_type, conf.aug_type)
        total_ser = 0
        # draw words of given gamma for all snrs
        transmitted_words, received_words, hs = self.channel_dataset.__getitem__(snr_list=[conf.val_snr])
        self.init_priors()
        ser_by_word = np.zeros(transmitted_words.shape[0])
        augmenter_wrapper = AugmenterWrapper(conf.aug_type, conf.fading_in_channel)
        for block_ind in range(conf.blocks_num):
            # get current word and channel
            transmitted_word = transmitted_words[block_ind]
            h = hs[block_ind]
            received_word = received_words[block_ind]
            # split words into data and pilot part
            x_pilot, x_data = transmitted_word[:conf.pilot_size], transmitted_word[conf.pilot_size:]
            y_pilot, y_data = received_word[:conf.pilot_size], received_word[conf.pilot_size:]
            if conf.is_online_training:
                # augment received words by the number of desired repeats
                augmenter_wrapper.update_hyperparams(y_pilot, x_pilot)
                y_aug, x_aug = augmenter_wrapper.augment_batch(h, y_pilot, x_pilot)
                # train
                self.online_training(x_aug, y_aug, h)
            # detect data part
            detected_word = self.forward(y_data, self.probs_vec)
            # calculate accuracy
            # ser, fer, err_indices = calculate_error_rates(detected_word, x_data[:, :received_word.shape[1]])
            ser, fer, err_indices = calculate_error_rates(detected_word, x_data[:, -received_word.shape[1]:])
            print('*' * 20)
            print(f'current: {block_ind, ser}')
            total_ser += ser
            ser_by_word[block_ind] = ser
            self.init_priors()

        total_ser /= conf.blocks_num
        print(f'Final ser: {total_ser}')
        return total_ser

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

    def plot_regions(self):
        # draw words of given gamma for all snrs
        transmitted_words, received_words, hs = self.channel_dataset.__getitem__(snr_list=[conf.val_snr])
        augmenter_wrapper = AugmenterWrapper(conf.aug_type, conf.fading_in_channel)
        for block_ind in range(conf.blocks_num):
            # get current word and channel
            transmitted_word = transmitted_words[block_ind]
            h = hs[block_ind]
            received_word = received_words[block_ind]
            # split words into data and pilot part
            x_pilot, x_data = transmitted_word[:conf.pilot_size], transmitted_word[conf.pilot_size:]
            y_pilot, y_data = received_word[:conf.pilot_size], received_word[conf.pilot_size:]
            # augment received words by the number of desired repeats
            augmenter_wrapper.update_hyperparams(y_pilot, x_pilot)
            y_aug, x_aug = augmenter_wrapper.augment_batch(h, y_pilot, x_pilot)
            # if online training flag is on - train using pilots part
            online_plotting(x_aug, y_aug, h)
