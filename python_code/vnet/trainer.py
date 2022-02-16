from time import time
from typing import Tuple, Union
from python_code.augmentations.augmenter_wrapper import AugmenterWrapper
from python_code.utils.config_singleton import Config
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.ecc.rs_main import decode
from python_code.utils.metrics import calculate_error_rates
from dir_definitions import WEIGHTS_DIR
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.optim import RMSprop, Adam, SGD
import numpy as np
import random
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)

MIN_MINIBATCH = 20
PRINT_FREQ = 10


class Trainer(object):
    def __init__(self):

        # initializes word and noise generator from seed
        self.n_states = 2 ** conf.memory_length

        # initialize matrices, datasets and detector
        self.initialize_weights_dir()
        self.initialize_dataloaders()
        self.initialize_detector()
        self.augmenter = AugmenterWrapper(conf.aug_type)

    def initialize_weights_dir(self):
        """
        Parse the config, load all attributes into the trainer
        :param config_path: path to config
        """
        self.weights_dir = os.path.join(WEIGHTS_DIR, conf.run_name)
        if not os.path.exists(self.weights_dir) and len(self.weights_dir):
            os.makedirs(self.weights_dir)

    def get_name(self):
        return self.__name__()

    def initialize_detector(self):
        """
        Every trainer must have some base detector model
        """
        self.detector = None
        pass

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
        if conf.loss_type == 'BCE':
            self.criterion = BCELoss().to(device)
        elif conf.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(device)
        elif conf.loss_type == 'MSE':
            self.criterion = MSELoss().to(device)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    def initialize_dataloaders(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.frames_per_phase = {'train': conf.train_frames, 'val': conf.val_frames}
        self.block_lengths = {'train': conf.train_block_length, 'val': conf.val_block_length}
        self.transmission_lengths = {
            'train': conf.train_block_length,
            'val': conf.val_block_length if not conf.use_ecc else conf.val_block_length + 8 * conf.n_symbols}
        self.channel_dataset = {
            phase: ChannelModelDataset(block_length=self.block_lengths[phase],
                                       transmission_length=self.transmission_lengths[phase],
                                       words=self.frames_per_phase[phase],
                                       use_ecc=conf.use_ecc,
                                       phase=phase)
            for phase in ['train', 'val']}
        self.dataloaders = {phase: torch.utils.data.DataLoader(self.channel_dataset[phase])
                            for phase in ['train', 'val']}

    def online_training(self, tx: torch.Tensor, rx: torch.Tensor, h, snr):
        pass

    def evaluate_at_point(self) -> float:
        """
        Monte-Carlo simulation over validation SNRs range
        :return: ber, fer, iterations vectors
        """
        print(f'Starts evaluation at gamma {conf.gamma}')
        start = time()
        # draw words of given gamma for all snrs
        transmitted_words, received_words, _ = self.channel_dataset['val'].__getitem__(snr_list=[conf.val_snr],
                                                                                       gamma=conf.gamma)

        # decode and calculate accuracy
        detected_words = self.detector(received_words, 'val')

        if conf.use_ecc:
            decoded_words = [decode(detected_word, conf.n_symbols) for detected_word in detected_words.cpu().numpy()]
            detected_words = torch.Tensor(decoded_words).to(device)

        ser, fer, err_indices = calculate_error_rates(detected_words, transmitted_words)
        print(f'Done. time: {time() - start}, ser: {ser}')
        return ser

    def evaluate(self) -> Union[float, np.ndarray]:
        if conf.is_online_training:
            self.deep_learning_setup()
        total_ser = 0
        # draw words of given gamma for all snrs
        transmitted_words, received_words, hs = self.channel_dataset['val'].__getitem__(snr_list=[conf.val_snr],
                                                                                        gamma=conf.gamma)
        ser_by_word = np.zeros(transmitted_words.shape[0])
        for count, (transmitted_word, received_word, h) in enumerate(zip(transmitted_words, received_words, hs)):
            # get current channel word and true transmitted word (unknown to the receiver)
            transmitted_word, received_word = transmitted_word.reshape(1, -1), received_word.reshape(1, -1)
            # split words into data and pilot part
            x_pilot, x_data = transmitted_word[:, :conf.pilot_size], transmitted_word[:, conf.pilot_size:]
            y_pilot, y_data = received_word[:, :conf.pilot_size], received_word[:, conf.pilot_size:]
            # if online training flag is on - train using pilots part
            if conf.is_online_training:
                self.online_training(x_pilot, y_pilot, h.reshape(1, -1), conf.val_snr)
            # detect data part
            detected_word = self.detector(y_data, 'val')
            # calculate accuracy
            ser, fer, err_indices = calculate_error_rates(detected_word, x_data)
            print('*' * 20)
            print(f'current: {count, ser}')
            total_ser += ser
            ser_by_word[count] = ser
            # print progress
            if (count + 1) % PRINT_FREQ == 0:
                print(f'Self-supervised: {count + 1}/{transmitted_words.shape[0]}, SER {total_ser / (count + 1)}')

        total_ser /= transmitted_words.shape[0]
        print(f'Final ser: {total_ser}')
        return ser_by_word

    def augment_words_wrapper(self, h, received_words, transmitted_words, total_size, n_repeats, phase):
        transmitted_words = transmitted_words.repeat(total_size, 1)
        received_words = received_words.repeat(total_size, 1)
        for i in range(total_size):
            upd_idx = i % n_repeats
            current_received = received_words[upd_idx].reshape(1, -1)
            current_transmitted = transmitted_words[upd_idx].reshape(1, -1)
            if i < n_repeats:
                received_words[i], transmitted_words[i] = self.augmenter.augment(current_received,
                                                                                 current_transmitted,
                                                                                 h, conf.train_snr,
                                                                                 update_hyper_params=(i == 0))
            else:
                received_words[i], transmitted_words[i] = current_received, current_transmitted
        return received_words, transmitted_words

    def run_train_loop(self, soft_estimation: torch.Tensor, transmitted_words: torch.Tensor):
        # calculate loss
        loss = self.calc_loss(soft_estimation=soft_estimation, transmitted_words=transmitted_words)
        # if loss is Nan inform the user
        if torch.sum(torch.isnan(loss)):
            print('Nan value')
            return np.nan
        current_loss = loss.item()
        # back propagation
        for param in self.detector.parameters():
            param.grad = None
        loss.backward()
        self.optimizer.step()
        return current_loss

    def save_weights(self, current_loss: float, snr: float, gamma: float):
        torch.save({'model_state_dict': self.detector.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': current_loss},
                   os.path.join(self.weights_dir, f'snr_{snr}_gamma_{gamma}.pt'))

    def load_weights(self, snr: float, gamma: float):
        """
        Loads detector's weights defined by the [snr,gamma] from checkpoint, if exists
        """
        if os.path.join(self.weights_dir, f'snr_{snr}_gamma_{gamma}.pt'):
            print(f'loading model from snr {snr} and gamma {gamma}')
            weights_path = os.path.join(self.weights_dir, f'snr_{snr}_gamma_{gamma}.pt')
            if not os.path.isfile(weights_path):
                # if weights do not exist, train on the synthetic channel. Then validate on the test channel.
                os.makedirs(self.weights_dir, exist_ok=True)
                self.train()
            checkpoint = torch.load(weights_path)
            try:
                self.detector.load_state_dict(checkpoint['model_state_dict'])
            except Exception:
                raise ValueError("Wrong run directory!!!")
        else:
            print(f'No checkpoint for snr {snr} and gamma {gamma} in run "{conf.run_name}", starting from scratch')

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
