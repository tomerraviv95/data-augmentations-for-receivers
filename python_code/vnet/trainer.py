from time import time
from typing import Tuple, Union
from python_code.channel.channel import ISIAWGNChannel
from python_code.utils.trellis_utils import calculate_states
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.channel.modulator import BPSKModulator
from python_code.ecc.rs_main import decode, encode
from python_code.utils.metrics import calculate_error_rates
from dir_definitions import CONFIG_PATH, WEIGHTS_DIR
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.optim import RMSprop, Adam, SGD
from shutil import copyfile
import random
import numpy as np
import itertools
import yaml
import torch
import os
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


class Trainer(object):
    def __init__(self, config_path=None, **kwargs):

        # general
        self.run_name = None

        # code parameters
        self.use_ecc = None
        self.n_symbols = None

        # channel
        self.memory_length = None
        self.channel_type = None
        self.channel_coefficients = None
        self.noisy_est_var = None
        self.fading_in_channel = None
        self.fading_in_decoder = None
        self.fading_taps_type = None
        self.subframes_in_frame = None
        self.gamma = None

        # validation hyperparameters
        self.val_block_length = None
        self.val_frames = None
        self.val_SNR_start = None
        self.val_SNR_end = None
        self.val_SNR_step = None
        self.eval_mode = None

        # training hyperparameters
        self.train_block_length = None
        self.train_frames = None
        self.train_minibatch_num = None
        self.train_minibatch_size = None
        self.train_SNR_start = None
        self.train_SNR_end = None
        self.train_SNR_step = None
        self.lr = None  # learning rate
        self.loss_type = None
        self.optimizer_type = None

        # self-supervised online training
        self.self_supervised = None
        self.self_supervised_iterations = None
        self.ser_thresh = None
        self.buffer_empty = None

        # augmentations
        self.augmentations = None

        # seed
        self.noise_seed = None
        self.word_seed = None

        # weights dir
        self.weights_dir = None

        # if any kwargs are passed, initialize the dict with them
        self.initialize_by_kwargs(**kwargs)

        # initializes all none parameters above from config
        self.param_parser(config_path)

        # initializes word and noise generator from seed
        self.rand_gen = np.random.RandomState(self.noise_seed)
        self.word_rand_gen = np.random.RandomState(self.word_seed)
        self.n_states = 2 ** self.memory_length

        # initialize matrices, datasets and detector
        self.initialize_dataloaders()
        self.initialize_detector()

        # regular training / augmentations repeats
        self.n_repeats = 100

    def initialize_by_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def param_parser(self, config_path: str):
        """
        Parse the config, load all attributes into the trainer
        :param config_path: path to config
        """
        if config_path is None:
            config_path = CONFIG_PATH

        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # set attribute of Trainer with every config item
        for k, v in self.config.items():
            try:
                if getattr(self, k) is None:
                    setattr(self, k, v)
            except AttributeError:
                pass

        if self.weights_dir is None:
            self.weights_dir = os.path.join(WEIGHTS_DIR, self.run_name)
            if not os.path.exists(self.weights_dir) and len(self.weights_dir):
                os.makedirs(self.weights_dir)
                # save config in output dir
                copyfile(config_path, os.path.join(self.weights_dir, "config.yaml"))

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
        if self.optimizer_type == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                  lr=self.lr)
        elif self.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                     lr=self.lr)
        elif self.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                 lr=self.lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")
        if self.loss_type == 'BCE':
            self.criterion = BCELoss().to(device)
        elif self.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(device)
        elif self.loss_type == 'MSE':
            self.criterion = MSELoss().to(device)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    def initialize_dataloaders(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.snr_range = {'train': np.arange(self.train_SNR_start, self.train_SNR_end + 1, step=self.train_SNR_step),
                          'val': np.arange(self.val_SNR_start, self.val_SNR_end + 1, step=self.val_SNR_step)}
        self.frames_per_phase = {'train': self.train_frames, 'val': self.val_frames}
        self.subframes_in_frame_phase = {'train': 1, 'val': self.subframes_in_frame}
        self.block_lengths = {'train': self.train_block_length, 'val': self.val_block_length}
        self.channel_coefficients = {'train': 'time_decay', 'val': self.channel_coefficients}
        self.transmission_lengths = {
            'train': self.train_block_length,
            'val': self.val_block_length if not self.use_ecc else self.val_block_length + 8 * self.n_symbols}
        self.channel_dataset = {
            phase: ChannelModelDataset(channel_type=self.channel_type,
                                       block_length=self.block_lengths[phase],
                                       transmission_length=self.transmission_lengths[phase],
                                       words=self.frames_per_phase[phase] * self.subframes_in_frame_phase[phase],
                                       memory_length=self.memory_length,
                                       channel_coefficients=self.channel_coefficients[phase],
                                       random=self.rand_gen,
                                       word_rand_gen=self.word_rand_gen,
                                       noisy_est_var=self.noisy_est_var,
                                       use_ecc=self.use_ecc,
                                       n_symbols=self.n_symbols,
                                       fading_taps_type=self.fading_taps_type,
                                       fading_in_channel=self.fading_in_channel,
                                       fading_in_decoder=self.fading_in_decoder,
                                       phase=phase,
                                       augmentations=self.augmentations if phase == 'train' else 'reg')
            for phase in ['train', 'val']}
        self.dataloaders = {phase: torch.utils.data.DataLoader(self.channel_dataset[phase])
                            for phase in ['train', 'val']}

    def online_training(self, tx: torch.Tensor, rx: torch.Tensor, h, snr):
        pass

    def single_eval_at_point(self, snr: float, gamma: float) -> float:
        """
        Evaluation at a single snr.
        :param snr: indice of snr in the snrs vector
        :return: ser for batch
        """
        # draw words of given gamma for all snrs
        transmitted_words, received_words, _ = self.channel_dataset['val'].__getitem__(snr_list=[snr], gamma=gamma)

        # decode and calculate accuracy
        detected_words = self.detector(received_words, 'val', snr, gamma)

        if self.use_ecc:
            decoded_words = [decode(detected_word, self.n_symbols) for detected_word in detected_words.cpu().numpy()]
            detected_words = torch.Tensor(decoded_words).to(device)

        ser, fer, err_indices = calculate_error_rates(detected_words, transmitted_words)

        return ser

    def gamma_eval(self, gamma: float) -> np.ndarray:
        """
        Evaluation at a single gamma value.
        :return: ser for batch.
        """
        ser_total = np.zeros(len(self.snr_range['val']))
        for snr_ind, snr in enumerate(self.snr_range['val']):
            self.load_weights(snr, gamma)
            ser_total[snr_ind] = self.single_eval_at_point(snr, gamma)
        return ser_total

    def evaluate_at_point(self) -> np.ndarray:
        """
        Monte-Carlo simulation over validation SNRs range
        :return: ber, fer, iterations vectors
        """
        ser_total = np.zeros(len(self.snr_range['val']))
        print(f'Starts evaluation at gamma {self.gamma}')
        start = time()
        ser_total += self.gamma_eval(self.gamma)
        print(f'Done. time: {time() - start}, ser: {ser_total}')
        return ser_total

    def eval_by_word(self, snr: float, gamma: float) -> Union[float, np.ndarray]:
        if self.self_supervised:
            self.deep_learning_setup()
        total_ser = 0
        # draw words of given gamma for all snrs
        transmitted_words, received_words, hs = self.channel_dataset['val'].__getitem__(snr_list=[snr], gamma=gamma)
        ser_by_word = np.zeros(transmitted_words.shape[0])
        # saved detector is used to initialize the decoder in meta learning loops
        # query for all detected words
        if self.buffer_empty:
            buffer_rx = torch.empty([0, received_words.shape[1]]).to(device)
            buffer_tx = torch.empty([0, received_words.shape[1]]).to(device)
            buffer_ser = torch.empty([0]).to(device)
        else:
            # draw words from different channels
            buffer_tx, buffer_rx = self.channel_dataset['train'].__getitem__(snr_list=[snr], gamma=gamma)
            buffer_ser = torch.zeros(buffer_rx.shape[0]).to(device)
            buffer_tx = torch.cat([
                torch.Tensor(encode(transmitted_word.int().cpu().numpy(), self.n_symbols).reshape(1, -1)).to(device) for
                transmitted_word in buffer_tx], dim=0)

        for count, (transmitted_word, received_word, h) in enumerate(zip(transmitted_words, received_words, hs)):
            transmitted_word, received_word = transmitted_word.reshape(1, -1), received_word.reshape(1, -1)
            # detect
            detected_word = self.detector(received_word, 'val', snr, gamma, count)
            # decode
            decoded_word = [decode(detected_word, self.n_symbols) for detected_word in detected_word.cpu().numpy()]
            decoded_word = torch.Tensor(decoded_word).to(device)
            # calculate accuracy
            ser, fer, err_indices = calculate_error_rates(decoded_word, transmitted_word)
            # encode word again
            decoded_word_array = decoded_word.int().cpu().numpy()
            encoded_word = torch.Tensor(encode(decoded_word_array, self.n_symbols).reshape(1, -1)).to(device)
            errors_num = torch.sum(torch.abs(encoded_word - detected_word)).item()
            print('*' * 20)
            print(f'current: {count, ser, errors_num}')
            total_ser += ser
            ser_by_word[count] = ser

            # save the encoded word in the buffer
            if ser <= self.ser_thresh:
                buffer_rx = torch.cat([buffer_rx, received_word])
                buffer_tx = torch.cat([buffer_tx,
                                       detected_word.reshape(1, -1) if ser > 0 else
                                       encoded_word.reshape(1, -1)],
                                      dim=0)
                buffer_ser = torch.cat([buffer_ser, torch.FloatTensor([ser]).to(device)])
                if not self.buffer_empty:
                    buffer_rx = buffer_rx[1:]
                    buffer_tx = buffer_tx[1:]
                    buffer_ser = buffer_ser[1:]

            if self.self_supervised and ser <= self.ser_thresh:
                # use last word inserted in the buffer for training
                self.online_training(buffer_tx[-1].reshape(1, -1), buffer_rx[-1].reshape(1, -1), h.reshape(1, -1), snr)

            if (count + 1) % 10 == 0:
                print(f'Self-supervised: {count + 1}/{transmitted_words.shape[0]}, SER {total_ser / (count + 1)}')

        total_ser /= transmitted_words.shape[0]
        print(f'Final ser: {total_ser}')
        return ser_by_word

    def evaluate(self) -> np.ndarray:
        """
        Evaluation either happens in a point aggregation way, or in a word-by-word fashion
        """
        # eval with training
        if self.eval_mode == 'by_word':
            if not self.use_ecc:
                raise ValueError('Only supports ecc')
            snr = self.snr_range['val'][0]
            self.load_weights(snr, self.gamma)
            return self.eval_by_word(snr, self.gamma)
        else:
            return self.evaluate_at_point()

    def train(self):
        """
        Main training loop. Runs in minibatches.
        Evaluates performance over validation SNRs.
        Saves weights every so and so iterations.
        """
        # batches loop
        for snr in self.snr_range['train']:
            print(f'SNR - {snr}, Gamma - {self.gamma}')

            # initialize weights and loss
            self.initialize_detector()
            self.deep_learning_setup()
            best_ser = math.inf

            # draw words
            transmitted_words, received_words, h = self.channel_dataset['train'].__getitem__(snr_list=[snr],
                                                                                             gamma=self.gamma)

            transmitted_words = transmitted_words.repeat(self.n_repeats, 1)
            received_words = received_words.repeat(self.n_repeats, 1)
            for minibatch in range(1, self.train_minibatch_num + 1):
                # run training loops
                loss = 0
                for i in range(self.train_frames * self.subframes_in_frame_phase['train'] * self.n_repeats):
                    current_received = received_words[i].reshape(1, -1)
                    current_transmitted = transmitted_words[i].reshape(1, -1)
                    loss += self.augmentations_wrapper(current_received, current_transmitted, h, snr)

                # evaluate performance
                ser = self.single_eval_at_point(snr, self.gamma)
                print(f'Minibatch {minibatch}, ser - {ser}, loss {loss}')
                # save best weights
                if ser < best_ser:
                    self.save_weights(loss, snr, self.gamma)
                    best_ser = ser

            print(f'best ser - {best_ser}')
            print('*' * 50)

    def augmentations_wrapper(self, current_received, current_transmitted, h, snr):
        if self.augmentations == 'reg':
            x, y = current_received, current_transmitted.reshape(1, -1)
        elif self.augmentations == 'aug1':
            x, y = self.augment_pair1(current_transmitted.reshape(1, -1), h,
                                      snr)
        elif self.augmentations == 'aug2':
            x, y = self.augment_pair2(current_received, current_transmitted, h)
        elif self.augmentations == 'aug3':
            x, y = self.augment_pair3(current_received, current_transmitted)
        else:
            raise ValueError("No sucn augmentation method!!!")
        # pass through detector
        soft_estimation = self.detector(x, 'train')
        current_loss = self.run_train_loop(soft_estimation, y)
        return current_loss

    def augment_pair1(self, transmitted_word, h, snr):
        binary_mask = torch.rand_like(transmitted_word) >= 0.5
        new_transmitted_word = (transmitted_word + binary_mask) % 2
        # encoding - errors correction code
        c = new_transmitted_word.cpu().numpy()
        # add zero bits
        padded_c = np.concatenate([c, np.zeros([c.shape[0], self.memory_length])], axis=1)
        # from channel dataset
        s = BPSKModulator.modulate(padded_c)
        # transmit through noisy channel
        new_received_word = ISIAWGNChannel.transmit(s=s, random=np.random.RandomState(),
                                                    h=h.cpu().numpy(),
                                                    snr=snr,
                                                    memory_length=self.memory_length)
        return torch.Tensor(new_received_word).to(device), new_transmitted_word

    def augment_pair2(self, received_word, transmitted_word, h):
        h = h.cpu().numpy()
        #### first calculate estimated noise pattern
        c = transmitted_word.cpu().numpy()
        # add zero bits
        padded_c = np.concatenate([c, np.zeros([c.shape[0], self.memory_length])], axis=1)
        # from channel dataset
        s = BPSKModulator.modulate(padded_c)
        blockwise_s = np.concatenate([s[:, i:-self.memory_length + i] for i in range(self.memory_length)],
                                     axis=0)
        trans_conv = np.dot(h[:, ::-1], blockwise_s)
        w_est = received_word.cpu().numpy() - trans_conv

        ### use the noise and add it to a new word
        binary_mask = torch.rand_like(transmitted_word) >= 0.5
        new_transmitted_word = (transmitted_word + binary_mask) % 2
        # encoding - errors correction code
        c = new_transmitted_word.cpu().numpy()
        # add zero bits
        padded_c = np.concatenate([c, np.zeros([c.shape[0], self.memory_length])], axis=1)
        # from channel dataset
        s = BPSKModulator.modulate(padded_c)
        blockwise_s = np.concatenate([s[:, i:-self.memory_length + i] for i in range(self.memory_length)],
                                     axis=0)
        new_trans_conv = np.dot(h[:, ::-1], blockwise_s)
        new_received_word = new_trans_conv + w_est
        return torch.Tensor(new_received_word).to(device), new_transmitted_word

    def augment_pair3(self, received_word, transmitted_word):
        #### first calculate estimated noise pattern
        gt_states = calculate_states(self.memory_length, transmitted_word)
        noise_samples = torch.empty_like(received_word)
        centers_est = torch.empty(2 ** self.memory_length).to(device)
        for state in torch.unique(gt_states):
            state_ind = (gt_states == state)
            state_received = received_word[0, state_ind]
            centers_est[state] = torch.mean(state_received)
            # centers_est[state] = classes_centers[15 - state]
            noise_samples[0, state_ind] = state_received - centers_est[state]

        new_transmitted_word = torch.rand_like(transmitted_word) >= 0.5
        new_gt_states = calculate_states(self.memory_length, new_transmitted_word)
        new_received_word = torch.empty_like(received_word)
        for state in torch.unique(new_gt_states):
            state_ind = (new_gt_states == state)
            new_received_word[0, state_ind] = centers_est[state] + noise_samples[0, state_ind]
        return new_received_word, new_transmitted_word

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
                self.fading_taps_type = 1
                os.makedirs(self.weights_dir, exist_ok=True)
                self.train()
                self.fading_taps_type = 2
            checkpoint = torch.load(weights_path)
            try:
                self.detector.load_state_dict(checkpoint['model_state_dict'])
            except Exception:
                raise ValueError("Wrong run directory!!!")
        else:
            print(f'No checkpoint for snr {snr} and gamma {gamma} in run "{self.run_name}", starting from scratch')

    def select_batch(self, gt_examples: torch.LongTensor, soft_estimation: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Select a batch from the input and gt labels
        :param gt_examples: training labels
        :param soft_estimation: the soft approximation, distribution over states (per word)
        :return: selected batch from the entire "epoch", contains both labels and the NN soft approximation
        """
        rand_ind = torch.multinomial(torch.arange(gt_examples.shape[0]).float(),
                                     self.train_minibatch_size).long().to(device)
        return gt_examples[rand_ind], soft_estimation[rand_ind]
