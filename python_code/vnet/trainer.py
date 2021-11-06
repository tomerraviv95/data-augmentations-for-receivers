from time import time
from typing import Tuple, Union
from python_code.augmentations.augmenter import Augmenter
from python_code.utils.config_singleton import Config
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.ecc.rs_main import decode, encode
from python_code.utils.metrics import calculate_error_rates
from dir_definitions import WEIGHTS_DIR
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.optim import RMSprop, Adam, SGD
import random
import numpy as np
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

N_UPDATES = 250


class Trainer(object):
    def __init__(self):

        # initializes word and noise generator from seed
        self.n_states = 2 ** conf.memory_length

        # initialize matrices, datasets and detector
        self.initialize_weights_dir()
        self.initialize_dataloaders()
        self.initialize_detector()

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
        detected_words = self.detector(received_words, 'val', conf.val_snr, conf.gamma)

        if conf.use_ecc:
            decoded_words = [decode(detected_word, conf.n_symbols) for detected_word in detected_words.cpu().numpy()]
            detected_words = torch.Tensor(decoded_words).to(device)

        ser, fer, err_indices = calculate_error_rates(detected_words, transmitted_words)
        print(f'Done. time: {time() - start}, ser: {ser}')
        return ser

    def eval_by_word(self, snr: float, gamma: float) -> Union[float, np.ndarray]:
        if conf.self_supervised:
            self.deep_learning_setup()
        total_ser = 0
        # draw words of given gamma for all snrs
        transmitted_words, received_words, hs = self.channel_dataset['val'].__getitem__(snr_list=[snr], gamma=gamma)
        ser_by_word = np.zeros(transmitted_words.shape[0])
        # saved detector is used to initialize the decoder in meta learning loops
        # query for all detected words
        if conf.buffer_empty:
            buffer_rx = torch.empty([0, received_words.shape[1]]).to(device)
            buffer_tx = torch.empty([0, received_words.shape[1]]).to(device)
            buffer_ser = torch.empty([0]).to(device)
        else:
            # draw words from different channels
            buffer_tx, buffer_rx = self.channel_dataset['train'].__getitem__(snr_list=[snr], gamma=gamma)
            buffer_ser = torch.zeros(buffer_rx.shape[0]).to(device)
            buffer_tx = torch.cat([
                torch.Tensor(encode(transmitted_word.int().cpu().numpy(), conf.n_symbols).reshape(1, -1)).to(device) for
                transmitted_word in buffer_tx], dim=0)

        for count, (transmitted_word, received_word, h) in enumerate(zip(transmitted_words, received_words, hs)):
            transmitted_word, received_word = transmitted_word.reshape(1, -1), received_word.reshape(1, -1)
            # detect
            detected_word = self.detector(received_word, 'val', snr, gamma, count)
            # decode
            decoded_word = [decode(detected_word, conf.n_symbols) for detected_word in detected_word.cpu().numpy()]
            decoded_word = torch.Tensor(decoded_word).to(device)
            # calculate accuracy
            ser, fer, err_indices = calculate_error_rates(decoded_word, transmitted_word)
            # encode word again
            decoded_word_array = decoded_word.int().cpu().numpy()
            encoded_word = torch.Tensor(encode(decoded_word_array, conf.n_symbols).reshape(1, -1)).to(device)
            errors_num = torch.sum(torch.abs(encoded_word - detected_word)).item()
            print('*' * 20)
            print(f'current: {count, ser, errors_num}')
            total_ser += ser
            ser_by_word[count] = ser

            # save the encoded word in the buffer
            if ser <= conf.ser_thresh:
                buffer_rx = torch.cat([buffer_rx, received_word])
                buffer_tx = torch.cat([buffer_tx,
                                       detected_word.reshape(1, -1) if ser > 0 else
                                       encoded_word.reshape(1, -1)],
                                      dim=0)
                buffer_ser = torch.cat([buffer_ser, torch.FloatTensor([ser]).to(device)])
                if not conf.buffer_empty:
                    buffer_rx = buffer_rx[1:]
                    buffer_tx = buffer_tx[1:]
                    buffer_ser = buffer_ser[1:]

            if conf.self_supervised and ser <= conf.ser_thresh:
                # use last word inserted in the buffer for training
                self.online_training(buffer_tx[-1].reshape(1, -1), buffer_rx[-1].reshape(1, -1), h.reshape(1, -1), snr)

            if (count + 1) % 10 == 0:
                print(f'Self-supervised: {count + 1}/{transmitted_words.shape[0]}, SER {total_ser / (count + 1)}')

        total_ser /= transmitted_words.shape[0]
        print(f'Final ser: {total_ser}')
        return ser_by_word

    def evaluate(self) -> float:
        """
        Evaluation either happens in a point aggregation way, or in a word-by-word fashion
        """
        self.load_weights(conf.val_snr, conf.gamma)
        # eval with training
        if conf.eval_mode == 'by_word':
            if not conf.use_ecc:
                raise ValueError('Only supports ecc')
            return self.eval_by_word(conf.val_snr, conf.gamma)
        else:
            return self.evaluate_at_point()

    def train(self):
        """
        Main training loop. Runs in minibatches.
        Evaluates performance over validation SNRs.
        Saves weights every so and so iterations.
        """
        # batches loop
        print(f'SNR - {conf.train_snr}, Gamma - {conf.gamma}')

        # initialize weights and loss
        self.initialize_detector()
        self.deep_learning_setup()
        best_ser = np.inf
        # draw words
        transmitted_words, received_words, h = self.channel_dataset['train'].__getitem__(snr_list=[conf.train_snr],
                                                                                         gamma=conf.gamma)
        # augment received words by the number of desired repeats
        transmitted_words = transmitted_words.repeat(conf.n_repeats, 1)
        received_words = received_words.repeat(conf.n_repeats, 1)
        for i in range(1, conf.n_repeats):
            current_received = received_words[i].reshape(1, -1)
            current_transmitted = transmitted_words[i].reshape(1, -1)
            received_words[i], transmitted_words[i] = Augmenter.augment(current_received, current_transmitted,
                                                                        conf.augmentations, h, conf.train_snr)

        for minibatch in range(1, conf.train_minibatch_num + 1):
            # run training loops
            loss = 0
            for upd_idx in range(N_UPDATES):
                i = upd_idx % conf.n_repeats  # the shape of the augmented received word
                current_received = received_words[i].reshape(1, -1)
                current_transmitted = transmitted_words[i].reshape(1, -1)
                # pass through detector
                soft_estimation = self.detector(current_received, 'train')
                current_loss = self.run_train_loop(soft_estimation, current_transmitted)
                loss += current_loss

            print(f'Minibatch {minibatch}, loss {loss}')
            # evaluate performance
            ser = self.evaluate_at_point()
            # save best weights
            if ser < best_ser:
                self.save_weights(loss, conf.train_snr, conf.gamma)
                best_ser = ser

        print('*' * 50)

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
