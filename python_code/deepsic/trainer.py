from python_code.utils.constants import Phase, HALF
from python_code.utils.metrics import calculate_error_rates
from python_code.deepsic.data.data_generator import DataGenerator
from python_code.utils.config_singleton import Config
from torch import nn
from typing import List
import numpy as np
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)


class Trainer:

    def __init__(self):
        self.test_frame_size = conf.pilot_size
        self.test_dg = DataGenerator(conf.val_block_length, phase=Phase.TEST, frame_num=conf.val_frames)
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference
        self.self_supervised = False
        self.phase = None

    def __str__(self):
        return 'trainer'

    def initialize_model(self) -> nn.Module:
        pass

    def initialize_single_detector(self) -> nn.Module:
        pass

    def copy_model(self, model: nn.Module):
        pass

    def train_model(self, single_model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int):
        pass

    def online_train_loop(self, model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int,
                          phase: Phase):
        pass

    def train_loop(self, model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int, phase: Phase):
        pass

    def predict(self, model: nn.Module, y: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        pass

    def evaluate(self, model: nn.Module, snr: int) -> List[float]:
        # generate data and declare sizes
        b_test, y_test = self.test_dg(snr=snr)
        c_pred = torch.zeros_like(y_test)
        b_pred = torch.zeros_like(b_test)
        c_frame_size = c_pred.shape[0] // conf.val_frames
        b_frame_size = b_pred.shape[0] // conf.val_frames
        probs_vec = HALF * torch.ones(c_frame_size - conf.pilot_size, y_test.shape[1]).to(device)

        # query for all detected words
        buffer_b, buffer_y = torch.empty([0, b_test.shape[1]]).to(device), torch.empty([0, y_test.shape[1]]).to(device)

        ber_list = []
        for frame in range(conf.val_frames - 1):
            # current word
            c_start_ind = frame * c_frame_size
            c_end_ind = (frame + 1) * c_frame_size
            current_y = y_test[c_start_ind:c_end_ind]
            b_start_ind = frame * b_frame_size
            b_end_ind = (frame + 1) * b_frame_size
            current_x = b_test[b_start_ind:b_end_ind]
            buffer_b, buffer_y, model = self.ecc_eval(model, buffer_b, buffer_y, probs_vec, ber_list, current_y,
                                                      current_x,
                                                      frame)

        return ber_list

    def ecc_eval(self, model: nn.Module, buffer_b: torch.Tensor, buffer_y: torch.Tensor, probs_vec: torch.Tensor,
                 ber_list: List[float], current_y: torch.Tensor, current_x: torch.Tensor, frame: int) -> [
        torch.Tensor, torch.Tensor]:
        # detect
        detected_word = self.predict(model, current_y, probs_vec)

        # calculate error rate
        ber = calculate_error_rates(detected_word, current_x)[0]
        ber_list.append(ber)
        print(frame, ber)

        # save the encoded word in the buffer
        if ber <= conf.ber_thresh:
            buffer_b = torch.cat([buffer_b, detected_word], dim=0)
            buffer_y = torch.cat([buffer_y, current_y], dim=0)

        # use last word inserted in the buffer for training
        if self.self_supervised and ber <= conf.ber_thresh:
            # use last word inserted in the buffer for training
            self.online_train_loop(model, detected_word, current_y, conf.self_supervised_epochs, self.phase)

        return buffer_b, buffer_y, model

    def main(self) -> List[float]:
        """
        Main train function. Generates data, initializes the model, trains it an evaluates it.
        :return: evaluated bers
        """
        all_bers = []  # Contains the ber
        print(f'training')
        print(f'snr {conf.val_snr}')
        model = self.initialize_model()
        self.phase = Phase.TEST
        ber = self.evaluate(model, conf.val_snr)
        all_bers.append(ber)
        print(f'\nber :{sum(ber) / len(ber)} @ snr: {conf.snr} [dB]')
        print(f'Training and Testing Completed\nBERs: {all_bers}')
        return all_bers
