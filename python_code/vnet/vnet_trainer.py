from python_code.vnet.vnet_detector import VNETDetector
from python_code.utils.trellis_utils import calculate_states
from python_code.vnet.trainer import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VNETTrainer(Trainer):
    """
    Trainer for the ViterbiNet model.
    """

    def __init__(self, config_path=None, **kwargs):
        super().__init__(config_path, **kwargs)

    def __name__(self):
        if self.noisy_est_var > 0:
            channel_state = ', CSI uncertainty'
        else:
            channel_state = ', perfect CSI'

        if not self.self_supervised:
            training = ', untrained'
        else:
            training = ''

        return 'ViterbiNet' + channel_state + training

    def initialize_detector(self):
        """
        Loads the ViterbiNet detector
        """
        self.detector = VNETDetector(n_states=self.n_states,
                                     transmission_lengths=self.transmission_lengths)

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param soft_estimation: [1,transmission_length,n_states], each element is a probability
        :param transmitted_words: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_states(self.memory_length, transmitted_words)
        gt_states_batch, input_batch = self.select_batch(gt_states, soft_estimation.reshape(-1, self.n_states))
        loss = self.criterion(input=input_batch, target=gt_states_batch)
        return loss

    def online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - train on the detected/re-encoded word only if the ser is below some threshold.
        Start from the saved meta-trained weights.
        :param tx: transmitted word
        :param rx: received word
        """

        if self.augmentations == 'reg':
            # run training loops
            current_loss = 0
            for i in range(self.train_frames * self.subframes_in_frame_phase['train'] * N_REPEATS):
                # pass through detector
                soft_estimation = self.detector(received_words[i].reshape(1, -1), 'train')
                current_loss += self.run_train_loop(soft_estimation, transmitted_words[i].reshape(1, -1))
        elif self.augmentations == 'aug1':
            current_loss = 0
            current_loss = self.augment1(N_REPEATS, current_loss, received_words, transmitted_words, h, snr)
        elif self.augmentations == 'aug2':
            current_loss = 0
            current_loss = self.augment2(N_REPEATS, current_loss, received_words, transmitted_words, h)
        elif self.augmentations == 'aug3':
            current_loss = 0
            current_loss = self.augment3(N_REPEATS, current_loss, received_words, transmitted_words)

        # run training loops
        for i in range(self.self_supervised_iterations):
            # calculate soft values
            soft_estimation = self.detector(rx, 'train')
            self.run_train_loop(soft_estimation=soft_estimation, transmitted_words=tx)


if __name__ == '__main__':
    dec = VNETTrainer()
    dec.train()
