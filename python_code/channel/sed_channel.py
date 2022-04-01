from python_code.channel.channels_hyperparams import N_ANT
import numpy as np


class SEDChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_user: int, frame_ind: int, fading: bool) -> np.ndarray:
        H_row = np.array([i for i in range(n_ant)])
        H_row = np.tile(H_row, [n_user, 1]).T
        H_column = np.array([i for i in range(n_user)])
        H_column = np.tile(H_column, [n_ant, 1])
        H = np.exp(-np.abs(H_row - H_column))
        if fading:
            H = SEDChannel.add_fading(H, n_ant, frame_ind)
        return H

    @staticmethod
    def add_fading(H: np.ndarray, n_ant: int, frame_ind: int) -> np.ndarray:
        degs_array = np.array([51, 39, 33, 21])
        center = 0.8
        fade_mat = center + (1 - center) * np.cos(2 * np.pi * frame_ind / degs_array)
        fade_mat = np.tile(fade_mat.reshape(1, -1), [n_ant, 1])
        return H * fade_mat

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float):
        sigma = 10 ** (-0.1 * snr)
        y = np.matmul(h, s) + np.sqrt(sigma) * np.random.randn(N_ANT, s.shape[1])
        return y
