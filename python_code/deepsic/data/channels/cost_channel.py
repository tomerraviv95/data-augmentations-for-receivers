from python_code.utils.constants import Phase
from dir_definitions import RESOURCES_DIR
from typing import Union
import numpy as np
import scipy.io
import os

SCALING_COEF = 0.25
COST_CONFIG_FRAMES = 10
MAX_FRAMES = 40


class COSTChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_user: int, frame_ind: int, phase: Phase) -> np.ndarray:
        total_h = np.empty([n_user, n_ant])
        main_folder = (1 + (frame_ind % MAX_FRAMES) // COST_CONFIG_FRAMES)
        for i in range(1, n_user + 1):
            path_to_mat = os.path.join(RESOURCES_DIR, f'{phase.value}_{main_folder}', f'h_{i}.mat')
            h_user = scipy.io.loadmat(path_to_mat)['norm_channel'][frame_ind % COST_CONFIG_FRAMES]
            total_h[i - 1] = SCALING_COEF * h_user
        total_h[np.arange(n_user), np.arange(n_user)] = 1
        return total_h
