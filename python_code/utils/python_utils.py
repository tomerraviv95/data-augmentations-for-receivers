import pickle as pkl

import numpy as np

from python_code.channel.channels_hyperparams import MODULATION_NUM_MAPPING
from python_code.utils.config_singleton import Config

conf = Config()


def save_pkl(pkls_path: str, array: np.ndarray):
    output = open(pkls_path, 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str):
    output = open(pkls_path, 'rb')
    return pkl.load(output)


def normalize_for_modulation(size):
    return size * 2 // MODULATION_NUM_MAPPING[conf.modulation_type]
