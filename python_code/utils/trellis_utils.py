import numpy as np
import torch

from python_code import DEVICE
from python_code.channel.channels_hyperparams import MODULATION_NUM_MAPPING
from python_code.utils.config_singleton import Config
import itertools

conf = Config()


def create_transition_table(n_states: int) -> np.ndarray:
    """
    creates transition table of size [n_states,2]
    previous state of state i and input bit b is the state in cell [i,b]
    """
    transition_table = np.concatenate([np.arange(n_states), np.arange(n_states)]).reshape(n_states, 2)
    return transition_table


def acs_block(in_prob: torch.Tensor, llrs: torch.Tensor, transition_table: torch.Tensor, n_states: int) -> [
    torch.Tensor, torch.LongTensor]:
    """
    Viterbi ACS block
    :param in_prob: last stage probabilities, [batch_size,n_states]
    :param llrs: edge probabilities, [batch_size,1]
    :param transition_table: transitions
    :param n_states: number of states
    :return: current stage probabilities, [batch_size,n_states]
    """
    transition_ind = transition_table.reshape(-1).repeat(in_prob.size(0)).long()
    batches_ind = torch.arange(in_prob.size(0)).repeat_interleave(2 * n_states)
    trellis = (in_prob + llrs)[batches_ind, transition_ind]
    reshaped_trellis = trellis.reshape(-1, n_states, 2)
    return torch.min(reshaped_trellis, dim=2)[0]


def calculate_siso_states(memory_length: int, transmitted_words: torch.Tensor) -> torch.Tensor:
    """
    calculates all states vector for the transmitted words
    :param memory_length: length of channel memory
    :param transmitted_words: channel transmitted words
    :return: vector of length of transmitted_words with values in the range of 0,1,...,n_states-1
    """
    states_enumerator = (2 ** torch.arange(memory_length)).reshape(1, -1).float().to(DEVICE)
    gt_states = torch.sum(transmitted_words * states_enumerator, dim=1).long()
    return gt_states


def calculate_mimo_states(n_user: int, transmitted_words: torch.Tensor) -> torch.Tensor:
    states_enumerator = (MODULATION_NUM_MAPPING[conf.modulation_type] ** torch.arange(n_user)).to(DEVICE)
    if conf.modulation_type == 'QPSK':
        transmitted_words = transmitted_words[::2] + 2 * transmitted_words[1::2]
    gt_states = torch.sum(transmitted_words * states_enumerator, dim=1).long()
    return gt_states


def calculate_symbols_from_states(state_size: int, gt_states: torch.Tensor) -> torch.Tensor:
    mask = 2 ** torch.arange(state_size).to(gt_states.device, gt_states.dtype)
    return gt_states.unsqueeze(-1).bitwise_and(mask).ne(0).long()


def break_transmitted_siso_word_to_symbols(memory_length: int, transmitted_words: np.ndarray) -> np.ndarray:
    padded = np.concatenate([np.zeros([transmitted_words.shape[0], memory_length - 1]), transmitted_words,
                             np.zeros([transmitted_words.shape[0], memory_length])], axis=1)
    unsqueezed_padded = np.expand_dims(padded, axis=1)
    blockwise_words = np.concatenate([unsqueezed_padded[:, :, i:-memory_length + i] for i in range(memory_length)],
                                     axis=1)
    return blockwise_words.squeeze().T


def generate_symbols_by_state(state, n_state):
    combinations = list(itertools.product([0, 1], repeat=n_state))
    return torch.Tensor(combinations[state][::-1]).reshape(1, n_state).to(DEVICE)


def break_received_siso_word_to_symbols(memory_length: int, received_words: np.ndarray) -> np.ndarray:
    padded = np.concatenate([np.ones([received_words.shape[0], memory_length - 1]), received_words,
                             np.ones([received_words.shape[0], memory_length])], axis=1)
    unsqueezed_padded = np.expand_dims(padded, axis=1)
    blockwise_words = np.concatenate([unsqueezed_padded[:, :, i:-memory_length + i] for i in range(memory_length)],
                                     axis=1)
    return blockwise_words.squeeze().T[:-memory_length + 1]
