from python_code.plotters.plotter_config import COLORS_DICT, LINESTYLES_DICT, MARKERS_DICT
from python_code.utils.config_singleton import Config
from python_code.utils.python_utils import load_pkl, save_pkl
from python_code.vnet.trainer import Trainer
from dir_definitions import FIGURES_DIR, PLOTS_DIR
import datetime
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import math

conf = Config()

MIN_BER_COEF = 0.2
MARKER_EVERY = 10


def get_ser_plot(dec: Trainer, run_over: bool, method_name: str, trial=None):
    print(method_name)
    # set the path to saved plot results for a single method (so we do not need to run anew each time)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    file_name = '_'.join([method_name, str(conf.channel_type)])
    if trial is not None:
        file_name = file_name + '_' + str(trial)
    plots_path = os.path.join(PLOTS_DIR, file_name + '.pkl')
    print(plots_path)
    # if plot already exists, and the run_over flag is false - load the saved plot
    if os.path.isfile(plots_path) and not run_over:
        print("Loading plots")
        ser_total = load_pkl(plots_path)
    else:
        # otherwise - run again
        print("calculating fresh")
        dec.train()
        ser_total = dec.evaluate()
        save_pkl(plots_path, ser_total)
    print(np.mean(ser_total))
    return ser_total


def plot_all_curves_aggregated(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], snr: float):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    plt.figure()
    min_block_ind = math.inf
    min_ber = math.inf
    max_block_ind = -math.inf
    # iterate all curves, plot each one
    for i, (ser, method_name, _) in enumerate(all_curves):
        print(method_name)
        print(len(ser))
        block_range = np.arange(1, len(ser) + 1)
        agg_ser = (np.cumsum(ser) / np.arange(1, len(ser) + 1))
        plt.plot(block_range, agg_ser,
                 label=method_name,
                 color=COLORS_DICT[method_name], marker=MARKERS_DICT[method_name],
                 linestyle=LINESTYLES_DICT[method_name], linewidth=2.2, markevery=MARKER_EVERY)
        min_block_ind = block_range[0] if block_range[0] < min_block_ind else min_block_ind
        max_block_ind = block_range[-1] if block_range[-1] > max_block_ind else max_block_ind
        min_ber = agg_ser[-1] if agg_ser[-1] < min_ber else min_ber
    plt.ylabel('Coded BER')
    plt.xlabel('Block Index')
    plt.xlim([min_block_ind - 0.1, max_block_ind + 0.1])
    plt.ylim(bottom=MIN_BER_COEF * min_ber)
    plt.yscale('log')
    plt.legend(loc='upper left', prop={'size': 15})
    plt.savefig(
        os.path.join(FIGURES_DIR, folder_name, f'coded_ber_versus_block_index - SNR {snr}.png'),
        bbox_inches='tight')
    plt.show()


def plot_by_values(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], field_name, values: List[float]):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    plt.figure()
    names = []
    for i in range(len(all_curves)):
        if all_curves[i][1] not in names:
            names.append(all_curves[i][1])

    for method_name in names:
        mean_sers = []
        for ser, cur_name in all_curves:
            mean_ser = np.mean(ser)
            if cur_name != method_name:
                continue
            mean_sers.append(mean_ser)
        plt.plot(values, mean_sers, label=method_name,
                 color=COLORS_DICT[method_name], marker=MARKERS_DICT[method_name],
                 linestyle=LINESTYLES_DICT[method_name], linewidth=2.2)

    plt.xticks(values, values)
    plt.xlabel(field_name)
    plt.ylabel('Coded BER')
    plt.grid(which='both', ls='--')
    plt.legend(loc='lower left', prop={'size': 15})
    plt.yscale('log')
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'coded_ber_versus_snrs.png'),
                bbox_inches='tight')
    # plt.ylim([1e-3, 1e-2])
    plt.show()
