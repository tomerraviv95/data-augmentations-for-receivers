from dir_definitions import CONFIG_RUNS_DIR
from python_code.plotters.plotter_utils import get_ser_plot, plot_all_curves_aggregated, plot_by_values
from python_code.utils.config_singleton import Config
from python_code.vnet.vnet_trainer import VNETTrainer
from python_code.plotters.plotter_config import *
import numpy as np
import os


def add_reg_viterbinet(all_curves, params_dict, run_over, trial_num):
    method_name = f'ViterbiNet - Regular Training'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'reg.yaml'))
    name = ''
    for field, value in params_dict.items():
        conf.set_value(field, value)
        name += f'_{field}_{value}'
    conf.set_value('run_name', method_name + name)

    print(method_name)
    total_ser = []
    for trial in range(trial_num):
        conf.set_value('seed', 1 + trial)
        dec = VNETTrainer()
        ser = get_ser_plot(dec, run_over=run_over,
                           method_name=method_name + name,
                           trial=trial)
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name))


def add_aug1_viterbinet(all_curves, params_dict, run_over, trial_num):
    method_name = f'ViterbiNet - Aug. 1'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'augmentation1.yaml'))
    name = ''
    for field, value in params_dict.items():
        conf.set_value(field, value)
        name += f'_{field}_{value}'
    conf.set_value('run_name', method_name + name)

    print(method_name)
    total_ser = []
    for trial in range(trial_num):
        conf.set_value('seed', 1 + trial)
        dec = VNETTrainer()
        ser = get_ser_plot(dec, run_over=run_over,
                           method_name=method_name + name,
                           trial=trial)
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name))


def add_aug2_viterbinet(all_curves, params_dict, run_over, trial_num):
    method_name = f'ViterbiNet - Aug. 2'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'augmentation2.yaml'))
    name = ''
    for field, value in params_dict.items():
        conf.set_value(field, value)
        name += f'_{field}_{value}'
    conf.set_value('run_name', method_name + name)

    print(method_name)
    total_ser = []
    for trial in range(trial_num):
        conf.set_value('seed', 1 + trial)
        dec = VNETTrainer()
        ser = get_ser_plot(dec, run_over=run_over,
                           method_name=method_name + name,
                           trial=trial)
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name))


def add_aug3_viterbinet(all_curves, params_dict, run_over, trial_num):
    method_name = f'ViterbiNet - Aug. 3'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'augmentation3.yaml'))
    name = ''
    for field, value in params_dict.items():
        conf.set_value(field, value)
        name += f'_{field}_{value}'
    conf.set_value('run_name', method_name + name)

    print(method_name)
    total_ser = []
    for trial in range(trial_num):
        conf.set_value('seed', 1 + trial)
        dec = VNETTrainer()
        ser = get_ser_plot(dec, run_over=run_over,
                           method_name=method_name + name,
                           trial=trial)
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name))


if __name__ == '__main__':
    run_over = False
    plot_by_block = False  # either plot by block, or by SNR
    trial_num = 7

    if plot_by_block:
        snr_values = [12]
    else:
        params_dicts = [{'n_repeats': 1, 'train_block_length': 80},
                        {'n_repeats': 5, 'train_block_length': 80},
                        {'n_repeats': 10, 'train_block_length': 80},
                        {'n_repeats': 25, 'train_block_length': 80}]

    all_curves = []

    for params_dict in params_dicts:
        print(params_dict)
        add_reg_viterbinet(all_curves, params_dict, run_over, trial_num)
        add_aug1_viterbinet(all_curves, params_dict, run_over, trial_num)
        add_aug2_viterbinet(all_curves, params_dict, run_over, trial_num)
        add_aug3_viterbinet(all_curves, params_dict, run_over, trial_num)

        # if plot_by_block:
        #     plot_all_curves_aggregated(all_curves, snr)

    if not plot_by_block:
        plot_by_values(all_curves, list(params_dicts[0].keys())[0],
                       [list(params_dict.values())[0] for params_dict in params_dicts])
