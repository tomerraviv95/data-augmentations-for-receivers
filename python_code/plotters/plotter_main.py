from dir_definitions import CONFIG_RUNS_DIR
from python_code.plotters.plotter_utils import get_ser_plot, plot_all_curves_aggregated, plot_by_values
from python_code.utils.config_singleton import Config
from python_code.vnet.vnet_trainer import VNETTrainer
from python_code.plotters.plotter_config import *
import numpy as np
import os


def set_method_name(conf, method_name, params_dict):
    name = ''
    for field, value in params_dict.items():
        conf.set_value(field, value)
        name += f'_{field}_{value}'
    conf.set_value('run_name', method_name + name)
    return name


def add_avg_ser(all_curves, conf, method_name, name, run_over, trial_num):
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


def add_reg_viterbinet(all_curves, params_dict, run_over, trial_num):
    method_name = f'ViterbiNet - Regular Training'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'reg.yaml'))
    name = set_method_name(conf, method_name, params_dict)
    print(method_name)
    add_avg_ser(all_curves, conf, method_name, name, run_over, trial_num)


def add_aug1_viterbinet(all_curves, params_dict, run_over, trial_num):
    method_name = f'ViterbiNet - Aug. 1'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'augmentation1.yaml'))
    name = set_method_name(conf, method_name, params_dict)
    print(method_name)
    add_avg_ser(all_curves, conf, method_name, name, run_over, trial_num)


def add_aug2_viterbinet(all_curves, params_dict, run_over, trial_num):
    method_name = f'ViterbiNet - Aug. 2'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'augmentation2.yaml'))
    name = set_method_name(conf, method_name, params_dict)
    print(method_name)
    add_avg_ser(all_curves, conf, method_name, name, run_over, trial_num)


def add_aug3_viterbinet(all_curves, params_dict, run_over, trial_num):
    method_name = f'ViterbiNet - Aug. 3'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'augmentation3.yaml'))
    name = set_method_name(conf, method_name, params_dict)
    print(method_name)
    add_avg_ser(all_curves, conf, method_name, name, run_over, trial_num)


if __name__ == '__main__':
    run_over = False
    plot_type = 'SNR'  # either plot by block, or by SNR

    if plot_type == 'SNR':
        trial_num = 5
        params_dicts = [
            {'train_snr': 9, 'val_snr': 9, 'train_block_length': 80, 'val_block_length': 80, 'val_frames': 100,
             'channel_coefficients': 'cost2100'},
            {'train_snr': 10, 'val_snr': 10, 'train_block_length': 80, 'val_block_length': 80, 'val_frames': 100,
             'channel_coefficients': 'cost2100'},
            {'train_snr': 11, 'val_snr': 11, 'train_block_length': 80, 'val_block_length': 80, 'val_frames': 100,
             'channel_coefficients': 'cost2100'},
            {'train_snr': 12, 'val_snr': 12, 'train_block_length': 80, 'val_block_length': 80, 'val_frames': 100,
             'channel_coefficients': 'cost2100'}
        ]
        label_name = 'SNR'
    elif plot_type == 'Repeats':
        trial_num = 7
        params_dicts = [{'n_repeats': 1, 'train_block_length': 280},
                        {'n_repeats': 5, 'train_block_length': 280},
                        {'n_repeats': 10, 'train_block_length': 280},
                        {'n_repeats': 15, 'train_block_length': 280},
                        {'n_repeats': 20, 'train_block_length': 280},
                        {'n_repeats': 25, 'train_block_length': 280}]
        label_name = 'Number of Unique Repeats'
    all_curves = []

    for params_dict in params_dicts:
        print(params_dict)
        add_reg_viterbinet(all_curves, params_dict, run_over, trial_num)
        add_aug1_viterbinet(all_curves, params_dict, run_over, trial_num)
        add_aug2_viterbinet(all_curves, params_dict, run_over, trial_num)
        add_aug3_viterbinet(all_curves, params_dict, run_over, trial_num)

    plot_by_values(all_curves, label_name,  # list(params_dicts[0].keys())[0]
                   [list(params_dict.values())[0] for params_dict in params_dicts])
