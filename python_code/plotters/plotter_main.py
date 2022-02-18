from dir_definitions import CONFIG_RUNS_DIR
from python_code.plotters.plotter_utils import get_ser_plot, plot_by_values
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
    method_name = 'ViterbiNet - Regular Training'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'reg.yaml'))
    name = set_method_name(conf, method_name, params_dict)
    print(method_name)
    add_avg_ser(all_curves, conf, method_name, name, run_over, trial_num)


def add_full_knowledge_augmenter_viterbinet(all_curves, params_dict, run_over, trial_num):
    method_name = 'ViterbiNet - FK Genie'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'fk_genie.yaml'))
    name = set_method_name(conf, method_name, params_dict)
    print(method_name)
    add_avg_ser(all_curves, conf, method_name, name, run_over, trial_num)


def add_partial_knowledge_augmenter_viterbinet(all_curves, params_dict, run_over, trial_num):
    method_name = 'ViterbiNet - PK Genie'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'pk_genie.yaml'))
    name = set_method_name(conf, method_name, params_dict)
    print(method_name)
    add_avg_ser(all_curves, conf, method_name, name, run_over, trial_num)


def add_self_supervised_scheme_viterbinet(all_curves, params_dict, run_over, trial_num):
    method_name = 'ViterbiNet - Self-Supervised Scheme'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'self_supervised_scheme.yaml'))
    name = set_method_name(conf, method_name, params_dict)
    print(method_name)
    add_avg_ser(all_curves, conf, method_name, name, run_over, trial_num)


if __name__ == '__main__':
    run_over = False
    plot_type = 'SNR_time_decay'  # either plot by block, or by SNR
    trial_num = 5

    if plot_type == 'SNR_time_decay':
        params_dicts = [
            {'val_snr': 9, 'val_frames': 300, 'channel_coefficients': 'time_decay'},
            {'val_snr': 10, 'val_frames': 300, 'channel_coefficients': 'time_decay'},
            {'val_snr': 11, 'val_frames': 300, 'channel_coefficients': 'time_decay'},
            {'val_snr': 12, 'val_frames': 300, 'channel_coefficients': 'time_decay'},
            {'val_snr': 13, 'val_frames': 300, 'channel_coefficients': 'time_decay'}
        ]
        label_name = 'SNR'
    elif plot_type == 'SNR_COST2100':
        params_dicts = [
            {'val_snr': 9, 'val_frames': 300, 'channel_coefficients': 'cost2100'},
            {'val_snr': 10, 'val_frames': 300, 'channel_coefficients': 'cost2100'},
            {'val_snr': 11, 'val_frames': 300, 'channel_coefficients': 'cost2100'},
            {'val_snr': 12, 'val_frames': 300, 'channel_coefficients': 'cost2100'},
            {'val_snr': 13, 'val_frames': 300, 'channel_coefficients': 'cost2100'}
        ]
        label_name = 'SNR'
    elif plot_type == 'Repeats':
        params_dicts = [{'n_repeats': 1},
                        {'n_repeats': 5},
                        {'n_repeats': 10},
                        {'n_repeats': 15},
                        {'n_repeats': 20},
                        {'n_repeats': 25}]
        label_name = 'Number of Unique Repeats'
    else:
        raise ValueError("No such plot type!!!")
    all_curves = []

    for params_dict in params_dicts:
        print(params_dict)
        add_reg_viterbinet(all_curves, params_dict, run_over, trial_num)
        add_self_supervised_scheme_viterbinet(all_curves, params_dict, run_over, trial_num)
        add_partial_knowledge_augmenter_viterbinet(all_curves, params_dict, run_over, trial_num)
        add_full_knowledge_augmenter_viterbinet(all_curves, params_dict, run_over, trial_num)

    plot_by_values(all_curves, label_name,  # list(params_dicts[0].keys())[0]
                   [list(params_dict.values())[0] for params_dict in params_dicts])
