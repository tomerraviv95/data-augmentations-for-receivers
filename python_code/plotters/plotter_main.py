import os

from dir_definitions import CONFIG_RUNS_DIR
from python_code.plotters.plotter_utils import get_ser_plot, plot_all_curves_aggregated, plot_by_values
from python_code.utils.config_singleton import Config
from python_code.vnet.vnet_trainer import VNETTrainer
from python_code.plotters.plotter_config import *


def add_reg_viterbinet(all_curves, snr, train_block_length):
    dec = VNETTrainer(run_name=f'reg_{train_block_length}', augmentations='reg', train_SNR_start=snr, train_SNR_end=snr,
                      val_SNR_start=snr,
                      val_SNR_end=snr, train_block_length=train_block_length)
    method_name = f'ViterbiNet - Regular Training'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + str(snr) + '_' + str(train_block_length))
    all_curves.append((ser, method_name, snr))


def add_aug1_viterbinet(all_curves, snr, train_block_length):
    dec = VNETTrainer(run_name=f'aug1_{train_block_length}', augmentations='aug1', train_SNR_start=snr,
                      train_SNR_end=snr, val_SNR_start=snr,
                      val_SNR_end=snr, train_block_length=train_block_length)
    method_name = f'ViterbiNet - Aug. 1'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + str(snr) + '_' + str(train_block_length))
    all_curves.append((ser, method_name, snr))


def add_aug2_viterbinet(all_curves, snr, train_block_length):
    dec = VNETTrainer(run_name=f'aug2_{train_block_length}', augmentations='aug2', train_SNR_start=snr,
                      train_SNR_end=snr, val_SNR_start=snr,
                      val_SNR_end=snr, train_block_length=train_block_length)
    method_name = f'ViterbiNet - Aug. 2'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + str(snr) + '_' + str(train_block_length))
    all_curves.append((ser, method_name, snr))


def add_aug3_viterbinet(all_curves, params_dict, run_over):
    method_name = f'ViterbiNet - Aug. 3'
    conf = Config()
    conf.load_config(os.path.join(CONFIG_RUNS_DIR, 'augmentation3.yaml'))
    name = ''
    for field, value in params_dict.items():
        conf.set_value(field, value)
        name += f'_{field}_{value}'
    conf.set_value('run_name', method_name + name)
    dec = VNETTrainer()
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + name)
    all_curves.append((ser, method_name))


if __name__ == '__main__':
    run_over = False
    plot_by_block = False  # either plot by block, or by SNR

    if plot_by_block:
        snr_values = [12]
    else:
        params_dicts = [{'n_repeats': 10},
                        {'n_repeats': 20},
                        {'n_repeats': 30},
                        {'n_repeats': 40},
                        {'n_repeats': 50},
                        {'n_repeats': 60},
                        {'n_repeats': 70},
                        {'n_repeats': 80},
                        {'n_repeats': 90},
                        {'n_repeats': 100}]
    all_curves = []

    for params_dict in params_dicts:
        print(params_dict)
        # add_reg_viterbinet(all_curves, snr, train_block_length)
        # add_aug1_viterbinet(all_curves, snr, train_block_length)
        # add_aug2_viterbinet(all_curves, snr, train_block_length)
        add_aug3_viterbinet(all_curves, params_dict, run_over)

        # if plot_by_block:
        #     plot_all_curves_aggregated(all_curves, snr)

    if not plot_by_block:
        plot_by_values(all_curves, list(params_dicts[0].keys())[0],
                       [list(params_dict.values())[0] for params_dict in params_dicts])
