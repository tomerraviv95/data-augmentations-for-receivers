from python_code.plotters.plotter_utils import get_ser_plot, plot_all_curves_aggregated, plot_by_snrs
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


def add_aug3_viterbinet(all_curves, snr, train_block_length):
    dec = VNETTrainer(run_name=f'aug3_{train_block_length}', augmentations='aug3', train_SNR_start=snr,
                      train_SNR_end=snr, val_SNR_start=snr,
                      val_SNR_end=snr, train_block_length=train_block_length)
    method_name = f'ViterbiNet - Aug. 3'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + str(snr) + '_' + str(train_block_length))
    all_curves.append((ser, method_name, snr))


if __name__ == '__main__':
    run_over = False
    plot_by_block = True  # either plot by block, or by SNR

    if plot_by_block:
        snr_values = [12]
    else:
        snr_values = [10, 11, 12, 13, 14, 15, 16]
    train_block_length = 120
    all_curves = []

    for snr in snr_values:
        print(snr)
        add_reg_viterbinet(all_curves, snr, train_block_length)
        add_aug1_viterbinet(all_curves, snr, train_block_length)
        add_aug2_viterbinet(all_curves, snr, train_block_length)
        add_aug3_viterbinet(all_curves, snr, train_block_length)

        if plot_by_block:
            plot_all_curves_aggregated(all_curves, snr)

    if not plot_by_block:
        plot_by_snrs(all_curves, snr_values)
