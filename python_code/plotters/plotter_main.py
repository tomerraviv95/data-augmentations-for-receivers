from python_code.plotters.plotter_utils import get_ser_plot, plot_all_curves_aggregated, plot_by_snrs
from python_code.vnet.vnet_trainer import VNETTrainer
from python_code.plotters.plotter_config import *


def add_reg_viterbinet(all_curves, snr):
    dec = VNETTrainer(run_name='reg', augmentations='reg', train_SNR_start=snr, train_SNR_end=snr, val_SNR_start=snr,
                      val_SNR_end=snr)
    method_name = f'ViterbiNet - Regular Training'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + str(snr))
    all_curves.append((ser, method_name, snr))


def add_aug1_viterbinet(all_curves, snr):
    dec = VNETTrainer(run_name='aug1', augmentations='aug1', train_SNR_start=snr, train_SNR_end=snr, val_SNR_start=snr,
                      val_SNR_end=snr)
    method_name = f'ViterbiNet - Aug. 1'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + str(snr))
    all_curves.append((ser, method_name, snr))


def add_aug2_viterbinet(all_curves, snr):
    dec = VNETTrainer(run_name='aug2', augmentations='aug2', train_SNR_start=snr, train_SNR_end=snr, val_SNR_start=snr,
                      val_SNR_end=snr)
    method_name = f'ViterbiNet - Aug. 2'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + str(snr))
    all_curves.append((ser, method_name, snr))


def add_aug3_viterbinet(all_curves, snr):
    dec = VNETTrainer(run_name='aug3', augmentations='aug3', train_SNR_start=snr, train_SNR_end=snr, val_SNR_start=snr,
                      val_SNR_end=snr)
    method_name = f'ViterbiNet - Aug. 3'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + str(snr))
    all_curves.append((ser, method_name, snr))


if __name__ == '__main__':
    run_over = True
    plot_by_block = False  # either plot by block, or by SNR

    if plot_by_block:
        snr_values = [(6)]
    else:
        snr_values = [11, 12, 13]
    all_curves = []

    for snr in snr_values:
        print(snr)
        add_reg_viterbinet(all_curves, snr)
        add_aug1_viterbinet(all_curves, snr)
        add_aug2_viterbinet(all_curves, snr)
        add_aug3_viterbinet(all_curves, snr)

        # if plot_by_block:
        #     plot_all_curves_aggregated(all_curves, snr)


    print(plot_by_block)
    if not plot_by_block:
        plot_by_snrs(all_curves, snr_values)
