from python_code.plotters.plotter_utils import get_ser_plot, plot_all_curves_aggregated, plot_schematic
from python_code.vnet.vnet_trainer import VNETTrainer
from python_code.plotters.plotter_config import *


def add_viterbinet(all_curves, snr, val_block_length):
    dec = VNETTrainer(augmentations='reg', train_SNR_start=snr, train_SNR_end=snr, val_SNR_start=snr, val_SNR_end=snr,
                      val_block_length=val_block_length)
    method_name = f'ViterbiNet'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + str(snr) + '_'
                                                           + str(val_block_length))
    all_curves.append((ser, method_name, snr, val_block_length))


def add_aug_viterbinet(all_curves, snr, val_block_length):
    dec = VNETTrainer(augmentations='aug', train_SNR_start=snr, train_SNR_end=snr, val_SNR_start=snr, val_SNR_end=snr,
                      val_block_length=val_block_length)
    method_name = f'ViterbiNet-Augmentations'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + str(snr) + '_'
                                                           + str(val_block_length))
    all_curves.append((ser, method_name, snr, val_block_length))


def add_ref_viterbinet(all_curves, snr, val_block_length):
    dec = VNETTrainer(augmentations='ref', train_SNR_start=snr, train_SNR_end=snr, val_SNR_start=snr, val_SNR_end=snr,
                      val_block_length=val_block_length)
    method_name = f'ViterbiNet-Reference'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + str(snr) + '_'
                                                           + str(val_block_length))
    all_curves.append((ser, method_name, snr, val_block_length))


if __name__ == '__main__':
    run_over = True
    plot_by_block = False  # either plot by block, or by SNR
    val_block_length = 120

    if plot_by_block:
        parameters = [(6, val_block_length)]
    else:
        parameters = [(5, val_block_length),
                      (6, val_block_length),
                      (7, val_block_length),
                      (8, val_block_length),
                      (9, val_block_length)]
    all_curves = []

    for snr, val_block_length in parameters:
        print(snr, val_block_length)
        add_viterbinet(all_curves, snr, val_block_length)
        add_aug_viterbinet(all_curves, snr, val_block_length)
        add_ref_viterbinet(all_curves, snr, val_block_length)

        if plot_by_block:
            plot_all_curves_aggregated(all_curves, val_block_length, snr)

    snr_values = [x[0] for x in parameters]
    print(plot_by_block)
    if not plot_by_block:
        plot_schematic(all_curves, snr_values)
