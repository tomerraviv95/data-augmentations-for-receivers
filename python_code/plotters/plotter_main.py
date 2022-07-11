from collections import namedtuple

from python_code.plotters.plotter_methods import compute_ser_for_method
from python_code.plotters.plotter_utils import plot_by_values
from python_code.utils.constants import ChannelModes, DetectorType

RunParams = namedtuple(
    "RunParams",
    "run_over trial_num",
    defaults=[False, 1]
)

if __name__ == '__main__':
    run_over = False  # whether to run over previous results
    trial_num = 5  # number of trials per point estimate, used to reduce noise by averaging results of multiple runs
    run_params_obj = RunParams(run_over=run_over,
                               trial_num=trial_num)
    label_name = 'SNR_linear_SISO'
    # figure 1a
    if label_name == 'SNR_linear_SISO':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
        ]
        methods_list = [
            'Regular Training',
            'Combined',
            'Extended Pilot Training'
        ]
        plot_by_field = 'val_snr'
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'SER'
    # figure 1b
    elif label_name == 'SNR_linear_MIMO':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100}
        ]
        methods_list = [
            'Regular Training',
            'Combined',
            'Extended Pilot Training'
        ]
        plot_by_field = 'val_snr'
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'SER'
    # figure 2a
    elif label_name == 'SNR_linear_synth_SISO_fading':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
        ]
        methods_list = [
            'Regular Training',
            'Combined',
            'Extended Pilot Training'
        ]
        plot_by_field = 'val_snr'
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'SER'
    # figure 2b
    elif label_name == 'SNR_linear_synth_MIMO_fading':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100},
        ]

        methods_list = [
            'Regular Training',
            'Negation',
            'Translation',
            'Geometric',
            'Combined',
            'Extended Pilot Training'
        ]
        plot_by_field = 'val_snr'
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'SER'
        # figure 3a
    elif label_name == 'SNR_non_linear_synth_SISO_fading':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
        ]
        methods_list = [
            'Regular Training',
            'Combined',
            'Extended Pilot Training'
        ]
        plot_by_field = 'val_snr'
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'SER'
    # figure 3b
    elif label_name == 'SNR_non_linear_synth_MIMO_fading':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False},
        ]

        methods_list = [
            'Regular Training',
            # 'Negation',
            # 'Translation',
            # 'Geometric',
            'Combined',
            'Extended Pilot Training'
        ]
        plot_by_field = 'val_snr'
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'SER'
    elif label_name == 'Pilots':
        params_dicts = [
            {'val_block_length': 5025, 'pilot_size': 25},
            {'val_block_length': 5050, 'pilot_size': 50},
            {'val_block_length': 5100, 'pilot_size': 100},
            {'val_block_length': 5150, 'pilot_size': 150},
            {'val_block_length': 5200, 'pilot_size': 200}
        ]
        methods_list = [
            'Regular Training',
            'Negation',
            'Translation',
            'Geometric',
            'Combined',
            'FK Genie'
        ]
        plot_by_field = 'pilot_size'
        xlabel, ylabel = 'Pilots', '-log( BER(Method) / BER(Regular) )'
    else:
        raise ValueError('No such plot type!!!')
    all_curves = []

    for method in methods_list:
        print(method)
        for params_dict in params_dicts:
            print(params_dict)
            compute_ser_for_method(all_curves, method, params_dict, run_params_obj)
    plot_by_values(all_curves, label_name, values, xlabel, ylabel)
