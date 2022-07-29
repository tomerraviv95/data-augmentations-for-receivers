from python_code.plotters.plotter_methods import compute_ser_for_method, RunParams
from python_code.plotters.plotter_utils import plot_by_values
from python_code.utils.constants import ChannelModes, DetectorType

if __name__ == '__main__':
    run_over = True  # whether to run over previous results
    trial_num = 1  # number of trials per point estimate, used to reduce noise by averaging results of multiple runs
    run_params_obj = RunParams(run_over=run_over,
                               trial_num=trial_num)
    # 'SNR_linear_SISO', 'SNR_linear_MIMO', 'SNR_linear_synth_SISO_fading', 'SNR_linear_synth_MIMO_fading',
    # 'SNR_non_linear_synth_SISO_fading', 'SNR_non_linear_synth_MIMO_fading','SNR_linear_COST_2100_SISO',
    # 'SNR_linear_COST_2100_MIMO', 'SNR_linear_synth_SISO_fading_ablation','SNR_linear_synth_MIMO_fading_ablation'
    # 'pilot_efficiency_siso','pilot_efficiency_mimo'
    label_name = 'SNR_linear_synth_SISO_fading'
    print(label_name)
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
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 1b
    elif label_name == 'SNR_linear_MIMO':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': True, 'blocks_num': 100, 'channel_model': 'Synthetic'},
        ]
        methods_list = [
            'Regular Training',
            # 'Combined',
            # 'Extended Pilot Training'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
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
            # 'Combined',
            # 'Extended Pilot Training'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 2b
    elif label_name == 'SNR_linear_synth_MIMO_fading':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
        ]

        methods_list = [
            'Regular Training',
            'Combined',
            'Extended Pilot Training'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 3a
    elif label_name == 'SNR_non_linear_synth_SISO_fading':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
        ]
        methods_list = [
            'Regular Training',
            'Combined',
            'Extended Pilot Training'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 3b
    elif label_name == 'SNR_non_linear_synth_MIMO_fading':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linearity': False,
             'channel_model': 'Synthetic'},
        ]

        methods_list = [
            'Regular Training',
            'Combined',
            'Extended Pilot Training'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 4a
    elif label_name == 'SNR_linear_COST_2100_SISO':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
        ]

        methods_list = [
            # 'Regular Training',
            'Combined',
            # 'Extended Pilot Training'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 4b
    elif label_name == 'SNR_linear_COST_2100_MIMO':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'linearity': True,
             'channel_model': 'Cost2100'},
        ]

        methods_list = [
            # 'Regular Training',
            'Combined',
            # 'Extended Pilot Training'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 5a - ablation study
    elif label_name == 'SNR_linear_synth_SISO_fading_ablation':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'online_repeats_n': 2000},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'online_repeats_n': 2000},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'online_repeats_n': 2000},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'online_repeats_n': 2000},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'online_repeats_n': 2000},
        ]
        methods_list = [
            'Regular Training',
            'Geometric',
            'Translation',
            'Rotation',
            'Combined',
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 5b - ablation
    elif label_name == 'SNR_linear_synth_MIMO_fading_ablation':
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Synthetic'},
        ]
        methods_list = [
            'Regular Training',
            'Geometric',
            'Translation',
            'Rotation',
            'Combined',
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    elif label_name == 'pilot_efficiency_siso':
        values = [100, 200, 300, 400, 500, 600]
        params_dicts = [
            {'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100,
             'pilot_size': val, 'val_block_length': int(10000 + val)}
            for val
            in values]
        methods_list = [
            'Regular Training',
            'Combined',
        ]
        xlabel, ylabel = 'Pilots Num', 'BER'
    elif label_name == 'pilot_efficiency_mimo':
        values = [300, 400, 500, 600]
        params_dicts = [
            {'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100,
             'pilot_size': val, 'val_block_length': int(10000 + val)}
            for val
            in values]
        methods_list = [
            'Regular Training',
            'Combined',
        ]
        xlabel, ylabel = 'Pilots Num', 'BER'
    else:
        raise ValueError('No such plot type!!!')
    all_curves = []

    for method in methods_list:
        print(method)
        for params_dict in params_dicts:
            print(params_dict)
            compute_ser_for_method(all_curves, method, params_dict, run_params_obj)
    plot_by_values(all_curves, values, xlabel, ylabel)
