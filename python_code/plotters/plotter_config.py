from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import ChannelModes, DetectorType


class PlotType(Enum):
    SNR_linear_SISO = 'SNR_linear_SISO'
    SNR_linear_MIMO = 'SNR_linear_MIMO'
    SNR_linear_synth_SISO_fading = 'SNR_linear_synth_SISO_fading'
    SNR_linear_synth_MIMO_fading = 'SNR_linear_synth_MIMO_fading'
    SNR_non_linear_synth_SISO_fading = 'SNR_non_linear_synth_SISO_fading'
    SNR_non_linear_synth_MIMO_fading = 'SNR_non_linear_synth_MIMO_fading'
    SNR_linear_COST_2100_SISO = 'SNR_linear_COST_2100_SISO'
    SNR_linear_COST_2100_MIMO = 'SNR_linear_COST_2100_MIMO'
    pilot_efficiency_siso = 'pilot_efficiency_siso'
    pilot_efficiency_mimo = 'pilot_efficiency_mimo'
    pilot_efficiency_siso_cost = 'pilot_efficiency_siso_cost'
    pilot_efficiency_mimo_cost = 'pilot_efficiency_mimo_cost'
    SNR_linear_synth_SISO_fading_ablation = 'SNR_linear_synth_SISO_fading_ablation'
    SNR_linear_synth_MIMO_fading_ablation = 'SNR_linear_synth_MIMO_fading_ablation'


def get_config(label_name: str) -> Tuple[List[Dict], list, list, str, str]:
    # figure 1a
    if label_name == PlotType.SNR_linear_SISO.name:
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
    elif label_name == PlotType.SNR_linear_MIMO.name:
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
            'Combined',
            'Extended Pilot Training'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 2a
    elif label_name == PlotType.SNR_linear_synth_SISO_fading.name:
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
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 2b
    elif label_name == PlotType.SNR_linear_synth_MIMO_fading.name:
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
    elif label_name == PlotType.SNR_non_linear_synth_SISO_fading.name:
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
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
    elif label_name == PlotType.SNR_non_linear_synth_MIMO_fading.name:
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
             'channel_model': 'Synthetic'},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'linear': False,
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
    elif label_name == PlotType.SNR_linear_COST_2100_SISO.name:
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Cost2100'},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Cost2100'},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Cost2100'},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Cost2100'},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Cost2100'},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Cost2100'},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Cost2100'},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Cost2100'},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Cost2100'},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': False, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Cost2100'},
        ]

        methods_list = [
            'Regular Training',
            'Combined',
            'Extended Pilot Training'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 4b
    elif label_name == PlotType.SNR_linear_COST_2100_MIMO.name:
        params_dicts = [
            {'val_snr': 9, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'from_scratch': False, 'channel_model': 'Cost2100', 'val_block_length': 11000, 'pilot_size': 1000},
            {'val_snr': 10, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'from_scratch': False, 'channel_model': 'Cost2100', 'val_block_length': 11000, 'pilot_size': 1000},
            {'val_snr': 11, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'from_scratch': False, 'channel_model': 'Cost2100', 'val_block_length': 11000, 'pilot_size': 1000},
            {'val_snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'from_scratch': False, 'channel_model': 'Cost2100', 'val_block_length': 11000, 'pilot_size': 1000},
            {'val_snr': 13, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'from_scratch': False, 'channel_model': 'Cost2100', 'val_block_length': 11000, 'pilot_size': 1000},
            {'val_snr': 9, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'from_scratch': False, 'channel_model': 'Cost2100', 'val_block_length': 11000, 'pilot_size': 1000},
            {'val_snr': 10, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'from_scratch': False, 'channel_model': 'Cost2100', 'val_block_length': 11000, 'pilot_size': 1000},
            {'val_snr': 11, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'from_scratch': False, 'channel_model': 'Cost2100', 'val_block_length': 11000, 'pilot_size': 1000},
            {'val_snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'from_scratch': False, 'channel_model': 'Cost2100', 'val_block_length': 11000, 'pilot_size': 1000},
            {'val_snr': 13, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'from_scratch': False, 'channel_model': 'Cost2100', 'val_block_length': 11000, 'pilot_size': 1000},
        ]

        methods_list = [
            'Regular Training',
            'Combined',
            'Extended Pilot Training'
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 5a - ablation study
    elif label_name == PlotType.SNR_linear_synth_SISO_fading_ablation.name:
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
        ]
        methods_list = [
            'Regular Training',
            'Geometric',
            'CC Rotation',
            'Translation',
            'Combined',
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    # figure 5b - ablation
    elif label_name == PlotType.SNR_linear_synth_MIMO_fading_ablation.name:
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
            'CC Rotation',
            'Translation',
            'Combined',
        ]
        values = list(range(9, 14))
        xlabel, ylabel = 'SNR', 'BER'
    elif label_name == PlotType.pilot_efficiency_siso.name:
        values = [100, 200, 300, 400, 500, 600]
        params_dicts = [
            {'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100,
             'pilot_size': val, 'val_block_length': int(10000 + val)}
            for val
            in values]
        params_dicts.extend([
            {'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100,
             'pilot_size': val, 'val_block_length': int(10000 + val)}
            for val
            in values])
        methods_list = [
            'Regular Training',
            'Combined',
        ]
        xlabel, ylabel = 'Pilots Num', 'BER'
    elif label_name == PlotType.pilot_efficiency_mimo.name:
        values = [512, 650, 800, 1000, 1200]
        params_dicts = [
            {'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100,
             'pilot_size': val, 'val_block_length': int(10000 + val)}
            for val
            in values]
        params_dicts.extend([
            {'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100,
             'pilot_size': val, 'val_block_length': int(10000 + val)}
            for val
            in values])
        methods_list = [
            'Regular Training',
            'Combined',
        ]
        xlabel, ylabel = 'Pilots Num', 'BER'
    elif label_name == PlotType.pilot_efficiency_siso_cost.name:
        values = [100, 200, 300, 400, 500, 600]
        params_dicts = [
            {'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Cost2100',
             'pilot_size': val, 'val_block_length': int(10000 + val)}
            for val
            in values]
        params_dicts.extend(
            [
                {'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
                 'fading_in_channel': True, 'from_scratch': False, 'channel_model': 'Cost2100',
                 'pilot_size': val, 'val_block_length': int(10000 + val)}
                for val
                in values]
        )
        methods_list = [
            'Regular Training',
            'Combined',
        ]
        xlabel, ylabel = 'Pilots Num', 'BER'
    elif label_name == PlotType.pilot_efficiency_mimo_cost.name:
        values = [512, 650, 800, 1000, 1200]
        params_dicts = [
            {'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'fading_in_channel': True, 'from_scratch': False, 'blocks_num': 100, 'channel_model': 'Cost2100',
             'pilot_size': val, 'val_block_length': int(10000 + val)}
            for val
            in values]
        params_dicts.extend(
            [
                {'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
                 'fading_in_channel': True, 'from_scratch': False, 'channel_model': 'Cost2100',
                 'pilot_size': val, 'val_block_length': int(10000 + val)}
                for val
                in values]
        )
        methods_list = [
            'Regular Training',
            'Combined',
        ]
        xlabel, ylabel = 'Pilots Num', 'BER'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, methods_list, values, xlabel, ylabel
