from collections import namedtuple

from python_code.plotters.plotter_methods import compute_ser_for_method
from python_code.plotters.plotter_utils import plot_by_values
from python_code.utils.constants import ChannelModes

RunParams = namedtuple(
    "RunParams",
    "run_over plot_type trial_num",
    defaults=[False, 'SISO', 1]
)

if __name__ == '__main__':
    run_over = False  # whether to run over previous results
    plot_type = ChannelModes.MIMO.name  # either SISO (ChannelModes.SISO.name) or MIMO (ChannelModes.MIMO.name)
    trial_num = 5  # number of trials per point estimate, used to reduce noise by averaging results of multiple runs
    methods_list = [
        'Regular Training',
        'Negation',
        'Flipping',
        'Geometric',
        'Combined',
        'PK Genie',
        # 'FK Genie',
        # 'Extended Pilot Regular Training'
    ]
    run_params_obj = RunParams(run_over=run_over,
                               plot_type=plot_type,
                               trial_num=trial_num)
    label_name = 'Pilots'
    if label_name == 'SNR':
        params_dicts = [
            {'val_snr': 9},
            {'val_snr': 10},
            {'val_snr': 11},
            {'val_snr': 12},
            {'val_snr': 13}
        ]
        methods_list = [
            'Regular Training',
            'Negation',
            'Flipping',
            'Geometric',
            'Combined',
            'PK Genie',
            'FK Genie',
            'Extended Pilot Regular Training'
        ]
        plot_by_field = 'val_snr'
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
            'Flipping',
            'Geometric',
            'Combined',
            'PK Genie',
            # 'FK Genie'
        ]
        plot_by_field = 'pilot_size'
    else:
        raise ValueError('No such plot type!!!')
    all_curves = []

    for method in methods_list:
        print(method)
        for params_dict in params_dicts:
            print(params_dict)
            compute_ser_for_method(all_curves, method, params_dict, run_params_obj)

    plot_by_values(all_curves, label_name, [params_dict[plot_by_field] for params_dict in params_dicts])
