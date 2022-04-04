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
    trial_num = 10  # number of trials per point estimate, used to reduce noise by averaging results of multiple runs
    methods_list = [
        'Regular Training',
        'SMOTE',
        'Flipping',
        'Adaptive',
        'Combined',
        # 'PK Genie',
        # 'FK Genie',
        # 'Extended Pilot Regular Training'
    ]
    run_params_obj = RunParams(run_over=run_over,
                               plot_type=plot_type,
                               trial_num=trial_num)
    params_dicts = [
        {'val_snr': 9},
        {'val_snr': 10},
        {'val_snr': 11},
        {'val_snr': 12},
        {'val_snr': 13}
    ]
    label_name = 'SNR'
    all_curves = []

    for method in methods_list:
        print(method)
        for params_dict in params_dicts:
            print(params_dict)
            compute_ser_for_method(all_curves, method, params_dict, run_params_obj)

    plot_by_values(all_curves, label_name, [list(params_dict.values())[0] for params_dict in params_dicts])
