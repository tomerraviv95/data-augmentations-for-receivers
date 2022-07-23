import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


def get_linestyle(method_name):
    if 'ViterbiNet' in method_name or 'DeepSIC' in method_name:
        return 'solid'
    elif 'RNN' in method_name or 'DNN' in method_name:
        return 'dashed'
    else:
        raise ValueError('No such detector!!!')


def get_marker(method_name):
    if 'Regular Training' in method_name:
        return '.'
    elif 'FK Genie' in method_name:
        return 'X'
    elif 'Geometric' in method_name:
        return '>'
    elif 'Translation' in method_name:
        return '<'
    elif 'Rotation' in method_name:
        return 'v'
    elif 'Combined' in method_name:
        return 'D'
    elif 'Extended Pilot Training' in method_name:
        return 'o'
    else:
        raise ValueError('No such method!!!')


def get_color(method_name):
    if 'Regular Training' in method_name:
        return 'b'
    elif 'FK Genie' in method_name:
        return 'black'
    elif 'Geometric' in method_name:
        return 'orange'
    elif 'Translation' in method_name:
        return 'pink'
    elif 'Rotation' in method_name:
        return 'green'
    elif 'Combined' in method_name:
        return 'red'
    elif 'Extended Pilot Training' in method_name:
        return 'royalblue'
    else:
        raise ValueError('No such method!!!')
