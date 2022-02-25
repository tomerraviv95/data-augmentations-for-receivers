import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

COLORS_DICT = {'ViterbiNet - Regular Training': 'green',
               'ViterbiNet - FK Genie': 'purple',
               'ViterbiNet - PK Genie': 'blue',
               'ViterbiNet - Adaptive Augmentation Scheme': 'red'}

MARKERS_DICT = {'ViterbiNet - Regular Training': 'o',
                'ViterbiNet - FK Genie': 'd',
                'ViterbiNet - PK Genie': 'x',
                'ViterbiNet - Adaptive Augmentation Scheme': '+'}

LINESTYLES_DICT = {'ViterbiNet - Regular Training': 'solid',
                   'ViterbiNet - FK Genie': 'solid',
                   'ViterbiNet - PK Genie': 'solid',
                   'ViterbiNet - Adaptive Augmentation Scheme': 'solid'}
