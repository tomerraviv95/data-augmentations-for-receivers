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

#

COLORS_DICT = {
    'ViterbiNet - Regular Training': 'green',
    'ViterbiNet - SMOTE': 'yellow',
    'ViterbiNet - Borderline SMOTE': 'orange',
    'ViterbiNet - Flipping': 'blue',
    'ViterbiNet - Adaptive': 'purple',
    'ViterbiNet - Adaptive + Flipping': 'red',
    'ViterbiNet - Adaptive + Flipping + Borderline SMOTE': 'black',
    'ViterbiNet - Extended Pilot Size (200)': 'pink',
    'ViterbiNet - Extended Pilot Size (400)': 'pink',
    'ViterbiNet - Max Pilot Size (2000)': 'pink',
}

MARKERS_DICT = {
    'ViterbiNet - Regular Training': 'o',
    'ViterbiNet - SMOTE': '>',
    'ViterbiNet - Borderline SMOTE': '>',
    'ViterbiNet - Flipping': 'x',
    'ViterbiNet - Adaptive': 'd',
    'ViterbiNet - Adaptive + Flipping': '+',
    'ViterbiNet - Adaptive + Flipping + Borderline SMOTE': '>',
    'ViterbiNet - Extended Pilot Size (200)': 'o',
    'ViterbiNet - Extended Pilot Size (400)': 'o',
    'ViterbiNet - Max Pilot Size (2000)': 'o',
}

LINESTYLES_DICT = {
    'ViterbiNet - Regular Training': 'dashed',
    'ViterbiNet - SMOTE': 'solid',
    'ViterbiNet - Borderline SMOTE': 'solid',
    'ViterbiNet - Flipping': 'solid',
    'ViterbiNet - Adaptive': 'solid',
    'ViterbiNet - Adaptive + Flipping': 'solid',
    'ViterbiNet - Adaptive + Flipping + Borderline SMOTE': 'solid',
    'ViterbiNet - Extended Pilot Size (200)': 'dashed',
    'ViterbiNet - Extended Pilot Size (400)': 'dashed',
    'ViterbiNet - Max Pilot Size (2000)': 'dashed'
}
