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
               'ViterbiNet - Adaptive': 'purple',
               'ViterbiNet - Flipping': 'blue',
               'ViterbiNet - Random OverSampling': 'black',
               'ViterbiNet - SMOTE': 'yellow',
               'ViterbiNet - Borderline SMOTE': 'orange',
               'ViterbiNet - Adaptive + Flipping': 'red'}

MARKERS_DICT = {'ViterbiNet - Regular Training': 'o',
                'ViterbiNet - Adaptive': 'd',
                'ViterbiNet - Random OverSampling': '>',
                'ViterbiNet - Flipping': 'x',
                'ViterbiNet - SMOTE': '>',
                'ViterbiNet - Borderline SMOTE': '>',
                'ViterbiNet - Adaptive + Flipping': '+'}

LINESTYLES_DICT = {'ViterbiNet - Regular Training': 'solid',
                   'ViterbiNet - Adaptive': 'solid',
                   'ViterbiNet - Random OverSampling': 'solid',
                   'ViterbiNet - Flipping': 'solid',
                   'ViterbiNet - SMOTE': 'solid',
                   'ViterbiNet - Borderline SMOTE': 'solid',
                   'ViterbiNet - Adaptive + Flipping': 'solid'}
