"""

To setup Freesurfer run
$ export FREESURFER_HOME=/Applications/freesurfer
$ source $FREESURFER_HOME/SetUpFreeSurfer.sh

"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import scipy.io as sio
from scipy.stats import mannwhitneyu
from surfer import Brain

# surfer parameters
subject_id = "fsaverage"

def panel_importance(importances, title, lines):
    plt.imshow(importances, interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto');
    plt.colorbar();
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    if lines:
        plt.axvline(x=19, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=24, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=32, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=4, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=10, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=27, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=56, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
    plt.xticks(np.arange(0, 48, 1.5), np.asarray((np.arange(0, 769, 24) - 256) / 512.0 * 1000, dtype='int'), size=11, rotation=90);
    plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=12);
    plt.ylabel('Frequency (Hz)', size=16);
    plt.xlabel('Time (ms)', size=16);
    plt.title(title, size=16);


def panel_time_profile(v, title, lines):
    plt.bar(range(len(v)), v, color="blue")
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    plt.xticks(np.arange(0, 48, 1.5), np.asarray((np.arange(0, 769, 24) - 256) / 512.0 * 1000, dtype='int'), size=11, rotation=90);
    #plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=12);
    plt.ylabel('Importance', size=16);
    plt.xlabel('Time (ms)', size=16);
    plt.title(title, size=16);