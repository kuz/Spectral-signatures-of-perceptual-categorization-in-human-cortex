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
from matplotlib.colors import LinearSegmentedColormap
import scipy.io as sio
from scipy.stats import mannwhitneyu
from surfer import Brain
import pdb

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
    if title is not None:
        plt.title(title, size=16);

def panel_fitfdiff(diff, title, lines):
    cm2rb = LinearSegmentedColormap.from_list('red_blue', ['blue', 'white', 'gray', 'red'], N=4)
    plt.imshow(diff, interpolation='none', origin='lower', cmap=cm2rb, aspect='auto', vmin=-1.0, vmax=2.0);
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
    if title is not None:
        plt.title(title, size=16);

def panel_curves(curves, types, colors, legend, title):
    for i in range(len(curves)):
        plt.plot(curves[i], linestyle=types[i], color=colors[i], linewidth=2, label=legend[i]);
    plt.legend();
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    plt.xticks(np.arange(0, 48, 1.5), np.asarray((np.arange(0, 769, 24) - 256) / 512.0 * 1000, dtype='int'), size=11, rotation=90);
    plt.xlabel('Time (ms)', size=16);
    plt.ylabel('Feature imporatance', size=16);
    plt.title(title, size=16);

def panel_time_profile(v, title, ylabel, lines):
    plt.bar(range(len(v)), v, color="blue")
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    plt.xticks(np.arange(0, 48, 2), np.asarray((np.arange(0, 769, 32) - 256) / 512.0 * 1000, dtype='int'), size=11, rotation=90);
    #plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=12);
    plt.ylabel(ylabel, size=16);
    plt.xlabel('Time (ms)', size=16);
    plt.title(title, size=16);

def panel_mesh_sagittal(foci, foci_colors, color_scales):
    foci_colors = np.array(foci_colors)
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.1, background='white');
    for color in np.unique(foci_colors):
        brain.add_foci(foci[foci_colors==color, :], hemi='rh', scale_factor=color_scales[color], color=color);
    brain.show_view('m');
    pic = brain.screenshot()
    plt.imshow(pic);
    plt.xlabel('MNI Y', size=16);
    plt.xticks(np.arange(0, 800, 20), np.asarray(np.arange(-75, 106, 4.5), dtype='int'), size=10, rotation=90);
    plt.ylabel('MNI Z', size=16);
    plt.yticks(np.arange(0, 800, 20), np.asarray(np.arange(96, -87, -4.372), dtype='int'), size=10);
    plt.xlim(0, 800);
    plt.ylim(800, 0);

def panel_mesh_axial(foci, foci_colors, color_scales):
    foci_colors = np.array(foci_colors)
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.1, background='white');
    for color in np.unique(foci_colors):
        brain.add_foci(foci[foci_colors==color, :], hemi='rh', scale_factor=color_scales[color], color=color);
    brain.show_view(view=dict(azimuth=0.0, elevation=0), roll=90);
    pic = brain.screenshot()
    plt.imshow(pic);
    plt.xlabel('MNI Y', size=16);
    plt.xticks(np.arange(0, 800, 20), np.asarray(np.arange(-75, 106, 4.5), dtype='int'), size=10, rotation=90);
    plt.ylabel('MNI X', size=16);
    plt.yticks(np.arange(0, 800, 20), np.asarray(np.arange(-94.5, 93.5, 4.7), dtype='int'), size=10);
    plt.xlim(0, 800);
    plt.ylim(800, 0);

def duoptych(foci, foci_colors, color_scales, filenames):
    
    fig = plt.figure(figsize=(16, 8), dpi=200);

    # sagittal
    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
    panel_mesh_sagittal(foci, foci_colors, color_scales)
    #ax1.text(10, 39, '1', fontsize=20)

    # axial
    ax1 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)
    panel_mesh_axial(foci, foci_colors, color_scales)
    #ax1.text(10, 39, '2', fontsize=20)

    # store the figure
    for filename in filenames:
        plt.savefig(filename, bbox_inches='tight');
    plt.clf();
    plt.close(fig);

def triptych_importance(tfdata, foci, foci_colors, color_scales, filenames, title=None, lines=True):
    
    fig = plt.figure(figsize=(28, 8), dpi=200);

    # importances
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)
    panel_importance(tfdata, title, lines)

    # sagittal
    ax1 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)
    panel_mesh_sagittal(foci, foci_colors, color_scales)

    # axial
    ax1 = plt.subplot2grid((1, 3), (0, 2), colspan=1, rowspan=1)
    panel_mesh_axial(foci, foci_colors, color_scales)

    # store the figure
    for filename in filenames:
        plt.savefig(filename, bbox_inches='tight');
    plt.clf();
    plt.close(fig);

def triptych_fitfdiff(tfdata, foci, foci_colors, color_scales, filenames, title=None, lines=True):
    fig = plt.figure(figsize=(28, 8), dpi=200);
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)
    panel_fitfdiff(tfdata, title, lines)
    ax1 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)
    panel_mesh_sagittal(foci, foci_colors, color_scales)
    ax1 = plt.subplot2grid((1, 3), (0, 2), colspan=1, rowspan=1)
    panel_mesh_axial(foci, foci_colors, color_scales)
    for filename in filenames:
        plt.savefig(filename, bbox_inches='tight');
    plt.clf();
    plt.close(fig);
