# -*- coding: utf-8 -*-

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
try:
    from surfer import Brain
    withsurfer = True
except:
    print "\nWARNING: PySurfer not available, 3D plotting will not work!\n"
    withsurfer = False
import pdb

# surfer parameters
subject_id = "fsaverage"

def panel_importance(importances, title, lines):
    plt.imshow(importances, interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto');
    cb = plt.colorbar();
    cb.ax.tick_params(labelsize=16);
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    if lines:
        plt.axvline(x=19, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=24, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=32, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=4, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=10, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=27, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=56, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
    
    # small axis 
    #plt.xticks(np.arange(0, 48, 1.5), np.asarray((np.arange(0, 769, 24) - 256) / 512.0 * 1000, dtype='int'), size=11, rotation=90);
    #plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=12);
    
    # med axis
    plt.xticks(np.arange(0, 48, 2), np.asarray((np.arange(0, 769, 32) - 256) / 512.0 * 1000, dtype='int'), size=18, rotation=90);
    plt.yticks(np.arange(0, 146, 7), np.arange(4, 150, 7), size=18);
    
    # big axis
    #plt.xticks(np.arange(0, 48, 3), np.asarray((np.arange(0, 769, 48) - 256) / 512.0 * 1000, dtype='int'), size=24, rotation=90);
    #plt.yticks(np.arange(0, 146, 9), np.arange(4, 150, 9), size=20);
    
    plt.ylabel('Frequency (Hz)', size=24);
    plt.xlabel('Time (ms)', size=24);
    if title is not None:
        plt.title(title, size=26);

def panel_activity(activity, title, lines, mask=None):
    plt.imshow(activity, interpolation='none', origin='lower', cmap=cm.bwr, aspect='auto');
    cb = plt.colorbar();
    plt.clim(-4.0, 4.0)
    if mask is not None:
        plt.contour(mask, mask, colors='black', vmin=0.0, vmax=1.0, extend='max', linewidths=1.0)
    cb.ax.tick_params(labelsize=16);
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    if lines:
        plt.axvline(x=19, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=24, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=32, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=4, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=10, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=27, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=56, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
    plt.xticks(np.arange(0, 48, 2), np.asarray((np.arange(0, 769, 32) - 256) / 512.0 * 1000, dtype='int'), size=18, rotation=90);
    plt.yticks(np.arange(0, 146, 7), np.arange(4, 150, 7), size=18);
    plt.ylabel('Frequency (Hz)', size=24);
    plt.xlabel('Time (ms)', size=24);
    if title is not None:
        plt.title(title, size=26);

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
    
    # small axis
    #plt.xticks(np.arange(0, 48, 1.5), np.asarray((np.arange(0, 769, 24) - 256) / 512.0 * 1000, dtype='int'), size=16, rotation=90);
    #plt.yticks(np.arange(0, 146, 7), np.arange(4, 150, 7), size=16);
    
    # big axis
    plt.xticks(np.arange(0, 48, 3), np.asarray((np.arange(0, 769, 48) - 256) / 512.0 * 1000, dtype='int'), size=24, rotation=90);
    plt.yticks(np.arange(0, 146, 9), np.arange(4, 150, 9), size=24);

    plt.ylabel('Frequency (Hz)', size=24);
    plt.xlabel('Time (ms)', size=24);
    if title is not None:
        plt.title(title, size=16);

def panel_curves(curves, types, colors, legend, title):
    for i in range(len(curves)):
        plt.plot(curves[i], linestyle=types[i], color=colors[i], linewidth=2, label=legend[i]);
    plt.legend();
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    plt.xticks(np.arange(0, 48, 2), np.asarray((np.arange(0, 769, 32) - 256) / 512.0 * 1000, dtype='int'), size=11, rotation=90);
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
    if not withsurfer:
        print "Cannot do 3D without PySurder. Exiting"
        exit()

    foci_colors = np.array(foci_colors)
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.1, background='white');
    for color in np.unique(foci_colors):
        brain.add_foci(foci[foci_colors==color, :], hemi='rh', scale_factor=color_scales[color], color=color);
    brain.show_view('m');
    pic = brain.screenshot()
    plt.imshow(pic);

    # small axis
    #plt.xlabel('MNI Y', size=24);
    #plt.xticks(np.arange(0, 800, 40), np.asarray(np.arange(-75, 106, 9.0), dtype='int'), size=18, rotation=90);
    #plt.ylabel('MNI Z', size=24);
    #plt.yticks(np.arange(0, 800, 40), np.asarray(np.arange(96, -87, -8.744), dtype='int'), size=18);
    
    # big axis
    plt.xlabel('MNI Y', size=24);
    plt.xticks(np.arange(0, 800, 50), np.asarray(np.arange(-75, 106, 11.3125), dtype='int'), size=24, rotation=90);
    plt.ylabel('MNI Z', size=24);
    plt.yticks(np.arange(0, 800, 50), np.asarray(np.arange(96, -87, -11.4375), dtype='int'), size=24);

    plt.xlim(0, 800);
    plt.ylim(800, 0);

def panel_mesh_axial(foci, foci_colors, color_scales):
    if not withsurfer:
        print "Cannot do 3D without PySurder. Exiting"
        exit()

    foci_colors = np.array(foci_colors)
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.1, background='white');
    for color in np.unique(foci_colors):
        brain.add_foci(foci[foci_colors==color, :], hemi='rh', scale_factor=color_scales[color], color=color);
    brain.show_view(view=dict(azimuth=0.0, elevation=0), roll=90);
    pic = brain.screenshot()
    plt.imshow(pic);

    # small axis
    #plt.xlabel('MNI Y', size=24);
    #plt.xticks(np.arange(0, 800, 40), np.asarray(np.arange(-75, 106, 9.0), dtype='int'), size=18, rotation=90);
    #plt.ylabel('MNI X', size=24);
    #plt.yticks(np.arange(0, 800, 40), np.asarray(np.arange(-94.5, 93.5, 9.4), dtype='int'), size=18);
    
    # big axis
    plt.xlabel('MNI Y', size=24);
    plt.xticks(np.arange(0, 800, 50), np.asarray(np.arange(-75, 106, 11.3125), dtype='int'), size=24, rotation=90);
    plt.ylabel('MNI Z', size=24);
    plt.yticks(np.arange(0, 800, 50), np.asarray(np.arange(-94.5, 93.5, 11.75), dtype='int'), size=24);

    plt.xlim(0, 800);
    plt.ylim(800, 0);

def panel_cluster_mean(cluster_id, cluster_means, cluster_predictive_score, cluster_poly_proportion, count, vlim, color, cid, lines=True, xlabels=True, ylabels=True):
    plt.imshow(cluster_means[cluster_id], interpolation='none', origin='lower', cmap=cm.bwr, aspect=0.3, vmin=-vlim, vmax=vlim);
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--');
    if lines:
        plt.axvline(x=19, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=24, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=32, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=4, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=10, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=27, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=56, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
    if xlabels:
        plt.xticks(np.arange(0, 48, 3), np.asarray((np.arange(0, 769, 48) - 256) / 512.0 * 1000, dtype='int'), size=15, rotation=90);
    else:
        plt.xticks([])
    if ylabels:
        plt.yticks(np.arange(0, 146, 10), np.arange(4, 150, 10), size=15);
        #plt.ylabel('Frequency (Hz)', size=16);
    else:
        plt.yticks([])
    plt.title('Activity of %s (%d) probes' % (color.upper(), count), size=16, color=color);
    plt.text(1, 129, str(cluster_id + 1), fontsize=20)
    per_cid_stds = [0.11, 0.10, 0.10, 0.09, 0.10, 0.08, 0.10, 0.12]
    plt.text(26, 132, 'F1 = %.4f ± %.2f' % (cluster_predictive_score[cluster_id], per_cid_stds[cid]), fontsize=15)
    plt.text(31, 118, 'Poly %d' % cluster_poly_proportion[cluster_id] + '%', fontsize=15)

def panel_boxplots(data, significance, title, xlab, ylab):
    if title == 'visage':
        title = 'face'

    bp = plt.boxplot(data, widths=0.5)
    plt.ylim((0.0, 1.0))    
    plt.setp(bp['boxes'], color='blue')
    plt.setp(bp['whiskers'], color='blue')
    plt.setp(bp['fliers'], color='blue')
    plt.setp(bp['medians'], color='blue')
    plt.setp(bp['caps'], color='blue')
    plt.title(title, size=12)
    
    # labels
    if ylab:
        plt.yticks(np.arange(0.0, 1.1, 0.1), np.round(np.arange(0.0, 1.1, 0.1), 1), size=8);
    else:
        plt.yticks([], []);

    if xlab:
        if len(data) == 3:
            plt.xticks([1, 2, 3], ['full\n4-150\nHz', 'bbgamma\n51-150\nHz', 'lowfreq\n4-50\nHz'], size=8, rotation=0);
        if len(data) == 4:
            plt.xticks([1, 2, 3, 4], ['C1', 'C2', 'C3', 'C4'], size=8, rotation=0);
    else:
        plt.xticks([], []);
    
    plt.axhline(y=0.125, linestyle='--', color='gray', linewidth=0.8)
    plt.axhline(y=0.390278, linestyle='--', color='gray', linewidth=0.8)

    # significance brackets
    h = 0.02

    if len(data) == 3:
        if significance[0] is not None:
            y = 0.83
            plt.plot([1, 1, 2-0.02, 2-0.02], [y, y+h, y+h, y], lw=0.5, c='black')
            plt.text((1 + 2) * .5, y + h, significance[0], ha='center', va='bottom', color='black', size=7)
        if significance[1] is not None:
            y = 0.83
            plt.plot([2+0.02, 2+0.02, 3, 3], [y, y+h, y+h, y], lw=0.5, c='black')
            plt.text((2 + 3) * .5, y + h, significance[1], ha='center', va='bottom', color='black', size=7)
        if significance[2] is not None:
            y = 0.91
            plt.plot([1, 1, 3, 3], [y, y+h, y+h, y], lw=0.5, c='black')
            plt.text((1 + 3) * .5, y + h, significance[2], ha='center', va='bottom', color='black', size=7)

    if len(data) == 4:
        threshold = 0.05 / (8 * 6)
        if significance[0] < threshold:
            y = 0.83
            plt.plot([1, 1, 2-0.02, 2-0.02], [y, y+h, y+h, y], lw=0.4, c='black')
            plt.text((1 + 2) * .5, y + h, '%.5f' % significance[0], ha='center', va='bottom', color='black', size=7)
        if significance[1] < threshold:
            y = 0.83
            plt.plot([2+0.02, 2+0.02, 3, 3], [y, y+h, y+h, y], lw=0.4, c='black')
            plt.text((2 + 3) * .5, y + h, '%.5f' % significance[1], ha='center', va='bottom', color='black', size=7)
        if significance[2] < threshold:
            y = 0.83
            plt.plot([3+0.02, 2+0.02, 3, 3], [y, y+h, y+h, y], lw=0.4, c='black')
            plt.text((2 + 3) * .5, y + h, '%.5f' % significance[2], ha='center', va='bottom', color='black', size=7)
        if significance[3] < threshold:
            y = 0.91
            plt.plot([1, 1, 3, 3], [y, y+h, y+h, y], lw=0.5, c='black')
            plt.text((1 + 3) * .5, y + h, '%.5f' % significance[3], ha='center', va='bottom', color='black', size=7)
        if significance[4] < threshold:
            y = 0.91
            plt.plot([2, 2, 4, 4], [y, y+h, y+h, y], lw=0.5, c='black')
            plt.text((2 + 4) * .5, y + h, '%.5f' % significance[4], ha='center', va='bottom', color='black', size=7)
        if significance[5] < threshold:
            y = 0.91
            plt.plot([1, 1, 4, 4], [y, y+h, y+h, y], lw=0.5, c='black')
            plt.text((1 + 4) * .5, y + h, '%.5f' % significance[4], ha='center', va='bottom', color='black', size=7)



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

def quadriptych(importances, foci, foci_colors, cluster_means, cluster_predictive_score, cluster_poly_proportion, cid, title, filenames, lines=True):
    
    fig = plt.figure(figsize=(40, 8), dpi=300);
    vlim = np.max([np.abs(np.min(cluster_means)), np.abs(np.max(cluster_means))]) * 1.2

    # overall importances
    ax1 = plt.subplot2grid((2, 8), (0, 0), colspan=2, rowspan=2)
    panel_importance(importances, title, lines)

    # 4 most prominents clusters of activity under the important regions
    ax1 = plt.subplot2grid((2, 8), (0, 2))
    panel_cluster_mean(0, cluster_means, cluster_predictive_score, cluster_poly_proportion, np.sum(foci_colors == 'green'), vlim, 'green', cid, lines=True, xlabels=False, ylabels=True)
    ax1 = plt.subplot2grid((2, 8), (0, 3))
    panel_cluster_mean(1, cluster_means, cluster_predictive_score, cluster_poly_proportion, np.sum(foci_colors == 'blue'), vlim, 'blue', cid, lines=True, xlabels=False, ylabels=False)
    ax1 = plt.subplot2grid((2, 8), (1, 2))
    panel_cluster_mean(2, cluster_means, cluster_predictive_score, cluster_poly_proportion, np.sum(foci_colors == 'red'), vlim, 'red', cid, lines=True, xlabels=True, ylabels=True)
    ax1 = plt.subplot2grid((2, 8), (1, 3))
    panel_cluster_mean(3, cluster_means, cluster_predictive_score, cluster_poly_proportion, np.sum(foci_colors == 'black'), vlim, 'black', cid, lines=True, xlabels=True, ylabels=False)

    # sagittal
    ax1 = plt.subplot2grid((2, 8), (0, 4), colspan=2, rowspan=2)
    panel_mesh_sagittal(foci, foci_colors, {'blue': 0.6, 'black': 0.6, 'red': 0.6, 'green': 0.6, 'whitesmoke': 0.6})

    # axial
    ax1 = plt.subplot2grid((2, 8), (0, 6), colspan=2, rowspan=2)
    panel_mesh_axial(foci, foci_colors, {'blue': 0.6, 'black': 0.6, 'red': 0.6, 'green': 0.6, 'whitesmoke': 0.6})

    # store the figure
    for filename in filenames:
        plt.savefig(filename, bbox_inches='tight');
    plt.clf();
    plt.close(fig);
