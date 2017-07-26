import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import scipy.io as sio
from surfer import Brain

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures'

# surfer parameters
subject_id = "fsaverage"
subjects_dir = os.environ["SUBJECTS_DIR"]

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/FT/Predictions'))
n_subjects = len(subjlist)
scores_spc = np.load('%s/../scores_sid_pid_cat.npy')

# 
importance_allctg = np.zeros((0, 146, 48))

# separate plot for each category
for cid, category in enumerate(categories):

    print 'Drawing "%s" ...' % category

    subdir = 'FT_importances_%d_%s' % (cid, category)
    try:
        os.mkdir('%s/%s' % (OUTDIR, subdir))
    except:
        pass

    importance = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid))
    successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))
    successful_mnis = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid))
    successful_areas = np.load('%s/%s' % (INDIR, 'FT_successful_areas_ctg%d.npy' % cid))
    
    importance_allctg = np.concatenate((importance_allctg, importance))

    # importance in time
    #most_important_moment = np.argmax(np.sum(importance[:, :, :], axis=1), axis=1)
    #most_important_moment = np.argsort(np.sum((importance[:, :, 17:48] * temporal_weights), axis=(1, 2)))
    #temporal_weights = np.tile(range(17,48), len(successful_areas)*146).reshape(len(successful_areas), 146, len(range(17,48))) / 47.0

    # Plot each probe's importances
    for i in range(importance.shape[0]):

        (pid, sid) = successful_probes[i]
        pid = pid - 1
        area = successful_areas[i]
        mni = successful_mnis[i]
        
        # each BA into own directory
        try:
            os.mkdir('%s/%s/BA%d' % (OUTDIR, subdir, area))
        except:
            pass

        # each Subject into own directory
        try:
            os.mkdir('%s/%s/Subject%d' % (OUTDIR, subdir, sid))
        except:
            pass

        # figure
        fig = plt.figure(figsize=(24, 6), dpi=300);

        # feature importances
        plt.subplot(1, 3, 1);
        plt.imshow(importance[i, :, :], interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto');
        plt.colorbar();
        plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--');
        plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
        plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=5);
        plt.ylabel('Frequency (Hz)', size=10);
        plt.xlabel('Time (30 ms bin)', size=10);
        plt.title('Importance of spectrotemporal features for "%s"\nsID: %d   pID: %d   BA: %d' % (categories[cid], sid, pid, area), size=11);

        # 3D mesh
        plt.subplot(1, 3, 2);
        brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.4, background='white');
        brain.add_foci(mni, hemi='rh', scale_factor=1.0, color='blue');
        brain.show_view('m');
        pic = brain.screenshot()
        plt.imshow(pic);

        # 3D mesh
        plt.subplot(1, 3, 3);
        brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.4, background='white');
        brain.add_foci(mni, hemi='rh', scale_factor=1.0, color='blue');
        brain.show_view(view=dict(azimuth=0.0, elevation=0), roll=90);
        pic = brain.screenshot()
        plt.imshow(pic);
        
        plt.savefig('%s/%s/FT_importances_%d_%s_s%d-p%d.png' % (OUTDIR, subdir, cid, category, sid, pid), bbox_inches='tight');
        plt.savefig('%s/%s/BA%d/FT_importances_%d_%s_s%d-p%d.png' % (OUTDIR, subdir, area, cid, category, sid, pid), bbox_inches='tight');
        plt.savefig('%s/%s/Subject%d/FT_importances_%d_%s_s%d-p%d.png' % (OUTDIR, subdir, sid, cid, category, sid, pid), bbox_inches='tight');
        plt.clf();
        plt.close(fig);


    # Mono and Poly predictive probes
    try:
        os.mkdir('%s/%s/Monopredictive' % (OUTDIR, subdir))
        os.mkdir('%s/%s/Polypredictive' % (OUTDIR, subdir))
    except:
        pass
    

    """
    for ba in np.unique(successful_areas):

        #most_important_moment = np.argmax(np.sum(np.mean(importance[successful_areas == ba, :, :], axis=0), axis=0))

        baprobes = np.mean(importance[successful_areas == ba, :, :], axis=0)
        maxfreqs = baprobes[:, 0].argsort()[::-1][:10]
        most_important_moment = np.argmax(np.sum(np.mean(importance[successful_areas == ba, :, :], axis=0)[maxfreqs, :], axis=0))

        fig = plt.figure(figsize=(16, 6), dpi=300);
        plt.subplot(1, 2, 1);
        plt.imshow(np.mean(importance[successful_areas == ba, :, :], 0), interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto');
        plt.colorbar();
        plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
        plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
        plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=5);
        plt.ylabel('Frequency (Hz)', size=10);
        plt.xlabel('Time (30 ms bin)', size=10);
        plt.title('Importance of spectrotemporal features for "%s"\nmean over %d probes in BA%d' % (categories[cid], np.sum(successful_areas == ba), ba), size=11);

        # 3D mesh
        plt.subplot(1, 2, 2);
        brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.4, background='white');
        brain.show_view(view=dict(azimuth=152.88, elevation=65.94), roll=101.58);
        brain.add_foci(successful_mnis[successful_areas == ba], hemi='rh', scale_factor=0.5, color='blue');
        pic = brain.screenshot()
        plt.imshow(pic);

        plt.savefig('%s/%s/Time/FT_importances_%d_%s_time%d_MEAN_BA%d.png' % (OUTDIR, subdir, cid, category, most_important_moment, ba), bbox_inches='tight');
        plt.clf();
        plt.close(fig);
    """

    # Mean over each BA
    for ba in np.unique(successful_areas):
    
        fig = plt.figure(figsize=(24, 6), dpi=300);
        plt.subplot(1, 3, 1);
        plt.imshow(np.mean(importance[successful_areas == ba, :, :], 0), interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto');
        plt.colorbar();
        plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
        plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
        plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=5);
        plt.ylabel('Frequency (Hz)', size=10);
        plt.xlabel('Time (30 ms bin)', size=10);
        plt.title('Importance of spectrotemporal features for "%s"\nmean over %d probes in BA%d' % (categories[cid], np.sum(successful_areas == ba), ba), size=11);

        # 3D mesh
        plt.subplot(1, 3, 2);
        brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.4, background='white');
        brain.add_foci(successful_mnis[successful_areas == ba], hemi='rh', scale_factor=0.5, color='blue');
        brain.show_view('m');
        pic = brain.screenshot()
        plt.imshow(pic);

        # 3D mesh
        plt.subplot(1, 3, 3);
        brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.4, background='white');
        brain.add_foci(successful_mnis[successful_areas == ba], hemi='rh', scale_factor=0.5, color='blue');
        brain.show_view(view=dict(azimuth=0.0, elevation=0), roll=90);
        pic = brain.screenshot()
        plt.imshow(pic);

        plt.savefig('%s/%s/FT_importances_%d_%s_MEAN_BA%d.png' % (OUTDIR, subdir, cid, category, ba), bbox_inches='tight');
        plt.clf();
        plt.close(fig);


    # Mean over all whole category
    fig = plt.figure(figsize=(24, 6), dpi=300);
    plt.subplot(1, 3, 1);
    plt.imshow(np.mean(importance, 0), interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto');
    plt.colorbar();
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
    plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=5);
    plt.ylabel('Frequency (Hz)', size=10);
    plt.xlabel('Time (30 ms bin)', size=10);
    plt.title('Importance of spectrotemporal features for "%s"' % categories[cid], size=11);

    # 3D mesh: 
    plt.subplot(1, 3, 2);
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.4, background='white');
    #brain.show_view(view=dict(azimuth=152.88, elevation=65.94), roll=101.58);
    brain.add_foci(successful_mnis, hemi='rh', scale_factor=0.5, color='blue');
    brain.show_view('m');
    pic = brain.screenshot()
    plt.imshow(pic);

    # 3D mesh: 
    plt.subplot(1, 3, 3);
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.4, background='white');
    #brain.show_view(view=dict(azimuth=152.88, elevation=65.94), roll=101.58);
    brain.add_foci(successful_mnis, hemi='rh', scale_factor=0.5, color='blue');
    brain.show_view(view=dict(azimuth=0.0, elevation=0), roll=90);
    pic = brain.screenshot()
    plt.imshow(pic);

    plt.savefig('%s/FT_importances_%d_%s_MEAN.png' % (OUTDIR, cid, category), bbox_inches='tight');
    plt.clf();
    plt.close(fig);
    