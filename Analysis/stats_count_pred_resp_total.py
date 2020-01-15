import os
import numpy as np
import scipy.io as sio

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Single Probe Classification/FT/Importances'

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
cat_number = [10, 20, 30, 40, 50, 60, 70, 90]
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/FT/Predictions'))

# results aggreagator
counts = {}

# process each subject
for subject in subjlist:
    sname = subject.replace('.npy', '')
    print 'Processing', sname, '...'
    counts[sname] = {}

    s_all = sio.loadmat('%s/%s/%s' % (DATADIR, 'LFP_8c_artif_bipolar_BA', subject.replace('.npy', '.mat')))
    mni_all = s_all['s']['probes'][0][0][0][0][2]
    mni_all = np.round(mni_all, 2)
    areas_all = np.ravel(s_all['s']['probes'][0][0][0][0][3])
    for pid, mni in enumerate(mni_all):
        mni_str = str(mni)
        counts[sname][mni_str] = {}
        counts[sname][mni_str]['area'] = areas_all[pid]
        counts[sname][mni_str]['resp'] = []
        counts[sname][mni_str]['pred'] = []

    for cid in range(len(categories)):
        s_res_ctg = sio.loadmat('%s/%s_cat%d/%s' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive', cat_number[cid], subject.replace('.npy', '.mat')))
        #s_res_ctg = sio.loadmat('%s/%s/%s' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive', subject.replace('.npy', '.mat')))
        mni_res_ctg = s_res_ctg['s']['probes'][0][0][0][0][2]
        mni_res_ctg = np.round(mni_res_ctg, 2)
        for mni in mni_res_ctg:
            mni_str = str(mni)
            try:
                counts[sname][mni_str]['resp'].append(cid)
            except:
                print "WARNING: ONE PROBE MISSING"
                    
    
# separate plot for each category
for cid, category in enumerate(categories):
    print 'Processing category', category, '...'

    predictive_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' %  cid))

    # Plot each probe's importances
    for i in range(predictive_probes.shape[0]):

        # load MNI information
        (pid, sid) = predictive_probes[i]
        pid = pid - 1
        s_pred = sio.loadmat('%s/%s/%s' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive', subjlist[sid].replace('.npy', '.mat')))
        sname = subjlist[sid].replace('.npy', '')
        mni_pred = s_pred['s']['probes'][0][0][0][0][2]
        mni_pred = np.round(mni_pred, 2)
        mni_str = str(mni_pred[pid])
        try:
            counts[sname][mni_str]['pred'].append(cid)
        except:
            print "WARNING: ONE PROBE MISSING"


# aggreate counts
table_pred = np.zeros((50, 8))
table_resp = np.zeros((50, 8))
for sname in counts.keys():
    for mni in counts[sname].keys():

        for cid in counts[sname][mni]['pred']:
            table_pred[counts[sname][mni]['area'], cid] += 1

        for cid in counts[sname][mni]['resp']:
            table_resp[counts[sname][mni]['area'], cid] += 1

table_pred = table_pred.astype(int)
table_resp = table_resp.astype(int)

pred_per_ba = np.sum(table_pred, 1)
resp_per_ba = np.sum(table_resp, 1)
pred_per_ctg = np.sum(table_pred, 0)
resp_per_ctg = np.sum(table_resp, 0)

# generate LaTeX output
for bid in range(50):
    line = 'BA %d &  & ' % (bid)
    for cid in range(8):
        line += str(table_pred[bid, cid]) + ' \\tiny{' + str(table_resp[bid, cid]) + '} &'
    line += '%d & %d \\\\' % (pred_per_ba[bid], resp_per_ba[bid])
    print line

line = ' & & '
for cid in range(8):
    line += str(pred_per_ctg[cid]) + ' \\tiny{' + str(resp_per_ctg[cid]) + '} &'
print line
