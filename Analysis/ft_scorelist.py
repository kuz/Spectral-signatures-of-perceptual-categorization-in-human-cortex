import os
import numpy as np
from sklearn.metrics import f1_score

FREQSET = 'lowfreq'
INDIR   = '../../Outcome/Single Probe Classification/%s/Predictions' % FREQSET
OUTDIR  = '../../Outcome/Single Probe Classification/%s' % FREQSET

n_classes = 8
filelist = sorted(os.listdir(INDIR))
n_subjects = len(filelist)
#threshold = 0.390278 # estimated as 99.999 percentile over permutation F1 scores
threshold = 0.0
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']

f1_scores = np.zeros((n_classes, 200, n_subjects))

for sid, filename in enumerate(filelist):
    data = np.load('%s/%s' % (INDIR, filename))
    for pid in data[()].keys():
        f1_scores[:, pid - 1, sid] = f1_score(data[()][pid]['true'], data[()][pid]['pred'], average=None)

succ_cat, succ_pid, succ_sid = np.where(f1_scores[:, :, :] > threshold)

scores = {}
for i in range(len(succ_cat)):
    if scores.get(succ_sid[i], None) is None:
        scores[succ_sid[i]] = {}
    if scores[succ_sid[i]].get(succ_pid[i], None) is None:
        scores[succ_sid[i]][succ_pid[i]] = {}
    scores[succ_sid[i]][succ_pid[i]][succ_cat[i]] = f1_scores[succ_cat[i], succ_pid[i], succ_sid[i]]

for sid in sorted(scores.keys()):
    print 'Subject %d - %s' % (sid, filelist[sid])
    for pid in sorted(scores[sid].keys()):
        print '    probe %d' % pid
        for cid in sorted(scores[sid][pid].keys()):
            print '        %s:\t%.4f' % (categories[cid][:6], scores[sid][pid][cid])

np.save('%s/scores_sid_pid_cat.npy' % OUTDIR, scores)