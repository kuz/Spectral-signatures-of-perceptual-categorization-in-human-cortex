import os
import numpy as np
from sklearn.metrics import f1_score

FREQSET = 'FT'
INDIR   = '../../Outcome/Single Probe Classification/%s/Predictions' % FREQSET
OUTDIR  = '../../Outcome/Single Probe Classification/%s' % FREQSET

n_classes = 8
filelist = sorted(os.listdir(INDIR))
n_subjects = len(filelist)
threshold = 0.390278 # estimated as 99.999 percentile over permutation F1 scores
#threshold = 0.0
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']

f1_scores = np.zeros((n_classes, 200, n_subjects))

standard_deviations = [[] for i in range(8)]
for sid, filename in enumerate(filelist):
    if sid in [9, 24, 36, 39, 69, 79]:
        continue
    print sid
    data = np.load('%s/%s' % (INDIR, filename), allow_pickle=True)
    for pid in data[()].keys():
        sc = np.where(f1_score(data[()][pid]['true'], data[()][pid]['pred'], average=None) > threshold)[0]
        if len(sc) > 0:
            for cid in sc:
                fold_scores = [f1_score(data[()][pid]['true'][fold[0]:fold[1]], data[()][pid]['pred'][fold[0]:fold[1]], average=None)[cid] for fold in [(0, 80), (80, 160), (160, 240), (240, 320), (320, 400)]]
                standard_deviations[cid].append(np.std(fold_scores))

all_stds = []
for cid in range(8):
    print categories[cid], ':', np.mean(standard_deviations[cid])
    all_stds += standard_deviations[cid]

print 'ALL :', np.mean(all_stds)