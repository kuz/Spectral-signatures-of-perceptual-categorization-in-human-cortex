import os
import argparse
import numpy as np
import scipy.io as sio
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

parser = argparse.ArgumentParser(description='Creates various datasets out of the processed intracranial data')
parser.add_argument('-f', '--featureset', dest='featureset', type=str, required=True, help='Directory with brain features (Processed/?)')
parser.add_argument('-s', '--sid', dest='sid', type=int, required=True, help='Subject ID')
parser.add_argument('-p', '--permutation', dest='permutation', type=bool, required=False, default=False, help='Whether to shuffle data for a permutation test')
args = parser.parse_args()
featureset = str(args.featureset)
sid = int(args.sid)
permutation = bool(args.permutation)

#: Paths
DATADIR = '../../Data/Intracranial/Processed'
if permutation:
    OUTDIR = '../../Outcome/Single Probe Classification/LFP/Permutation'
else:
    OUTDIR = '../../Outcome/Single Probe Classification/LFP/Predictions'

print OUTDIR

#: Data parameters
nstim = 401
n_cv = 5

#: List of subjects
subjects = os.listdir('%s/%s/' % (DATADIR, featureset))
if len(subjects) == 0:
    raise Exception('Data for featureset "%s" does not exist.' % self.featureset)


# prepare the data structure
cms = {}

# load data
sfile = subjects[sid]
s = sio.loadmat('%s/%s/%s' % (DATADIR, featureset, sfile))
sname = s['s']['name'][0][0][0]
data = s['s']['data'][0][0]
areas = np.ravel(s['s']['probes'][0][0][0][0][3])
stimgroups = np.ravel(s['s']['stimgroups'][0][0])
nprobes = len(areas)

if nprobes > 0:

    # subtract baseline mean from the signal
    means = np.mean(data[:, :, 50:230], axis=2).reshape((401, nprobes, 1))
    signal = data - np.broadcast_to(means, (401, nprobes, 768))
    #signal = signal[:, :, 256:]

    # partition the data for CV
    skf = StratifiedKFold(stimgroups, n_folds=n_cv)

    for pid in range(data.shape[1]):
        clf = RandomForestClassifier(n_estimators=1000, n_jobs=4)
        
        if permutation:
            predicted = cross_val_predict(clf, signal[:, pid, :], np.random.permutation(stimgroups), cv=skf)
        else:
            predicted = cross_val_predict(clf, signal[:, pid, :], stimgroups, cv=skf)
        
        cms[pid] = {'true':list(stimgroups), 'pred':list(predicted)}
        print pid, '\t', np.max(f1_score(stimgroups, predicted, average=None))

np.save('%s/%s.npy' % (OUTDIR, sname), cms)

