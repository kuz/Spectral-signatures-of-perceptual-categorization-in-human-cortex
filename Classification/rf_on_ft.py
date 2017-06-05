import os
import argparse
import numpy as np
import scipy.io as sio
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

parser = argparse.ArgumentParser(description='Train separate RF on each probe of a given subject and store predictions')
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
    OUTDIR = '../../Outcome/Single Probe Classification/FT/Permutation'
else:
    OUTDIR = '../../Outcome/Single Probe Classification/FT/Predictions'

#: Data parameters
nstim = 401
n_cv = 5
n_freqs = 146

#: List of subjects
subjects = os.listdir('%s/%s/' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive'))
if len(subjects) == 0:
    raise Exception('Data for featureset "%s" does not exist.' % featureset)

# prepare the data structure
cms = {}

# load neural responses
sfile = subjects[sid]
s = sio.loadmat('%s/%s/%s' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive', sfile))
sname = s['s']['name'][0][0][0]
print 'Processing', sname
areas = np.ravel(s['s']['probes'][0][0][0][0][3])
n_probes = len(areas)
stimgroups = np.ravel(s['s']['stimgroups'][0][0])

if n_probes > 0:

    skf = StratifiedKFold(stimgroups, n_folds=n_cv)
    
    for pid in range(1, n_probes + 1):

        # load frequency data for the probe
        ft = sio.loadmat('%s/%s/%s-%d.mat' % (DATADIR, featureset, sname, pid))

        # normalize by baseline
        baseline = ft['ft'][:, :, 0:13]
        means = np.mean(baseline, axis=2).reshape((401, n_freqs, 1))
        signal = ft['ft'] / np.broadcast_to(means, (401, n_freqs, 48))
        signal[np.isnan(signal)] = 0.0

        # reshape for RF
        signal = signal.reshape(signal.shape[0], signal.shape[1] * signal.shape[2])

        # train a classifier
        clf = RandomForestClassifier(n_estimators=3000, n_jobs=10)
        if permutation:
            predicted = cross_val_predict(clf, signal, np.random.permutation(stimgroups), cv=skf)
        else:
            predicted = cross_val_predict(clf, signal, stimgroups, cv=skf)
        clf = None
        
        # store and show results
        cms[pid] = {'true':list(stimgroups), 'pred':list(predicted)}
        print pid, '\t', np.max(f1_score(stimgroups, predicted, average=None))

np.save('%s/%s.npy' % (OUTDIR, sname), cms)

