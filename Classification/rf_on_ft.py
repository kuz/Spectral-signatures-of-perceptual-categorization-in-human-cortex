import os
import argparse
import numpy as np
import scipy.io as sio
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

parser = argparse.ArgumentParser(description='Creates various datasets out of the processed intracranial data')
parser.add_argument('-f', '--featureset', dest='featureset', type=str, required=True, help='Directory with brain features (Processed/?)')
args = parser.parse_args()
featureset = str(args.featureset)

#: Paths
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = 

#: Data parameters
nstim = 401
n_cv = 5

#: List of subjects
subjects = os.listdir('%s/%s/' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive'))
if len(self.subjects) == 0:
    raise Exception('Data for featureset "%s" does not exist.' % self.featureset)

# prepare the data structure
#dataset = {}
#dataset['neural_responses'] = np.zeros((nstim, 0))
#dataset['areas'] = np.zeros(0)
#dataset['subjects'] = []

# load neural responses
for sfile in subjects:
    s = sio.loadmat('%s/%s/%s' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive', sfile))
    sname = s['s']['name'][0][0][0]
    areas = np.ravel(s['s']['probes'][0][0][0][0][3])
    stimgroups = np.ravel(s['s']['stimgroups'][0][0])
    skf = StratifiedKFold(stimgroups, n_folds=n_cv)
    
    for pid in range(1, len(areas) + 1):

        # load frequency data for the probe
        ft = sio.loadmat('%s/%s/%s-%d.mat' % (DATADIR, featureset, sname, pid))
        ft = ft['ft'].reshape(ft['ft'].shape[0], ft['ft'].shape[1] * ft['ft'].shape[2])

        # train a classifier
        clf = RandomForestClassifier(n_estimators=1000, n_jobs=8)
        predicted = cross_val_predict(clf, ft, stimgroups, cv=skf)
        
        #print confusion_matrix(stimgroups, predicted)
        #print pid, '\t', f1_score(stimgroups, predicted, average='weighted')
        print pid, '\t', np.max(f1_score(stimgroups, predicted, average=None))

