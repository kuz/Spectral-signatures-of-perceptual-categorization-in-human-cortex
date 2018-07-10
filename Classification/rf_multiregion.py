import numpy as np
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.externals import joblib
import json
import argparse

# parse cmd arguments
parser = argparse.ArgumentParser(description='Build feature matrix from featureset')
parser.add_argument('-m', '--ms', dest='ms', type=str, required=True, help='Windows of the featuresets (Processed/?)')
args = parser.parse_args()
ms = str(args.ms)

# parameters
DATADIR = '../../Outcome/Feature matrices'
n_cv = 5

alldata = {}
alldata['image_category'] = None
alldata['neural_responses'] = None
#for hz in ['4hz8', '9hz14', '15hz30', '31hz70', '70hz150']:
for hz in ['4hz8', '9hz14', '15hz19', '20hz24', '25hz29', '30hz34', '35hz39', '40hz44', '45hz49', '50hz54', '55hz59', '60hz64',
           '65hz69', '70hz74', '75hz79', '80hz84', '85hz89', '90hz94', '95hz99', '100hz104', '105hz109', '110hz114', '115hz119',
           '120hz124', '125hz129', '130hz134', '135hz139', '140hz144', '145hz149']:
    with open('%s/featurematrix_mean_%s_%s_LFP_8c_artif_bipolar_BA_responsive.pkl' % (DATADIR, hz, ms), 'rb') as f:
        data = cPickle.load(f)
        for dropstim in [80]:
            keepidx = data['image_category'] != dropstim
            data['image_category'] = data['image_category'][keepidx]
            data['neural_responses'] = data['neural_responses'][keepidx]

        if alldata['neural_responses'] is None:
            alldata['image_category'] = data['image_category']
            alldata['neural_responses'] = data['neural_responses']
        else:
            alldata['neural_responses'] = np.concatenate((alldata['neural_responses'], data['neural_responses']), axis=1)

# uncomment for a permutation test
#alldata['image_category'] = np.random.permutation(alldata['image_category'])

# obtain CV predictions
print 'Data size', alldata['neural_responses'].shape
clf = RandomForestClassifier(n_estimators=3000)
predicted = cross_validation.cross_val_predict(clf, alldata['neural_responses'], alldata['image_category'], cv=n_cv, pre_dispatch=1, n_jobs=1)

# display results
print confusion_matrix(alldata['image_category'], predicted)
print f1_score(alldata['image_category'], predicted, average='weighted')

# store the model
#joblib.dump(clf, 'rf-multi.pkl') 