import os
import numpy as np
import scipy.io as sio

# paths
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Data/Intracranial/Calculated'

# parameters
featureset = 'LFP_8c_artif_bipolar_BA_responsive'

# load subject list
subjects = os.listdir('%s/%s/' % (DATADIR, featureset))
if len(subjects) == 0:
    raise Exception('Data for featureset "%s" does not exist.' % self.featureset)

dataset = {}
dataset['mnis'] = np.zeros((0, 3))

# load neural responses
for sfile in subjects:
    print 'Processing %s' % sfile
    s = sio.loadmat('%s/%s/%s' % (DATADIR, featureset, sfile))
    sname = s['s']['name'][0][0][0]
    mnis = s['s']['probes'][0][0][0][0][2]
    if mnis.shape[0] > 0:
        dataset['mnis'] = np.concatenate((dataset['mnis'], mnis))

# store the dataset
np.save('%s/all_mnis.npy' % OUTDIR, dataset['mnis'])
