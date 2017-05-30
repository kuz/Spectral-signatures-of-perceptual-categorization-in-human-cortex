import os
import numpy as np
import scipy.io as sio

DATADIR = '../../Data/Intracranial/Processed'
subjects = os.listdir('%s/%s/' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive'))
featureset = 'ft_4hz150_LFP_8c_artif_bipolar_BA_responsive'

for sfile in subjects:
    print '---', sfile, '---'
    s = sio.loadmat('%s/%s/%s' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive', sfile))
    areas = np.ravel(s['s']['probes'][0][0][0][0][3])
    sname = s['s']['name'][0][0][0]
    n_probes = len(areas)
    for pid in range(1, n_probes + 1):
        fname = '%s/%s/%s-%d.mat' % (DATADIR, featureset, sname, pid)
        if not os.path.isfile(fname):
            print fname, 'not found!'