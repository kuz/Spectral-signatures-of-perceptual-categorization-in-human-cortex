import os
import argparse
import numpy as np
import scipy.io as sio
import cPickle


class DataBuilder:

    #: Paths
    DATADIR = '../Data/Intracranial/Processed'
    OUTDIR = '../Outcome'

    #: List of subjects
    subjects = None

    #: Data parameters
    featureset = None
    nstim = 269

    def __init__(self, featureset):
        self.featureset = featureset
        self.subjects = os.listdir('%s/%s/' % (self.DATADIR, self.featureset))

    def build_stim_probe_category():

        # prepare the data structure
        dataset = {}
        dataset['neural_responses'] = np.zeros((self.nstim, 0))
        dataset['areas'] = np.zeros(0)
        dataset['subjects'] = []

        # load neural responses
        for sfile in self.subjects:
            s = sio.loadmat('%s/%s/%s' % (self.DATADIR, self.featureset, sfile))
            sname = s['s']['name'][0][0][0]
            data = s['s']['data'][0][0]
            areas = np.ravel(s['s']['probes'][0][0][0][0][3])
            if len(areas) > 0:
                dataset['neural_responses'] = np.concatenate((dataset['neural_responses'], data), axis=1)
                dataset['areas'] = np.concatenate((dataset['areas'], areas))
                dataset['subjects'] += [sname] * len(areas)

        # load class labels
        dataset['image_category'] = np.loadtxt('%s/../stimgroups.txt' % DATADIR, dtype='int')

        # store the dataset
        with open('%s/stim_probe_category_ON_%s.pkl' % (self.OUTDIR, self.featureset), 'wb') as outfile:
            cPickle.dump(dataset, outfile)


if __name__ == '__main__':

    # meangamma_bipolar_noscram_artif_brodmann_resppositive

    parser = argparse.ArgumentParser(description='Creates various datasets out of the processed intracranial data')
    parser.add_argument('-f', '--featureset', dest='featureset', type=str, required=True, help='Directory with brain features (Processed/?)')
    args = parser.parse_args()
    featureset = str(args.featureset)

    db = DataBuilder(featureset)
    db.build_stim_probe_category