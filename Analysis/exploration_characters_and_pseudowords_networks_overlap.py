import os
import numpy as np
import pdb
from functions_plotting import duoptych
from matplotlib import pylab as plt


# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
OUTDIR = '../../Outcome/Figures'

# compute overlap
successful_mnis_pseud = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % 5))
successful_mnis_chars = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % 6))

# drop 0,0,0 electrode if it creeps in
drop_idx = np.unique(np.where(successful_mnis_pseud == [0, 0, 0])[0])
successful_mnis_pseud = np.delete(successful_mnis_pseud, drop_idx, 0)
drop_idx = np.unique(np.where(successful_mnis_chars == [0, 0, 0])[0])
successful_mnis_chars = np.delete(successful_mnis_chars, drop_idx, 0)

# keep unique
successful_mnis_pseud = np.unique(successful_mnis_pseud, axis=0)
successful_mnis_chars = np.unique(successful_mnis_chars, axis=0)

# prepare lists
full_list = np.unique(np.vstack((successful_mnis_pseud, successful_mnis_chars)), axis=0)
only_in_chars = np.empty((0, 3))
only_in_pseud = np.empty((0, 3))
in_both = np.empty((0, 3))

# conver to Python lists for "in" operator to work
full_list = full_list.tolist()
successful_mnis_pseud = successful_mnis_pseud.tolist()
successful_mnis_chars = successful_mnis_chars.tolist()

# split into 3 groups
for mni in full_list:
    if (mni in successful_mnis_chars) and (mni in successful_mnis_pseud):
        in_both = np.vstack((in_both, mni))
    elif (mni in successful_mnis_chars) and (mni not in successful_mnis_pseud):
        only_in_chars = np.vstack((only_in_chars, mni))
    elif (mni not in successful_mnis_chars) and (mni in successful_mnis_pseud):
        only_in_pseud = np.vstack((only_in_pseud, mni))
    else:
        print "Error: this can't be"

# print stats
print "Unique chars ", len(successful_mnis_chars)
print "Unique pseud ", len(successful_mnis_pseud)
print "Unique total ", len(full_list)
print "Only in chars", only_in_chars.shape[0]
print "Only in pseud", only_in_pseud.shape[0]
print "In both      ", in_both.shape[0]

# plot
fig = plt.figure(figsize=(16, 8), dpi=200);
foci = np.vstack((in_both, only_in_pseud, only_in_chars))
colors = ['black'] * in_both.shape[0] + ['blue'] * only_in_pseud.shape[0] + ['red'] * only_in_chars.shape[0]
duoptych(foci, np.array(colors), {'black': 0.4, 'blue': 0.4, 'red': 0.4}, ["%s/pseudo_and_character_network.png" % OUTDIR])
