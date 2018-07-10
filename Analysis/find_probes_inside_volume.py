import os
import numpy as np
import pdb

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
INDIR_ALL = '../../Data/Intracranial/Calculated'

# Visual Word Form
'''
regions = {'left' : {'c': [-44, -55, -23], 'r': 6},
           'right': {'c': [  0,   0,   0], 'r': 0}}
'''

# Fusiform Face Form
'''left_x =  [-44, -38]
left_y =  [-61, -50]
left_z =  [-24, -15]
right_x = [ 36,  43]
right_y = [-55, -49]
right_z = [-25, -13]'''

# ? (houses and scenes)
'''left_x =  [-44, -38]
left_y =  [-61, -50]
left_z =  [-24, -15]
right_x = [ 36,  43]
right_y = [-55, -49]
right_z = [-25, -13]'''

# DMN
default_radius = 4
regions = {# ventral medial prefrontal cortex 
           '24L': {'c': [ -5,   1,  32], 'r': 3},
           '24R': {'c': [  5,   5,  31], 'r': 4},
           '10L': {'c': [-23,  55,   4], 'r': 7},
           '20R': {'c': [ 23,  55,   7], 'r': 7},
           '32L': {'c': [ -5,  39,  20], 'r': 5},
           '32R': {'c': [  6,  33,  16], 'r': 4},
           # posterior cingulate
           '30L': {'c': [-12, -43,   8], 'r': default_radius}, 
           '30R': {'c': [ 12, -45,   8], 'r': default_radius},
           '23L': {'c': [-10, -45,  24], 'r': default_radius},
           '23R': {'c': [  9, -45,  24], 'r': default_radius},
           '31L': {'c': [ -8, -49,  38], 'r': default_radius},
           '31R': {'c': [  8, -48,  39], 'r': default_radius},
           # Inferior parietal lobule
           '39L': {'c': [-36, -60,  33], 'r': default_radius},
           '39R': {'c': [ 46, -59,  31], 'r': default_radius},
           '40L': {'c': [-53, -32,  33], 'r': default_radius},
           '40R': {'c': [ 51, -33,  34], 'r': default_radius},
           # Laterial temporal cortex
           '21L': {'c': [-59, -25, -13], 'r': default_radius},
           '21R': {'c': [ 60, -27,  -9], 'r': default_radius},
           # Dorsal medial prefrontal cortex
           '24L': {'c': [ -5,   1,  32], 'r': default_radius},
           '24R': {'c': [  5,   5,  31], 'r': default_radius},
            '9L': {'c': [-39,  34,  37], 'r': default_radius},
            '9R': {'c': [ 35,  39,  31], 'r': default_radius},
           # Hippocampal formation
           'HCL': {'c': [-24, -22, -20], 'r': default_radius},
           'HCR': {'c': [ 24, -22, -20], 'r': default_radius}}

# Total number of probes within the volume
all_mnis = np.load('%s/%s' % (INDIR_ALL, 'all_mnis.npy'))
all_mnis_in_volume = np.empty((0, 3))
for rname in regions:
    r = regions[rname]
    all_mnis_in_region = all_mnis[(all_mnis[:, 0] > r['c'][0] - r['r'])  & (all_mnis[:, 0] < r['c'][0] + r['r']) &
                                  (all_mnis[:, 1] > r['c'][1] - r['r'])  & (all_mnis[:, 1] < r['c'][1] + r['r']) &
                                  (all_mnis[:, 2] > r['c'][2] - r['r'])  & (all_mnis[:, 2] < r['c'][2] + r['r'])]
    all_mnis_in_volume = np.concatenate((all_mnis_in_volume, all_mnis_in_region), axis=0)
print "Total number of probes: %d" % all_mnis_in_volume.shape[0]

# Number of successfull probes per category inside the volume 
for cid in range(8):

    # load probe data for the given category
    successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))
    successful_mnis = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid))

    # find MNIs within the volume
    succ_mnis_in_volume = np.empty((0, 3))
    for rname in regions:
        r = regions[rname]
        mnis_in_region = successful_mnis[(successful_mnis[:, 0] > r['c'][0] - r['r'])  & (successful_mnis[:, 0] < r['c'][0] + r['r']) &
                                         (successful_mnis[:, 1] > r['c'][1] - r['r'])  & (successful_mnis[:, 1] < r['c'][1] + r['r']) &
                                         (successful_mnis[:, 2] > r['c'][2] - r['r'])  & (successful_mnis[:, 2] < r['c'][2] + r['r'])]
        succ_mnis_in_volume = np.concatenate((succ_mnis_in_volume, mnis_in_region), axis=0)


    # print out corresponding probe IDs
    print 'CID %d' % cid
    for i in range(succ_mnis_in_volume.shape[0]):
        (pid, sid) = successful_probes[np.where(successful_mnis == succ_mnis_in_volume[i])[0][0]]
        print "Subject %d - probe %d" % (sid, pid - 1)    
