import os
import numpy as np

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'

# Visual Word Form Area
'''cid = 7
left_x = [-50, -38]
left_y = [-61, -50]
left_z = [-30, -16]
right_x = [ 0, 0]
right_y = [ 0, 0]
right_z = [ 0, 0]'''

# Fusiform Face Area
'''cid = 7
left_x =  [-44, -38]
left_y =  [-61, -50]
left_z =  [-24, -15]
right_x = [ 36,  43]
right_y = [-55, -49]
right_z = [-25, -13]'''

# Parahippocampal Place Area
cid = 7
left_x =  [-31, -22]
left_y =  [-55, -49]
left_z =  [-12,  -6]
right_x = [ 21,  32]
right_y = [-54, -45]
right_z = [-12,  -6]

# load probe data for the given category
successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))
successful_mnis = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid))

# find MNIs within the volume
left_mnis  = successful_mnis[(successful_mnis[:, 0] > left_x[0])  & (successful_mnis[:, 0] <  left_x[1]) &
                             (successful_mnis[:, 1] > left_y[0])  & (successful_mnis[:, 1] <  left_y[1]) &
                             (successful_mnis[:, 2] > left_z[0])  & (successful_mnis[:, 2] <  left_z[1])]
right_mnis = successful_mnis[(successful_mnis[:, 0] > right_x[0]) & (successful_mnis[:, 0] < right_x[1]) &
                             (successful_mnis[:, 1] > right_y[0]) & (successful_mnis[:, 1] < right_y[1]) &
                             (successful_mnis[:, 2] > right_z[0]) & (successful_mnis[:, 2] < right_z[1])]

# print out corresponding probe IDs
print 'CID %d' % cid
for i in range(left_mnis.shape[0]):
    (pid, sid) = successful_probes[np.where(successful_mnis == left_mnis[i])[0][0]]
    print "Subject %d - probe %d" % (sid, pid - 1)
for i in range(right_mnis.shape[0]):
    (pid, sid) = successful_probes[np.where(successful_mnis == right_mnis[i])[0][0]]
    print "Subject %d - probe %d" % (sid, pid - 1)
    
