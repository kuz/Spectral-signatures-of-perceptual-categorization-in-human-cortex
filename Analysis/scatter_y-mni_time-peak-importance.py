import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
areas_of_interest = [17, 18, 19, 37, 20]

# load category data
for cid in range(8):
    importance = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid))
    successful_areas = np.load('%s/%s' % (INDIR, 'FT_successful_areas_ctg%d.npy' % cid))
    successful_mnis = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid))

    # focus on probes inside areas of interest
    probes_of_interest = np.array(range(successful_mnis.shape[0]))
    bool_index = [a in areas_of_interest for a in successful_areas]
    probes_of_interest = probes_of_interest[bool_index]

    # for every probe in AOI read Y MNI coordinate and time of peak importance
    y_mni_coord = []
    time_of_peak_importance = []
    for i in probes_of_interest:
        y_mni_coord.append(successful_mnis[i, 1])
        time_of_peak_importance.append(np.argmax(np.mean(importance[i, 66:, :], axis=0)))

    fig = plt.figure(figsize=(8, 6), dpi=100);
    plt.scatter(time_of_peak_importance, y_mni_coord);
    plt.xlim([0, 48]);
    plt.ylim([-90, 0])
    plt.savefig('test-%d.png' % cid, bbox_inches='tight');
    plt.clf();
    plt.close(fig);