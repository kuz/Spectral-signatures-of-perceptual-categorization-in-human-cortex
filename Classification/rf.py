import numpy as np
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import json
import argparse

# parse cmd arguments
parser = argparse.ArgumentParser(description='Build feature matrix from featureset')
parser.add_argument('-f', '--featureset', dest='featureset', type=str, required=True, help='Name of the featureset (Processed/?)')
args = parser.parse_args()
featureset = str(args.featureset)

# parameters
DATADIR = '../../Outcome/Feature matrices/'
n_cv = 5

with open('%s/featurematrix_%s.pkl' % (DATADIR, featureset), 'rb') as f:
    data = cPickle.load(f)

#
# Subselect probes from previous experiments
#

# ??
selectors = json.loads('{"AL_25FEV13N": [2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 23], "AM_10JAN12G": [4, 5, 6], "AP_15FEV11G": [3, 4, 5, 6, 20, 21, 22, 23, 24, 25], "BD_15MAR11G": [1, 2, 3, 4, 7, 8], "CC_2JUL12G": [8, 9, 10], "CM_19NOV12G": [2, 3, 34], "CP_12NOV12G": [2, 21, 26, 31, 32, 33], "CQ_24JAN12G": [3, 4, 5, 13, 14, 15, 16, 17, 27, 28, 30, 31, 32, 36, 37, 38, 41, 42], "EG_16AVR12G": [25, 26, 27, 28, 29, 30, 31, 32, 42], "FM_04AVR13N": [19, 20, 21, 36, 37, 39, 44, 48, 49, 50, 51, 52, 60], "FS_12MARS12G": [10, 11, 35, 36], "HL_080211G": [2, 3, 4, 9, 10], "HT_18MAR13G": [3, 20, 21, 22, 34], "JC_16AVR13N": [19, 40, 41, 42, 64, 65, 66, 67, 68, 69, 70, 71, 72], "JH_26AVR13G": [27], "JP_21JAN13G": [10, 13], "LYONNEURO_2013_KAMM": [1, 2], "LYONNEURO_2013_LEFC": [0, 10, 22, 24, 25, 26, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 78, 87, 88, 89, 90], "LYONNEURO_2013_ROTP": [2, 6, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 33, 34], "LYONNEURO_2013_SEIT": [8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26], "LYONNEURO_2013_SEMC": [14], "LYONNEURO_2013_TANS": [15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 33, 34, 35, 36, 37, 38, 39, 40], "LYONNEURO_2013_VACJ": [5, 18, 19, 20, 21], "LYONNEURO_2014_BATG": [19, 20, 21, 36, 43], "LYONNEURO_2014_BENC": [8, 9, 10, 11, 12, 13, 15], "LYONNEURO_2014_BLAP": [1, 2, 13, 21, 22, 23, 24, 25, 26, 27], "LYONNEURO_2014_BOBX": [6, 10, 16, 41, 42, 43, 44, 45, 46], "LYONNEURO_2014_CHAB": [4, 11, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46], "LYONNEURO_2014_CHER": [6, 7, 9, 10, 11, 17, 18, 23, 24, 26, 27, 28, 29], "LYONNEURO_2014_DESM": [2, 3, 4], "LYONNEURO_2014_FAUD": [15], "LYONNEURO_2014_FEPK": [1, 3, 4, 13, 23], "LYONNEURO_2014_LADC": [1, 2], "LYONNEURO_2014_LIBI": [1, 4, 6, 11, 32], "LYONNEURO_2014_MONM": [13, 14, 15, 16, 18, 19], "LYONNEURO_2014_PERRA": [4, 7, 8, 13, 14, 23], "LYONNEURO_2014_PIRJ": [6], "LYONNEURO_2014_QUEA": [8, 9, 10, 11, 12, 13], "LYONNEURO_2014_RENT": [3, 4, 5, 11, 12, 13, 14], "LYONNEURO_2014_REYD": [6, 7, 8], "LYONNEURO_2014_TAMN": [4, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18], "LYONNEURO_2014_THUV": [13, 14, 15, 16], "LYONNEURO_2015_GAUC": [89], "LYONNEURO_2015_GRER": [1], "LYONNEURO_2015_SELS": [7, 8, 9], "MC_30SEP09G": [5, 6, 11, 12, 13], "MM_15SEP09G": [4, 5, 12], "MM_22OCT12G": [0, 1, 3, 4, 7], "MM_6MARS12G": [2, 3, 4, 5, 13, 15, 16, 17, 18], "MP_22MAR11G": [5], "NB_14JAN13G": [19, 26, 34, 35, 36, 37, 38, 39, 43, 45, 46, 47], "NR_14FEV12G": [2, 3, 4, 6, 8, 9, 10, 11], "NR_21JUN11G": [19, 20], "NS_15OCT12G": [7, 34, 35, 36, 37], "OP_17SEP12G": [2, 5], "PC_18NOV13G": [7, 8, 9, 27], "PD_01FEV13N": [2, 3, 4], "PM_17JAN12G": [3, 4, 5, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 29, 30, 31, 32, 33, 34, 35], "PP_08FEV13N": [15, 16, 17, 18, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46], "PW_24FEV14G": [10, 26, 27], "RG_13AVR10G": [4, 5, 6, 7, 9, 14, 17, 23, 24, 25, 29, 30, 31], "RK_15SEP10G": [3, 17, 19, 34, 35], "SR_30MAI12G": [4, 5, 11], "TH_01MAR11G": [4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17]}')

# ??
# selectors = json.loads('{"AL_25FEV13N": [1,6,10,15,16,17,18,19,20,21,22,23,24,25,35,44],"AM_10JAN12G": [6,7,8],"AP_15FEV11G": [6,7,8,9,16,19,37,38,39,40,41,42],"BD_15MAR11G": [4,5,6,7,12,13],"CC_2JUL12G": [14,15,16],"CM_19NOV12G": [2,3,17,39],"CP_12NOV12G": [3,28,39,44,45,46],"CQ_24JAN12G": [3,4,5,17,18,19,20,21,35,36,38,39,40,44,45,46,51,52],"EG_16AVR12G": [29,30,31,32,33,34,35,36,48],"FM_04AVR13N": [27,28,30,56,57,59,67,75,76,77,78,80,90],"FS_12MARS12G": [14,15,47,48],"HL_080211G": [2,3,4,10,11],"HT_18MAR13G": [5,31,32,33,49],"JCC_17JUI13N": [10],"JC_16AVR13N": [23,48,49,50,77,78,79,80,81,82,83,84,85,86],"JH_26AVR13G": [37],"JP_21JAN13G": [16,19,54],"LYONNEURO_2013_KAMM": [1,2],"LYONNEURO_2013_LEFC": [0,15,29,30,35,36,37,85,89,90,91,92,93,94,95,96,97,98,99,100,101,107,123,124,125,126],"LYONNEURO_2013_ROTP": [4,14,15,16,17,18,24,25,26,35,36,37,38,39,40,41,42,45,46,49,50,51,52,54,55,56],"LYONNEURO_2013_SEIT": [13,21,22,23,24,25,26,27,28,29,30,33,34],"LYONNEURO_2013_SEMC": [6,17,33],"LYONNEURO_2013_TANS": [1,28,34,35,37,38,39,40,41,46,47,48,49,54,55,56,57,58,59,60,61],"LYONNEURO_2013_VACJ": [9,31,32,33,34],"LYONNEURO_2014_BATG": [35,36,37,53,61,62],"LYONNEURO_2014_BENC": [19,32,33,34,35,36,37,40],"LYONNEURO_2014_BLAP": [1,2,21,26,35,36,37,38,39,40,41],"LYONNEURO_2014_BOBX": [13,18,26,56,57,58,59,60,61],"LYONNEURO_2014_CHAB": [1,7,12,24,45,47,50,51,52,53,54,55,56,57,58,59,62,63,64,65],"LYONNEURO_2014_CHER": [10,11,13,14,15,21,23,36,37,39,40,41,42,43],"LYONNEURO_2014_DESM": [9,10,11,12,13],"LYONNEURO_2014_FAUD": [21],"LYONNEURO_2014_FEPK": [1,3,4,7,13,15,21,28,33,34,46,66,69],"LYONNEURO_2014_LADC": [9,11],"LYONNEURO_2014_LIBI": [5,6,10,16,25,59],"LYONNEURO_2014_MARB": [11],"LYONNEURO_2014_MONM": [19,20,21,22,24,25],"LYONNEURO_2014_PERR": [2],"LYONNEURO_2014_PERRA": [7,10,11,16,17,22,35],"LYONNEURO_2014_PIRJ": [15],"LYONNEURO_2014_QUEA": [13,14,15,16,17,18],"LYONNEURO_2014_RENT": [11,12,13,19,20,21,22],"LYONNEURO_2014_REYD": [16,17,18],"LYONNEURO_2014_SIEJ": [0,15,28],"LYONNEURO_2014_TAMN": [16,20,21,22,23,24,31,32,33,34,35,36],"LYONNEURO_2014_THUV": [3,4,23,38,39,40,41],"LYONNEURO_2015_BOUC": [67],"LYONNEURO_2015_GAUC": [102],"LYONNEURO_2015_GRER": [1],"LYONNEURO_2015_SELS": [2,12,13,14],"MC_30SEP09G": [6,7,14,15,16],"MM_15SEP09G": [8,9,18],"MM_22OCT12G": [4,5,7,8,15],"MM_6MARS12G": [5,6,7,8,16,18,19,20,21],"MP_22MAR11G": [7],"NB_14JAN13G": [27,34,42,43,44,45,46,47,51,53,54,55],"NR_14FEV12G": [2,3,4,7,10,11,12,13],"NR_21JUN11G": [24,25],"NS_15OCT12G": [13,40,41,42,43],"OP_17SEP12G": [5,10],"PC_18NOV13G": [16,17,18,41],"PD_01FEV13N": [11,12,13],"PM_17JAN12G": [4,5,6,16,17,18,19,20,21,26,27,28,29,30,34,35,36,37,38,40,41],"PP_08FEV13N": [27,28,29,30,58,59,60,61,62,63,64,65,66,67,68],"PW_24FEV14G": [14,34,35],"RG_13AVR10G": [7,8,9,10,15,21,25,37,38,39,44,45,46],"RK_15SEP10G": [3,19,21,39,40],"SR_30MAI12G": [12,13,24],"TH_01MAR11G": [6,7,8,9,10,11,15,16,17,18,19,20]}')

# Probes in DMN areas

keep_probes = []
for sname in selectors.keys():
    subject_probes = np.array([i for i,x in enumerate(data['subjects']) if x == sname])
    subject_probes = subject_probes[selectors[sname]]
    keep_probes += list(subject_probes)
data['neural_responses'] = data['neural_responses'][:, keep_probes]

# filter out the target stimuli (80 -- fruit)
#for dropstim in [60, 70, 80, 90]:
for dropstim in [80]:
    keepidx = data['image_category'] != dropstim
    data['image_category'] = data['image_category'][keepidx]
    data['neural_responses'] = data['neural_responses'][keepidx]

# uncomment for a permutation test
#data['image_category'] = np.random.permutation(data['image_category'])

# obtain CV predictions
print 'Data size', data['neural_responses'].shape
clf = RandomForestClassifier(n_estimators=3000)
predicted = cross_validation.cross_val_predict(clf, data['neural_responses'], data['image_category'], cv=n_cv)

# display results
print confusion_matrix(data['image_category'], predicted)
print f1_score(data['image_category'], predicted, average='weighted')
