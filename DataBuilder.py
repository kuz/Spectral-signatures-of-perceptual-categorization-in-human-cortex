import argparse
import numpy as np


class DataBuilder

    #: Paths
    DATADIR = '../../Data/Intracranial/Processed'

    def __init__()
        pass

    def 


if __name__ == '__main__':

    # meangamma_bipolar_noscram_artif_brodmann_resppositive

    parser = argparse.ArgumentParser(description='Creates various datasets out of the processed intracranial data')
    parser.add_argument('-f', '--featureset', dest='featureset', type=str, required=True, help='Directory with brain features (Processed/?)')
    parser.add_argument('-d', '--distance', dest='distance', type=str, required=True, help='The distance metric to use')
    parser.add_argument('-o', '--onwhat', dest='onwhat', type=str, required=True, help='image or matrix depending on which you to compute the correlation on')
    parser.add_argument('-t', '--threshold', dest='threshold', type=float, required=True, help='Significance level a score must have to be counter (1.0 to store all)')
    parser.add_argument('-s', '--statistic', dest='statistic', type=str, required=True, help='Type of score to compute when aggregating: varexp, corr')
    parser.add_argument('-p', '--permfilter', dest='permfilter', type=str, required=True, help='Whether to filter the results with permutation test results')
    
    args = parser.parse_args()
    backbone = str(args.backbone)
    featureset = str(args.featureset)
    distance = str(args.distance)
    onwhat = str(args.onwhat)
    threshold = float(args.threshold)
    statistic = str(args.statistic)
    suffix = ''
    permfilter = bool(args.permfilter == 'True')


    db = DataBuilder()
    db