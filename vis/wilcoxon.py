#!/usr/bin/env python
#
# File: wilcoxon.py
#
import argparse
import pickle
from pprint import pprint
import numpy as np
from scipy.stats import wilcoxon


def filter_outlier(data):
    new_dat = []
    for d in data:
        if d < -50:
            d = -50
        new_dat.append(d)
    return new_dat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file1', type=str)
    parser.add_argument('file2', type=str)
    parser.add_argument('--filter', action='store_true')
    args = parser.parse_args()

    with open(args.file1, 'rb') as f:
        retlist1 = pickle.load(f, encoding='latin1')['retlist'][:50]
        if args.filter:
            retlist1 = filter_outlier(retlist1)

    with open(args.file2, 'rb') as f:
        retlist2 = pickle.load(f, encoding='latin1')['retlist'][:50]
        if args.filter:
            retlist2 = filter_outlier(retlist2)

    mean1 = np.mean(retlist1)
    std1 = np.std(retlist1) / np.sqrt(50)
    mean2 = np.mean(retlist2)
    std2 = np.std(retlist2) / np.sqrt(50)
    z_stat, p_val = wilcoxon(retlist1, retlist2)

    pprint({
        args.file1: {
            'mean': mean1,
            'std': std1
        },
        args.file2: {
            'mean': mean2,
            'std': std2
        },
        'p_val': p_val
    })
    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    main()
