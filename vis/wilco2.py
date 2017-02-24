#!/usr/bin/env python
#
# File: wilcoxon.py
#
import argparse
import pickle
import os.path
from pprint import pprint
from itertools import combinations
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
    parser.add_argument('dir', type=str)
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    control_params = ['centralized', 'decentralized', 'concurrent']
    nn_params = ['gru', 'mlp']
    total_params = ['{}-{}'.format(cp, np) for cp in control_params
                    for np in nn_params] + ['heuristic']

    retlist = dict.fromkeys(total_params)
    for tp in total_params:
        pkl_file = os.path.join(args.dir, tp + '.pkl')
        try:
            with open(pkl_file, 'rb') as f:
                retl = pickle.load(f, encoding='latin1')['retlist'][:50]

            if args.filter:
                retl = filter_outlier(retl)

            retlist[tp] = {
                'retl': retl,
                'mean': np.mean(retl),
                'std': np.std(retl) / np.sqrt(50),
            }
        except Exception as e:
            print(e)

    pvals = {}
    for p1, p2 in combinations(total_params, 2):
        if retlist[p1] and retlist[p2]:
            _, p_val = wilcoxon(retlist[p1]['retl'], retlist[p2]['retl'])
            pvals[p1, p2] = p_val

    with open(os.path.join(args.dir, 'results.pkl'), 'wb') as f:
        pickle.dump({'retlist': retlist, 'pvals': pvals}, f)

    for ret in retlist:
        if retlist[ret]:
            print('{}: {}, {}'.format(ret, retlist[ret]['mean'], retlist[ret]['std']))
            if args.all:
                pprint(retlist[ret]['retl'])
    print('######')
    pprint(pvals)

    index = []
    comp = []
    for row in total_params:
        index.append(row)
        rlist = []
        for col in total_params:
            if (row, col) in pvals:
                rlist.append(pvals[row, col])
            else:
                rlist.append(None)
        comp.append(rlist)

    import pandas as pd
    pd.set_option('expand_frame_repr', False)
    print(pd.DataFrame(comp, index=index, columns=index))


if __name__ == '__main__':
    main()
