#!/usr/bin/env python
#
# File: bar_plot.py
#
import argparse
import os
import pickle
import matplotlib

params = {
    'axes.labelsize': 12,
    'font.size': 16,
    'font.family': 'serif',
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    #'text.usetex': True,
    'figure.figsize': [4.5, 4.5]
}
matplotlib.rcParams.update(params)

import matplotlib.pyplot as plt
plt.style.use('grayscale')

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--random', type=float, default=None)
    args = parser.parse_args()

    if args.random:
        random = args.random
    else:
        random = 0
    with open(os.path.join(args.dir, 'results.pkl'), 'rb') as f:
        res = pickle.load(f)['retlist']
    control_params = ['decentralized', 'concurrent', 'centralized']
    nn_params = ['mlp', 'gru', 'heuristic']

    header = ['training']
    for nnp in nn_params:
        header.append(nnp)
        header.append(nnp + '_error')

    mat = []
    for cp in control_params:
        row = [cp]
        for nnp in nn_params:
            key = cp + '-' + nnp
            if key in res:
                if res[key]:
                    row.append(res[key]['mean'] - random)
                    row.append(res[key]['std'])
                else:
                    row.append(None)
                    row.append(None)
            elif 'heuristic' in key:
                row.append(res['heuristic']['mean'] - random)
                row.append(res['heuristic']['std'])
        mat.append(row)

    dat = pd.DataFrame(mat, columns=header)
    print(dat.to_csv(index=False, float_format='%.3f', na_rep='nan'))

    csv_errors = dat[['mlp_error', 'gru_error']].rename(
        columns={'mlp_error': 'mlp',
                 'gru_error': 'gru'})
    ax = dat[['mlp', 'gru']].plot(kind='bar', title='', legend=True, yerr=csv_errors, alpha=0.7)
    ax.plot(dat['heuristic'], linewidth=1.2, linestyle='--', alpha=0.7)
    leg = ax.legend(['Heuristic', 'MLP', 'GRU'])
    leg.get_frame().set_alpha(0.7)
    ax.set_xticklabels(dat.training, rotation=0)
    ax.set_ylabel('Normalized Returns')
    plt.savefig(os.path.join(args.dir, 'bar.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    main()
