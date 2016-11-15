#!/usr/bin/env python3
#
# File: showlog.py
#
import argparse
import pandas as pd


def runplot(args):
    assert len(set(args.logfiles)) == len(args.logfiles), 'Log files must be unique'

    fname2log = {}
    for fname in args.logfiles:
        df = pd.read_csv(fname)
        if 'Iteration' in df.keys():
            df.set_index('Iteration', inplace=True)
        elif 'Epoch' in df.keys():
            df.set_index('Epoch', inplace=True)
        else:
            raise NotImplementedError()
        if not args.fields == 'all':
            df = df.loc[:, args.fields.split(',')]
        fname2log[fname] = df

    if not args.noplot:
        import matplotlib
        if args.plotfile is not None:
            matplotlib.use('Agg')

        import matplotlib.pyplot as plt
        plt.style.use('seaborn-colorblind')

    ax = None
    for fname, df in fname2log.items():
        with pd.option_context('display.max_rows', 9999):
            print(fname)
            print(df[-1:])

        if not args.noplot:
            if ax is None:
                ax = df.plot(subplots=True, title=','.join(args.logfiles))
            else:
                df.plot(subplots=True, title=','.join(args.logfiles), ax=ax, legend=False)

            if args.plotfile is not None:
                plt.savefig(args.plotfile, transparent=True, bbox_inches='tight', dpi=300)
            else:
                plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logfiles', type=str, nargs='+')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--fields', type=str, default='all')
    parser.add_argument('--plotfile', type=str, default=None)
    args = parser.parse_args()
    runplot(args)
