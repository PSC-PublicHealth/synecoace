#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import pickle as pickle


def mkSamps(df, nSamp):
    fracWt = df['FWC']/df['FWC'].sum()
    choices = np.random.choice(len(df), nSamp, p=fracWt)
    return df.iloc[choices].drop(columns=['FWC'])


def scatter(idx, vals, target):
    """target[idx] += vals, but allowing for repeats in idx"""
    np.add.at(target, idx.ravel(), vals.ravel())


def createBinner(colL, range_d=None):
    #print(colL)
    if range_d is None:
        get_range = lambda c : 2
    else:
        get_range = lambda c : range_d[c]
    def binner(sampV, col_dict=None):
        if col_dict is None:
            assert isinstance(sampV, pd.DataFrame), 'A column dictionary is needed'
        if isinstance(sampV, pd.DataFrame):
            get_col = lambda c : sampV[c].values
        else:
            get_col = lambda c : sampV[:, col_dict[c]]

        nBins = 1
        binV = np.zeros(len(sampV), dtype=np.int)
        #print('zeroV: ', zeroV)
        #print('oneV: ', oneV)
        #print('binV: ', binV)
        #print('colL (%d elts) %s' % (len(colL), colL))
        for col in colL:
            range = get_range(col)
            #range = 2 if range_d is None else range_d[col]
            choices = get_col(col)
            #print('col %s range %d' % (col, range))
            #binV = (range * binV) + np.where(choices, oneV, zeroV)
            binV = (range * binV) + choices.astype(int)
            nBins *= range
            #print('binV: ', binV)
        #print('binner nBins = %d' % nBins)
        return binV, nBins
    return binner


def toHisto(sampV, whichBin):
    """Generate a histogram of sample bins"""
    binV, nBins = whichBin(sampV)
    targ = np.zeros([nBins], dtype=np.int32)
    vals = np.ones([len(sampV)], dtype=np.int32)
    scatter(binV, vals, targ)
    return targ


def toProbV(sampV, whichBin):
    sampH = toHisto(sampV, whichBin)
    probV = sampH.astype(np.float64)
    probV /= np.sum(probV)
    return probV


def test_binner():
    df = pd.DataFrame({'bincol':[0,1,1,0],'scalcol':[0,1,2,3], 'bincol2':[0,1,0,1]})
    print(df)
    range_d = {'bincol':2, 'scalcol':4, 'bincol2':2}
    whichBin = createBinner(['bincol', 'scalcol', 'bincol2'], range_d)
    print(whichBin(df))
    
    sampV = np.array([[0,1,1,0], [0,1,2,3], [0,1,0,1]]).transpose()
    print(sampV)
    col_d = {'bincol':0, 'scalcol':1, 'bincol2':2}
    # range_d is unchanged
    whichBin = createBinner(['bincol', 'scalcol', 'bincol2'], range_d)
    print(whichBin(sampV, col_dict=col_d))


def main():
    test_binner()

if __name__ == "__main__":
    main()

