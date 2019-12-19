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


def createBinner(colL):
    #print(colL)
    def binner(sampV, col_dict=None):
        if col_dict is None:
            assert isinstance(sampV, pd.DataFrame), 'A column dictionary is needed'
        nBins = 1
        zeroV = np.zeros(len(sampV), dtype=np.int)
        oneV = np.ones(len(sampV), dtype=np.int)
        binV = zeroV.copy()
        #print('zeroV: ', zeroV)
        #print('oneV: ', oneV)
        #print('binV: ', binV)
        if isinstance(sampV, pd.DataFrame):
            print('colL (%d elts) %s' % (len(colL), colL))
            for col in colL:
                choices = sampV[col]
                print('col: ',col)
                binV = (2 * binV) + np.where(choices, oneV, zeroV)
                nBins *= 2
                #print('binV: ', binV)
        else:
            for col in colL:
                choices = sampV[:, col_dict[col]]
                #print('col: ',col, choices)
                binV = (2 * binV) + np.where(choices, oneV, zeroV)
                nBins *= 2
                #print('binV: ', binV)
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


