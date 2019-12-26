#! /usr/bin/env python

import sys
import numpy as np
import scipy.sparse as sp
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import pickle as pickle

from sampling import toProbV
from mutator import Mutator

def mutualInfo(sampVX, sampVY, whichBin, binnerParams={}):
    assert len(sampVX) == len(sampVY), 'Sample vector lengths do not match'
    binVX, nBinsX = whichBin(sampVX, **binnerParams)
    binVY, nBinsY = whichBin(sampVY, **binnerParams)
    assert nBinsX == nBinsY, 'Unexpectedly got different bin counts?'
    cA = np.zeros([nBinsX, nBinsX], dtype=np.int32)
    idxV = np.ravel_multi_index(np.array([binVX, binVY]), (nBinsX, nBinsX))
    np.add.at(cA.ravel(), idxV, np.ones(len(idxV), dtype=np.int32).ravel())
    pA = cA.astype(np.float32)
    pA /= sum(pA.ravel())
    xPV = toProbV(sampVX, whichBin)
    yPV = toProbV(sampVY, whichBin)
    xyPA = np.einsum('i,j->ij', xPV, yPV)  # einsum is my new favorite function
    oldErr = np.seterr(invalid='ignore', divide='ignore')
    prodA = pA * np.nan_to_num(np.log(pA / xyPA))  # element-wise calculation
    np.seterr(**oldErr)
    return np.sum(prodA.ravel())


########
# This implementation (from stackoverflow) produces the same MI values as the one above
# stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
########
# from scipy.stats import chi2_contingency
# 
# def calc_MI(x, y, bins):
#     c_xy = np.histogram2d(x, y, bins)[0]
#     g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
#     mi = 0.5 * g / c_xy.sum()
#     return mi
# 
# def mutualInfo(sampVX, sampVY, whichBin):
#     #assert len(sampVX) == len(sampVY), 'Sample vector lengths do not match'
#     binVX, nBinsX = whichBin(sampVX)
#     binVY, nBinsY = whichBin(sampVY)
#     assert nBinsX == nBinsY, 'Unexpectedly got different bin counts?'
#     return calc_MI(binVX, binVY, nBinsX)

#print mutualInfo(sampX.values, sampY.values)
#print mutualInfo(sampX.values, sampX.values)
#print mutualInfo(sampY.values, sampY.values)


# def lnLik(samps1V, samps2V, wtSerV):
#     """
#     funV has the right shape to fill the role of likelihood in the Metropolis algorithm.  We'll
#     take the log, and use it as a log likelihood.
#     """
#     #print 'lnLik samps1V', samps1V
#     #print 'samps2V', samps2V
#     #print 'wtSerV', wtSerV
#     wtA = wtSerV
#     #offset = samps1V.shape[1] - wtSer.shape[0]
#     offset = samps1V.shape[1] - wtSerV.shape[0]
#     #print 'offset', samps1V.shape, wtSerV.shape, offset
#     samp1A = samps1V[:, offset:]
#     #print 'samp1A', samp1A
#     samp2A = samps2V[:, offset:]
#     #print 'samp2A', samp2A
#     delta = samp1A - samp2A
#     delta *= delta
#     return np.asarray((-np.asmatrix(wtA) * np.asmatrix(delta).transpose())).reshape((-1, 1))


def genRawMetropolisSamples(nSamp, nIter, guess, lnLikFun, lnLikParams, mutator, mutatorParams,
                            verbose=False):
    """
    Parameters:
        nSamp: number of samples in the vector of current samples
        nIter: the number of iterations- this many vectors of raw samples will be drawn
        guess: the initial vector of samples to be mutated
        lnLikFun: a function returning the log likelihood, used in the Metropolis ratio
        
            lnLik = lnLikFun(samples, **lnLikParams)
            
        lnLikParms: passed to lnLikFun as above
        mutator: function implementing the Metropolis mutation, of the form
        
            newSamps = mutator(oldSamps, **mutatorParams)

        mutatorParams: passed to mutator as above
    Returns:
        A: list of vectors of samples after Metropolis acceptance
        acceptanceRate: integer vector of acceptance rate for each element of the vectors in A
    """
    # Metropolis-Hastings with nIter iterations.
    accepted  = np.zeros([nSamp, 1], dtype=np.int)
    onesV = np.ones([nSamp], dtype=np.int).reshape((-1, 1))
    zerosV = np.zeros([nSamp], dtype=np.int).reshape((-1, 1))
    A = [guess]
    for n in range(nIter):
        oldAlpha  = A[-1]  # old parameter value as array
        #print('oldAlpha: ')
        #print(oldAlpha.head())
        oldLnLik = lnLikFun(oldAlpha, **lnLikParams)
        newAlpha = mutator.apply(oldAlpha, **mutatorParams)
        #print('newAlpha: ')
        #print(newAlpha.head())
        newLnLik = lnLikFun(newAlpha, **lnLikParams)
        #print('newLnLik: ', newLnLik)
        if verbose and (n % 100 == 0):
            print('%s: %s' % (n, np.sum(newLnLik)))
        llDelta = newLnLik - oldLnLik
        llDelta = np.minimum(llDelta, 0.0)
        #choices = np.logical_or(newLnLik > oldLnLik,
        #                        np.random.random(newLnLik.shape) < np.exp(newLnLik - oldLnLik))
        choices = np.logical_or(newLnLik > oldLnLik,
                                np.random.random(newLnLik.shape) < np.exp(llDelta.astype(np.float)))
        rslt = np.choose(choices, [oldAlpha, newAlpha])
        rsltDF = pd.DataFrame(rslt, columns=oldAlpha.columns.copy())
        #print('point 1 result: ')
        #print(rsltDF.head())
        A.append(rsltDF)
        accepted += np.choose(choices, [zerosV, onesV])
    acceptanceRate = accepted/float(nIter)
    return A, acceptanceRate


def genMetropolisSamples(nSamp, nIter, guess, lnLikFun, lnLikParams, mutator, mutatorParams,
                        mutationsPerSamp=10, burninMutations=10, verbose=False):
    """
    Generates independent samples by metropolis sampling with genRawMetropolisSamples and
    discarding enough samples to assure independence.
    Parameters:
        nSamp: number of samples in the vector of current samples
        nIter: the number of iterations- this many vectors of raw samples will be drawn
        guess: the initial vector of samples to be mutated
        lnLikFun: a function returning the log likelihood, used in the Metropolis ratio
        
            lnLik = lnLikFun(samples, **lnLikParams)
            
        lnLikParms: passed to lnLikFun as above
        mutator: function implementing the Metropolis mutation, of the form
        
            newSamps = mutator(oldSamps, mutatorParams)

        mutatorParams: passed to mutator as above
        mutatationsPerSamp: minimum number of accepted mutations between retained samples (on average)
        burninMutations: number of independent samples to discard during burn-in sampling
    Returns:
        cleanSamps: a list of vectors of independent samples
    """
    clean = []
    while True:
        A, acceptanceRate = genRawMetropolisSamples(nSamp, nIter, guess, lnLikFun, lnLikParams,
                                                    mutator, mutatorParams, verbose=verbose)
        print('acceptanceRate: ',
              np.quantile(acceptanceRate, 0.75),
              np.quantile(acceptanceRate, 0.5),
              np.quantile(acceptanceRate, 0.25),
              acceptanceRate.min())
        nKeep = int((acceptanceRate * nIter).min() / mutationsPerSamp)
        if nKeep:
            keepStep = nIter//nKeep
            burnIn = burninMutations * keepStep
            if burnIn >= nIter:
                burninMutations -= nIter//keepStep
                print("Not enough mutations for burn-in; acceptance rate %s; %d discards remain"
                      % (acceptanceRate.min(), burninMutations))
            else:
                for idx, sV in enumerate(A[burnIn:]):
                    if idx % keepStep == 0:
                        clean.append(sV)
                break
        else:
            print("NO GOOD MUTATIONS; acceptance rate %s; continuing"
                  % acceptanceRate.min())
            print(acceptanceRate.min(), nIter, (acceptanceRate * nIter).min())
    return clean


def test_mi():
    from sampling import createBinner, createSparseBinner
    nSamp = 10000
    sampXDF = pd.DataFrame({'A':np.random.randint(0,12,nSamp),
                            'B':np.random.randint(0,7,nSamp),
                            'C':np.random.randint(0,3,nSamp)})
    sampYDF = pd.DataFrame({'A':np.random.randint(0,12,nSamp),
                            'B':np.random.randint(0,7,nSamp),
                            'C':np.random.randint(0,3,nSamp)})
    binner = createBinner(['A', 'B', 'C'], {'A':12, 'B':7, 'C':3})
    print(mutualInfo(sampXDF, sampYDF, binner))
    binner = createSparseBinner(['A', 'B', 'C'], {'A':12, 'B':7, 'C':3})
    print(mutualInfo(sampXDF, sampYDF, binner))

def main():
    test_mi()

if __name__ == "__main__":
    main()



