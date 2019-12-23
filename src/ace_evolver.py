#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import pickle as pickle

from data_conversions import selectData, transformData, floatifyData
from sampling import mkSamps
from metropolis import mutualInfo, lnLik, genRawMetropolisSamples, genMetropolisSamples


# Now we need a mutator
def mutate(sampV, df, stepSzV):
    """
    Return a 'mutated' version of sampV, based on the given step sizes.  Unfortunately our samples
    are discrete and come from a table, so I'm not sure how to do this unless we first generate
    a proximity network of some sort, so for the moment let's just generate a new set of samples-
    this corresponds to an infinitely wide mutator.
    """
    return mkSamps(df, len(sampV)).values

def sampleAndCalcMI(wtSer, nSamp, nIter, sampler, testSampParams, genSampParams,
                    mutator, mutatorParams, lnLik,
                    drawGraph=False, verbose=False):
    testSamps = sampler(nSamp, **testSampParams)
    guess = sampler(nSamp, **genSampParams)
    lnLikParams = {'samps2V': testSamps.values, 'wtSerV': wtSer}
    cleanSamps = genMetropolisSamples(nSamp, nIter, guess, lnLik, lnLikParams,
                                      mutator, mutatorParams, verbose=verbose)
    cleanV = np.concatenate(cleanSamps)
    expandedTestV = np.concatenate([testSamps.values] * len(cleanSamps))
    
    if drawGraph:
        testBins, nBins = whichBin(expandedTestV)
        rsltBins = whichBin(cleanV)[0]
        hM, xEdges, yEdges = np.histogram2d(testBins, rsltBins, bins=64)
        plt.imshow(np.log(hM + 1))
        plt.show()

    return mutualInfo(cleanV, expandedTestV)


def whichBin(sampV):
    """
    Input is an ndarray of sample values
    Maybe check out Pandas 'cut'
    """
    fplBinWidth = 50
    fplMin = 50
    bin = np.abs((sampV[:, COLUMN_DICT['FPL']] - fplMin) // 50).astype('int')
    assert (bin >= 0).all() and (bin < 8).all(), 'FPL out of range?'
    nBins = 8
    # Each of the following is either 1.0 or 2.0
    bin = 2 * bin + (sampV[:, COLUMN_DICT['K4Q32X01']] == 1.0)
    nBins *= 2
    bin = 2 * bin + (sampV[:, COLUMN_DICT['K7Q30']] == 1.0)
    nBins *= 2
    bin = 2 * bin + (sampV[:, COLUMN_DICT['K7Q31']] == 1.0)
    nBins *= 2
    return bin, nBins


def generateSamples(oldSamps, wtVec, workingCols,
                    nIter, sampler, genSampParams, srcDF,
                    mutator, mutatorParams,
                    verbose=False):

    # get the right index order but no extra entries
    wtSer = pd.Series({key: val for key, val in zip(workingCols, list(range(len(workingCols))))},
                     index=srcDF.columns)
    # wtSer = pd.Series({'YEAR': wtVec[0],
    #                   'FPL': wtVec[1],
    #                   'SC_AGE_YEARS': wtVec[2],
    #                   'K4Q32X01': wtVec[3],
    #                   'K7Q30': wtVec[4],
    #                   'K7Q31': wtVec[5],
    #                   'AGEPOS4': wtVec[6]},
    #                  index=srcDF.columns)
    dropL = [col for col in wtSer.index if col not in workingCols]
    wtSer = wtSer.drop(labels=dropL)

    nSamp = len(oldSamps)
    guess = sampler(nSamp, **genSampParams)
    lnLikParams = {'samps2V': oldSamps, 'wtSerV': wtSer}
    cleanSamps = genMetropolisSamples(nSamp, nIter, guess, lnLik, lnLikParams,
                                      mutator, mutatorParams, verbose=verbose,
                                     mutationsPerSamp=2, burninMutations=4)
    cleanV = np.concatenate(cleanSamps)

    return cleanV


def main(argv = None):
    if argv is None:
        argv = sys.argv

    outID = int(sys.argv[1])
    print("Result ID will be {}".format(outID))

    print('loading subject data and subsetting')
    fullDF = pd.read_csv('/home/welling/git/synecoace/data/nsch_2016_topical.csv')
    print(fullDF.columns)

    fullDF = fullDF.reset_index()
    print("All available columns:")
    print(fullDF.columns)

    subDF, acesL, boolColL, scalarColL = transformData(selectData(fullDF))
    
    subDF, acesL, boolColL, scalarColL = floatifyData(subDF, acesL, boolColL, scalarColL)

    print("Selected columns:")
    print(subDF.columns)
    COLUMN_DICT = {key : idx for idx, key in enumerate(mkSamps(subDF, 1).columns)}
    print(COLUMN_DICT)
    INV_COLUMN_DICT = {val:key for key, val in list(COLUMN_DICT.items())}
    print(INV_COLUMN_DICT)


    print("Number of records after selection and missing data removal: {}".format(len(subDF)))

    print("Loading the precalculated guide functions")
    with open('rsltd_20190408.pkl', 'rb') as f:
        rsltD = pickle.load(f, encoding='latin1')
    print("Found the following year pairs: {}".format(list(rsltD.keys())))
    #print rsltD[(6, 7)].x
    ageMin = min([a for a,b in rsltD])
    ageMax = max([b for a,b in rsltD])
    print('age range: ', ageMin, ageMax)
    for age in range(ageMin, ageMax):
        assert (age, age+1) in rsltD, "Age step {1} -> {2} is missing from precalculated guide functions".format(age,age+1)

    ageDFD = {}
    for age in range(ageMin, ageMax+1):
        ageDFD[age] = subDF[subDF.AGE==age]
    print('sample counts by age:')
    for age in range(ageMin, ageMax+1):
        print('  %s: %s' % (age, len(ageDFD[age])))

    nSampTarget = 30
    nSampPerBatch = 10
    nIter = 120000
    stepsizes = np.empty([nSampPerBatch])
    stepsizes.fill(0.005)
    
    def sampler(nSampPerBatch, df):
        return mkSamps(df, nSampPerBatch)

    startYear = 6
    endYear = 11
    #endYear = 17
    startDF = ageDFD[startYear]
    startDF = startDF[startDF.FPL <= 100]
    startSamps = sampler(nSampPerBatch, startDF).values # this defines agent initial state
    sampsByYearD = {startYear: startSamps}
    #startSamps = sampsByYearD[startYear]
    lowYear = startYear
    oldSamps = startSamps  # just the numpy array without pandas wrapping
    while True:
        highYear = lowYear + 1
        genSampParams = {'df': ageDFD[highYear]}
        mutatorParams = {'stepSzV': stepsizes, 'df': ageDFD[highYear]}

        print('samps at low year %s: ' % lowYear)
        print(oldSamps[:, COLUMN_DICT['index']])
        wtVec = rsltD[(lowYear, highYear)].x

        newSamps = None
        workingCols = ['YEAR', 'FPL', 'SC_AGE_YEARS', 'K4Q32X01', 'K7Q30', 'K7Q31', 'AGEPOS4']
        while True:
            try:
                batch = generateSamples(oldSamps, wtVec, workingCols,
                                        nIter, sampler,
                                        genSampParams, subDF,
                                        mutate, mutatorParams)
                print('got %s' % len(batch))
                if newSamps is None:
                    newSamps = batch.copy()
                else:
                    newSamps = np.concatenate((newSamps, batch))
                if len(newSamps) >= nSampTarget:
                    break
            except AssertionError as e:
                print('pass failed to produce useable samples: %s' % e)

        print('set of new: ', newSamps[:, COLUMN_DICT['index']])
        # Trim down to size
        newSamps = newSamps[np.random.choice(newSamps.shape[0], len(oldSamps), replace=True), :]
        print('reduced newSamps: ', newSamps[:, COLUMN_DICT['index']])
        sampsByYearD[highYear] = newSamps

        if highYear >= endYear:
            break
        else:
            lowYear = highYear
            oldSamps = newSamps

    #mkSamps test
    nSamp = 10
    print(mkSamps(subDF, nSamp))

    for k, v in list(sampsByYearD.items()):
        print('%s: %s' % (k, v))
    with open('sav_%s.pkl' % outID, 'w') as f:
        pickle.dump(sampsByYearD, f)

if __name__ == "__main__":
    main()
