#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import cPickle as pickle


def mkSamps(df, nSamp):
    fracWt = df['FWC']/df['FWC'].sum()
    choices = np.random.choice(len(df), nSamp, p=fracWt)
    return df.iloc[choices].drop(columns=['FWC'])


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


def scatter(idx, vals, target):
    """target[idx] += vals, but allowing for repeats in idx"""
    np.add.at(target, idx.ravel(), vals.ravel())


def toHisto(sampV):
    """Generate a histogram of sample bins"""
    binV, nBins = whichBin(sampV)
    targ = np.zeros([nBins], dtype=np.int32)
    vals = np.ones([len(sampV)], dtype=np.int32)
    scatter(binV, vals, targ)
    return targ


def toProbV(sampV):
    sampH = toHisto(sampV)
    probV = sampH.astype(np.float64)
    probV /= np.sum(probV)
    return probV


def mutualInfo(sampVX, sampVY):
    assert len(sampVX) == len(sampVY), 'Sample vector lengths do not match'
    binVX, nBinsX = whichBin(sampVX)
    binVY, nBinsY = whichBin(sampVY)
    assert nBinsX == nBinsY, 'Unexpectedly got different bin counts?'
    cA = np.zeros([nBinsX, nBinsX], dtype=np.int32)
    idxV = np.ravel_multi_index(np.array([binVX, binVY]), (nBinsX, nBinsX))
    np.add.at(cA.ravel(), idxV, np.ones(len(idxV), dtype=np.int32).ravel())
    pA = cA.astype(np.float32)
    pA /= sum(pA.ravel())
    xPV = toProbV(sampVX)
    yPV = toProbV(sampVY)
    xyPA = np.einsum('i,j->ij', xPV, yPV)  # einsum is my new favorite function
    oldErr = np.seterr(invalid='ignore', divide='ignore')
    prodA = pA * np.nan_to_num(np.log(pA / xyPA))  # element-wise calculation
    np.seterr(**oldErr)
    return np.sum(prodA.ravel())


def lnLik(samps1V, samps2V, wtSerV):
    """
    funV has the right shape to fill the role of likelihood in the Metropolis algorithm.  We'll
    take the log, and use it as a log likelihood.
    """
    #print 'lnLik samps1V', samps1V
    #print 'samps2V', samps2V
    #print 'wtSerV', wtSerV
    wtA = wtSerV
    #offset = samps1V.shape[1] - wtSer.shape[0]
    offset = samps1V.shape[1] - wtSerV.shape[0]
    #print 'offset', samps1V.shape, wtSerV.shape, offset
    samp1A = samps1V[:, offset:]
    #print 'samp1A', samp1A
    samp2A = samps2V[:, offset:]
    #print 'samp2A', samp2A
    delta = samp1A - samp2A
    delta *= delta
    return np.asarray((-np.asmatrix(wtA) * np.asmatrix(delta).transpose())).reshape((-1, 1))


# Now we need a mutator
def mutate(sampV, df, stepSzV):
    """
    Return a 'mutated' version of sampV, based on the given step sizes.  Unfortunately our samples
    are discrete and come from a table, so I'm not sure how to do this unless we first generate
    a proximity network of some sort, so for the moment let's just generate a new set of samples-
    this corresponds to an infinitely wide mutator.
    """
    return mkSamps(df, len(sampV)).values


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
    A = [guess.values] # List of vectors of samples
    for n in range(nIter):
        oldAlpha  = A[-1]  # old parameter value as array
        #print 'start: ', oldAlpha
        oldLnLik = lnLikFun(oldAlpha, **lnLikParams)
        newAlpha = mutator(oldAlpha, **mutatorParams)
        #print 'newAlpha: ', newAlpha
        newLnLik = lnLikFun(newAlpha, **lnLikParams)
        #print 'newLnLik: ', newLnLik
        if verbose and (n % 100 == 0):
            print '%s: %s' % (n, np.sum(newLnLik))
        llDelta = newLnLik - oldLnLik
        llDelta = np.minimum(llDelta, 0.0)
        #choices = np.logical_or(newLnLik > oldLnLik,
        #                        np.random.random(newLnLik.shape) < np.exp(newLnLik - oldLnLik))
        choices = np.logical_or(newLnLik > oldLnLik,
                                np.random.random(newLnLik.shape) < np.exp(llDelta))
        rslt = np.choose(choices, [oldAlpha, newAlpha])
        A.append(rslt)
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
    A, acceptanceRate = genRawMetropolisSamples(nSamp, nIter, guess, lnLikFun, lnLikParams,
                                                mutator, mutatorParams, verbose=verbose)
    #print acceptanceRate, nIter, (acceptanceRate * nIter).min()
    nKeep = int((acceptanceRate * nIter).min() / mutationsPerSamp)
    keepStep = nIter//nKeep
    burnIn = burninMutations * keepStep
    assert burnIn < nIter, 'Not enough iterations for burn-in (%s vs %s)' % (nIter, burnIn)
    clean = []
    for idx, sV in enumerate(A[burnIn:]):
        if idx % keepStep == 0:
            clean.append(sV)
    return clean


# This function takes lnLik from the environment!
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


def generateSamples(oldSamps, wtVec, workingCols,
                    nIter, sampler, genSampParams, srcDF,
                    mutator, mutatorParams,
                    verbose=False):

    # get the right index order but no extra entries
    wtSer = pd.Series({key: val for key, val in zip(workingCols, range(len(workingCols)))},
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


def selectData(fullDF):

#     subDF=fullDF[['ACE1', 'ACE3', 'ACE4', 'ACE5', 'ACE6', 'ACE7', 'ACE8', 'ACE9', 'ACE10', 'FWC', 'index',
#                   'YEAR', 'FPL', 'SC_AGE_YEARS','K4Q32X01', 'K7Q30', 'K7Q31', 'AGEPOS4']].copy()

    subDF=fullDF[fullDF.FIPSST == 45][['ACE3', 'ACE4', 'ACE5', 'ACE6', 'ACE7',
                                   'ACE8', 'ACE9', 'ACE10',
                                   'FWC', 'YEAR', 'FPL', 'SC_AGE_YEARS',
                                   'K4Q30_R', 'K4Q32X01',
                                   'K7Q30', 'K7Q31', 'AGEPOS4',
                                   'HHLANGUAGE', # 1=English 2=Spanish 3=Other
                                   'SC_CSHCN', # selected child special health care needs (1=yes, 2=no)
                                   'SC_RACE_R', # selected child race (1=white, 2=black, 3=native 4=asian 5=islands, 6=other, 7=mixed)
                                   'SC_HISPANIC_R', # 1=latinx, 2=not
                                   'TOTCSHCN', # Total children with special health care needs
                                   'TOTNONSHCN', # Total children without special health care needs
                                   'TOTMALE', # Total male children
                                   'TOTFEMALE', # Total female childrem
                                   'K2Q05', # Born 3 weeks before due date (1=yes, 2=no)
                                   'BIRTHWT_VL', # Birth weight very low (1=yes, 2=no)
                                   'BIRTHWT_L', # Birth weight low (1=yes, 2=no)
                                   'MOMAGE', # Age of mother at birth (scalar)
                                   'S4Q01', # Doctor visit in last 12 months (1=yes, 2=no)
                                   'index'
                                  ]]
    return subDF


def transformData(data):
#     data['ACETOT'] = data['ACE1'] + 15 - (data['ACE3'] + data['ACE4'] + data['ACE5'] + data['ACE6']
#                             + data['ACE7'] + data['ACE8'] + data['ACE9'] + data['ACE10'])

    data = data.rename(columns={'AGEPOS4':'BIRTHORDER', 'SC_AGE_YEARS':'AGE', 'K4Q30_R':'DENTALCARE',
                               'K4Q32X01':'VISIONCARE', 'K7Q30':'SPORTSTEAMS', 'K7Q31':'CLUBS',
                                'ACE3':'PARENTDIVORCED',
                                'ACE4':'PARENTDIED', 'ACE5':'PARENTJAIL', 'ACE6':'SEEPUNCH',
                               'ACE7':'VIOLENCE', 'ACE8': 'MENTALILL', 'ACE9':'DRUGSALCOHOL',
                               'ACE10':'RACISM'})

#     out = pd.cut(data['FPL'], 8, labels=False)
#     data['FPL_quantized'] = out
#     data['FPL_quantized'].unique()
    
    data = data.dropna()

    """
    TOTCSHCN plus TOTNONSHCN: convert to HHNCHILDREN (scalar)
    
                                       'TOTCSHCN', # Total children with special health care needs
                                       'TOTNONSHCN', # Total children without special health care needs
    
    """

    massaged_data = data[['FWC', 'AGE', 'FPL', 'index']].copy()
    massaged_data['DENTALCARE'] = (data['DENTALCARE'] != 3.0)
    massaged_data['VISIONCARE'] = (data['VISIONCARE'] == 1.0)
    massaged_data['SPORTSTEAMS'] = (data['SPORTSTEAMS'] == 1.0)
    massaged_data['CLUBS'] = (data['CLUBS'] == 1.0)
    massaged_data['BIRTHORDER'] = data['BIRTHORDER']
    massaged_data['SC_CSHCN'] = (data['SC_CSHCN'] == 1.0)
    massaged_data['BIRTHWT_VL'] = (data['BIRTHWT_VL'] == 1.0)
    massaged_data['BIRTHWT_L'] = (data['BIRTHWT_L'] == 1.0)
    massaged_data['HHLANGUAGE_ENGLISH'] = (data['HHLANGUAGE'] == 1.0)
    massaged_data['HHLANGUAGE_SPANISH'] = (data['HHLANGUAGE'] == 2.0)
    massaged_data['DOCTORVISIT'] = (data['S4Q01'] == 1.0)
    massaged_data['PREMATURE'] = (data['K2Q05'] == 1.0)
    massaged_data['SC_RACE_WHITE'] = (data['SC_RACE_R'] == 1.0)
    massaged_data['SC_RACE_BLACK'] = (data['SC_RACE_R'] == 2.0)
    massaged_data['SC_RACE_NATIVE'] = (data['SC_RACE_R'] == 3.0)
    massaged_data['SC_RACE_ASIAN'] = (data['SC_RACE_R'] == 4.0)
    massaged_data['SC_RACE_ISLANDS'] = (data['SC_RACE_R'] == 5.0)
    massaged_data['SC_RACE_OTHER'] = (data['SC_RACE_R'] == 6.0)
    massaged_data['SC_RACE_MIXED'] = (data['SC_RACE_R'] == 7.0)
    massaged_data['SC_RACE_HISPANIC'] = (data['SC_HISPANIC_R'] == 1.0)
    massaged_data['TOTKIDS'] = data['TOTCSHCN'] + data['TOTNONSHCN'] # scalar
    massaged_data['TOTCSHCN'] = data['TOTCSHCN']  # scalar
    massaged_data['MOMAGE'] = data['MOMAGE'] # scalar
    
    
    acesL = ['PARENTDIVORCED', 'PARENTDIED', 'PARENTJAIL', 'SEEPUNCH', 'VIOLENCE', 'MENTALILL',
             'DRUGSALCOHOL', 'RACISM']
    acesL.sort()
    for ace in acesL:
        massaged_data[ace] = (data[ace] == 1)
    
    boolColL = ['DENTALCARE', 'VISIONCARE', 'SPORTSTEAMS', 'CLUBS', 'SC_CSHCN',
                'BIRTHWT_VL', 'BIRTHWT_L', 'HHLANGUAGE_ENGLISH', 'HHLANGUAGE_SPANISH',
                'DOCTORVISIT', 'PREMATURE',
                'SC_RACE_WHITE', 'SC_RACE_BLACK', 'SC_RACE_HISPANIC', 'SC_RACE_NATIVE', 'SC_RACE_ASIAN',
                'SC_RACE_ISLANDS', 'SC_RACE_OTHER', 'SC_RACE_MIXED']
    boolColL.sort()

    scalarColL = ['FPL', 'BIRTHORDER', 'AGE', 'TOTCSHCN', 'TOTKIDS', 'MOMAGE']
    scalarColL.sort()
    
    print('Before transformation:')
    print data.head()
    print('After transformation:')
    print massaged_data.head()

    return massaged_data, acesL, boolColL, scalarColL


def floatifyData(data, acesL, boolColL, scalarColL):
    """
    Things in acesL and boolColL are boolean; convert to 1.0/0.0
    """
    convertS = set(acesL + boolColL)
    out_data = data[[col for col in data.columns if col not in convertS]].copy()
    for col in convertS:
        out_data[col] = np.choose(data[col], [0.0, 1.0])
    return out_data, acesL, boolColL, scalarColL


def boolifyData(data, acesL, boolColL, scalarColL, fpl_mode='binned'):
    """
    Things in acesL and boolColL are already boolean.  Must convert the values in scalarColL.
    
    Supported fpl_mode values are 'binned', 'ascending', 'descending'
    """
    convertS = set(scalarColL)
    out_data = data[[col for col in data.columns if col not in convertS]].copy()
    print 'out_data columns 1: {}'.format(out_data.columns)
    drop_these = []
    for col in convertS:
        if col == 'MOMAGE':  # special case
            out_data['MOMAGE_LT_20'] = (data[col] < 20)
            out_data['MOMAGE_GT_39'] = (data[col] > 39)
            boolColL = boolColL[:] + ['MOMAGE_LT_20', 'MOMAGE_GT_39']
        elif col == 'FPL':  # special case
            out_data['FPL_quantized'] = pd.cut(data['FPL'], 8, labels=False)

            if fpl_mode == 'binned':
                cat_list = pd.get_dummies(out_data['FPL_quantized'], prefix='FPL')
                data1 = out_data.join(cat_list)
                out_data = data1
                drop_these += ['FPL_quantized']
            elif fpl_mode == 'ascending':
                out_data['FPL_AT_LEAST_7'] = out_data['FPL_quantized'] >= 7
                out_data['FPL_AT_LEAST_6'] = out_data['FPL_quantized'] >= 6
                out_data['FPL_AT_LEAST_5'] = out_data['FPL_quantized'] >= 5
                out_data['FPL_AT_LEAST_4'] = out_data['FPL_quantized'] >= 4
                out_data['FPL_AT_LEAST_3'] = out_data['FPL_quantized'] >= 3
                out_data['FPL_AT_LEAST_2'] = out_data['FPL_quantized'] >= 2
                out_data['FPL_AT_LEAST_1'] = out_data['FPL_quantized'] >= 1
                drop_these += ['FPL_quantized']
            elif fpl_mode == 'descending':
                out_data['FPL_LESS_THAN_1'] = out_data['FPL_quantized'] < 1
                out_data['FPL_LESS_THAN_2'] = out_data['FPL_quantized'] < 2
                out_data['FPL_LESS_THAN_3'] = out_data['FPL_quantized'] < 3
                out_data['FPL_LESS_THAN_4'] = out_data['FPL_quantized'] < 4
                out_data['FPL_LESS_THAN_5'] = out_data['FPL_quantized'] < 5
                out_data['FPL_LESS_THAN_6'] = out_data['FPL_quantized'] < 6
                out_data['FPL_LESS_THAN_7'] = out_data['FPL_quantized'] < 7
                drop_these += ['FPL_quantized']
            else:
                raise RuntimeError('Unknown FPL_mode')

        else:
            cat_list = pd.get_dummies(data[col], prefix=col)
            data1 = out_data.join(cat_list)
            out_data = data1

    out_data1 = out_data.drop(columns=drop_these)
    out_data = out_data1
    print 'out_data columns 2: {}'.format(out_data.columns)


    return out_data, acesL, boolColL, scalarColL


def main(argv = None):
    if argv is None:
        argv = sys.argv

    outID = int(sys.argv[1])
    print "Result ID will be {}".format(outID)

    print 'loading subject data and subsetting'
    fullDF = pd.read_csv('/home/welling/git/synecoace/data/nsch_2016_topical.csv')
    print fullDF.columns

    fullDF = fullDF.reset_index()
    print "All available columns:"
    print fullDF.columns

    subDF, acesL, boolColL, scalarColL = transformData(selectData(fullDF))
    
    subDF, acesL, boolColL, scalarColL = floatifyData(subDF, acesL, boolColL, scalarColL)

    print "Selected columns:"
    print subDF.columns
    COLUMN_DICT = {key : idx for idx, key in enumerate(mkSamps(subDF, 1).columns)}
    print COLUMN_DICT
    INV_COLUMN_DICT = {val:key for key, val in COLUMN_DICT.items()}
    print INV_COLUMN_DICT


    print "Number of records after selection and missing data removal: {}".format(len(subDF))

    print "Loading the precalculated guide functions"
    with open('rsltd_20190408.pkl', 'rU') as f:
        rsltD = pickle.load(f)
    print "Found the following year pairs: {}".format(rsltD.keys())
    #print rsltD[(6, 7)].x
    ageMin = min([a for a,b in rsltD])
    ageMax = max([b for a,b in rsltD])
    print 'age range: ', ageMin, ageMax
    for age in range(ageMin, ageMax):
        assert (age, age+1) in rsltD, "Age step {1} -> {2} is missing from precalculated guide functions".format(age,age+1)

    ageDFD = {}
    for age in range(ageMin, ageMax+1):
        ageDFD[age] = subDF[subDF.AGE==age]
    print 'sample counts by age:'
    for age in range(ageMin, ageMax+1):
        print '  %s: %s' % (age, len(ageDFD[age]))

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

        print 'samps at low year %s: ' % lowYear
        print oldSamps[:, COLUMN_DICT['index']]
        wtVec = rsltD[(lowYear, highYear)].x

        newSamps = None
        workingCols = ['YEAR', 'FPL', 'SC_AGE_YEARS', 'K4Q32X01', 'K7Q30', 'K7Q31', 'AGEPOS4']
        while True:
            try:
                batch = generateSamples(oldSamps, wtVec, workingCols,
                                        nIter, sampler,
                                        genSampParams, subDF,
                                        mutate, mutatorParams)
                print 'got %s' % len(batch)
                if newSamps is None:
                    newSamps = batch.copy()
                else:
                    newSamps = np.concatenate((newSamps, batch))
                if len(newSamps) >= nSampTarget:
                    break
            except AssertionError as e:
                print 'pass failed to produce useable samples: %s' % e

        print 'set of new: ', newSamps[:, COLUMN_DICT['index']]
        # Trim down to size
        newSamps = newSamps[np.random.choice(newSamps.shape[0], len(oldSamps), replace=True), :]
        print 'reduced newSamps: ', newSamps[:, COLUMN_DICT['index']]
        sampsByYearD[highYear] = newSamps

        if highYear >= endYear:
            break
        else:
            lowYear = highYear
            oldSamps = newSamps

    #mkSamps test
    nSamp = 10
    print mkSamps(subDF, nSamp)

    for k, v in sampsByYearD.items():
        print '%s: %s' % (k, v)
    with open('sav_%s.pkl' % outID, 'w') as f:
        pickle.dump(sampsByYearD, f)

if __name__ == "__main__":
    main()
