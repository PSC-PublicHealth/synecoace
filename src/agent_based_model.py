#! /usr/bin/env python

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from data_conversions import selectData, transformData
from data_conversions import floatifyData, boolifyData, quantizeData
from sampling import mkSamps, toHisto, toProbV
from sampling import createSparseBinner as createBinner
from metropolis import mutualInfo, genMetropolisSamples


def select_subset(df, match_dct):
    df = df.copy()
    print('begin select_subset: %d records, %d unique'
          % (int(df.shape[0]), len(df.index.unique())))
    for k, v in match_dct.items():
        df = df[df[k] == v]
        print('%s == %s: %d entries, %d unique'
              % (k, v, int(df.shape[0]), len(df.index.unique())))
    return df


def createWeightSer(colL, range_d, vals=None):
    if vals is None:
        wtSer = pd.Series({col:1.0/range_d[col] for col in colL})
    else:
        wtSer = pd.Series({col:v for col, v in zip(colL, vals)})
    print('wtSer: %s' % wtSer.values)
    return wtSer


def lnLik(samps1V, samps2V, wtSerV):
    """
    funV has the right shape to fill the role of likelihood in the Metropolis algorithm.  We'll
    take the log, and use it as a log likelihood.
    """
    try:
        wtA = wtSerV.values
        sub1DF = samps1V[wtSerV.index]
        sub2DF = samps2V[wtSerV.index]
        delta = sub1DF.values - sub2DF.values
        delta *= delta
    except Exception as e:
        print('samps1V: ')
        print(samps1V)
        print('wtSerV.index:')
        print(wtSerV.index)
        raise
        
    return np.asarray((-np.asmatrix(wtA) * np.asmatrix(delta).transpose())).reshape((-1, 1))


def mutate(sampV, df, stepSzV):
    """
    Return a 'mutated' version of sampV, based on the given step sizes.  Unfortunately our samples
    are discrete and come from a table, so I'm not sure how to do this unless we first generate
    a proximity network of some sort, so for the moment let's just generate a new set of samples-
    this corresponds to an infinitely wide mutator.
    """
    return mkSamps(df, len(sampV))


# This function takes lnLik from the environment!
def sampleAndCalcMI(wtSer, nSamp, nIter, sampler, testSampParams, genSampParams,
                    binner, binnerParams,
                    mutator, mutatorParams, drawGraph=False, verbose=False):
    testSamps = sampler(**testSampParams)
    guess = sampler(**genSampParams)
    lnLikParams = {'samps2V': testSamps, 'wtSerV': wtSer}
    cleanSamps = genMetropolisSamples(nSamp, nIter, guess, lnLik, lnLikParams,
                                      mutate, mutatorParams, verbose=verbose)
    if isinstance(cleanSamps[0], pd.DataFrame):
        cleanV = pd.concat(cleanSamps)
        expandedTestV = pd.concat([testSamps] * len(cleanSamps))
    else:
        cleanV = np.concatenate(cleanSamps)
        expandedTestV = np.concatenate([testSamps.values] * len(cleanSamps))
    
    if drawGraph:
        testBins, nBins = whichBin(expandedTestV)
        rsltBins = whichBin(cleanV)[0]
        hM, xEdges, yEdges = np.histogram2d(testBins, rsltBins, bins=64)
        plt.imshow(np.log(hM + 1))
        plt.show()

    return mutualInfo(cleanV, expandedTestV, binner, binnerParams=binnerParams)


def minimizeMe(wtVec,
               nSamp, nIter, colL, sampler, testSampParams, genSampParams,
               binner, binnerParams,
               mutator, mutatorParams):
    wtSer = createWeightSer(colL, {}, wtVec)
    mI = sampleAndCalcMI(wtSer, nSamp, nIter, sampler, testSampParams, genSampParams,
                         binner, binnerParams,
                         mutator, mutatorParams, drawGraph=False, verbose=False)
    print('minimizeMe -> ', mI)
    return -mI


class Agent(object):
    def __init__(self, prototype, all_samples, samp_gen, fixed_l, aces_l, age,
                 range_d = None):
        fixed_d = {elt: prototype.iloc[0][elt] for elt in fixed_l}
        self.fixed_keys = fixed_d 
        self.samp_gen = samp_gen
        self.age = age # whatever is in the prototype
        self.range_d = range_d
        self.inner_cohort = samp_gen(prototype)
        self.outer_cohort = samp_gen(select_subset(all_samples, self.fixed_keys))
        advancing_l = []
        for elt in aces_l:
            if prototype.iloc[0][elt]:
                fixed_d[elt] = True
            else:
                advancing_l.append(elt)
        self.advancing_l = advancing_l
        open_l = [k for k in self.outer_cohort.columns]
        for elt in list(fixed_d) + advancing_l:
            open_l.remove(elt)
        self.open_l = open_l
        print('initial outer_cohort unique entries: ', self.outer_cohort.index.unique())
        # set the fixed keys according to the prototype
        # the inner cohort is the prototype
        # the outer cohort is all samples with the same fixed keys
    
    def age_transition(self, new_all_samples):
        # each fixed key must get updated according to its own rule
        # - gender, birthorder, etc. stay fixed
        # the outer cohort is all samples at the new age with the same fixed keys
        print('new_all_samples unique entries: ', new_all_samples.index.unique())
        all_col_l = self.open_l + self.advancing_l
        which_bin = createBinner(all_col_l, range_d=self.range_d)
        wt_ser = createWeightSer(all_col_l, range_d=self.range_d)
        samples_subset = select_subset(new_all_samples, self.fixed_keys)
        print('samples_subset unique entries: ', new_all_samples.index.unique())
        new_outer_cohort = self.samp_gen(samples_subset)
        #print('inner cohort keys: ', self.inner_cohort.columns)
        #print('outer cohort keys: ', self.outer_cohort.columns)
#         print('old inner vs. outer: ', mutualInfo(self.inner_cohort, self.outer_cohort,
#                                                   which_bin))
#         print('old outer vs. new outer: ', mutualInfo(self.outer_cohort, new_outer_cohort,
#                                                       which_bin))
#         print('old inner vs. new outer: ', mutualInfo(self.inner_cohort, new_outer_cohort,
#                                                       which_bin))
#         print('old outer vs. old outer: ', mutualInfo(self.outer_cohort, self.outer_cohort,
#                                                       which_bin))
#         print('new outer vs. new outer: ', mutualInfo(new_outer_cohort, new_outer_cohort,
#                                                       which_bin))
#         print('lnLik: follows')
#         print(lnLik(self.inner_cohort, new_outer_cohort, wt_ser))
        
        new_age = self.age + 1
        print('------------------')
        print('new outer cohort unique entries: ', new_outer_cohort.index.unique())
        print('starting %s -> %s' % (self.age, new_age))
        print('------------------')
        nSamp = len(self.inner_cohort)
        nIter = 1000
        stepsizes = np.empty([nSamp])
        stepsizes.fill(0.005)
        testSampParams = {'df': self.inner_cohort}
        genSampParams = {'df': new_outer_cohort}
        binnerParams = {}
        mutatorParams = {'stepSzV': stepsizes, 'df': new_outer_cohort}
    
        guess = wt_ser.values.copy()
        rslt = minimize(minimizeMe, guess, (nSamp, nIter, all_col_l,
                                            self.samp_gen,
                                            testSampParams, genSampParams,
                                            which_bin, binnerParams,
                                            mutate, mutatorParams), method='powell')
        print('------------------')
        print('Optimization result:')
        print(rslt)
        print('------------------')

        bestWtSer = createWeightSer(all_col_l, {}, rslt.x)
        lnLikParams = {'samps2V': self.inner_cohort, 'wtSerV': bestWtSer}
        cleanSamps = genMetropolisSamples(nSamp, nIter, self.samp_gen(**genSampParams), 
                                          lnLik, lnLikParams,
                                          mutate, mutatorParams, verbose=True)
        if isinstance(cleanSamps[0], pd.DataFrame):
            newCleanV = pd.concat(cleanSamps)
        else:
            newCleanV = np.concatenate(cleanSamps)
        new_inner_cohort = self.samp_gen(newCleanV)
        
        print('new inner cohort unique entries: ', new_inner_cohort.index.unique())
        print('------------------')

        self.age = new_age
        self.outer_cohort = new_outer_cohort
        self.inner_cohort = new_inner_cohort
        

def createWeightedSamplesGenerator(n_samp):
    def rsltF(df):
        rslt = mkSamps(df, n_samp)
        if 'index' in rslt.columns:
            rslt = rslt.drop(columns=['index'])
        return rslt
    return rsltF


def main():
    fullDF = pd.read_csv('/home/welling/git/synecoace/data/nsch_2016_topical.csv',
                         encoding='utf-8')
    print(fullDF.columns)
    fipsD = {'SC':45, 'NC':37, 'TN':47, 'GA':13,
             'AL':1, 'VA':51, 'LA':22, 'AR':5}
    subDF, acesL, boolColL, scalarColL = transformData(selectData(fullDF.reset_index(),
                                                                  fipsL=fipsD.values(),
                                                                  includeFips=True))
    # columns we think will be fixed throughout life
#     fixedL = ['BIRTHWT_VL', 'BIRTHWT_L', 'PREMATURE',
#               'HHLANGUAGE_ENGLISH', 'HHLANGUAGE_SPANISH',
#               'SC_FEMALE', 'SC_RACE_NATIVE', 'SC_RACE_ASIAN',
#               'SC_RACE_ISLANDS', 'SC_RACE_OTHER', 'SC_RACE_MIXED',
#               'SC_RACE_HISPANIC', 'MOMAGE_LT_20', 'MOMAGE_GT_39',
#               'BIRTHORDER_2', 'BIRTHORDER_3',
#               'BIRTHORDER_4']
    fixedL = ['BIRTHWT_VL', 'PREMATURE',
              'HHLANGUAGE_ENGLISH',
              'SC_FEMALE', 'SC_RACE_NATIVE', 'SC_RACE_ASIAN',
              'SC_RACE_ISLANDS', 'SC_RACE_OTHER', 'SC_RACE_MIXED',
              'SC_RACE_HISPANIC', 'MOMAGE_LT_20', 'MOMAGE_GT_39']
    dropL = ['DRUGSALCOHOL', 'MENTALILL', 'PARENTDIED', 'PARENTDIVORCED',
             'PARENTJAIL', 'RACISM', 'SEEPUNCH', 'VIOLENCE',
             'SC_RACE_WHITE', 'TOTCSHCN', 'TOTACES']
    subDF = subDF.drop(columns=dropL)
    for col in dropL:
        if col in acesL: acesL.remove(col)
        if col in boolColL: boolColL.remove(col)
        if col in scalarColL: scalarColL.remove(col)

    #print(scalarColL)
    ageL = subDF['AGE'].unique()
    print('ages: ', ageL)
    ageMin = int(min(ageL))
    ageMax = int(max(ageL))
    scalarColL.remove('AGE')
    print('scalar columns: ', scalarColL)
    ageDFD = {}
    range_d = None
    for age in range(ageMin, ageMax+1):
        ageDF = subDF[subDF.AGE==age].drop(columns=['AGE', 'FIPSST'])
        ageDFD[age], _, _, _, dct = quantizeData(ageDF, acesL, boolColL, scalarColL)
        if range_d is None:
            range_d = dct
        else:
            assert dct == range_d, 'Quantized ranges do not match?'

    weightedSampGen = createWeightedSamplesGenerator(1000)

    df = subDF[subDF.AGE==ageMin].drop(columns='AGE')
    df = df[df.FIPSST == fipsD['SC']].drop(columns=['FIPSST'])
    df, _, _, _, dct = quantizeData(df, acesL, boolColL, scalarColL)
    assert dct == range_d, 'Quantized ranges do not match?'
    scSampGen = createWeightedSamplesGenerator(1)
    prototype = scSampGen(ageDFD[ageMin])
    print('prototype columns: ', prototype.columns)

    agent = Agent(prototype, ageDFD[ageMin], weightedSampGen, fixedL, acesL, ageMin,
                  range_d=range_d)
    while agent.age < ageMax:
        new_age = agent.age + 1
        agent.age_transition(ageDFD[new_age])

#     wrkSamps = mkSamps(ageDFD[age], 100000).drop(columns=['index'])
#     #print(wrkSamps.columns)
#     print('-------- age {}: {} samples ----------'.format(age, len(ageDFD[age])))
#     for col in acesL:
#         print(col, ' : ', float(wrkSamps[col].sum())/float(len(wrkSamps)))

if __name__ == "__main__":
    main()

        