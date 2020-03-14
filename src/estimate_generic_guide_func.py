#! /usr/bin/env python

import sys
import os
from optparse import OptionParser
import uuid
import yaml
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from  scipy.stats import describe
import matplotlib.pyplot as plt

from data_conversions import selectData, transformData
from data_conversions import floatifyData, boolifyData, quantizeData
from sampling import mkSamps, toHisto, toProbV
from sampling import createSparseBinner as createBinner
from metropolis import mutualInfo, genMetropolisSamples
from mutator import FreshDrawMutator, MSTMutator
from agent_based_model import load_dataset, createWeightSer


FIPS_DCT = {'SC':45, 'NC':37, 'TN':47, 'GA':13,
            'AL':1, 'VA':51, 'LA':22, 'AR':5}


#N_SAMP = 1000 # length of sample vectors
N_SAMP = 100


def get_unique_entries(cohort):
    if 'RECIDX' in cohort.columns:
        lst = [int(v) for v in cohort.drop_duplicates()['RECIDX']]
    else:
        lst = [int(v) for v in cohort.index.unique()]
    lst.sort()
    return lst


def select_subset(df, match_dct):
    df = df.copy()
    print('begin select_subset: %d records, %d unique'
          % (int(df.shape[0]), len(get_unique_entries(df))))
    for k, v in match_dct.items():
        df = df[df[k] == v]
        print('%s == %s: %d entries, %d unique'
              % (k, v, int(df.shape[0]), len(get_unique_entries(df))))
    return df


def lnLik(samps1V, samps2V, wtSerV):
    """
    funV has the right shape to fill the role of likelihood in the Metropolis algorithm.  We'll
    take the log, and use it as a log likelihood.
    """
    try:
        wtA = wtSerV.values
        sub1DF = samps1V[wtSerV.index]
        sub2DF = samps2V[wtSerV.index]
        delta = (sub1DF.values - sub2DF.values).astype(np.float)
        delta *= delta
        rslt = -np.einsum('ij,j -> i', delta, wtA)
#         print('wtA:')
#         print(wtA)
#         print('delta:')
#         print(delta)
#         print('rslt: ')
#         print(rslt)
        return rslt
    except Exception as e:
        print('samps1V: ')
        print(samps1V)
        print('wtSerV.index:')
        print(wtSerV.index)
        raise
          
#     return np.asarray((-np.asmatrix(wtA) * np.asmatrix(delta).transpose())).reshape((-1, 1))


# def lnLik(samps1V, samps2V, wtSerV):
#     """
#     funV has the right shape to fill the role of likelihood in the Metropolis algorithm.  We'll
#     take the log, and use it as a log likelihood.
#     """
#     try:
#         wtA = wtSerV.values.astype(np.float)
#         sub1DF = samps1V[wtSerV.index]
#         sub1M = sub1DF.values.astype(np.float)
#         sub2DF = samps2V[wtSerV.index]
#         sub2M = sub2DF.values.astype(np.float)
#         rslt = np.einsum('ia,a,ja -> i', sub1M, wtA, sub2M)
#     except Exception as e:
#         print('samps1V: ')
#         print(samps1V)
#         print('wtSerV.index:')
#         print(wtSerV.index)
#         import pdb
#         pdb.Pdb().set_trace()
#         raise
#          
#     return (1.0/float(sub2M.shape[0])) * rslt.reshape((-1,1))


# def gen_samples_using(samp_gen, target_cohort, pool_cohort,
#                       mutator, mutator_params,
#                       wt_ser, niter=1000):
#         ln_lik_params = {'samps2V': target_cohort, 'wtSerV': wt_ser}
#         gen_samp_params = {'df': pool_cohort}
#         
#         nsamp = len(target_cohort)
#         cleanSamps = genMetropolisSamples(nsamp, niter, samp_gen(**gen_samp_params), 
#                                           lnLik, ln_lik_params,
#                                           mutator, mutator_params, verbose=True)
#         if isinstance(cleanSamps[0], pd.DataFrame):
#             newCleanV = pd.concat(cleanSamps)
#         else:
#             newCleanV = np.concatenate(cleanSamps)
#         return samp_gen(newCleanV)
        

# This function takes lnLik from the environment!
def sampleAndCalcMI(wtSer, nSamp, nIter, sampler, testSampParams, genSampParams,
                    binner, binnerParams,
                    mutator, mutatorParams, drawGraph=False, verbose=False):
    tdf = testSampParams['df']
    n_samp = testSampParams['n_samp']
    assert len(tdf) < n_samp, 'test df is too big'
    gp1 = tdf.drop(columns='FWC')
    gp2 = mkSamps(tdf, n_samp - len(tdf))
    testSamps = pd.concat([gp1, gp2], axis=0)
    #testSamps = sampler(**testSampParams)
    guess = sampler(**genSampParams)
    lnLikParams = {'samps2V': testSamps, 'wtSerV': wtSer}
    cleanSamps = genMetropolisSamples(nSamp, nIter, guess, lnLik, lnLikParams,
                                     mutator, mutatorParams, verbose=verbose)
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


def weightedSampGen(df, n_samp):
    rslt = mkSamps(df, n_samp)
    if 'index' in rslt.columns:
        rslt = rslt.drop(columns=['index'])
    return rslt


def createWeightedSamplesGenerator(n_samp):
    def rsltF(df):
        rslt = mkSamps(df, n_samp)
        if 'index' in rslt.columns:
            rslt = rslt.drop(columns=['index'])
        return rslt
    return rsltF


# def get_rslt_path():
#     pth = os.path.join(os.getcwd(), str(uuid.uuid1()))
#     print('Result full path: ', pth)
#     return pth



def fitness(self, vec, **kwargs):
    return -mutualInfo(self.vec_to_df(vec), kwargs['target'],
                       self.binner, self.binnerParams)

def vec_to_df(self, vec):
    return self.samples_indexed.loc[vec]


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


MONITOR_LOG=[]

def monitorDE(xk, convergence):
    global MONITOR_LOG
    print('MONITOR: %s %s' % (xk, convergence))
    MONITOR_LOG.append([xk.copy(), convergence])


def main():
    global MONITOR_LOG
    parser = OptionParser(usage="""
    %prog [--alg=ALG] [seed]
    """)
#     parser.add_option('--fixed', action='store_true',
#                       help="Use a fixed generic guide function")
    parser.add_option('--alg', action='store',
                      choices=['gradient', 'fixed', 'genetic', 'd_e'],
                      help="Algorithm - one of gradient, fixed, genetic, d_e [default %default]",
                      default="fixed")
    opts, args = parser.parse_args()
    if len(args) > 1:
        parser.error('Extra arguments found')
    if len(args) == 1:
        try:
            seed_row = int(args[0])
            seed_file = None
        except ValueError:
            if os.path.exists(args[0]):
                seed_row = None
                seed_file = args[0]
            else:
                parser.error('seed must be an integer row number or a pkl file containing one record')
    else:
        parser.error('required argument is missing')
        seed_row = None
        seed_file = None

    parser.destroy()
    
#     rslt_path = get_rslt_path()
#     os.makedirs(rslt_path)

    subDF, acesL, boolColL, scalarColL, fixedL, passiveL, ageL = load_dataset()

    #print(scalarColL)
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

#    weightedSampGen = createWeightedSamplesGenerator(N_SAMP)

#    scSampGen = createWeightedSamplesGenerator(1)
#    prototype = scSampGen(pd.read_pickle(seed_file))
    prototype = weightedSampGen(pd.read_pickle(seed_file), 1)

    fixed_l = fixedL
    fixed_d = {elt: prototype.iloc[0][elt] for elt in fixed_l}
    passive_l = passiveL
    aces_l = acesL
    advancing_l = [elt for elt in aces_l
                   if elt not in list(fixed_d) + passive_l]
    open_l = [k for k in prototype.columns]
    for elt in list(fixed_d) + advancing_l + passive_l:
        open_l.remove(elt)
    samp_gen = weightedSampGen
    nIter = 1000
    all_col_l = open_l + advancing_l
    which_bin = createBinner(all_col_l, range_d=range_d)
    binnerParams = {}

    for age in range(ageMin, ageMax):
        transition = '%d-%d' % (age, age+1)
        outer_cohort = select_subset(ageDFD[age], fixed_d)
        new_outer_cohort = select_subset(ageDFD[age+1], fixed_d).drop(columns=['FWC'])
        mutator = MSTMutator(new_outer_cohort)
        mutatorParams = {'nsteps': 2}
        testSampParams = {'df': outer_cohort, 'n_samp': N_SAMP}
        genSampParams = {'df': new_outer_cohort, 'n_samp': N_SAMP}
        initial_wt_ser = createWeightSer(all_col_l, range_d=range_d)
        MONITOR_LOG = []
        rslt = differential_evolution(minimizeMe, 
                                      [(0.5*v, 4.0*v) for v in initial_wt_ser.values],
                                      (N_SAMP, nIter, all_col_l,
                                       weightedSampGen,
                                       testSampParams, genSampParams,
                                       which_bin, binnerParams,
                                       mutator, mutatorParams),
                                      maxiter=10, popsize=10,
                                      disp=True, polish=False,
                                      callback=monitorDE)
        print(rslt)
        with open('%s_generic.pkl' % transition, 'wb') as f:
            pickle.dump([rslt, MONITOR_LOG], f)


if __name__ == "__main__":
    main()

        