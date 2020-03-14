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


FIPS_DCT = {'SC':45, 'NC':37, 'TN':47, 'GA':13,
            'AL':1, 'VA':51, 'LA':22, 'AR':5}


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


# def lnLik(samps1V, samps2V, wtSerV):
#     """
#     funV has the right shape to fill the role of likelihood in the Metropolis algorithm.  We'll
#     take the log, and use it as a log likelihood.
#     """
#     try:
#         wtA = wtSerV.values
#         sub1DF = samps1V[wtSerV.index]
#         sub2DF = samps2V[wtSerV.index]
#         delta = sub1DF.values - sub2DF.values
#         delta *= delta
#         import pdb
#         pdb.Pdb().set_trace()
#     except Exception as e:
#         print('samps1V: ')
#         print(samps1V)
#         print('wtSerV.index:')
#         print(wtSerV.index)
#         raise
#          
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
#         print('index: ', sub1DF.index)
#         print('columns: ', sub1DF.columns)
#         print('shape: ', sub1DF.values.shape)
#         print(sub1DF.values)
#         print('FPL column?')
#         print(sub1DF.values[:,-2])
        base_fpl_col = sub2DF.values[:,-2]
        delta_fpl_col = delta[:, -2]
        delta_fpl_col = np.where(base_fpl_col < 1.0, delta_fpl_col - 0.3, delta_fpl_col)
        delta[:, -2] = delta_fpl_col
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


def lik(samps1V, samps2V, wtSerV):
    return np.exp(lnLik(samps1V, samps2V, wtSerV))


# This function takes lnLik from the environment!
def sampleAndCalcMI(wtSer, nSamp, nIter, sampler, testSampParams, genSampParams,
                    binner, binnerParams,
                    mutator, mutatorParams, drawGraph=False, verbose=False):
    testSamps = sampler(**testSampParams)
    guess = sampler(**genSampParams)
    lnLikParams = {'samps2V': testSamps, 'wtSerV': wtSer}
    cleanSamps = genMetropolisSamples(nSamp, nIter, guess, lik, lnLikParams,
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


def minimizeMe(wtVec,
               nSamp, nIter, colL, sampler, testSampParams, genSampParams,
               binner, binnerParams,
               mutator, mutatorParams):
    wtSer = createWeightSer(colL, {}, wtVec)
    mI = sampleAndCalcMI(wtSer, nSamp, nIter, sampler, testSampParams, genSampParams,
                         binner, binnerParams,
                         mutator, mutatorParams, drawGraph=False, verbose=False)
    #print('minimizeMe -> ', mI)
    return -mI


class Agent(object):
    def __init__(self, prototype, all_samples, samp_gen, fixed_l, aces_l, passive_l,
                 age, range_d = None, algorithm=None, opt_dct={}):
        assert algorithm is not None, 'algorithm should be set'
        self.algorithm = algorithm
        self.opt_dct = opt_dct.copy()
        fixed_d = {elt: prototype.iloc[0][elt] for elt in fixed_l}
        self.fixed_d = fixed_d
        self.samp_gen = samp_gen
        self.age = age # whatever is in the prototype
        self.range_d = range_d
        self.passive_l = passive_l
        self.outer_cohort = samp_gen(select_subset(all_samples, fixed_d))
        advancing_l = [elt for elt in aces_l
                       if elt not in list(self.fixed_d) + passive_l]
        self.advancing_l = advancing_l
        open_l = [k for k in self.outer_cohort.columns]
        for elt in list(fixed_d) + advancing_l + passive_l:
            open_l.remove(elt)
        self.open_l = open_l
        self.mutual_info = None
        print('KEYS: ', prototype.index)
        self.inner_cohort = samp_gen(prototype)
#         self.inner_cohort = self.gen_samples_using(samp_gen(prototype),
#                                                    self.outer_cohort,
#                                                    MSTMutator(self.outer_cohort),
#                                                    {'nsteps': 2},
#                                                    createWeightSer(self.open_l + self.advancing_l,
#                                                                    range_d=self.range_d))
        print('initial outer_cohort unique entries: ', get_unique_entries(self.outer_cohort))
        print('initial inner_cohort unique entries: ', get_unique_entries(self.inner_cohort))
    
    def write_state(self, path):
        outer_cohort_unique = get_unique_entries(self.outer_cohort)
        inner_cohort_unique = get_unique_entries(self.inner_cohort)

        dct = {'fixed_d': {k : int(v) for k, v in self.fixed_d.items()},
               'age': self.age,
               'range_d': {k : int(v) for k, v in self.range_d.items()},
               'passive_l': self.passive_l,
               'advancing_l': self.advancing_l,
               'open_l': self.open_l,
               'outer_cohort_unique_entries': outer_cohort_unique,
               'inner_cohort_unique_entries': inner_cohort_unique,
               'algorithm': self.algorithm,
               'mutualinfo': self.mutual_info
               }
        with open(os.path.join(path, 'state.yml'), 'w') as f:
            yaml.dump(dct, f)
        self.inner_cohort.to_pickle(os.path.join(path, 'inner_cohort.pkl'))

    def fitness(self, vec, **kwargs):
        return -mutualInfo(self.vec_to_df(vec), kwargs['target'],
                           self.binner, self.binnerParams)
    
    def vec_to_df(self, vec):
        return self.samples_indexed.loc[vec]
        

    def age_transition(self, new_all_samples):
        new_age = self.age + 1
        print('------------------')
        print('new_all_samples unique entries: ', len(get_unique_entries(new_all_samples)))
        all_col_l = self.open_l + self.advancing_l
        which_bin = createBinner(all_col_l, range_d=self.range_d)
        self.binner = which_bin
        self.binnerParams = {}
        wt_ser = createWeightSer(all_col_l, range_d=self.range_d)
        samples_subset = select_subset(new_all_samples, self.fixed_d)
        print('------------------')
        print('samples_subset unique entries: ', len(get_unique_entries(samples_subset)))
        new_outer_cohort = self.samp_gen(samples_subset)        
        print('------------------')
        print('new outer cohort unique entries: ', get_unique_entries(new_outer_cohort))
        print('starting %s -> %s' % (self.age, new_age))
        print('------------------')

        if self.algorithm == 'genetic':
            print('samples_subset columns:', samples_subset.columns)
            print(samples_subset.head())
            self.pool = samples_subset['RECIDX'].unique()
            self.samples_indexed = samples_subset.drop(columns=['FWC']).set_index('RECIDX')

            popsz = 50  # 2
            grpsz = len(self.inner_cohort)
            crossp = 0.5  # Fraction of array replaced in a crossing op
            crossfrac = 0.8  # Fraction of the replacement which comes from an existing pop member

            pop = {idx : np.random.choice(self.pool, grpsz) for idx in range(popsz)}
            #target = np.random.choice(self.pool, grpsz) # for test purposes
            target = self.inner_cohort
            fitnessParams = {'target' : target}
            fitnessV = np.asarray([self.fitness(pop[idx], **fitnessParams) for idx in range(popsz)])
            best_idx = np.argmin(fitnessV)
            best = pop[best_idx]
            print('Best: ', best_idx, fitnessV[best_idx])
            niter = 100
            for iter in range(niter):
                cross_ct = 0
                for j in range(popsz):
                    mutant = np.random.choice(self.pool, grpsz)  # a fresh mutant
                    idxs = [idx for idx in range(popsz) if idx != j]
                    nbr = pop[np.random.choice(idxs)]  # random member of the population
                    nbr_mutant_cross_points = np.random.rand(grpsz) < crossfrac
                    nbr_mutant_cross = np.where(nbr_mutant_cross_points, nbr, mutant)
                    cross_points = np.random.rand(grpsz) < crossp
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, grpsz)] = True
                    trial = np.where(cross_points, nbr_mutant_cross, pop[j])
                    f = self.fitness(trial, **fitnessParams)
                    if f < fitnessV[j]:
                        fitnessV[j] = f
                        pop[j] = trial
                        cross_ct += 1
                        if f < fitnessV[best_idx]:
                            best_idx = j
                            best = trial
                #yield best, fitness[best_idx]
                print('Iter ', iter, ':', best_idx, fitnessV[best_idx], cross_ct)
            new_inner_cohort = self.vec_to_df(best)
            self.mutual_info = fitnessV[best_idx]

        elif self.algorithm in ['fixed', 'd_e', 'gradient']:  # all of which use explicit guide func
            #mutator = FreshDrawMutator()
            #mutatorParams = {'df': new_outer_cohort}
            mutator = MSTMutator(new_outer_cohort)
            #mutator.plot_tree()
            mutatorParams = {'nsteps': 2, 'df': new_outer_cohort}
            if self.algorithm == 'fixed':
                print('------------------')
                assert 'seed' in self.opt_dct, 'no seed so I cannot find fixed wt_ser values'
                seed_pth, seed_root = os.path.split(self.opt_dct['seed'])
                pkl_path = os.path.join(seed_pth,
                                        'deep_' + seed_root + '_generic_guides',
                                        '%d-%d_generic.pkl' % (self.age, new_age))
                print('Opening %s' % pkl_path)
                with open(pkl_path, 'rb') as f:
                    stuff = pickle.load(f)
                    if isinstance(stuff, list):
                        stuff = stuff[0]
                    fixed_wts = stuff.x
                wt_ser = createWeightSer(all_col_l, {}, fixed_wts)
                print('Using fixed guide function:')
                print(wt_ser.values)
                print('------------------')
                bestWtSer = wt_ser
            elif self.algorithm in ['d_e', 'gradient']:
                nSamp = len(self.inner_cohort)
                nIter = 1000
                stepsizes = np.empty([nSamp])
                stepsizes.fill(0.005)
                testSampParams = {'df': self.inner_cohort}
                genSampParams = {'df': new_outer_cohort}
                binnerParams = {}
                if self.algorithm == 'd_e':
                    rslt = differential_evolution(minimizeMe, 
                                                  [(0.5*v, 4.0*v) for v in wt_ser.values],
                                                  (nSamp, nIter, all_col_l,
                                                   self.samp_gen,
                                                   testSampParams, genSampParams,
                                                   which_bin, binnerParams,
                                                   mutator, mutatorParams),
                                                  maxiter=10, popsize=10,
                                                  disp=True)
                else:  # gradient
                    rslt = minimize(minimizeMe, wt_ser.values.copy(),
                                    (nSamp, nIter, all_col_l,
                                     self.samp_gen,
                                     testSampParams, genSampParams,
                                     which_bin, binnerParams,
                                     mutator, mutatorParams),
                                    method='L-BFGS-B',
                                    bounds=[(0.5*v, 4.0*v) for v in wt_ser.values],
                                    options={'eps':0.01})
                print('------------------')
                print('Optimization result:')
                print(rslt)
                print('------------------')
                bestWtSer = createWeightSer(all_col_l, {}, rslt.x, niter=nIter)
    
#             lnLikParams = {'samps2V': self.inner_cohort, 'wtSerV': bestWtSer}
#             cleanSamps = genMetropolisSamples(nSamp, nIter, self.samp_gen(**genSampParams), 
#                                               lnLik, lnLikParams,
#                                               mutator, mutatorParams, verbose=True)
#             if isinstance(cleanSamps[0], pd.DataFrame):
#                 newCleanV = pd.concat(cleanSamps)
#             else:
#                 newCleanV = np.concatenate(cleanSamps)
#             new_inner_cohort = self.samp_gen(newCleanV)
            new_inner_cohort = self.gen_samples_using(self.inner_cohort,
                                                      new_outer_cohort,
                                                      mutator, mutatorParams,
                                                      bestWtSer)
            self.mutual_info =  -mutualInfo(new_inner_cohort, self.inner_cohort,
                                            self.binner, self.binnerParams)

        lst = get_unique_entries(new_inner_cohort)
        print('new inner cohort unique entries (%d): %s' % (len(lst), lst))
        print('------------------')

        self.age = new_age
        self.outer_cohort = new_outer_cohort
        self.inner_cohort = new_inner_cohort
        

    def gen_samples_using(self, target_cohort, pool_cohort,
                          mutator, mutator_params,
                           wt_ser, niter=1000):
            ln_lik_params = {'samps2V': target_cohort, 'wtSerV': wt_ser}
            gen_samp_params = {'df': pool_cohort}
            
            nsamp = len(target_cohort)
            cleanSamps = genMetropolisSamples(nsamp, niter, self.samp_gen(**gen_samp_params), 
                                              lnLik, ln_lik_params,
                                              mutator, mutator_params, verbose=True)
            if isinstance(cleanSamps[0], pd.DataFrame):
                newCleanV = pd.concat(cleanSamps)
            else:
                newCleanV = np.concatenate(cleanSamps)
            return self.samp_gen(newCleanV)
        

def createWeightedSamplesGenerator(n_samp):
    def rsltF(df):
        rslt = mkSamps(df, n_samp)
        if 'index' in rslt.columns:
            rslt = rslt.drop(columns=['index'])
        return rslt
    return rsltF


def createWeightSer(colL, range_d, vals=None):
    if vals is None:
        wtSer = pd.Series({col:1.0/(range_d[col] * range_d[col])
                           for col in colL})
    else:
        assert len(vals) == len(colL), 'values do not match col count'
        wtSer = pd.Series({col:v for col, v in zip(colL, vals)})
    #print('wtSer: %s' % wtSer.values)
    return wtSer


def get_rslt_path():
    pth = os.path.join(os.getcwd(), str(uuid.uuid1()))
    print('Result full path: ', pth)
    return pth


def load_dataset():
    fullDF = pd.read_csv('/home/welling/git/synecoace/data/nsch_2016_topical.csv',
                         encoding='utf-8')
    print(fullDF.columns)
    subDF, acesL, boolColL, scalarColL = transformData(selectData(fullDF.reset_index(),
                                                                  fipsL=FIPS_DCT.values(),
                                                                  includeFips=True))
    assert 'RECIDX' not in subDF.columns, 'RECIDX already exists?'
    subDF = subDF.reset_index().rename(columns={'index':'RECIDX'}).drop(columns=['level_0'])
    passiveL = ['RECIDX']
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
    passiveL += ['DRUGSALCOHOL', 'MENTALILL', 'PARENTDIED', 'PARENTDIVORCED',
                 'PARENTJAIL', 'RACISM', 'SEEPUNCH', 'VIOLENCE',
                 'SC_RACE_WHITE', 'TOTCSHCN', 'TOTACES']
    ageL = subDF['AGE'].unique()
    return subDF, acesL, boolColL, scalarColL, fixedL, passiveL, ageL


def main():
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
        seed_row = None
        seed_file = None

    parser.destroy()
    
    rslt_path = get_rslt_path()
    os.makedirs(rslt_path)

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

    weightedSampGen = createWeightedSamplesGenerator(1000)

    scSampGen = createWeightedSamplesGenerator(1)
    if seed_file is None:
        assert opts.alg != 'fixed', 'alg fixed requires a seed file from which to generate data dir name'
        opt_dct = {}
        if seed_row is None:
            df = subDF[subDF.AGE==ageMin]
            df = df[df.FIPSST == FIPS_DCT['SC']].drop(columns=['AGE', 'FIPSST'])
            df, _, _, _, dct = quantizeData(df, acesL, boolColL, scalarColL)
            assert dct == range_d, 'Quantized ranges do not match?'
            prototype = scSampGen(ageDFD[ageMin])
        else:
            df = subDF[subDF.AGE==ageMin].drop(columns=['AGE', 'FIPSST'])
            df = df[df.RECIDX == seed_row]
            assert df.shape[0] == 1, 'Seed matched %d samples' % df.count()
            df, _, _, _, dct = quantizeData(df, acesL, boolColL, scalarColL)
            assert dct == range_d, 'Quantized ranges do not match?'
            prototype = scSampGen(df)
    else:
        df = pd.read_pickle(seed_file)
        prototype = scSampGen(df)
        opt_dct = {'seed':os.path.splitext(seed_file)[0]}
        
    print('prototype columns: ', prototype.columns)

    agent = Agent(prototype, ageDFD[ageMin], weightedSampGen, fixedL, acesL, passiveL,
                  ageMin, range_d=range_d, algorithm=opts.alg, opt_dct=opt_dct)
    while agent.age < ageMax:
        new_age = agent.age + 1
        sub_path = os.path.join(rslt_path, '%d_%d' % (agent.age, new_age))
        os.makedirs(sub_path)
        agent.write_state(sub_path)
        agent.age_transition(ageDFD[new_age])


if __name__ == "__main__":
    main()

        