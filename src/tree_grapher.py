#! /usr/bin/env python

import sys
import os
import uuid
import yaml
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from data_conversions import selectData, transformData
from data_conversions import floatifyData, boolifyData, quantizeData
from sampling import mkSamps, toHisto, toProbV
from sampling import createSparseBinner as createBinner
from metropolis import mutualInfo, genMetropolisSamples
from mutator import FreshDrawMutator, MSTMutator
from agent_based_model import load_dataset, select_subset



class Agent(object):
    def __init__(self, prototype, all_samples, samp_gen, fixed_l, aces_l, passive_l,
                 age, range_d = None):
        fixed_d = {elt: prototype.iloc[0][elt] for elt in fixed_l}
        self.fixed_d = fixed_d
        self.samp_gen = samp_gen
        self.age = age # whatever is in the prototype
        self.range_d = range_d
        self.passive_l = passive_l
        self.inner_cohort = samp_gen(prototype)
        self.outer_cohort = samp_gen(select_subset(all_samples, fixed_d))
        advancing_l = [elt for elt in aces_l
                       if elt not in list(self.fixed_d) + passive_l]
        self.advancing_l = advancing_l
        open_l = [k for k in self.outer_cohort.columns]
        for elt in list(fixed_d) + advancing_l + passive_l:
            open_l.remove(elt)
        self.open_l = open_l
        print('KEYS: ', prototype.index)
        print('initial outer_cohort unique entries: ', self.outer_cohort.index.unique())
        # set the fixed keys according to the prototype
        # the inner cohort is the prototype
        # the outer cohort is all samples with the same fixed keys
    
    def write_state(self, path):
        dct = {'fixed_d': {k : int(v) for k, v in self.fixed_d.items()},
               'age': self.age,
               'range_d': {k : int(v) for k, v in self.range_d.items()},
               'passive_l': self.passive_l,
               'advancing_l': self.advancing_l,
               'open_l': self.open_l,
               'outer_cohort_unique_entries': [int(v) for v in self.outer_cohort.drop_duplicates()['RECIDX']],
               'inner_cohort_unique_entries': [int(v) for v in self.inner_cohort.drop_duplicates()['RECIDX']],
               }
        with open(os.path.join(path, 'state.yml'), 'w') as f:
            yaml.dump(dct, f)
        self.inner_cohort.to_pickle(os.path.join(path, 'inner_cohort.pkl'))
    
    def age_transition(self, new_all_samples):
        # each fixed key must get updated according to its own rule
        # - gender, birthorder, etc. stay fixed
        # the outer cohort is all samples at the new age with the same fixed keys
        print('new_all_samples unique entries: ', new_all_samples.index.unique())
        all_col_l = self.open_l + self.advancing_l
        which_bin = createBinner(all_col_l, range_d=self.range_d)
        wt_ser = createWeightSer(all_col_l, range_d=self.range_d)
        samples_subset = select_subset(new_all_samples, self.fixed_d)
        print('samples_subset unique entries: ', new_all_samples.index.unique())
        new_outer_cohort = self.samp_gen(samples_subset)        
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
        #mutator = FreshDrawMutator()
        mutator = MSTMutator(new_outer_cohort)
        mutator.plot_tree()
        mutatorParams = {'nsteps': 2, 'df': new_outer_cohort}
    
        rslt = minimize(minimizeMe, wt_ser.values.copy(),
                        (nSamp, nIter, all_col_l,
                         self.samp_gen,
                         testSampParams, genSampParams,
                         which_bin, binnerParams,
                         mutator, mutatorParams),
                        method='L-BFGS-B',
                        bounds=[(0.25*v, 4.0*v) for v in wt_ser.values])
        print('------------------')
        print('Optimization result:')
        print(rslt)
        print('------------------')
 
        bestWtSer = createWeightSer(all_col_l, {}, rslt.x)
        lnLikParams = {'samps2V': self.inner_cohort, 'wtSerV': bestWtSer}
        cleanSamps = genMetropolisSamples(nSamp, nIter, self.samp_gen(**genSampParams), 
                                          lnLik, lnLikParams,
                                          mutator, mutatorParams, verbose=True)
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
        

def main(argv):
    rslt_path = argv[1]
    print(rslt_path)
    subDF, acesL, boolColL, scalarColL, fixedL, passiveL, ageL = load_dataset()

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

    state_dct = {}
    inner_cohort_dct = {}
    outer_cohort_dct = {}
    for age in range(ageMin, ageMax+1):
        path = os.path.join(rslt_path, '%d_%d' % (age, age+1))
        if os.path.exists(path) and os.path.isdir(path):
            with open(os.path.join(path, 'state.yml')) as f:
                state_dct[age] = yaml.safe_load(f)
            inner_cohort_dct[age] = pd.read_pickle(os.path.join(path,
                                                                'inner_cohort.pkl'))
            outer_cohort_dct[age] = select_subset(ageDFD[age],
                                                  state_dct[age]['fixed_d'])
            print(state_dct[age])
        else:
            break
    
    for age in range(ageMin, ageMax+1):
        if age in state_dct:
            mutator = MSTMutator(outer_cohort_dct[age])
            mutator.plot_tree(weights=inner_cohort_dct[age],
                              title="Age {}".format(age))

if __name__ == "__main__":
    main(sys.argv)

        