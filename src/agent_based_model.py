#! /usr/bin/env python

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

from data_conversions import selectData, transformData, floatifyData, boolifyData
from sampling import mkSamps, toHisto, toProbV, createBinner
from metropolis import mutualInfo


def select_subset(df, match_dct):
    df = df.copy()
    print('begin select_subset: %d records, %d unique'
          % (int(df.shape[0]), len(df.index.unique())))
    for k, v in match_dct.items():
        df = df[df[k] == v]
        print('%s == %s: %d entries, %d unique'
              % (k, v, int(df.shape[0]), len(df.index.unique())))
    return df


class Agent(object):
    def __init__(self, prototype, all_samples, fixed_l, aces_l, age):
        fixed_d = {elt: prototype.iloc[0][elt] for elt in fixed_l}
        advancing_l = []
        for elt in aces_l:
            if prototype.iloc[0][elt]:
                fixed_d[elt] = True
            else:
                advancing_l.append(elt)
        open_l = [k for k in all_samples.columns]
        for elt in list(fixed_d) + advancing_l:
            open_l.remove(elt)
        self.fixed_keys = fixed_d 
        self.advancing_l = advancing_l
        self.open_l = open_l
        self.inner_cohort = prototype
        self.outer_cohort = select_subset(all_samples, self.fixed_keys)
        print('outer_cohort unique entries: ', self.outer_cohort.index.unique())
        self.age = age # whatever is in the prototype
        # set the fixed keys according to the prototype
        # the inner cohort is the prototype
        # the outer cohort is all samples with the same fixed keys
    
    def age_transition(self, new_outer_cohort):
        # each fixed key must get updated according to its own rule
        # - gender, birthorder, etc. stay fixed
        # the outer cohort is all samples at the new age with the same fixed keys
        which_bin = createBinner(self.open_l + self.advancing_l)
        print('inner cohort keys: ', self.inner_cohort.columns)
        print('outer cohort keys: ', self.outer_cohort.columns)
        print('old inner vs. outer: ', mutualInfo(self.inner_cohort, self.outer_cohort,
                                                  which_bin))
        self.age += 1
        

def createWeightedSamplesGenerator(full_df_dct, n_samp):
    def rsltF(age):
        return mkSamps(full_df_dct[age], n_samp).drop(columns=['index'])
    return rsltF


def main():
    fullDF = pd.read_csv('/home/welling/git/synecoace/data/nsch_2016_topical.csv',
                         encoding='utf-8')
    print(fullDF.columns)
    fipsD = {'SC': 45, 'NC':37, 'TN':47, 'GA':13,
             'AL':1, 'VA':51, 'LA':22, 'AR':5}
    subDF, acesL, boolColL, scalarColL = transformData(selectData(fullDF.reset_index(),
                                                                  fipsL=fipsD.values(),
                                                                  includeFips=True))
    #print(scalarColL)
    ageL = subDF['AGE'].unique()
    print('ages: ', ageL)
    ageMin = int(min(ageL))
    ageMax = int(max(ageL))
    scalarColL.remove('AGE')
    print('scalar columns: ', scalarColL)
    ageDFD = {}
    for age in range(ageMin, ageMax+1):
        ageDF = subDF[subDF.AGE==age].drop(columns=['AGE', 'FIPSST'])
        ageDFD[age] = boolifyData(ageDF, acesL, boolColL, scalarColL)[0]
    ageDFD[7].head()

    weightedSampGen = createWeightedSamplesGenerator(ageDFD, 100000)

    df = subDF[subDF.AGE==ageMin].drop(columns='AGE')
    df = df[df.FIPSST == fipsD['SC']].drop(columns=['FIPSST'])
    df = boolifyData(df, acesL, boolColL, scalarColL)[0]
    scSampGen = createWeightedSamplesGenerator({ageMin:df}, 1)
    prototype = scSampGen(ageMin)
    print('prototype columns: ', prototype.columns)

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

    agent = Agent(prototype, weightedSampGen(ageMin), fixedL, acesL, ageMin)
    while agent.age < ageMax:
        new_age = agent.age + 1
        agent.age_transition(weightedSampGen(new_age))
        break

#     wrkSamps = mkSamps(ageDFD[age], 100000).drop(columns=['index'])
#     #print(wrkSamps.columns)
#     print('-------- age {}: {} samples ----------'.format(age, len(ageDFD[age])))
#     for col in acesL:
#         print(col, ' : ', float(wrkSamps[col].sum())/float(len(wrkSamps)))

if __name__ == "__main__":
    main()

        