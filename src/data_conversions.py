#! /usr/bin/env python

import numpy as np
import pandas as pd

def selectData(fullDF,fipsL=None, includeFips=False):
    if fipsL is None:
        fipsL = [45]

#     subDF=fullDF[['ACE1', 'ACE3', 'ACE4', 'ACE5', 'ACE6', 'ACE7', 'ACE8', 'ACE9', 'ACE10', 'FWC', 'index',
#                   'YEAR', 'FPL', 'SC_AGE_YEARS','K4Q32X01', 'K7Q30', 'K7Q31', 'AGEPOS4']].copy()
    keyL = ['ACE3', 'ACE4', 'ACE5', 'ACE6', 'ACE7',
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
            'SC_SEX', # child sex (1=male, 2=female)
            'index'
            ]
    if includeFips:
        keyL.append('FIPSST')
    subDF=fullDF[fullDF['FIPSST'].isin(fipsL)][keyL]
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

    copy_l = ['FWC', 'AGE', 'FPL', 'index']
    if 'FIPSST' in data.columns:
        copy_l.append('FIPSST')
    massaged_data = data[copy_l].copy()
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
    massaged_data['SC_FEMALE'] = (data['SC_SEX'] == 1.0)
    
    
    acesL = ['PARENTDIVORCED', 'PARENTDIED', 'PARENTJAIL', 'SEEPUNCH', 'VIOLENCE', 'MENTALILL',
             'DRUGSALCOHOL', 'RACISM']
    acesL.sort()
    for ace in acesL:
        massaged_data[ace] = (data[ace] == 1)
    massaged_data['TOTACES'] = massaged_data[acesL].astype(int).sum()
    
    boolColL = ['DENTALCARE', 'VISIONCARE', 'SPORTSTEAMS', 'CLUBS', 'SC_CSHCN',
                'BIRTHWT_VL', 'BIRTHWT_L', 'HHLANGUAGE_ENGLISH', 'HHLANGUAGE_SPANISH',
                'DOCTORVISIT', 'PREMATURE',
                'SC_RACE_WHITE', 'SC_RACE_BLACK', 'SC_RACE_HISPANIC', 'SC_RACE_NATIVE', 'SC_RACE_ASIAN',
                'SC_RACE_ISLANDS', 'SC_RACE_OTHER', 'SC_RACE_MIXED', 'SC_FEMALE']
    boolColL.sort()

    scalarColL = ['FPL', 'BIRTHORDER', 'AGE', 'TOTCSHCN', 'TOTKIDS', 'MOMAGE', 'TOTACES']
    scalarColL.sort()
    
#     print('Before transformation:')
#     print data.head()
#     print('After transformation:')
#     print massaged_data.head()

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
    drop_these = []
    for col in convertS:
        if col == 'MOMAGE':  # special case
            out_data['MOMAGE_LT_20'] = (data[col] < 20)
            out_data['MOMAGE_GT_39'] = (data[col] > 39)
            boolColL = boolColL[:] + ['MOMAGE_LT_20', 'MOMAGE_GT_39']
        elif col == 'FPL':  # special case
            intervals = pd.interval_range(start=0, periods=8, closed='left')
            bin_map = {val:idx for idx, val in enumerate(intervals)}
            x = pd.cut(0.01 * data['FPL'], intervals, labels=False)
            out_data['FPL_quantized'] = x.apply(lambda y: bin_map[y])
            if fpl_mode == 'binned':
                cat_list = pd.get_dummies(out_data['FPL_quantized'],
                                          prefix='FPL', drop_first=True)
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

        elif col == 'TOTCSHCN':
            intervals = pd.interval_range(start=0, periods=6, closed='left')
            bin_map = {val:idx for idx, val in enumerate(intervals)}
            x = pd.cut(data[col], intervals, labels=False)
            out_data[col + '_quantized'] = x.apply(lambda y: bin_map[y])
            cat_list = pd.get_dummies(out_data[col + '_quantized'],
                                      prefix=col, drop_first=True)
            data1 = out_data.join(cat_list)
            out_data = data1
            drop_these += [col + '_quantized']

        else:
            # All of these deal with number of kids, so minimum is 1
            intervals = pd.interval_range(start=1, periods=5, closed='left')
            bin_map = {val : idx + 1 for idx, val in enumerate(intervals)}
            x = pd.cut(data[col], intervals, labels=False)
            out_data[col + '_quantized'] = x.apply(lambda y: bin_map[y])
            cat_list = pd.get_dummies(out_data[col + '_quantized'],
                                      prefix=col, drop_first=True)
            data1 = out_data.join(cat_list)
            out_data = data1
            drop_these += [col + '_quantized']

    out_data1 = out_data.drop(columns=drop_these)
    out_data = out_data1

    return out_data, acesL, boolColL, scalarColL


def quantizeData(data, acesL, boolColL, scalarColL):
    """
    Things in acesL and boolColL are already boolean.  Must convert the values in scalarColL.
    """
    range_d = {}
    colS = frozenset(acesL + boolColL + scalarColL)
    rslt = data[[col for col in data.columns if col not in colS]].copy()
    for col in acesL + boolColL:
        rslt[col] = data[col].astype(int)
        range_d[col] = 2

    for col in scalarColL:
        if col == 'MOMAGE':  # special case
            rslt['MOMAGE_LT_20'] = (data[col] < 20).astype(int)
            rslt['MOMAGE_GT_39'] = (data[col] > 39).astype(int)
            boolColL = boolColL[:] + ['MOMAGE_LT_20', 'MOMAGE_GT_39']
            range_d['MOMAGE_LT_20'] = 2
            range_d['MOMAGE_GT_39'] = 2
            
        elif col == 'FPL':  # special case
            intervals = pd.interval_range(start=0, periods=8, closed='left')
            bin_map = {val:idx for idx, val in enumerate(intervals)}
            x = pd.cut(0.01 * data['FPL'], intervals, labels=False)
            rslt['FPL'] = x.apply(lambda y: bin_map[y] )
            range_d['FPL'] = 8
 
        elif col == 'TOTCSHCN':
            intervals = pd.interval_range(start=0, periods=6, closed='left')
            bin_map = {val:idx for idx, val in enumerate(intervals)}
            x = pd.cut(data[col], intervals, labels=False)
            rslt[col] = x.apply(lambda y: bin_map[y])
            range_d[col] = 6
 
        elif col == 'TOTACES':
            rslt[col] = data[col]
            range_d[col] = 9
 
        else:
            # All of these deal with number of kids, so minimum is 1
            intervals = pd.interval_range(start=1, periods=5, closed='left')
            bin_map = {val : idx for idx, val in enumerate(intervals)}
            x = pd.cut(data[col], intervals, labels=False)
            rslt[col + '_minusone'] = x.apply(lambda y: bin_map[y])
            range_d[col + '_minusone'] = 5

    return rslt, acesL, boolColL, scalarColL, range_d


