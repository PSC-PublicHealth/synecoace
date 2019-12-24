#! /usr/bin/env python

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

from sampling import mkSamps

class Mutator(object):
    def go(self, mutateThisDF, **kwargs):
        return df

class FreshDrawMutator(Mutator):
    """
    This mutator returns a completely new set of records drawn from
    a pool provided in kwargs.
    """
    def apply(self, mutateThisDF, **kwargs):
        """
        Return a 'mutated' version of sampV, based on the given step sizes.  Unfortunately our samples
        are discrete and come from a table, so I'm not sure how to do this unless we first generate
        a proximity network of some sort, so for the moment let's just generate a new set of samples-
        this corresponds to an infinitely wide mutator.
        """
        df = kwargs['df']
        stepSzV = kwargs['stepSzV']
        return mkSamps(df, len(mutateThisDF))

class MSTMutator(Mutator):
    def __init__(self, df):
        idx_df = df.reset_index().drop_duplicates()
        nrows = len(idx_df)
        mtx = np.zeros((nrows, nrows))
        offset_d = {offset : row['index'] for offset, (idx, row) in enumerate(idx_df[['index']].iterrows())}
        idx_d = {v : k for k, v in offset_d.items()}
        idx_df.index = idx_df['index']
        unique_df = idx_df.drop(columns=['index'])
        #G = nx.Graph()
        for off0 in range(nrows):
            row0 = unique_df.loc[offset_d[off0]]
            v0 = row0.values
            for off1 in range(nrows):
                row1 = unique_df.loc[offset_d[off1]]
                v1 = row1.values
                wt = sum(np.abs(v1 - v0))
                if off0 != off1:
                    wt += 1
                mtx[off0, off1] = wt
                #G.add_edge(off0, off1, weight=wt)
        sp_mtx = minimum_spanning_tree(mtx)
        #MT = nx.minimum_spanning_tree(G)
        sp_mtx = sp_mtx.todense()
        def mutate_fun(idx, ssm, idx_d, offset_d):
            try:
                row_off = idx_d[idx]
            except Exception as e:
                import pdb
                pdb.Pdb().set_trace()
            row = ssm[row_off, :]
            candidates = row.nonzero()[1]
            offset = np.random.choice(candidates)
            return offset_d[offset]
        self.df = df
        self.ssm = sp_mtx + sp_mtx.transpose()  # symmetrize
        self.offset_d = offset_d
        self.idx_d = idx_d
        self.mutate_vec = np.vectorize(mutate_fun,
                                       excluded=['ssm', 'idx_d', 'offset_d'])    
    
    def apply(self, mutateThisDF, **kwargs):
        """
        Return a 'mutated' version of sampV, based on the given step sizes.  Unfortunately our samples
        are discrete and come from a table, so I'm not sure how to do this unless we first generate
        a proximity network of some sort, so for the moment let's just generate a new set of samples-
        this corresponds to an infinitely wide mutator.
        """
        df = kwargs['df']
        assert df is self.df, 'DataFrame is not the one from which this instance was built'
        print('input follows')
        print(mutateThisDF)
        print('df follows')
        print(df)
        print('inner df follows')
        print(self.df)
        idxV = self.mutate_vec(mutateThisDF.index.values,
                               ssm=self.ssm, idx_d=self.idx_d, offset_d=self.offset_d)
        print('idxV follows: ', idxV.shape, idxV.dtype)
        print(idxV)
        try:
            src = df.copy().reset_index().drop_duplicates()
            src = src.set_index('index')
            rslt = src.loc[idxV]
        except Exception as e:
            import pdb
            pdb.Pdb().set_trace()
            raise
        print('rslt follows')
        print(rslt)
        return rslt
    
    def plot_tree(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.to_networkx_graph(self.ssm)
        plt.figure()
        nx.draw_networkx(G)
        plt.show()
        

    
    