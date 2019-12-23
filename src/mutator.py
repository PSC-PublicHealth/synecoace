#! /usr/bin/env python

import numpy as np
import pandas as pd

from sampling import mkSamps

class Mutator(object):
    def go(self, mutateThisDF, **kwargs):
        return df

class FreshDrawMutator(Mutator):
    """
    This mutator returns a completely new set of records drawn from
    a pool provided in kwargs.
    """
    def go(self, mutateThisDF, **kwargs):
        """
        Return a 'mutated' version of sampV, based on the given step sizes.  Unfortunately our samples
        are discrete and come from a table, so I'm not sure how to do this unless we first generate
        a proximity network of some sort, so for the moment let's just generate a new set of samples-
        this corresponds to an infinitely wide mutator.
        """
        df = kwargs['df']
        stepSzV = kwargs['stepSzV']
        return mkSamps(df, len(mutateThisDF))

    