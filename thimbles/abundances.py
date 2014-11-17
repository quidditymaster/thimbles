import pandas as pd
from thimbles.ptable import ptable



class AbundanceVector(object):
    
    def __init__(self, abundances):
        if isinstance(abundances, (float, int)):
            abundances = solar_abundances + abundances
        self._data = pd.DataFrame(abundances)

#solar_abundances = AbundanceVector(ptable.logeps)

