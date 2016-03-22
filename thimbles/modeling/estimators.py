from .derivatives import deriv

from thimbles.sqlaimports import *
from thimbles.modeling.distributions import NormalDistribution


class Estimator(object):
    
    def __init__(self, informants, informed, set_value=True):
        self.informants = informants
        self.informed = informed
    
    def run(self):
        raise NotImplementedError("use a subclass")


class IterativeDecoupledLinearEstimator(object):
    
    def __init__(
            self,
            informants,
            informed,
    ):
        self.informants = informants
        self.informed = informed
    
    def __call__(self):
        #find the informant distributions and parameters
        
        #build the derivative matrix
        
        dmat = modeling.deriv(self.distribution_informants)
        

