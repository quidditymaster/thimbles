import numpy as np
from thimbles.binning import CoordinateBinning
from thimbles.utils.partitioning import 
from modeling import Model, parameter

class PartitionedMatrixModel(Model):
    
    def __init__(self, matrix_factory, ordinal_function, min_delta=0.0):
        self.matrix_factory = matrix_factory
        self.ordinal_function = ordinal_function
        
        #with no information start with just one bin
        self.coefficients = None
    
    @parameter(free=True)
    def coefficients_p(self):
        return self.coefficients
    
    @coefficients_p.setter
    def set_coefficients(self, value):
        self.coefficients = value
    
    @coefficients_p.parameterizer
    def optimize_partition(self, fit_state):
        pass