import numpy as np
from spectrum import WavelengthSolution

class BiSpectrumOperationPolicy(object)
    
    def __init__(self, binning_p, 
                 left_valuation_p, 
                 right_valuation_p):
        self.bining_p = binning_p
        self.left_valuation_p = left_valuation_p
        self.right_valuation_p = right_valuation_p

class StandardBinningPolicy(object):
    
    def __init__(self, bound_type="intersection", solution_type="log-linear", sampling_ratio=2.0):
        self.oversample=oversample
    
    def __call__(self, spec1, spec2):
        wv1 = spec1.wv
        wv2 = spec2.wv
        
        edge1 = [wv1[0], wv1[-1]]
        edge2 = [wv2[0], wv2[-1]]
        min1, max1 = np.min(edge1), np.max(edge1)
        min2, max2 = np.min(edge2), np.max(edge2)
        
        if bound_type == "intersection":
            out_min = np.max([min1, min2])
            out_max = np.min([max1, max2])
        if bound_type == "union":
            out_min = np.min([min1, min2])
            out_max = np.max([max1, max2])
        
        n1captured = spec1.bounding_indexes(
        n2captured = spec2.bounding_ndexes
