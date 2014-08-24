import unittest
import numpy as np
import matplotlib.pyplot as plt
from thimbles.modeling.modeling import Model, ModelChain, Modeler
import scipy.sparse

class LineModel(Model):
    
    def __init__(self, slope, offset):
        self.slope = slope
        self.offset = offset
    
    def __call__(self, input, x):
        return x*self.slope + self.offset
    
    def as_linear_op(self, input, x):
        return None
    
    def parameter_damping(self, input, x):
        return np.zeros(2), 10.0*np.ones(2)

    def parameter_expansion(self, input, x):
        x = np.asarray(x)
        mat = np.array([x, np.ones(len(x))]).transpose()
        return scipy.sparse.csr_matrix(mat)
    
    def set_pvec(self, pvec):
        slope, offset = pvec
        self.slope = slope
        self.offset=offset
    
    def get_pvec(self):
        return np.array([self.slope, self.offset])

class LineFitTest(unittest.TestCase):

    def setUp(self):
        npts = 200
        x = np.linspace(-1, 1, npts)
        self.x = x
        slope = 3.1415926535
        self.slope=slope
        intercept = -2.71828
        self.offset = intercept
        noise_level = 0.1
        noise = np.random.normal(size=(npts,))*noise_level
        self.y_var = np.ones(npts, dtype=float)*noise_level**2
        y = slope*x + intercept + noise
        self.y = y
        
        start_slope = 0.0
        start_offset = 0.0
        self.lmod = LineModel(start_slope, start_offset)
        chain = ModelChain([self.lmod], target_data=y, target_inv_covar=1.0/self.y_var, kw_inputs=[{"x":x}])
        
        self.modeler = Modeler()
        self.modeler.add_chain(chain)
        
        self.slope = slope
        self.intercept = intercept
    
    def test_line_fit(self):
        #import pdb; pdb.set_trace()
        for i in range(5):
            self.modeler.iterate(self.lmod)
            #print "slope {} , offset {}".format(self.lmod.slope, self.lmod.offset)
        self.assertAlmostEqual(self.lmod.slope, self.slope, delta=0.1)
        self.assertAlmostEqual(self.lmod.offset, self.offset, delta=0.1)
        

if __name__ == "__main__":
    unittest.main()
