import unittest
import numpy as np
import matplotlib.pyplot as plt
from thimbles.utils.piecewise_polynomial import PiecewisePolynomial
from thimbles.utils import partitioning
from thimbles.modeling.modeling import parameter
from thimbles.modeling.modeling import Model, DataRootedModelTree, DataModelNetwork

from thimbles.sqlaimports import *
import scipy.sparse

class LineModel(Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    
    def __init__(self, slope, intercept):
        super(LineModel, self).__init__()
        self.slope = slope
        self.intercept = intercept
    
    def __call__(self, input, x):
        return x*self.slope + self.intercept
    
    @parameter(free=True)
    def slope_p(self):
        return self.slope
    
    @slope_p.setter
    def set_slope(self, value):
        self.slope = value
    
    @parameter(free=True)
    def intercept_p(self):
        return self.intercept
    
    @intercept_p.setter
    def set_intercept(self, value):
        self.intercept = value    

class PiecewiseConstantModel:#(Model):
    
    def __init__(self, constants, break_pts):
        self.ppol = PiecewisePolynomial(self.constants, self._break_pts)
    
    @parameter(free=True)
    def constants_p(self):
        return self.ppol.constants
    
    @constants_p.setter
    def set_constants(self, constants, break_pts=None):
        constants = np.asarray(constants)
        self.ppol.coefficients = constants.reshape((-1, 1))
        if not break_pts is None:
            self.ppol.set_break_pts(break_pts) 
    
    def __call__(self, input_vec):
        return self.ppol(input_vec)

#class PiecewiseConstantTest(unittest.TestCase):
#    
#    def setUp(self):
#        self.npts = 200
#        self.noise_level = 0.01
#        self.brkpts = [0.5, 0.75, 0.85]
#        self.values = np.array([1.0, 0.5, -1.0, 1.5])
#        self.x = np.linspace(0, 1, self.npts)
#        self.ppol = PiecewisePolynomial(self.values.reshape((-1, 1)), self.brkpt#s)
#        self.y = self.ppol(self.x)
#        self.y_noisy = self.y + np.random.normal(size=self.y.shape)
#    
#    def test_parameterize(self):
#        pass


class LineFitTest(unittest.TestCase):

    def setUp(self):
        npts = 200
        x = np.linspace(-1, 1, npts)
        self.x = x
        slope = 3.1415926535
        self.slope=slope
        intercept = -2.71828
        self.intercept = intercept
        noise_level = 0.55
        noise = np.random.normal(size=(npts,))*noise_level
        self.y_var = np.ones(npts, dtype=float)*noise_level**2
        y = slope*x + intercept + noise
        self.y = y
        
        start_slope = 0.0
        start_offset = 0.0
        self.lmod = LineModel(start_slope, start_offset)
        tree = DataRootedModelTree([self.lmod], target_data=y, data_weight=1.0/self.y_var, kw_inputs=[{"x":x}])
        self.model_net = DataModelNetwork([tree])
        
        self.slope = slope
        self.intercept = intercept
    
    def test_line_fit(self):
        #import pdb; pdb.set_trace()
        self.model_net.converge()
        #print "slope {} , intercept {}".format(self.lmod.slope, self.lmod.intercept)
        self.assertAlmostEqual(self.lmod.slope, self.slope, delta=0.1)
        self.assertAlmostEqual(self.lmod.intercept, self.intercept, delta=0.1)


class TestParameterize(unittest.TestCase):
    
    def setUp(self):
        pass

if __name__ == "__main__":
    unittest.main()
