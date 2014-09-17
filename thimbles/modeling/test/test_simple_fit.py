import unittest
import numpy as np
import matplotlib.pyplot as plt
from thimbles.modeling.modeling import parameter
from thimbles.modeling.modeling import Model, DataRootedModelTree, DataModelNetwork
import scipy.sparse

class LineModel(Model):
    
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
