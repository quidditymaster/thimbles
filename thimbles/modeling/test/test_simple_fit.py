import unittest
import numpy as np
import matplotlib.pyplot as plt
from ..models import Model, ModelSpace
import scipy.sparse as sp

class LineFitTest(unittest.TestCase):

    def setUp(self):
        npts = 100
        x = np.linspace(-1, 1, npts)
        slope = 1.5
        intercept = -2.0
        noise_level = 1.5
        noise = np.random.normal(size=(npts,))*noise_level
        self.y_var = noise_level**2
        y = slope*x + intercept + noise
        self.x = x
        self.y = y
        self.slope = slope
        self.intercept = intercept
    
    def test_line_fit(self):
        slope_name = "slope"
        intercept_name = "intercept"
        x_name = "x"
        y_name = "y"
        
        def get_x():
            return self.x
        
        def get_y():
            return self.y
        
        def get_y_var():
            return self.y_var
        
        def line_function(**kwargs):
            x = kwargs[x_name]
            slope, intercept = kwargs[slope_name], kwargs[intercept_name]
            return slope*x + intercept
        
        data_model = Model(outputs={y_name:get_y}, map_var={y_name:get_y_var})
        
        #make the model to be fit
        line_model = Model(inputs={x_name:get_x, slope_name:Model()},
                   outputs={y_name:line_function})
        
        #build a model space for them and put the base parameters in it
        msp = ModelSpace()
        msp.add_global_values({x_name:self.x, slope_name:1.0, intercept_name:0.0})
        
        #add the models to the space
        msp.add_models([data_model, line_model])
        
        #build a concordance between the parameterized model and the data
        msp.add_concordance(line_model, data_model) #should take arbitrary positional args
        
        #fill in the values in the space
        msp.execute()
        
        msp.fit_iteration()
        

if __name__ == "__main__":
    unittest.main()
