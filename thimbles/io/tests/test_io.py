
# Standard Library
import unittest
import os

# 3rd Party
import numpy as np
from astropy.io import fits

# Internal
import thimbles
from thimbles.io import ExtractWavelengthCoefficients

resources_path = os.path.join(os.path.dirname(thimbles.__file__),"resources/test_data")

# ########################################################################### #

class TestWavelengthFunctions (unittest.TestCase):
    
    
    def setUp(self):
        unittest.TestCase.setUp(self)

class TestExtractWavlengthCoefficients (unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        
    def test_hd221170_read (self):
        filename = "hd221170_3500sm_header.fits"
        hdu = fits.open(os.path.join(resources_path,filename))[0]
        ewc = ExtractWavelengthCoefficients(hdu.header)
           
        spectre_history = ewc.get_SPECTRE_history()
        
        self.assert_(len(spectre_history)==1, "Found too many different tags")
        
        key = spectre_history.keys()[0]
        self.assert_(key==1121065200.0, "Found wrong time stamp")
        
        extra_info,func_type,coeffs = spectre_history[key]
        self.assertEqual(func_type,'chebyshev poly', "wrong spectre function type")
        
        solution = np.array([3497.88543141, 30.9881846801, -0.666763321767, -0.0528877896848, 0.000482608059658, 0.0])
        self.assert_(np.all(np.array(coeffs)==solution),"Wrong coefficients found")

        
        
        
        
        
        
        
        
