import unittest
import numpy as np
try:
    import ipdb as pdb
except ImportError:
    import pdb
from thimbles.spectrum import LineSpreadFunction,GaussianLSF, BoxLSF

# ########################################################################### #

# TODO: add more tests
# np.testing.assert_array_almost_equal_nulp(x, y, nulp)


class TestSpectrum (unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
       
    def test_reference_frame (self):
        pass 
    
    def test_wv_solution (self):
        pass
    
    def test_spectrum (self):
        pass
    
    def test_continuum (self):
        pass
    
class TestLSF (unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        np.random.seed(5)
        self.dmsg = "({},{})"
                
    def test_gaussian_lsf (self):
        kws = dict(widths = np.random.normal(10,5,10),
                   max_sigma = 5,
                   wv_soln = None)
        glsf = GaussianLSF(**kws)
        
        value = glsf.get_integral(0,5)
        correct = 0.65895878128927521
        msg = self.dmsg.format(value,correct)+ " incorrect integral output"
        self.assertAlmostEqual(value,correct,8,msg)
        
        correct = (-108.7692796751945, 112.7692796751945)
        value = glsf.get_coordinate_density_range(2)
        msg = self.dmsg.format(value,correct)+" get_coordinate_density"
        self.assertEqual(value, correct,msg)
        
        correct = 22.153855935038898
        value = glsf.get_rms_width(2)
        msg = self.dmsg.format(value,correct)+" get_rms_width"
        self.assertEqual(value, correct,msg)
        
    def test_box_lsf (self):
        blsf = BoxLSF(None)
        
        values = [(blsf.get_integral(5,10),1),
                  (blsf.get_integral(5,2),0),
                  (blsf.get_integral(5,5.25),0.75)]
        
        for first,second in values:
            msg = self.dmsg.format(first,second)+" wrong get_integral"
            self.assertEqual(first, second, msg)
        
        
        # TODO:  add more tests
        
if __name__ == "__main__":
    unittest.main()