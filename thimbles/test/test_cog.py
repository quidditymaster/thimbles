import h5py
import numpy as np
import thimbles as tmb
from thimbles.utils  import piecewise_polynomial
import matplotlib.pyplot as plt
import unittest

tpath = tmb.__path__[0]

class invertible_cog_tester(unittest.TestCase):
    
    def setUp(self):
        cog_ppol_hf = h5py.File("%s/resources/cog_ppol.h5" % tpath)
        coeffs, knots, centers, scales = np.array(cog_ppol_hf["coefficients"]), np.array(cog_ppol_hf["knots"]), np.array(cog_ppol_hf["centers"]), np.array(cog_ppol_hf["scales"])
        self.iqp = piecewise_polynomial.InvertiblePiecewiseQuadratic(coeffs, knots, centers=centers, scales=scales)

    def test_cog_inversion(self):
        x = np.linspace(-10, -2, 500)
        y = self.iqp(x) 
        y_inv = self.iqp.inverse(y)
        dev_sum = np.sum(np.abs(y_inv-x))
        self.assertTrue(dev_sum < 1e-5)

if __name__ == "__main__":
    unittest.main()
