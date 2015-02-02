import unittest
import thimbles as tmb
from thimbles.pseudonorms import *
import numpy as np
import matplotlib.pyplot as plt

class DummySpectrum(object):
    
    def __init__(self, flux):
        self.flux = flux

    def __len__(self):
        return len(self.flux)

class TestSortNorm(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_true_normal(self):
        npts = 50
        y_mean = 1.0
        psnorm_vals = []
        for i in range(200):
            y = np.random.normal(size=(npts,))*0.1+y_mean
            spec = DummySpectrum(y)
            snorm = sorting_norm(spec)
            psnorm_vals.append(snorm[0])
        #import matplotlib.pyplot as plt
        #plt.hist(psnorm_vals, 71)
        #plt.show()
        self.assertTrue(y_mean - 0.1 < np.mean(psnorm_vals) < y_mean + 0.1)

if __name__ == "__main__":
    unittest.main()
