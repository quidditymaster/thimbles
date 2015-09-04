import unittest
import thimbles as tmb
from thimbles.pseudonorms import *
import numpy as np
import matplotlib.pyplot as plt

show_diagnostics = False

class DummySpectrum(object):
    
    def __init__(self, flux):
        self.flux = flux

    def __len__(self):
        return len(self.flux)

class TestSortNorm(unittest.TestCase):
    bias_tolerance = 0.05

    def setUp(self):
        pass
    
    def test_true_normal(self):
        npts = 50
        y_mean = 1.0
        psnorm_vals = []
        y_accum = []
        snorm_accum = []
        for i in range(200):
            y = np.random.normal(size=(npts,))*0.1+y_mean
            y_accum.append(y)
            spec = DummySpectrum(y)
            snorm = sorting_norm(spec)
            snorm_accum.append(snorm)
            psnorm_vals.append(np.mean(snorm))
        if show_diagnostics:
            fig, ax = plt.subplots()
            for i in range(len(y_accum)):
                plt.plot(y_accum[i], c="blue", alpha=0.3)
                plt.plot(snorm_accum[i], c="orange", alpha=0.5)
            fig, ax = plt.subplots()
            plt.hist(np.array(psnorm_vals)-y_mean, 21)
            plt.xlabel("norm bias")
            plt.show()
        self.assertTrue(y_mean - self.bias_tolerance < np.mean(psnorm_vals) < y_mean + self.bias_tolerance)

if __name__ == "__main__":
    unittest.main()
