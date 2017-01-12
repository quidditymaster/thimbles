import unittest
import numpy as np
from thimbles.coordinatization import *
import thimbles as tmb
from thimbles.spectrum import Spectrum
from thimbles.resampling import *

class TestSampling(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_same_same(self):
        x = [0, 1, 2, 3]
    
    def test_every_other_pixel(self):
        x = [0, 1, 2, 3, 4]
        x_prime = [0, 2, 4]
        tolerance = 1e-7
        rebin_matrix = [
            [1, 0.5, 0, 0, 0],
            [0, 0.5, 1.0, 1, 0],
            [0, 0.0, 0.5, 1.0, 0.5],
        ]
