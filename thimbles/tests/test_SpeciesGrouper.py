import unittest
import numpy as np
try:
    import ipdb as pdb
except ImportError:
    import pdb
from thimbles.features import SpeciesGrouper

# ########################################################################### #

class TestSpeciesGrouper (unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
    
    def test_Grouper_no_inputs(self):
        spg = SpeciesGrouper([])
        sp = np.array([26.0, 26.1, 22.0, 22.1])
        spres = np.array([2.0, 3.0, 0.0, 1.0])
        self.assertTrue(np.all(spg(sp) == spres))
        
if __name__ == "__main__":
    unittest.main()
