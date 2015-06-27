
import thimbles as tmb
import unittest
import numpy as np
import os
from numpy.testing import assert_almost_equal

examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_files")

class TestReadWrite(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_read_ascii_single_spec(self):
        spec ,= tmb.io.spec_io.read_ascii(os.path.join(examples_dir, "two_column.txt"))
        spec_wvs = spec.wvs
        lin_wvs = np.linspace(5000, 5007, len(spec))
        assert_almost_equal(lin_wvs, spec_wvs)

    def test_read_ascii_multi_spec(self):
        spectra = tmb.io.spec_io.read_ascii(os.path.join(examples_dir, "two_column_multi_spec.txt"))
        self.assertTrue(len(spectra) == 3)
        
        

if __name__ == "__main__":
    unittest.main()
