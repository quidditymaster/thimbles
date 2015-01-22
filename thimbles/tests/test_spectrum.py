import unittest
import numpy as np
try:
    import ipdb as pdb
except ImportError:
    import pdb
from thimbles.spectrum import Spectrum, WavelengthSolution

# ########################################################################### #
# np.testing.assert_almost_equal(x, y, nulp)


class TestSpectrum (unittest.TestCase):
    
    def setUp(self):
        npts = 100
        self.npts = npts
        self.wvs = np.linspace(0, 10, npts)
        self.flux = np.random.random(npts)
        self.ivar = np.ones(npts)
        self.spec = Spectrum(self.wvs, self.flux, self.ivar)
    
    def test_properties(self):
        spec = self.spec 
        self.assertTrue(len(self.spec) == self.npts)
        np.testing.assert_almost_equal(self.flux, spec.flux)
        np.testing.assert_almost_equal(self.wvs, spec.wvs)
        np.testing.assert_almost_equal(self.ivar, spec.ivar)
        np.testing.assert_almost_equal(1.0/self.ivar, spec.var)
    
    def test_repr(self):
        repr(self.spec)
    
    def test_create_wo_ivar(self):
        wvs = np.linspace(1, 1000, 100)
        randvals = 5.0+np.random.normal(size=(100,))
        spec = Spectrum(wvs, randvals)
        var_mean = np.mean(spec.var)
        #import pdb; pdb.set_trace()
        #self.assertTrue(0.8 < var_mean < 1.2)
    
    def test_bounded_view(self):
        min_wv = 3.2
        max_wv = 5.0
        bspec = self.spec.sample([min_wv, max_wv], mode="bounded_view")
        bwvs = bspec.wvs
        xdelt = self.wvs[1]-self.wvs[0]
        self.assertTrue((min_wv - xdelt) <= bwvs[0] <= (min_wv+xdelt))
        self.assertTrue(max_wv-xdelt <= bwvs[-1] <= max_wv+xdelt)

if __name__ == "__main__":
    unittest.main()
