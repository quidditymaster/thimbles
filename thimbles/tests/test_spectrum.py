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
        self.min_wv = 0
        self.max_wv = 10
        self.wvs = np.linspace(self.min_wv, self.max_wv, npts)
        self.flux_slope = 5
        self.flux_offset = 1.2
        self.flux = self.wvs*self.flux_slope + self.flux_offset
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
    
    def test_len(self):
        assert len(self.spec) == self.npts
    
    def test_create_wo_ivar(self):
        wvs = np.linspace(1, 1000, 100)
        randvals = 5.0+np.random.normal(size=(100,))
        spec = Spectrum(wvs, randvals)
        var_mean = np.mean(spec.var)
        #import pdb; pdb.set_trace()
        #self.assertTrue(0.8 < var_mean < 1.2)
    
    def test_sample_mode_errors(self):
        def call_sample(mode):
            self.spec.sample([3, 5], mode=mode)
        self.assertRaises(ValueError, call_sample, mode="not a mode")
    
    def test_bounded_view(self):
        min_wv = self.min_wv+0.37*(self.max_wv-self.min_wv)
        max_wv = self.min_wv+0.89*(self.max_wv-self.min_wv)
        bspec = self.spec.sample([min_wv, max_wv], mode="bounded")
        bwvs = bspec.wvs
        xdelt = self.wvs[2]-self.wvs[0]
        self.assertTrue((min_wv - xdelt) <= bwvs[0] <= (min_wv+xdelt))
        self.assertTrue(max_wv-xdelt <= bwvs[-1] <= max_wv+xdelt)
    
    def test_pseudonorm(self):
        psnorm = self.spec.pseudonorm()
        nspec = self.spec.normalized()
        np.testing.assert_almost_equal(self.spec.flux/psnorm, nspec.flux)
    
    def test_interpolate(self):
        min_wv = self.min_wv+0.37*(self.max_wv-self.min_wv)
        max_wv = self.min_wv+0.89*(self.max_wv-self.min_wv)
        iterp_wvs = np.linspace(min_wv, max_wv, 200)
        perfect_iterp = self.flux_slope*iterp_wvs + self.flux_offset
        iterp_res = self.spec.sample(iterp_wvs, mode="interpolate")
        np.testing.assert_almost_equal(iterp_res.flux, perfect_iterp)
        
        iterp_res, iterp_mat = self.spec.sample(iterp_wvs, mode="interpolate", return_matrix = True)
        np.testing.assert_almost_equal(iterp_mat*self.spec.flux, perfect_iterp)


if __name__ == "__main__":
    unittest.main()
