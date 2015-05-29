import unittest
import numpy as np
try:
    import ipdb as pdb
except ImportError:
    import pdb
from thimbles.coordinatization import *
import thimbles as tmb
from thimbles.spectrum import Spectrum, WavelengthSolution

# ########################################################################### #
# np.testing.assert_almost_equal(x, y, nulp)


class TestWavelengthSolution(unittest.TestCase):
    
    def setUp(self):
        npts = 100
        self.min_wv = 5000
        self.max_wv = 5500
        flux = np.ones(npts)
        ivar = np.ones(npts)
        self.linear_wvs = np.linspace(self.min_wv, self.max_wv, npts)
        self.log_linear_wvs = np.exp(np.linspace(np.log(self.min_wv), np.log(self.max_wv), npts))
        self.poly_wvs = self.linear_wvs + np.linspace(0, 1, npts)**2
        
        self.linspec = Spectrum(self.linear_wvs, flux, ivar)
        self.logspec = Spectrum(self.log_linear_wvs, flux, ivar)
        self.polyspec = Spectrum(self.poly_wvs, flux, ivar)
        
        self.spectra = [self.linspec, self.logspec, self.polyspec]
        
    def test_coordinatization_types(self):
        lin_idxer = self.linspec.wv_sample.wv_soln.indexer
        assert isinstance(lin_idxer, LinearCoordinatization)
        log_idxer = self.logspec.wv_sample.wv_soln.indexer
        assert isinstance(log_idxer, LogLinearCoordinatization)
        poly_idxer = self.polyspec.wv_sample.wv_soln.indexer
        assert isinstance(poly_idxer, ArbitraryCoordinatization)
    
    def test_spec_passthrough(self):
        for spec in self.spectra:
            wvs = np.linspace(self.min_wv, self.max_wv, 5)
            indexes = spec.get_index(wvs)
            wv_sol_indexes = spec.wv_sample.wv_soln.get_index(wvs)
            np.testing.assert_almost_equal(indexes, wv_sol_indexes)
            spec_back_wvs = spec.get_wvs(indexes)
            wv_soln_back_wvs = spec.wv_sample.wv_soln.get_wvs(indexes)
            np.testing.assert_almost_equal(spec_back_wvs, wv_soln_back_wvs)
            np.testing.assert_almost_equal(wvs, spec_back_wvs)
    
    def test_scalar_indexing(self):
        for spec in self.spectra:
            wvsoln = spec.wv_sample.wv_soln
            cent_wv = 0.5*(self.max_wv+self.min_wv)
            cent_idx = wvsoln.get_index(cent_wv)
            back_wv = wvsoln.get_wvs(cent_idx)
            assert np.abs(back_wv-cent_wv) < 1e-7
    
    def test_index_snap_clip(self):
        for spec in self.spectra:
            wvsoln = spec.wv_sample.wv_soln
            wvs = np.linspace(self.min_wv, self.max_wv, 7)
            res = wvsoln.get_index(wvs, snap=True)
            assert res.astype(int).dtype == res.dtype
            res = wvsoln.get_index(wvs-10.0, clip=True)
            assert np.min(res) == 0
            res = wvsoln.get_index(wvs+10.0, clip=True)
            assert np.max(res) == len(wvsoln)-1

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
        import pdb; pdb.set_trace()
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

    def test_rebin(self):
        rebin_wvs = self.spec.wvs[::5]
        rebin_spec = self.spec.sample(rebin_wvs, mode="rebin")
        #will still be a line since we preserve normalization
        perfect_rebin = self.flux_slope*rebin_wvs[2:-2]+self.flux_offset
        np.testing.assert_almost_equal(rebin_spec.flux[2:-2], perfect_rebin)


class TestRVSetting(unittest.TestCase):
    min_wv = 5000
    max_wv = 6000
    npts = 100
    
    def setUp(self):
        pass
    
    def make_spec(self, rv, dh, rv_shifted, dh_shifted):
        return tmb.Spectrum(
            np.linspace(5000, 6000, 100), 
            np.ones(100), 
            np.ones(100), 
            rv=rv, 
            delta_helio=dh, 
            rv_shifted=rv_shifted, 
            helio_shifted=dh_shifted
        )
    
    def test_set_rv_shifts(self):
        spec = self.make_spec(0.0, 0.0, rv_shifted=True, dh_shifted=True) 
        pre_shift_wvs = spec.wvs
        shift_vel = 100
        spec.set_rv(shift_vel)
        wvs = spec.wvs
        assert wvs[0] < self.min_wv
        assert wvs[-1] < self.max_wv
        assert wvs[0] < wvs[-1]
        np.testing.assert_almost_equal(wvs, pre_shift_wvs*(1.0-shift_vel/tmb.speed_of_light))
        spec.set_rv(-shift_vel)
        wvs = spec.wvs
        assert wvs[0] > self.min_wv
        assert wvs[-1] > self.max_wv
        assert wvs[-1] > wvs[0]
        np.testing.assert_almost_equal(wvs, pre_shift_wvs*(1.0+shift_vel/tmb.speed_of_light))
    
    def test_initialize_rv(self):
        shift_vel = 100.0
        spec = self.make_spec(rv=shift_vel, dh=0.0, rv_shifted=True, dh_shifted=True)
        rest_min = self.min_wv
        obs_min = self.min_wv/(1.0-shift_vel/tmb.speed_of_light)
        np.testing.assert_almost_equal(rest_min, spec.wvs[0])
        rest_indexer = spec.wv_sample.wv_soln.indexer
        spec_min_p = rest_indexer.inputs["min"]
        obs_min_p = spec_min_p.mapped_models[0].inputs["wvs"]
        np.testing.assert_almost_equal(obs_min, obs_min_p.value)
        spec_min_p.invalidate()
        np.testing.assert_almost_equal(rest_min, spec_min_p.value)
    
    def test_initialize_vs_set_rv(self):
        shift_vel = 100.0
        spec1 = self.make_spec(rv=0.0, dh=0.0, rv_shifted=True, dh_shifted=True)
        spec1.set_rv(shift_vel)
        spec2 = self.make_spec(rv=shift_vel, dh=0.0, rv_shifted=False, dh_shifted=True)
        np.testing.assert_almost_equal(spec1.wvs, spec2.wvs)

    def test_initialize_vs_set_dh(self):
        shift_vel = 100.0
        spec1 = self.make_spec(rv=0.0, dh=0.0, rv_shifted=True, dh_shifted=True)
        spec1.wv_sample.wv_soln.delta_helio_p.value = shift_vel
        spec2 = self.make_spec(rv=0.0, dh=shift_vel, rv_shifted=True, dh_shifted=False)
        np.testing.assert_almost_equal(spec1.wvs, spec2.wvs)


if __name__ == "__main__":
    unittest.main()
