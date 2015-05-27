import unittest
import numpy as np
from numpy.testing import assert_almost_equal as almost_equal
from thimbles.spectrographs import SpectrographModel
from thimbles.spectrum import FluxParameter
import thimbles as tmb

class TestSpectrographModel(unittest.TestCase):
    min_wv = 100
    max_wv = 200
    npts_spec = 30
    npts_model = 100    
    
    def setUp(self):
        spec_wvs = np.linspace(self.min_wv, self.max_wv, self.npts_spec)
        model_wvs = np.linspace(self.min_wv, self.max_wv, self.npts_model)
        self.spec = tmb.Spectrum(spec_wvs, np.ones(self.npts_spec), np.ones(self.npts_spec))
        inp_mod_flux = FluxParameter(model_wvs, np.ones(self.npts_model))
        self.spectrograph_mod = SpectrographModel(
            self.spec, 
            model_wvs, 
            model_flux_p=inp_mod_flux
        )
    
    def test_sampling_matrix(self):
        samp_mat_p = self.spectrograph_mod.inputs["sampling_matrix"]
        res = samp_mat_p.value*np.ones(self.npts_model)
        np.testing.assert_almost_equal(res, np.ones(self.npts_spec))
    
    def test_fine_norm(self):
        fine_norm_p = self.spectrograph_mod.inputs["fine_norm"]
        almost_equal(fine_norm_p.value, np.ones(self.npts_spec))
    
    def test_executes(self):
        model_val = self.spectrograph_mod.output_p.value
        almost_equal(model_val, np.ones(self.npts_spec))

if __name__ == "__main__":
    unittest.main()
