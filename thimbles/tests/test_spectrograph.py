import unittest
import numpy as np
from numpy.testing import assert_almost_equal as almost_equal
from thimbles.spectrographs import SamplingModel
import thimbles as tmb

class TestSamplingMatrixhModel(unittest.TestCase):
    min_wv = 100
    max_wv = 200
    npts_spec = 30
    npts_model = 100    
    
    def setUp(self):
        pass
    
    def test_sampling_matrix(self):
        spec_wvs = tmb.coordinatization.as_coordinatization(np.linspace(self.min_wv, self.max_wv, self.npts_spec))
        spec_wvs_p = tmb.modeling.Parameter(spec_wvs)
        model_wvs = tmb.coordinatization.as_coordinatization(np.linspace(self.min_wv, self.max_wv, self.npts_model))
        model_wvs_p = tmb.modeling.Parameter(model_wvs)
        spec = tmb.Spectrum(spec_wvs, np.ones(self.npts_spec), np.ones(self.npts_spec))
        inp_mod_flux = tmb.modeling.Parameter(np.ones(self.npts_model))
        output_p = tmb.modeling.Parameter()
        samp_mat_mod = SamplingModel(
            output_p=output_p,
            input_wvs_p=model_wvs_p,
            output_wvs_p=spec_wvs_p,
            input_lsf_p=tmb.modeling.FloatParameter(1.0),
            output_lsf_p=tmb.modeling.FloatParameter(1.0),
            )
        samp_mat = output_p.value
        res = samp_mat*np.ones(self.npts_model)
        np.testing.assert_almost_equal(res, np.ones(self.npts_spec))
    

if __name__ == "__main__":
    unittest.main()
