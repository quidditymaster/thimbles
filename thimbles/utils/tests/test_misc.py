import unittest
from thimbles.utils.misc import *
import scipy.sparse as sparse

class TestIRLS(unittest.TestCase):
    
    def setUp(self):
        self.npts = 200
        self.x = np.linspace(-1, 1, self.npts)
        self.slope = 3.0
        self.offset = -0.5
        self.y = self.slope*self.x + self.offset
        self.fit_matrix = np.array([np.ones(self.npts), self.x]).transpose()
    
    def _check_result(self, res_vec, tol):
        fit_offset = res_vec[0]
        fit_slope = res_vec[1]
        self.assertTrue(np.abs(fit_slope-self.slope) < tol)
        self.assertTrue(np.abs(fit_offset-self.offset) < tol)
    
    def test_noiseless(self):
        #import pdb; pdb.set_trace()
        res_vec = irls(self.fit_matrix, self.y, 1.0)
        self._check_result(res_vec, tol=1e-10)
    
    def test_cauchy_noise(self):
        #import pdb; pdb.set_trace()
        y_cauchy = np.random.standard_cauchy(size=(self.npts,)) + self.y
        res_vec = irls(self.fit_matrix, y_cauchy, 1.0)
        self._check_result(res_vec, 0.5)
    
    def test_gaussian_noise(self):
        y_gauss = np.random.normal(size=(self.npts,)) + self.y
        res_vec = irls(self.fit_matrix, y_gauss, 1.0)
        self._check_result(res_vec, 0.5)
    
    def test_sparse(self):
        res_vec = irls(sparse.csr_matrix(self.fit_matrix), self.y, 1.0)
        self._check_result(res_vec, 1e-10)


class TestUnweightedVoigtFit(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_random_voigt(self):
        #sig = np.random.uniform(1.0, 5.0)
        #gam = np.random.uniform(1.0, 5.0)
        sig = 1.0
        gam = 2.0
        ew = 3.0
        x = np.linspace(-10, 10, 25)
        vgt = voigt(x, 0.0, sig, gam)*ew
        
        res = unweighted_voigt_fit(x, vgt)
        self.assertTrue(np.abs(sig-res[0]) < 1e-4)
        self.assertTrue(np.abs(gam-res[1]) < 1e-4)
        self.assertTrue(np.abs(ew-res[2]) < 1e-4)
        
class TestNoiseEstimates(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_smoothed_mad_error(self):
        npts = 1000
        rpts = 2.0+np.random.random(size=(npts,))
        smerr = smoothed_mad_error(rpts)
        #import pdb; pdb.set_trace()

if __name__ == "__main__":
    unittest.main()
