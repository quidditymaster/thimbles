import unittest
from thimbles.utils.misc import *

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
        res_vec = irls(self.fit_matrix, self.y, 1.0)
        self._check_result(res_vec, tol=1e-5)
    
    def test_cauchy_noise(self):
        import pdb; pdb.set_trace()
        y_cauchy = np.random.standard_cauchy(size=(self.npts,)) + self.y
        res_vec = irls(self.fit_matrix, y_cauchy, 1.0)
        self._check_result(res_vec, 0.5)
    
    def test_gaussian_noise(self):
        y_gauss = np.random.normal(size=(self.npts,)) + self.y
        res_vec = irls(self.fit_matrix, y_gauss, 1.0)
        self._check_result(res_vec, 0.2)

if __name__ == "__main__":
    unittest.main()
