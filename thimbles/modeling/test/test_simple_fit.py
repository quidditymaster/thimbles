import unittest
import numpy as np
import matplotlib.pyplot as plt
import thimbles as tmb
import scipy.sparse as sp

class SimpleFitTest(unittest.TestCase):

    def setUp(self):
        self.true_lparams = np.asarray([0.0, 0.6, 0.125])
        self.lparams = np.asarray([0.0, 1.0, 0.125])
        self.prof = tmb.line_profiles.Voigt(0.0, self.lparams)
        self.npts = 100
        self.x = np.linspace(-10, 10, self.npts)
        
        self.ev_func = lambda a, b, p: self.get_gmat(a)*b
        self.op_func = lambda a, b, p: self.get_gmat(a)
    
    def get_gmat(self, alpha_vec):
        sigma, gamma = alpha_vec
        self.lparams[1:] = sigma, gamma
        self.prof.set_parameters(self.lparams)
        self.prof_vec = self.prof(self.x)
        gmat = sp.bsr_matrix(np.hstack((np.ones((self.npts, 1)) , self.prof_vec.reshape((-1, 1)))))
        return gmat
    
    def test_exact_fit(self):
        start_params = np.array([0.05, 0.0])
        y = self.get_gmat(start_params)
        
        DM = tmb.modeling.data_models.DataModel
        dm = DM()
        
        gmod = tmb.modeling.LocallyLinearModel(self.lparams, np.ones(2), None, self.ev_func, self.op_func)

if __name__ == "__main__":
    unittest.main()
