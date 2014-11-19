import unittest
import thimbles as tmb
import numpy as np
from thimbles.hypergrid import HyperGridInterpolator
from scipy.interpolate import LinearNDInterpolator

class testHyperGridInterpolator2D(unittest.TestCase):
    
    def setUp(self):
        self.nx, self.ny = 8, 7
        self.y_vals = np.arange(self.nx*self.ny).reshape((self.nx, self.ny))
        self.row_min = -1
        self.row_max = 1.2
        self.col_min = -10.1
        self.col_max = 3.3
        self.row_coords = np.linspace(self.row_min, self.row_max, self.nx)
        self.col_coords = np.linspace(self.col_min, self.col_max, self.ny)
    
    def test_scipy_compare(self):
        #setup the LinearNDInterpolator
        x_mesh, y_mesh = np.meshgrid(self.col_coords, self.row_coords)
        coords = np.hstack([y_mesh.reshape((-1, 1)), x_mesh.reshape((-1, 1))])
        values = self.y_vals.reshape((-1,))
        lndinterp = LinearNDInterpolator(coords, values)
        
        #set up our hypergrid interpolator
        hgrid = HyperGridInterpolator([self.row_coords, self.col_coords], self.y_vals)
        
        scale_vec = np.array([self.row_max-self.row_min, self.col_max-self.col_min])
        rand_pts = np.random.random((50, 2))*scale_vec
        rand_pts += np.array([self.row_min, self.col_min])
        
        import pdb; pdb.set_trace()
        lnd_results = lndinterp(rand_pts)
        hgrid_results = hgrid(rand_pts)
        self.assertTrue(np.testing.assert_allclose(lnd_results, hgrid_results))

if __name__ == "__main__":
    unittest.main()
