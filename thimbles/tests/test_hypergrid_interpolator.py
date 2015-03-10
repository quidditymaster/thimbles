import unittest
import thimbles as tmb
import matplotlib.pyplot as plt
import numpy as np
from thimbles.hypergrid import HyperGridInterpolator
from scipy.interpolate import LinearNDInterpolator

plot_comparison = False

class testHyperGridInterpolator2D(unittest.TestCase):
    
    def setUp(self):
        self.nx, self.ny = 7, 5
        self.x_slope = 3.2
        self.y_slope = -1.2
        self.const = 1.1
        self.x_min = -1
        self.x_max = 1.2
        self.y_min = -10.1
        self.y_max = 3.3
        self.x_coords = np.linspace(self.x_min, self.x_max, self.nx)
        self.y_coords = np.linspace(self.y_min, self.y_max, self.ny)
        #x_mesh, y_mesh = np.meshgrid(self.y_coords, self.x_coords)
        x_mesh = self.x_coords.reshape((-1, 1))*np.ones((1, self.ny))
        y_mesh = self.y_coords.reshape((1, -1))*np.ones((self.nx, 1))
        self.target_vals = self.const + self.x_slope*x_mesh + self.y_slope*y_mesh
    
    def test_interpolate(self):
        hgrid = HyperGridInterpolator([self.x_coords, self.y_coords], self.target_vals)
        
        n_mult = 5
        x_coords = np.linspace(self.x_min, self.x_max, n_mult*self.nx)
        y_coords = np.linspace(self.y_min, self.y_max, n_mult*self.ny)
        x_mesh = x_coords.reshape((-1, 1))*np.ones((1, n_mult*self.ny))
        y_mesh = y_coords.reshape((1, -1))*np.ones((n_mult*self.nx, 1))
        target_vals = self.const + self.x_slope*x_mesh + self.y_slope*y_mesh
        
        coords = np.dstack((x_mesh, y_mesh))
        hgrid_vals = hgrid(coords)
        if plot_comparison:
            fig, axes = plt.subplots(1, 3)
            vmin = np.min(target_vals)
            vmax = np.max(target_vals)
            axes[0].imshow(target_vals, cmap="afmhot", vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[1].imshow(hgrid_vals, cmap="afmhot", interpolation="nearest", vmin=vmin, vmax=vmax)
            axes[2].imshow(hgrid_vals-target_vals, cmap="afmhot", interpolation="nearest", vmin=vmin, vmax=vmax)
            plt.show()
        #import pdb; pdb.set_trace()
        np.testing.assert_almost_equal(target_vals, hgrid_vals)

if __name__ == "__main__":
    unittest.main()
