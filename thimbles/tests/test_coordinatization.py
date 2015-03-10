import unittest
import thimbles.coordinatization as coord
from thimbles.coordinatization import as_coordinatization
import numpy as np

class TestEdgeCenterConversion(unittest.TestCase):

    def setUp(self):
        self.edges = np.asarray([0, 1, 2, 3, 4])
        self.centers = np.asarray([0.5, 1.5, 2.5, 3.5])
        self.tol = 1e-10
    
    def test_edges_to_centers(self):
        fcents = coord.edges_to_centers(self.edges)
        self.assertTrue(np.sum(np.abs(fcents - self.centers)) < self.tol)
    
    def test_centers_to_edges(self):
        fedges = coord.centers_to_edges(self.centers)
        self.assertTrue(np.sum(np.abs(fedges - self.edges)) < self.tol)
    
    def test_there_and_back(self):
        fedges = coord.edges_to_centers(coord.centers_to_edges(self.edges))
        fcents = coord.centers_to_edges(coord.edges_to_centers(self.centers))
        edge_diff_sum = np.sum(np.abs(fedges - self.edges))
        cent_diff_sum = np.sum(np.abs(fcents - self.centers))
        self.assertTrue(edge_diff_sum < self.tol)
        self.assertTrue(cent_diff_sum < self.tol)

class TestEdgeCenterConversionReversed(TestEdgeCenterConversion):
    
    def setUp(self):
        self.edges = np.asarray([4, 3, 2, 1, 0])
        self.centers = np.asarray([3.5, 2.5, 1.5, 0.5])
        self.tol = 1e-10

class TestIndexConversion(unittest.TestCase):
    
    def setUp(self):
        self.min = 5000.0
        self.max = 12000.0
        self.npts = 53
        self.x = np.linspace(self.min, self.max, self.npts)
        self.coord_obj = coord.ArbitraryCoordinatization(self.x)
        self.tol = 1e-13
    
    def test_setting(self):
        #test setting the normal way
        c_obj = coord.ArbitraryCoordinatization(self.x)
        self.assertTrue(np.std(c_obj.coordinates - self.x) < self.tol)
        #test setting as bin edges
        c_obj = coord.ArbitraryCoordinatization(coord.centers_to_edges(self.x), as_edges=True)
        self.assertTrue(np.std(c_obj.coordinates - self.x) < self.tol)
    
    def test_min_max(self):
        self.assertTrue(self.coord_obj.min == self.min)
        self.assertTrue(self.coord_obj.max == self.max)
    
    def test_get_coord(self):
        coords = self.coord_obj.get_coord(np.arange(self.npts))
        diffs = coords - self.x
        np.testing.assert_almost_equal(coords, self.x)
    
    def test_get_index(self):
        indexes = self.coord_obj.get_index(self.x)
        np.testing.assert_almost_equal(indexes, np.arange(self.npts))
    
    def nest_coord_index(self):
        xi = self.coord_obj.get_coord(self.coord_obj.get_index(self.x))
        ix = self.coord_obj.get_index(self.coord_obj.get_coord(np.arange(self.npts)))
        np.testing.assert_almost_equal(xi , self.x)
        np.testing.assert_almost_equal(ix , np.arange(self.npts))
    
    def test_nested_extrapolation(self):
        t_idxs = np.array([-1, -5, -10, 21, 32, 97])
        outer_coords = self.coord_obj.get_coord(t_idxs)
        res_idxs = self.coord_obj.get_index(outer_coords)
        np.testing.assert_almost_equal(t_idxs, res_idxs)
    
    def test_interpolant(self):
        tslope, toff= 3.2, 1.3
        test_y = self.x*tslope + toff
        test_x = np.sort(np.random.uniform(self.min, self.max, size=(100,)))
        interp_mat = self.coord_obj.interpolant_sampling_matrix(test_x)
        target_y = test_x*tslope + toff
        #import matplotlib.pyplot as plt
        #plt.plot(test_x, interp_mat*test_y)
        #plt.plot(test_x, target_y)
        #plt.show()
        #import pdb; pdb.set_trace()
        np.testing.assert_almost_equal(target_y, interp_mat*test_y)


class TestLinearCoordinatization(TestIndexConversion):
    
    def setUp(self):
        self.min = 100.0
        self.max = 120.0
        self.npts = 53
        self.x = np.linspace(self.min, self.max, self.npts)
        self.coord_obj = coord.LinearCoordinatization(self.x)
        self.tol = 1e-13

class TestLinearCoordinatizationInit(unittest.TestCase):
    
    def setUp(self):
        self.min = 5000.0
        self.max = 12000.0
        self.npts = 53
        self.dx = (self.max-self.min)/(self.npts - 1)
    
    def validate_coord(self, coord_instance):
        ci = coord_instance
        self.assertAlmostEqual(ci.min, self.min)
        self.assertAlmostEqual(ci.max, self.max)
        self.assertAlmostEqual(ci.npts, self.npts)
        self.assertAlmostEqual(ci.dx, self.dx)
    
    def test_coord_make(self):
        #test making from a coordinate vector
        x = np.linspace(self.min, self.max, self.npts)
        ci = coord.LinearCoordinatization(x)
        self.validate_coord(ci)
        
        #make using any combination of min max dx and npts
        ci = coord.LinearCoordinatization(min=self.min, max=self.max, npts=self.npts)
        self.validate_coord(ci)
        
        ci = coord.LinearCoordinatization(min=self.min, max=self.max, dx=self.dx)
        self.validate_coord(ci)
        
        ci = coord.LinearCoordinatization(max=self.max, npts=self.npts, dx=self.dx)
        self.validate_coord(ci)
        
        ci = coord.LinearCoordinatization(min=self.min, npts=self.npts, dx=self.dx)
        self.validate_coord(ci)


class TestAsCoordinatization(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_to_linear(self):
        res = as_coordinatization(np.linspace(10, 35, 101))
        self.assertTrue(isinstance(res, coord.LinearCoordinatization))
    
    def test_to_log_linear(self):
        #import pdb; pdb.set_trace()
        res = as_coordinatization(np.exp(np.linspace(1, 5, 26)))
        self.assertTrue(isinstance(res, coord.LogLinearCoordinatization))

if __name__ == "__main__":
    unittest.main()
