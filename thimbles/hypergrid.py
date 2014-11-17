import binning
import numpy as np

class HyperGridInterpolator:
    
    """A class for doing quick linear interpolation over a large D dimensional
     hypercube of vector outputs.
    
    inputs: 
    
    coordinates: list of arrays
      locations of sampled grid points in each dimension
      e.g. coordinates = [ [x1, x2, x3, ... xn], [y1, y2, y3 ... ym]]
      coordinates must be monotonically increasing
    grid_data: numpy.ndarray
      a hypercube of data with a shape 
      len(coordinates[0]), len(coordinates[1]), ..., len(output_vector)
    extrapolation: string
       'nearest' for parameters outside the grid coordinates clip to closest
       grid index in each dimension and return that result.
       
        
    """    
    def __init__(self, coordinates, grid_data, extrapolation="nearest"):
        self.binner = binning.TensoredBinning(coordinates)
        self.n_dims_in = len(coordinates)
        self.grid_data = grid_data
        return
    
    def __call__(self, coord_vec):
        idx_val = self.binner.coordinates_to_indicies(coord_vec)[0]
        min_idx = np.asarray(idx_val, dtype = int)
        alphas = idx_val-min_idx
        transform = np.zeros((self.n_dims_in, self.n_dims_in))
        c_vertex = np.zeros(self.n_dims_in, dtype = int)
        #temp, promote_order = zip(*sorted(zip(alphas, range(self.n_dims_in)), reverse = True))
        promote_order = np.argsort(-alphas)
        #import pdb;pdb.set_trace()
        for i in xrange(self.n_dims_in):
            c_vertex[promote_order[i]] = 1.0
            transform[i] = c_vertex
        secondary_weights = np.dot(np.linalg.inv(transform), alphas)
        first_weight = 1.0 - np.sum(secondary_weights)
        output = first_weight * self.grid_data[tuple(min_idx)]
        for i in xrange(self.n_dims_in):
            vertex_data = self.grid_data[tuple(min_idx + transform[i])]
            output += secondary_weights[i] * vertex_data
        return output
