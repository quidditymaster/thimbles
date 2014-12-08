import coordinatization as co
import numpy as np

class HyperGridInterpolator:
    
    """A class for doing quick linear interpolation over a large D dimensional
     hypercube of vector outputs.
    
    inputs: 
    
    coordinates: list of arrays or Coordinatization objects
      locations of sampled grid points in each dimension
      e.g. coordinates = [ [x1, x2, x3, ... xn], [y1, y2, y3 ... ym]]
    grid_data: numpy.ndarray
      a hypercube of data with a shape 
      len(coordinates[0]), len(coordinates[1]), ..., len(output_vector)
    extrapolation: string
       'nearest' for parameters outside the grid coordinates clip to closest
       grid index in each dimension and return that result.
       
        
    """    
    def __init__(self, coordinates, grid_data, extrapolation="nearest"):
        self.indexer = co.TensoredCoordinatization(coordinates)
        self.n_dims_in = len(coordinates)
        self.grid_data = grid_data
    
    def __call__(self, coord_vec, return_weights=False):
        coord_vec = np.atleast_2d(coord_vec)
        continuous_idxs = self.indexer.get_index(coord_vec)
        min_vec = np.ones(self.n_dims_in)
        max_vec = np.asarray(self.indexer.shape) - 1
        nearest_idxs = np.around(np.clip(continuous_idxs, min_vec, max_vec)).astype(int)
        idx_deltas = continuous_idxs - nearest_idxs
        delta_int = np.where(idx_deltas > 0, 1, -1)
        neighbor_idxs = delta_int + nearest_idxs
        alpha = np.abs(idx_deltas) #dimension by dimension alphas
        
        interped_data = self.grid_data[nearest_idxs]
        for dim_idx in range(self.grid_data):
            cur_iterp = nearest_idxs
        
        
        coord_vec = np.atleast_2d(coord_vec)
        idx_val = self.indexer.get_index(coord_vec)
        min_idx = np.asarray(idx_val, dtype = int)
        alphas = idx_val-min_idx
        array_promotion_ord = np.argsort(alphas, axis=1)
        interped_data = np.zeros((len(coord_vec), self.grid_data.shape[-1]))
        for i in range(len(coord_vec)):
            c_transform = np.zeros((self.n_dims_in, self.n_dims_in))
            promoted_idxs = []
            for j in range(self.n_dims_in):
                promoted_idxs.append(promotion_ord[i]
                c_transform[i, []
            interped_data[i] = 

        transform = np.zeros((self.n_dims_in, self.n_dims_in))
        c_vertex = np.zeros(self.n_dims_in, dtype = int)
        #temp, promote_order = zip(*sorted(zip(alphas, range(self.n_dims_in)), reverse = True))
        promote_order = np.argsort(-alphas)
        #import pdb;pdb.set_trace()
        for i in xrange(self.n_dims_in):
            c_vertex[promote_order[i]] = 1.0
            transform[i] = c_vertex
        secondary_weights = np.dot(np.linalg.pinv(transform), alphas)
        first_weight = 1.0 - np.sum(secondary_weights)
        output = first_weight * self.grid_data[tuple(min_idx)]
        for i in xrange(self.n_dims_in):
            vertex_data = self.grid_data[tuple(min_idx + transform[i])]
            output += secondary_weights[i] * vertex_data
        return output
