from copy import copy

import numpy as np

from . import coordinatization as co


class HyperGridInterpolator:
    
    """A class for doing quick linear interpolation over a large D dimensional
     hypercube of vector outputs.
    
    inputs: 
    
    coordinates: list of arrays or Coordinatization objects
      locations of sampled grid points in each dimension
      e.g. coordinates = [ [x1, x2, x3, ... xn], [y1, y2, y3 ... ym]]
    grid_data: numpy.ndarray
      a hypercube of data with any number of dimensions. The first few
      dimensions of the array are assumed to correspond to the coordinate dimensions such that a change in the index of that dimesnsion of the array corresponds to a change in the coordinates of the input coordinates.
      Any remaining dimensions of the grid_data are assumed to be the intended
      array shape of the output data.

    for example if we wish to interpolate a single value over two input dimensions we would input as the grid data a 2D grid of values n_x by n_y, if we wish to interpolate a 5 dimensional quantity over one dimension we would input a data grid of n_x by 5 array. etc.
    """    
    def __init__(self, coordinates, grid_data, extrapolation="nearest"):
        self.indexer = co.TensoredCoordinatization(coordinates)
        self.n_dims_in = len(coordinates)
        self.grid_data = grid_data
    
    def __call__(self, coord_vec):
        coord_vec = np.atleast_2d(coord_vec)
        input_shape = coord_vec.shape
        if not input_shape[-1] == self.n_dims_in:
            raise ValueError("final dimension of input coordinates does not match number of specified coordinate dimensions")
        if len(input_shape) > 2:
            coord_vec = coord_vec.reshape((-1, coord_vec.shape[-1]))
        continuous_idxs = self.indexer.get_index(coord_vec)
        min_vec = np.ones(self.n_dims_in)
        max_vec = np.asarray(self.indexer.shape) - 2
        nearest_idxs = np.around(np.clip(continuous_idxs, min_vec, max_vec)).astype(int)
        idx_deltas = continuous_idxs - nearest_idxs
        delta_int = np.where(idx_deltas > 0, 1, -1)
        neighbor_idxs = nearest_idxs + delta_int
        neighbor_weights = np.abs(idx_deltas)
        nearest_weight = 1.0-np.sum(neighbor_weights, axis=1)
        #import pdb;pdb.set_trace()
        n_idx_tup = [nearest_idxs[:, i] for i in range(self.n_dims_in)]
        interped_data = self.grid_data[n_idx_tup]*nearest_weight
        for dim_idx in range(self.n_dims_in):
            neighbor_idx_tup = copy(n_idx_tup)
            neighbor_idx_tup[dim_idx] = neighbor_idxs[:, dim_idx]
            interped_data += self.grid_data[neighbor_idx_tup]*neighbor_weights[:, dim_idx]
        if len(input_shape) > 2:
            data_shape = interped_data.shape[1:]
            output_shape = list(input_shape[:-1])
            output_shape.extend(data_shape)
            output_shape = tuple(output_shape)
            interped_data = interped_data.reshape(output_shape)
        return interped_data
