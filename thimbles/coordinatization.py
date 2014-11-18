import numpy as np

from thimbles import logger

class CoordinatizationError(Exception):
    pass

def edges_to_centers(edges):
    """convert from an array of bin edges to bin centers.
    """
    edges = np.asarray(edges)
    return 0.5*(edges[:-1]+edges[1:])

def centers_to_edges(centers):
    """convert from an array of coordinates to an array of bin edges 
    centered on those coordinates.
    """
    centers = np.asarray(centers)
    if len(centers) == 0:
        return np.zeros(0)
    edges = np.zeros(len(centers) + 1, dtype = float)
    edges[1:-1] = 0.5*(centers[1:] + centers[:-1])
    edges[0] = centers[0] - 0.5*(centers[1]-centers[0])
    edges[-1] = centers[-1] + 0.5*(centers[-1] - centers[-2])
    return edges

def as_coordinatization(coordinates, delta_max=0, force_linear=False, force_log_linear=False):
    """convert a set of coordinates into a coordinatization
    or if passed a coordinatization return it. This function detects
    whether the given coordinates are sufficiently close to 
    linear or linear in the log and returns a LinearCoordinatization
    or LogLinearCoordinitization where appropriate.
    
    parameters
    
    coordinates: ndarray or Coordinatization
      the array of coordinates to be coordinatized 
    
    delta_max: float
      if the maximum absolute deviation of a linear or log-linear 
      coordinatization of the given coordinates would be less than 
      delta_max then return the approximating LinearCoordinatization 
      or LogLinearCoordinatization.
      If delta_max <= 0 no checks are carried out.
    
    force_linear: bool
      if true return a LinearCoordinatization regardless of how well
      or how badly it describes the input coordinates, simply go from
      min to max in a linear way.
    
    force_log_linear: bool
      if true return a LogLinearCoordinatization regardless of how well
      it fits, go from min to max on a logarithmic scale.
    
    """
    if isinstance(coordinates, Coordinatization):
        return coordinates
    coordinates = np.asarray(coordinates)
    if len(coordinates) < 2:
        raise CoordinatizationError("cannot coordinatize arrays of size 1 or less")
    
    if force_linear:
        return LinearCoordinatization(coordinates)
    if force_log_linear:
        return LogLinearCoordinatization(coordinates)
    if len(coordinates == 2):
        #perfectly described by a linear coordinatization no need to check
        return LinearCoordinatization(coordinates)
    
    if delta_max <= 0:
        return Coordinatization(coordinates)
    
    for coord_class in [LinearCoordinatization,
                        LogLinearCoordinatization]:
        test_coord_instance = coord_class(coordinates)
        test_coords = test_lin.get_coord(np.arange(len(coordinates)))
        if np.max(np.abs(test_coords - coordinates)) < delta_max:
            return test_coord_instance
    
    return Coordinatization(coordinates)


class Coordinatization(object):
    
    def __init__(self, coordinates, as_edges=False, check_ordered=False):
        """a class for representing coordinatizations of ordered arrays.
        
        coordinates: ndarray
          the coordinates associated to each element of an array
          in order to ensure the ability to be able to go from 
          coordinates to indexes and back these coordinates should be
          monotonically increasing or decreasing.
        as_edges: bool
          if as_edges is True then the passed in coordinates are assumed
          to be the edges of a set of bins from which the centers of those
          bins will be constructed to and used as the coordinates for
          this coordinatization.
        check_ordered: bool
          if True check that the coordinates array is monotonic therefore
          admitting an invertible mapping from coordinate to index.
        
        Note: instead of instantiating this class directly consider
        the convenience function as_coordinatization in this module.
        It will automatically detect for the common cases of linear
        coordinatization and LogLinear coordinatization and apply 
        those if appropriate.
        """
        coordinates = np.asarray(coordinates)
        if check_ordered:
            gradient = scipy.gradient(coordinates)
            gsum = np.sum(gradient > 0)
            if not ((gsum == 0) or gsum == len(coordinates)):
                raise CoordinatizationError("coordinates not monotonic no allowable coordinatization")
        
        if not as_edges:
            self.coordinates = coordinates
            self.bins = centers_to_edges(coordinates)
        else:
            self.bins = coordinates
            self.coordinates = edges_to_centers(coordinates)
        self._cached_prev_bin = (self.bins[0], self.bins[1])
        self._cached_prev_bin_idx = 0
        self._start_dx = self.bins[1] - self.bins[0]
        self._end_dx = self.bins[-1] - self.bins[-2]
    
    __doc__ = __init__.__doc__
    
    def __len__(self):
        return len(self.coordinates)
    
    @property
    def min(self):
        return self.coordinates[0]
    
    @property
    def max(self):
        return self.coordinates[-1]
    
    def get_coord(self, index, clip=False, snap=False):
        """convert array indexes to coordinates
        
        parameters
        
        index: ndarray or number
          the index within the array
        clip: bool
          if true returned coordate values are clipped to lie between
          the minimum and maximum coordinate values of this coordinatization.
        snap: bool
          if true the input indexes are rounded to the nearest integer values
          so that the output coordinates lie exactly on the coordinate centers.
        """
        #TODO:dump this operation out to cython
        index = np.asarray(index)
        if snap:
            index = np.around(index)
        input_shape = index.shape
        index = np.atleast_1d(index)
        out_coordinates = np.zeros(index.shape, dtype=float)
        min_x = self.min
        max_idx = len(self)-1
        for i in range(len(index)):
            alpha = index[i]%1
            int_part = int(index[i])
            if index[i] < 0:
                out_coordinates[i] = min_x + self._start_dx*index[i]
            elif index[i] > len(self.coordinates)-2:
                out_coordinates[i] = self.coordinates[max_idx] + (index[i] - max_idx)*self._end_dx
            else:
                out_coordinates[i] = self.coordinates[int_part]*(1.0-alpha)
                out_coordinates[i] += self.coordinates[int_part+1]*alpha
        if clip:
            out_coordinates = np.clip(out_coordinates, self.min, self.max)
        return out_coordinates
    
    def get_index(self, coord, clip=False, snap=False):
        """convert array of coordinates to the associated indexes
        
        parameters
        
        coord: ndarray or number
          an array of coordinates
        clip: bool
          if true returned coordate values are clipped to lie between
          0 and len(self) - 1
        snap: bool
          if true the returned indexes are rounded to the nearest integer
          and the returned array has dtype int.
        """
        coord = np.asarray(coord)
        in_shape = coord.shape
        coord = np.atleast_1d(coord)
        out_index = np.zeros(in_shape, dtype=float)
        coord_idxs = np.argsort(coord)
        cur_idx = 0
        min_x = self.min
        max_x = self.max
        max_idx = len(self)-1
        for x_i in range(len(coord)):
            cur_i = coord_idxs[x_i]
            cur_x = coord[cur_i]
            if cur_x <= self.min:
                idx = (cur_x-min_x)/self._start_dx
            elif cur_x >= self.max:
                idx = (cur_x-max_x)/self._end_dx + max_idx
            else:
                bounded = False
                while not bounded:
                    lb, ub = self.coordinates[cur_idx:cur_idx+2]
                    if lb <= cur_x <= ub:
                        bounded = True
                    else:
                        cur_idx += 1
                idx = cur_idx + (cur_x - lb)/(ub-lb)
            out_index[cur_i] = idx
        if snap:
            out_coordinates = np.around(out_index).astype(int)
        if clip:
            out_coordinates = np.clip(out_coordinates, 0, len(self)-1)
        return out_index.reshape(in_shape)
        
        #TODO:dump this to cython
        # get the upper and lower bounds for the coordinates
        lb,ub = self.bins[0],self.bins[-1]
        xv = np.asarray(coord)
        out_idx_vals = np.zeros(len(xv.flat), dtype = float)
        for x_idx in xrange(len(xv.flat)):
            cur_x = xv.flat[x_idx]
            #check if the last solution still works
            if self._cached_prev_bin[0] <= cur_x <= self._cached_prev_bin[1]:
                #if so find the fraction through the pixel
                alpha = (cur_x-self._cached_prev_bin[0])/(self._cached_prev_bin[1]-self._cached_prev_bin[0])
                out_idx_vals[x_idx] = self._cached_prev_bin_idx + alpha
                continue
            #make sure that the x value is inside the bin range or extrapolate
            if lb > cur_x:
                if extrapolation == "linear":
                    out_idx_vals[x_idx] = (cur_x-self.coordinates[0])/self._start_dx
                elif extrapolation == "nan":
                    out_idx_vals[x_idx] = np.nan
                elif extrapolation == "nearest":
                    out_idx_vals[x_idx] = 0
                else:
                    raise ValueError("extrapolation value not understood")
                continue
            if ub < cur_x:
                if extrapolation == "linear":
                    out_idx_vals[x_idx] = (len(self.coordinates) -1)+(cur_x-self.coordinates[-1])/self._end_dx
                elif extrapolation == "nearest":
                    out_idx_vals[x_idx] = len(self.bins)-2
                elif extrapolation == "nan":
                    out_idx_vals[x_idx] = np.nan
                else:
                    raise ValueError("extrapolation value not understood")
                continue
            lbi, ubi = 0, len(self.bins)-1
            while True:
                mididx = (lbi+ubi)//2
                midbound = self.bins[mididx]
                if midbound <= cur_x:
                    lbi = mididx
                else:
                    ubi = mididx
                if self.bins[lbi] <= cur_x <= self.bins[lbi+1]:
                    self._cached_prev_bin = self.bins[lbi], self.bins[lbi+1]
                    self._cached_prev_bin_idx = lbi
                    break
            alpha = (cur_x-self._cached_prev_bin[0])/(self._cached_prev_bin[1]-self._cached_prev_bin[0])
            out_idx_vals[x_idx] = lbi + alpha
        return out_idx_vals
    
    def get_bin_index(self, coords):
        """ find the bin index to which the coordinate belongs
        """
        out_c = np.around(self.coordinates_to_indicies(coords))
        return np.array(out_c, dtype=int)
    
    def interpolant_matrix(self, coords):
        raise NotImplementedError("I'm getting to it...")
        index_vals = self.coordinates_to_indicies(coords, extrapolation="nan")
        upper_index = np.ceil(index_vals)
        lower_index = np.floor(index_vals)
        alphas = index_vals - lower_index
        interp_vals =  self.flux[upper_index]*alphas
        interp_vals += self.flux[lower_index]*(1-alphas)
        var = self.get_var()
        sampled_var = var[upper_index]*alphas**2
        sampled_var += var[lower_index]*(1-alphas)**2
        return Spectrum(wvs, interp_vals, misc.var_2_inv_var(sampled_var))


class LinearCoordinatization(Coordinatization):
    
    def __init__(self, coordinates=None, min=None, max=None, npts=None, dx=None):
        """a class representing a linear mapping between a coordinate and
        the index number of an array.
        
        coordinates: ndarray
          an input coordinate array if specified none of min, max, npts, or dx
          may be specified. If coordinates is None, any three of min, max, npts
          and dx may be specified but not all four.
        min: float
          the minimum coordinate
        max: float
          the maximum coordinate
        npts: integer
          the number of points in the coordinatization
        dx: float
          the difference between consecutive positions in the array
        
        Note: if npts is the value left unspecified dx will adjusted 
        to allow for an integer npts with npts >= 2.
        
        """
        if not (coordinates is None):
            if not all([(val is None) for val in [min, max, npts, dx]]):
                raise ValueError("if coordinates are specified min, max, npts and dx may not be")
            self.npts = len(coordinates)
            self.min, self.max =sorted([coordinates[0], coordinates[-1]])
            self.dx = float(self.max-self.min)/(self.npts-1)
            if np.sign(coordinates[-1]-coordinates[0]) < 0:
                self.dx = -self.dx
        else:
            if max is None:
                if not all([not (val is None) for val in [min, npts, dx]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")
                self.min = min
                self.max = np.abs(dx)*(npts-1) + self.min
                self.npts = npts
                self.dx = dx
            elif min is None:
                if not all([not (val is None) for val in [max, npts, dx]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")
                self.max = max
                self.min = self.max - np.abs(dx)*(npts-1)
                self.npts = npts
                self.dx = dx
            elif npts is None:
                if not all([not (val is None) for val in [min, max, dx]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")
                self.min = min
                self.max = max
                self.npts = int(round(float(self.max-self.min)/dx)) + 1
                if self.npts <= 1:
                    self.npts = 2
                self.dx = (self.max-self.min)/(self.npts - 1)
            elif dx is None:
                if not all([not (val is None) for val in [min, max, npts]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")        
                self.min = min
                self.max = max
                self.npts = npts
                self.dx = float(self.max-self.min)/(self.npts -1)
                   
    
    def __len__(self):
        return self.npts
    
    @property
    def min(self):
        return self._min

    @min.setter 
    def min(self, value):
        self._min = value
    
    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self._max = value

    def get_index(self, coord):
        return (np.asarray(coord) - self.min)/self.dx
    
    def get_coord(self, index):
        return np.asarray(index)*self.dx + self.min

class TensoredCoordinatization(object):
    """A class for handling coordinatizations in multiple dimensions.
    The multidimensional coordinates are built up by tensoring together
    coordinates along each dimension. 
    """
    
    def __init__(self, bin_centers_list):
        """    
        inputs
            bins_list: a list of the bins along each dimension
        """
        self.coordinatizations = [as_coordinatization(bin_centers) for bin_centers in bin_centers_list]
        self.shape = tuple([len(b) for b in self.coordinatizations])
    
    def get_index(self, coord, clip=False, snap=False):
        coord = np.asarray(coord)
        in_shape = coord.shape
        coord = np.atleast_2d(coord)
        indexes_list = []
        for dim_idx in range(len(self.coordinatizations)):
            cur_coord = self.coordinatizations[dim_idx]
            index_vec = cur_coord.get_index(coord[:, dim_idx], clip=clip, snap=snap)
            indexes_list.append(index_vec.reshape((-1, 1)))
        return np.hstack(indexes_list).reshape(in_shape)
    
    def get_coord(self, index, clip=False, snap=False):
        index = np.asarray(index)
        in_shape = index.shape
        index = np.atleast_2d(index)
        coord_list = []
        for dim_idx in range(len(self.binnings)):
            cur_cdn = self.coordinatizations[dim_idx]
            coord_vec = cur_cdn.get_coord(index[:, dim_idx], clip=clip,snap=snap)
            coord_list.append(coord_vec.reshape((-1, 1)))
        return np.hstack(indexes_list).reshape(in_shape)
