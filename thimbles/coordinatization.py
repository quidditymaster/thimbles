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
          if true the returned coordinates are snapped to match the nearest
          coordinate center exactly.
        """
        #TODO:dump this operation out to cython
        index = np.asarray(index)
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
        return out_coordinates
    
    def get_index(self, coord, clip=False, snap=False):
        """convert coordinates to array indexes.
        """
        
        """assign continuous indexes to the input_coordinates 
        which place them on to the indexing of these coordinates.
        
        coords: ndarray
          coordinates to convert to the index
        extralpolation: string
          'linear': linearly extrapolate the indexes to indexes < 0 and
             greater than len(self.coordinates)-1
          'nan': coordinates outside of the bin boundaries are set to np.nan
          'nearest': a value of 0 is placed for coordinates less than 
            the lower limit and a value of len(self.coordinates)-1 is placed 
            for coordinates greater than the upper limit.
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
    
    def __init__(self, coordinates):
        """approximates the input coordinates with a linear
        coordinatization which simply linearly interpolates
        between the first and last values.
        """
        self.x0 = coordinates[0]
        self.npts = len(coordinates)
        self.dx = (coordinates[-1]-coordinates[0])/(self.npts-1)
    
    def __len__(self):
        return self.npts
    
    def coordinates_to_indicies(self, input_coordinates):
        return (np.asarray(input_coordinates) - self.x0)/self.dx
    
    def indicies_to_coordinates(self, input_indicies):
        return np.asarray(input_indicies)*self.dx + self.x0
    
    #def map_indicies(self, input_coordinates):
    #    return (input_coordinates-self.x0)/self.dx
    
    def get_bin_index(self, input_coordinates):
        #TODO make this handle negatives properly
        return np.asarray(self.coordinates_to_indicies(input_coordinates), dtype = int)


class TensoredBinning(object):
    """A binning class for handling bins in multiple dimensions.
    The multidimensional bins are built up by tensoring together
    bins along each dimension. That is if we have the bins in x
    (0, 1), (1, 2) and in y the bins (0, 3), (3,5) then the tensored
    binning will be [[(0, 1), (0, 3)], [(0, 1), (3, 5)], [(1, 2), (0, 3)]
    [(1, 2), (3, 5)]]. That is we get one bin for each possible pairing
    of the input bins. 
    """
    
    def __init__(self, bin_centers_list):
        """    
        inputs
            bins_list: a list of the bins along each dimension
        """
        self.binnings = [CoordinateBinning(bin_centers) for bin_centers in bin_centers_list]
        self.shape = tuple([len(b) for b in self.binnings])
    
    def coordinates_to_indicies(self, xcoords, extrapolation="linear"):
        xcoords = np.asarray(xcoords)
        if len(xcoords.shape) != 2:
            if len(self.shape) == len(xcoords):
                xcoords = xcoords.reshape((-1, len(self.shape)))
        indexes_list = []
        for binning_idx in range(len(self.binnings)):
            xv = xcoords[:, binning_idx]
            index_vec = self.binnings[binning_idx].coordinates_to_indicies(xv, extrapolation=extrapolation)
            indexes_list.append(index_vec.reshape((-1, 1)))
        return np.hstack(indexes_list)
    
