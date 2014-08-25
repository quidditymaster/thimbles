import numpy as np

from . import verbosity

def bins_to_centers(bins):
    bins = np.asarray(bins)
    return 0.5*(bins[:-1]+bins[1:])

def centers_to_bins(coord_centers):
    coord_centers = np.asarray(coord_centers)
    if len(coord_centers) == 0:
        return np.zeros(0)
    bins = np.zeros(len(coord_centers) + 1, dtype = float)
    bins[1:-1] = 0.5*(coord_centers[1:] + coord_centers[:-1])
    bins[0] = coord_centers[0] - 0.5*(bins[1]-coord_centers[0])
    bins[-1] = coord_centers[-1] + 0.5*(coord_centers[-1] - coord_centers[-2])
    return bins

class CoordinateBinning (object):
    """a container that associates a set of coordinates to their indexes
    and vice versa.
    """
    
    def __init__(self, coordinates):
        """coordinates must be a monotonic function of index
        """
        self.coordinates = coordinates
        self.bins = centers_to_bins(coordinates)
        self._cached_prev_bin = (self.bins[0], self.bins[1])
        self._cached_prev_bin_idx = 0
        self._start_dx = self.coordinates[1] - self.coordinates[0]
        self._end_dx = self.coordinates[-1] - self.coordinates[-2]
    
    def __len__(self):
        return len(self.coordinates)
    
    def indicies_to_coordinates(self, input_indicies):
        #TODO:dump this operation out to cython
        input_indicies = np.asarray(input_indicies)
        out_coordinates = np.zeros(len(input_indicies), dtype=float)
        for i in range(len(input_indicies)):
            alpha = input_indicies[i]%1
            int_part = int(input_indicies[i])
            if input_indicies[i] < 0:
                out_coordinates[i] = self.coordinates[0] + alpha*self._start_dx
            elif input_indicies[i] > len(self.coordinates)-2:
                out_coordinates[i] = self.coordinates[-1] + alpha*self._end_dx
            else:
                out_coordinates[i] = self.coordinates[int_part]*(1.0-alpha)
                out_coordinates[i] += self.coordinates[int_part+1]*alpha
        return out_coordinates
    
    def coordinates_to_indicies(self, coords, extrapolation="linear"):
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
        #TODO:dump this to cython
        # get the upper and lower bounds for the coordinates
        lb,ub = self.bins[0],self.bins[-1]
        xv = np.asarray(coords)
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


class LinearBinning1d(CoordinateBinning):
    
    def __init__(self, x0, dx):
        self.x0 = x0
        self.dx = dx
    
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
    