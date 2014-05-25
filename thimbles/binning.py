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
    bins[0] = coord_centers[0] - (bins[1]-coord_centers[0])
    bins[-1] = coord_centers[-1] + 0.5*(coord_centers[-1] - coord_centers[-2])
    return bins

class CoordinateBinning1D (object):
    """a container that associates a set of coordinates to their indexes
    and vice versa. Also fascilitates fast interpolation
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
    
    def coordinates_to_indicies(self, input_coordinates):
        """assign continuous indexes to the input_coordinates 
        which place them on to the indexing of these coordinates.
        """
        #TODO:dump this to cython
        # get the upper and lower bounds for the coordinates
        lb,ub = self.bins[0],self.bins[-1]
        xv = np.asarray(input_coordinates)
        out_idx_vals = np.zeros(len(xv.flat), dtype = int)
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
                out_idx_vals[x_idx] = (cur_x-self.coordinates[0])/self._start_dx
                continue
            if ub < cur_x:
                out_idx_vals[x_idx] = (cur_x-self.coordinates[1])/self._end_dx
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
    
    def get_bin_index(self, xvec):
        """uses interval binary search to quickly find the bin belonging to the 
        input coordinates. If a coordinate outside of the bins is asked for a 
        linear extrapolation of the bin index is returned. (so be warned 
        indexes can be less than 0 and greater than len!)
        """
        # get the upper and lower bounds for the coordinates
        lb,ub = self.bins[0],self.bins[-1]
        xv = np.asarray(xvec)
        out_idxs = np.zeros(len(xv.flat), dtype = int)
        for x_idx in xrange(len(xv.flat)):
            cur_x = xvec[x_idx]
            #check if the last solution still works
            if self._cached_prev_bin[0] <= cur_x <= self._cached_prev_bin[1]:
                out_idxs[x_idx] = self._cached_prev_bin_idx
                continue
            #make sure that the x value is inside the bin range
            if lb > cur_x:
                out_idxs[x_idx] = int((cur_x-lb)/self._start_dx -1)
            if ub > cur_x:
                out_idxs[x_idx] = int((cur_x-ub)/self._end_dx)
            lbi, ubi = 0, self.n_bounds-1
            while True:
                mididx = (lbi+ubi)/2
                midbound = self.bins[mididx]
                if midbound <= cur_x:
                    lbi = mididx
                else:
                    ubi = mididx
                if self.bins[lbi] <= cur_x <= self.bins[lbi+1]:
                    self._cached_prev_bin = self.bins[lbi], self.bins[lbi+1]
                    self._cached_prev_bin_idx = lbi
                    break
            out_idxs[x_idx] = lbi
        return out_idxs
    
    def interpolate(self, coords, vals, variance=None, fill_val=0.0, fill_var=None):
        return None


class LinearBinning1d(CoordinateBinning1D):
    
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
