import numpy as np
import scipy

from thimbles import logger
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.sqlaimports import *

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

def as_coordinatization(coordinates, delta_max=1e-10, force_linear=False, force_log_linear=False):
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
    if len(coordinates) == 2:
        #perfectly described by a linear coordinatization no need to check
        return LinearCoordinatization(coordinates)
    
    if delta_max <= 0:
        return ArbitraryCoordinatization(coordinates)
    
    for coord_class in [LinearCoordinatization,
                        LogLinearCoordinatization]:
        test_coord_instance = coord_class(coordinates)
        test_coords = test_coord_instance.get_coord(np.arange(len(coordinates)))
        if np.max(np.abs(test_coords - coordinates)) < delta_max:
            return test_coord_instance
    
    return ArbitraryCoordinatization(coordinates)

class Coordinatization(ThimblesTable, Base):
    min = Column(Float)
    max = Column(Float)
    npts = Column(Integer)
    _coordinatization_type = Column(String)
    __mapper_args__ = {
        "polymorphic_identity":"coordinatization",
        "polymorphic_on": _coordinatization_type
    }
    
    def get_index(self, coord):
        raise NotImplementedError("abstract class")
    
    def get_coord(self, index):
        raise NotImplementedError("abstract class")
    
    def __len__(self):
        return self.npts
    
    def interpolant_sampling_matrix(self, sample_coords, extrapolate=False):
        """generates a  matrix which carries a vector sampled at coordinates
        corresponding to this coordinatization to a linear interpolation 
        of those values sampled at coordinates coresponding to sample_coords.
        
        parameters:
        
        sample_coords: ndarray
          the coordinates for which the linear interpolation is desired.
        """
        input_coordinatization = self    
        sample_coordinatization = as_coordinatization(sample_coords)
        #the input coordinates most closely lie at these coordinates in the sampling coordinatization
        float_input_cols = input_coordinatization.get_index(sample_coordinatization.coordinates)
        clipped_snapped_input_cols = np.clip(np.around(float_input_cols).astype(int), 1, len(input_coordinatization)-2)
        #col_idx_val = input_coord.get_index(self.coordinates)
        #snap_idx = np.clip(np.around(col_idx_val).astype(int), 1, len(input_coord)-2)
        snap_delta = float_input_cols - clipped_snapped_input_cols
        snap_direction = np.where(snap_delta > 0, 1, -1)
        neighbor_idxs = clipped_snapped_input_cols + snap_direction
        neighbor_alpha = np.abs(snap_delta)
        snap_alpha = 1.0-neighbor_alpha
        if not extrapolate:
            close_enough = neighbor_alpha < 2.0
            neighbor_alpha = np.where(close_enough, neighbor_alpha, 0.0)
            snap_alpha = np.where(close_enough, snap_alpha, 0.0)
        #import pdb; pdb.set_trace()
        nrows = len(sample_coordinatization)
        ncols = len(input_coordinatization)
        mat_shape = (nrows, ncols)
        mat_dat = np.hstack([snap_alpha, neighbor_alpha])
        row_idxs = np.hstack([np.arange(nrows), np.arange(nrows)])
        col_idxs = np.hstack([clipped_snapped_input_cols, neighbor_idxs])
        interp_mat = scipy.sparse.coo_matrix((mat_dat, (row_idxs, col_idxs)), shape=mat_shape).tocsr()
        return interp_mat

class ArbitraryCoordinatization(Coordinatization):
    _id = Column(Integer, ForeignKey("Coordinatization._id"), primary_key=True)
    coordinates = Column(PickleType)
    _start_dx = Column(Float)
    _end_dx = Column(Float)
    __mapper_args__={
        "polymorphic_identity":"arbitrarycoordinatization",
    }
    
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
        min, max = sorted([self.coordinates[0], self.coordinates[-1]])
        self.min = min
        self.max = max
        self.npts = len(self.coordinates)
        self._cached_prev_bin = (self.bins[0], self.bins[1])
        self._cached_prev_bin_idx = 0
        self._start_dx = self.bins[1] - self.bins[0]
        self._end_dx = self.bins[-1] - self.bins[-2]
    
    __doc__ = __init__.__doc__
    
    def __len__(self):
        return len(self.coordinates)
        
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
        out_index = np.zeros(coord.shape, dtype=float)
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
            out_index = np.around(out_index).astype(int)
        if clip:
            out_index = np.clip(out_index, 0, len(self)-1)
        return out_index.reshape(in_shape)

class LinearCoordinatization(Coordinatization):
    _id = Column(Integer, ForeignKey("Coordinatization._id"), primary_key=True)
    dx = Column(Float)
    __mapper_args__={
        "polymorphic_identity":"linearcoordinatization"
    }
    
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
        self._recalc_coords = True
        if not (coordinates is None):
            if not all([(val is None) for val in [min, max, npts, dx]]):
                raise ValueError("if coordinates are specified min, max, npts and dx may not be")
            self._from_coord_vec(coordinates)
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
    
    def _from_coord_vec(self, coordinates):
        self.npts = len(coordinates)
        self.min, self.max =sorted([coordinates[0], coordinates[-1]])
        self.dx = float(self.max-self.min)/(self.npts-1)
        if np.sign(coordinates[-1]-coordinates[0]) < 0:
            self.dx = -self.dx
    
    def __len__(self):
        return self.npts
    
    @property
    def coordinates(self):
        if self._recalc_coords:
            self._coords = np.linspace(self.min, self.max, self.npts) 
        return self._coords
    
    @coordinates.setter
    def coordinates(self, value):
        self._from_coord_vec(value)
        self._recalc_coords = True
    
    def get_index(self, coord, clip=False, snap=False):
        indexes = (np.asarray(coord) - self.min)/self.dx
        if clip:
            np.clip(indexes, 0, len(self)-1, out=indexes)
        if snap:
            indexes = np.around(indexes).astype(int)
        return indexes
    
    def get_coord(self, index, clip=False, snap=False):
        if snap:
            index = np.around(index).astype(int)
        coords = np.asarray(index)*self.dx + self.min
        if clip:
            coords = np.clip(coords, self.min, self.max)
        return coords

class LogLinearCoordinatization(Coordinatization):
    _id = Column(Integer, ForeignKey("Coordinatization._id"), primary_key=True)
    R = Column(Float)
    __mapper_args__={
        "polymorphic_identity":"loglinearcoordinatization"
    }
    
    def __init__(self, coordinates=None, min=None, max=None, npts=None, R=None):
        self._recalc_coords = True
        if not (coordinates is None):
            if not all([(val is None) for val in [min, max, npts, R]]):
                raise ValueError("if coordinates are specified none of min, max, npts or R may be")
            self._from_coord_vec(coordinates)
        else:
            if max is None:
                if any([val is None for val in [min, npts, R]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")
                raise NotImplementedError("haven't gotten to it yet")
                self.min = min
                #np.log(max/min)*R = npts
                self.max = doot#
                self.npts = npts
                self.R = R
            elif min is None:
                raise NotImplementedError("haven't gotten to it yet")
                if any([val is None for val in [max, npts, R]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")
                self.max = max
                self.min = self.max - np.abs(dx)*(npts-1)
                self.npts = npts
                self.R = R
            elif npts is None:
                if any([val is None for val in [min, max, R]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")
                self.min = min
                self.max = max
                self.npts = int(round(np.log(self.max/self.min)*R))
                if self.npts <= 1:
                    self.npts = 2
                self.R = (self.npts -1)/np.log(self.max/self.min)
            elif R is None:
                if not all([not (val is None) for val in [min, max, npts]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")        
                self.min = min
                self.max = max
                self.npts = npts
                self.R = (self.npts -1)/np.log(self.max/self.min)
    
    def _from_coord_vec(self, coord_vec):
        min, max = sorted([coord_vec[0], coord_vec[-1]])
        npts = len(coord_vec)
        if coord_vec[-1] > coord_vec[0]:
            R = (npts-1)/np.log(max/min)
        else:
            R = (npts-1)/np.log(max/min)
        self.min = min
        self.max = max
        self.npts = npts
        self.R = R
    
    @property
    def coordinates(self):
        if self._recalc_coords:
            self._coords = np.exp(np.linspace(np.log(self.min), np.log(self.max), self.npts))
        return self._coords
    
    @coordinates.setter
    def coordinates(self, value):
        self._from_coord_vec(value)
        self._recalc_coords = True
    
    def get_index(self, coord, clip=False, snap=False):
        indexes = np.log(np.asarray(coord)/self.min)*self.R
        if clip:
            np.clip(indexes, 0, len(self)-1, out=indexes)
        if snap:
            indexes = np.around(indexes).astype(int)
        return indexes
    
    def get_coord(self, index, clip=False, snap=False):
        if snap:
            index = np.around(index).astype(int)
        coords = np.exp(np.asarray(index)/self.R)*self.min
        if clip:
            coords = np.clip(coords, self.min, self.max)
        return coords

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
