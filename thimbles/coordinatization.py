import numpy as np
import scipy

import thimbles as tmb
from thimbles.thimblesdb import ThimblesTable, Base
from thimbles.modeling import Parameter, FloatParameter
from thimbles.sqlaimports import *
from functools import reduce

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

def as_coordinatization(coordinates, delta_max=1e-5, force_linear=False, force_log_linear=False):
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

class Coordinatization(tmb.modeling.Model):
    _id = Column(Integer, ForeignKey("Model._id"), primary_key=True)
    npts = Column(Integer)
    __mapper_args__ = {
        "polymorphic_identity":"Coordinatization",
    }
    
    def get_index(self, coord):
        raise NotImplementedError("abstract class")
    
    def get_coord(self, index):
        raise NotImplementedError("abstract class")
    
    #def __getitem__(self, index):
    #    if isinstance(index, slice):
    #        return self.coordinates[index]
    #    else:
    #        if index < 0:
    #            print("warning Coordinatization[-n] does not wrap around")
    #        return self.get_coord(index)
    
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
    _start_dx = None
    _end_dx = None
    __mapper_args__={
        "polymorphic_identity":"ArbitraryCoordinatization",
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
        self.output_p = Parameter()
        
        coordinates = np.asarray(coordinates)
        if check_ordered:
            gradient = scipy.gradient(coordinates)
            gsum = np.sum(gradient > 0)
            if not ((gsum == 0) or gsum == len(coordinates)):
                raise CoordinatizationError("coordinates not monotonic no allowable coordinatization")
        
        if as_edges:
            coordinates = edges_to_centers(coordinates)
        
        self.add_input("coordinates", Parameter(coordinates))
        self.npts = len(coordinates)
    
    __doc__ = __init__.__doc__
    
    @property
    def coordinates(self):
        return self.output_p.value
    
    @property
    def min(self):
        return self.coordinates[0]
    
    @property
    def max(self):
        return self.coordinates[-1]
    
    @property
    def start_dx(self):
        if self._start_dx is None:
            self._start_dx = self.coordinates[1]-self.coordinates[0]
        return self._start_dx
    
    @property
    def end_dx(self):
        if self._end_dx is None:
            self._end_dx = self.coordinates[-1]-self.coordinates[-2]
        return self._end_dx
    
    def __call__(self, vprep=None):
        vdict = self.get_vdict(vprep)
        return vdict[self.inputs["coordinates"][0]]
    
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
        #TODO:pipe this out to a jit compiled numba routine
        index = np.asarray(index)
        if snap:
            index = np.around(index)
        input_shape = index.shape
        index = np.atleast_1d(index)
        out_coordinates = np.zeros(index.shape, dtype=float)
        min_x = self.coordinates[0]
        max_idx = len(self)-1
        for i in range(len(index)):
            alpha = index[i]%1
            int_part = int(index[i])
            if index[i] < 0:
                out_coordinates[i] = min_x + self.start_dx*index[i]
            elif index[i] > len(self.coordinates)-2:
                out_coordinates[i] = self.coordinates[max_idx] + (index[i] - max_idx)*self.end_dx
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
                idx = (cur_x-min_x)/self.start_dx
            elif cur_x >= self.max:
                idx = (cur_x-max_x)/self.end_dx + max_idx
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
        "polymorphic_identity":"LinearCoordinatization"
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
        self.add_input("min", FloatParameter())
        self.add_input("max", FloatParameter())
        self.output_p = Parameter()
        
        if not (coordinates is None):
            if not all([(val is None) for val in [min, max, npts, dx]]):
                raise ValueError("if coordinates are specified min, max, npts and dx may not be")
            self._from_coord_vec(coordinates)
        else:
            if max is None:
                if not all([not (val is None) for val in [min, npts, dx]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")
                self.min = min
                self.max = np.abs(dx)*(npts-1) + min
                self.npts = npts
            elif min is None:
                if not all([not (val is None) for val in [max, npts, dx]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")
                self.max = max
                self.min = max - np.abs(dx)*(npts-1)
                self.npts = npts
            elif npts is None:
                if not all([not (val is None) for val in [min, max, dx]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")
                self.min = min
                self.max = max
                self.npts = int(round(float(max-min)/dx)) + 1
            elif dx is None:
                if not all([not (val is None) for val in [min, max, npts]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")        
                self.min = min
                self.max = max
                self.npts = npts
            else:
                raise Exception("Coordinatization initialized improprerly")
    
    @property
    def min(self):
        return self.inputs["min"][0].value
    
    @min.setter
    def min(self, value):
        self.inputs["min"][0].value = value
    
    @property
    def max(self):
        return self.inputs["max"][0].value
    
    @max.setter
    def max(self, value):
        self.inputs["max"][0].value = value
    
    @property
    def dx(self):
        return (self.max-self.min)/(self.npts-1.0)
    
    def _from_coord_vec(self, coordinates):
        self.npts = len(coordinates)
        self.inputs["min"][0].value = coordinates[0]
        self.inputs["max"][0].value = coordinates[-1]
    
    def __len__(self):
        return self.npts
    
    def __call__(self, vprep=None):
        vdict=self.get_vdict()
        min_val = vdict[self.inputs["min"][0]]
        max_val = vdict[self.inputs["max"][0]]
        return np.linspace(min_val, max_val, self.npts) 
    
    @property
    def coordinates(self):
        return self.output_p.value
    
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
    __mapper_args__={
        "polymorphic_identity":"loglinearcoordinatization"
    }
    
    def __init__(self, coordinates=None, min=None, max=None, npts=None, R=None):
        self.add_input("min", FloatParameter())
        self.add_input("max", FloatParameter())
        self.output_p = Parameter()
        
        min_p ,= self.inputs["min"]
        max_p ,= self.inputs["max"]
        
        if not (coordinates is None):
            if not all([(val is None) for val in [min, max, npts, R]]):
                raise ValueError("if coordinates are specified none of min, max, npts or R may be")
            self._from_coord_vec(coordinates)
        else:
            if max is None:
                if any([val is None for val in [min, npts, R]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")
                raise NotImplementedError("haven't gotten to it yet")
                min_p.value = min
                #np.log(max/min)*R = npts
                max_p.value = doot
                self.npts = npts
            elif min is None:
                raise NotImplementedError("haven't gotten to it yet")
                if any([val is None for val in [max, npts, R]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")
                max_p.value = max
                min_p.value = max - np.abs(dx)*(npts-1)
                self.npts = npts
            elif npts is None:
                if any([val is None for val in [min, max, R]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")
                min_p.value = min
                max_p.value = max
                self.npts = int(round(np.log(max/min)*R))
            elif R is None:
                if not all([not (val is None) for val in [min, max, npts]]):
                    raise ValueError("three of min, max, npts, and dx must be specified")        
                min_p.value = min
                max_p.value = max
                self.npts = npts
    
    def _from_coord_vec(self, coord_vec):
        self.min = coord_vec[0]
        self.max= coord_vec[-1]
        self.npts = len(coord_vec)
    
    @property
    def R(self):
        return (self.npts -1.0)/np.log(self.max/self.min)
    
    @property
    def min(self):
        return self.inputs["min"][0].value
    
    @min.setter
    def min(self, value):
        self.inputs["min"][0].value = value
    
    @property
    def max(self):
        return self.inputs["max"][0].value
    
    @max.setter
    def max(self, value):
        self.inputs["max"][0].value = value
    
    @property
    def coordinates(self):
        return self.output_p.value
    
    def __call__(self, vprep=None):
        print("coordinates regenerated")
        vdict = self.get_vdict(vprep)
        min_val = vdict[self.inputs["min"][0]]
        max_val = vdict[self.inputs["max"][0]]
        return np.exp(np.linspace(np.log(min_val), np.log(max_val), self.npts))
    
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



tensored_coord_assoc = sa.Table(
    "tensored_coord_assoc", 
    Base.metadata,
    Column("coordinatization_id", Integer, ForeignKey("Coordinatization._id")),
    Column("tensored_coordinatization_id", Integer, ForeignKey("TensoredCoordinatization._id")),
)

class TensoredCoordinatization(ThimblesTable, Base):
    """A class for handling coordinatizations in multiple dimensions.
    The multidimensional coordinates are built up by tensoring together
    independent coordinatizations for each dimension. 
    """
    coordinatizations = relationship("Coordinatization", secondary=tensored_coord_assoc)
    
    _shape = None
    
    def __init__(self, coordinates_list):
        """    
        inputs
            bins_list: a list of the bins along each dimension
        """
        self.coordinatizations = [as_coordinatization(bin_centers) for bin_centers in coordinates_list]
    
    @property
    def shape(self):
        if self._shape is None:
            self._shape = tuple([len(b) for b in self.coordinatizations])
        return self._shape
    
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
    
    @property
    def coordinates(self):
        coordinates = []
        ones_block = np.ones(self.shape)
        for coord_idx in range(len(self.coordinatizations)):
            coord_n = self.coordinatizations[coord_idx].coordinates
            reshape_tup = np.ones(len(self.shape), dtype=int)
            reshape_tup[coord_idx] = -1
            reshape_tup = tuple(reshape_tup)
            coord_n = coord_n.reshape(reshape_tup)
            cur_c = ones_block * coord_n
            coordinates.append(cur_c)
        coordinates = np.dstack(coordinates)
        return coordinates
    
    @property
    def indexes(self):
        indexes = []
        ones_block = np.ones(self.shape, dtype=int)
        for coord_idx in range(len(self.coordinatizations)):
            idx_n = np.arange(self.shape[coord_idx])
            reshape_tup = np.ones(len(self.shape), dtype=int)
            reshape_tup[coord_idx] = -1
            reshape_tup = tuple(reshape_tup)
            coord_n = idx_n.reshape(reshape_tup)
            cur_c = ones_block * coord_n
            indexes.append(cur_c)
            indexes = np.dstack(indexes)
        return indexes    
    
    def interpolant_sampling_matrix(self, coord_arr, extrapolate=False):
        coord_vec = np.atleast_2d(coord_arr)
        input_shape = coord_vec.shape
        if not input_shape[-1] == len(self.shape):
            raise ValueError("final dimension of input coordinates does not match number of specified coordinate dimensions")
        if len(input_shape) > 2:
            raise ValueError("can't make a sparse matrix if coord_arr.ndim > 2 reshape your array to be (npts x ndim) and then reshape back again after applying sampling matrix""")
        continuous_idxs = self.get_index(coord_vec)
        max_vec = np.asarray(self.shape) - 2
        nearest_idxs = np.around(np.clip(continuous_idxs, 1, max_vec)).astype(int)
        idx_deltas = continuous_idxs - nearest_idxs
        delta_int = np.where(idx_deltas > 0, 1, -1)
        neighbor_idxs = nearest_idxs + delta_int
        neighbor_weights = np.abs(idx_deltas)
        nearest_weight = 1.0-np.sum(neighbor_weights, axis=1)
        #import pdb;pdb.set_trace()
        if not extrapolate:
            close_enough = np.abs(nearest_weight) < 2.0
            neighbor_weights *= close_enough.reshape((-1, 1))
            nearest_weight *= close_enough
        
        nrows = len(coord_vec)
        ncols = reduce(lambda x, y: x*y , self.shape)
        n_entries = nrows*(len(self.shape)+1)
        row_idxs = np.zeros(n_entries)
        col_idxs = np.zeros(n_entries)
        mat_entries = np.zeros(n_entries)
        mat_shape = (nrows, ncols)
        
        clb = 0
        cub = clb + nrows
        n_idx_tup = [nearest_idxs[:, i] for i in range(len(self.shape))]
        col_idxs[clb:cub] = np.ravel_multi_index(n_idx_tup, self.shape)
        row_idxs[clb:cub] = np.arange(nrows)
        mat_entries[clb:cub] = nearest_weight
        
        #mat_dat = np.hstack([snap_alpha, neighbor_alpha])
        #row_idxs = np.hstack([np.arange(nrows), np.arange(nrows)])
        #col_idxs = np.hstack([clipped_snapped_input_cols, neighbor_idxs])
        
        for dim_idx in range(len(self.shape)):
            clb = cub
            cub = clb + nrows
            neighbor_idx_tup = copy(n_idx_tup)
            neighbor_idx_tup[dim_idx] = neighbor_idxs[:, dim_idx]
            col_idxs[clb:cub] = np.ravel_multi_index(neighbor_idx_tup, self.shape)
            row_idxs[clb:cub] = np.arange(nrows)
            mat_entries[clb:cub] = neighbor_weights[:, dim_idx]
            
            #interped_data += self.grid_data[neighbor_idx_tup]*neighbor_weights[:, dim_idx]
        interp_mat = scipy.sparse.coo_matrix((mat_entries, (row_idxs, col_idxs)), shape=mat_shape).tocsr()
        return interp_mat
    
    
    def unraveled_curvature_matrices(self):
        curvature_mats = []
        nrows = reduce(lambda x, y: x*y , self.shape)
        index_brick = self.indexes.reshape((-1, len(self.shape)))
        identity_idxs = [index_brick[:, i] for i in range(len(self.shape))]
        mat_data = np.repeat(-0.5, 3*nrows)
        mat_data[:nrows] = 1.0
        for dim_idx in range(len(self.shape)):
            to_ravel_plus = copy(identity_idxs)
            to_ravel_plus[dim_idx] = to_ravel_plus[dim_idx]+1
            plus_idxs = np.ravel_multi_index(to_ravel_plus, self.shape, mode="clip")
            to_ravel_minus = copy(identity_idxs)
            to_ravel_minus[dim_idx] = to_ravel_minus[dim_idx]-1
            minus_idxs = np.ravel_multi_index(to_ravel_minus, self.shape, mode="clip")       
            row_idxs = np.hstack([np.arange(nrows), plus_idxs, minus_idxs])
            col_idxs = np.hstack([np.arange(nrows), np.arange(nrows), np.arange(nrows)])
            cmat = scipy.sparse.coo_matrix((mat_data, (row_idxs, col_idxs)), shape=(nrows, nrows))
            curvature_mats.append(cmat)
        return curvature_mats

