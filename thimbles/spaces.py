import numpy as np
import copy
from collections import Iterable

class SubspaceError(Exception):
    pass

class Dimension(object):
    
    def __init__(self, name, shape, mask=None):
        self.name=name
        self.shape = shape
        if mask is None:
            mask = np.ones(shape, dtype=bool)
        self.mask=mask.reshape((-1,))
    
    @property
    def nfree(self):
        #TODO cache this value
        return int(np.sum(self.mask))

class Space(object):
    
    def __init__(self, dimensions=None):
        if dimensions is None:
            dimensions = []
        self.dimensions=dimensions
        self.refresh_name_index()
    
    def refresh_name_index(self):
        self.name_to_idx = {}
        for dim_i in range(len(self.dimensions)):
            dim = self.dimensions[dim_i]
            self.name_to_idx[dim.name]=dim_i  
    
    def subspace(self, sub_indexes):
        if not isinstance(sub_indexes, Space):
            if isinstance(sub_indexes, Dimension):
                sub_indexes = Space(dimensions=[sub_indexes])
            elif not isinstance(sub_indexes, Iterable):
                sub_indexes = [sub_indexes]
            sub_dims = []
            for dim in self.dimensions:
                dim_idx = self.name_to_idx.get(dim.name)
                if not dim_idx is None:
                    sub_dims.append(self.dimensions[dim_idx])
                else:
                    raise SubspaceError("dimension %s is not part of this space" % dim.name)
            sub_indexes = Space(sub_dims)
        out_dims = []
        for dim in sub_indexes:
            and_mask = dim.mask*self[dim].mask
            new_dim = Dimension(name=dim.name, shape=dim.shape, mask=and_mask)
            out_dims.append(new_dim)
        return Space(out_dims)
    
    def __getitem__(self, index):
        if isinstance(index, Dimension):
            didx = self.name_to_index[index.name]
            return self.dimensions[didx]
        if isinstance(index, basestring):
            didx = self.name_to_index[index]
            return self.dimensions[didx]
    
    def __iter__(self):
        for dim in self.dimensions:
            yield dim
    
    def __len__(self):
        return len(self.dimensions)
    
    def __add__(self, other):
        """add two parameter spaces together.
        """
        if not isinstance(other, Space):
            raise NotImplementedError("operation not implemented")
        assert set([d.name for d in self.dimensions]).intersection([d.name for d in other.dimensions]) == set() 
        joint_dims = copy(self.dimensions)
        joint_dims.extend(other.dimensions)
        return Space(joint_dims)

class Vector(object):
    
    def __init__(self, data, space=None,):
        self._data = {}
        for k, v in data:
            v = np.asarray(v)
            self._data[k] = v.reshape((-1,))
        if space is None:
            dims = []
            for k, v in self._data:
                dim = Dimension(k, v.shape)
                dims.append(dim)
            space = Space(dims)
        self.space = space
    
    def asarray(self):
        out = []
        for dim in self.space:
            if dim.nfree > 0:
                out.append(self._data[dim.name][dim.mask])
        return np.hstack(out)
    
    def __getitem__(self, index):
        subspace = self.space.subspace(index)
        out = []
        for dim in subspace:
            if dim.nfree > 0:
                out.append(self._data[dim.name][dim.mask])
        return np.hstack(out)
    
    def _set(self, subspace, value):
        value = np.asarray(value).reshape((-1,))
        lbi, ubi = 0, 0
        for dim in subspace:
            nfree = dim.nfree
            if nfree > 0:
                lbi = ubi
                ubi += nfree
                self._data[dim.name][dim.mask]=value[lbi:ubi]
    
    def __setitem__(self, index, value):
        subspace = self.space.subspace(index)
        self._set(subspace, value)
    
    def vector_set(self, vector):
        self._set(vector.space, vector.asarray())
    
    def subvector(self, subspace):
        subspace = self.space.subspace(subspace)
        vdat = {}
        for dim in subspace:
            dname = dim.name
            vdat[dname] = self._data[dim.name].copy()
        sv = Vector(vdat, subspace)
        return sv
    
    def copy(self):
        return copy.deepcopy(self)