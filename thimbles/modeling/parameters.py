import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.orm.collections import collection
from functools import reduce

mult_func = lambda x, y: x*y
def flat_size(shape_tup):
    if shape_tup == tuple():
        return 1
    return reduce(mult_func, shape_tup)


class ParameterGroup(object):
    
    def parameter_index(self, parameter):
        return self.parameters.index(parameter)
    
    def get_pvec(self, pdict=None):
        """convert from a dictionary of parameter mapped values to
        a parameter vector in this embedding space.
        
        """
        parameters = self.parameters
        if pdict is None:
            pdict = {}
        pvals = []
        for p in parameters:
            pval = pdict.get(p)
            if pval is None:
                pval = p.get()
            flat_pval = np.asarray(pval).reshape((-1,))
            pvals.append(flat_pval)
        return np.hstack(pvals)
    
    def set_pvec(self, pvec, attr=None, as_delta=False):
        parameters = self.parameters
        pshapes = [p.shape for p in parameters]
        nvals = [flat_size(pshape) for pshape in pshapes]
        break_idxs = np.cumsum(nvals)[:-1]
        if as_delta:
            pvec = pvec + self.get_pvec()
        flat_params = np.split(pvec, break_idxs)
        if attr is None:
            for p_idx in range(len(parameters)):
                param = parameters[p_idx]
                pshape = pshapes[p_idx]
                flat_val = flat_params[p_idx]
                if pshape == tuple():
                    to_set = float(flat_val)
                else:
                    to_set = flat_val.reshape(pshape)
                param.set(to_set)
        elif isinstance(attr, str):
            for p_idx in range(len(parameters)):
                param = parameters[p_idx]
                pshape = pshapes[p_idx]
                flat_val = flat_params[p_idx]
                if pshape == tuple():
                    setattr(param, attr, float(flat_val))
                else:
                    setattr(param, attr, flat_val.reshape(pshape))
        else:
            raise ValueError("attr must be a string if set")
    
    def get_pdict(
            self, 
            value_replacements=None, 
            attr=None, 
            name_as_key=False
    ):
        
        if free_only:
            parameters = self.free_parameters
        else:
            parameters = self.parameters
        if name_as_key:
            keys = [p.name for p in parameters]
        else:
            keys = parameters
        if attr is None:
            values = [p.get() for p in parameters]
        else:
            values = [getattr(p, attr) for p in parameters]
        pdict = dict(list(zip(keys, values))) 
        return pdict
    
    def set_pdict(self, val_dict, attr=None):
        for p in val_dict:
            if attr is None:
                p.set(val_dict[p])
            else:
                setattr(p, attr, val_dict[p])

class InformedModels(object):
    
    def __init__(self):
        self.models = []
        self._aliases = []
    
    def append(self, param_alias):
        self.models.append(param_alias.model)
        self._aliases.append(param_alias)
    
    def remove(self, param_alias):
        self.models.remove(param_alias.model)
        self._aliases.remove(param_alias)
    
    def __len__(self):
        return len(self.models)
    
    def __getitem__(self, index):
        return self.models[index]
    
    def __iter__(self):
        for mod in self.models:
            yield mod
    
    @collection.iterator
    def _iter_aliases(self):
        for alias in self._aliases:
            yield alias

class Parameter(ThimblesTable, Base):
    parameter_class = Column(String)
    __mapper_args__={
        "polymorphic_identity":"parameter",
        "polymorphic_on": parameter_class
    }
    _value = None
    models = relationship(
        "InputAlias",
        collection_class = InformedModels
    )
    
    def __init__(self, value=None,):
        self._value = value
    
    def fire_upstream(self):
        """fire the model immediately upstream from this parameter.
        Often this will require also causing the models upstream from
        it to fire as well and so on causing a cascade.
        """
        m_models = self.mapped_models
        if len(m_models) >= 1:
            if len(m_models) > 1:
                print("warning parameter value regeneration is non-unique consider changing the model hierarchy to use cached parameter values.")
            mod = m_models[0]
            mod.fire() #should populate our self._value attribute
    
    def invalidate_downstream(self):
        """run through all parameters downstream of this one
        marking them as invalid, so that subsequent calls asking
        for their value will trigger an update cascade.
        """
        parameter_front = [mod.output_p for mod in self.models if (not mod.output_p is None) and (not mod.output_p.invalid())]
        while len(parameter_front) > 0:
            param = parameter_front.pop(0)
            param.invalidate()
            #find all the downstream parameters
            for mod in param.models:
                out_p = mod.output_p
                #check that the model actually maps to something.
                if not out_p is None:
                    #if parameter is not already invalid add it to the queue.
                    if not out_p.invalid():
                        parameter_front.append(out_p)
    
    @property
    def value(self):
        return self.get()
    
    @value.setter
    def value(self, value):
        self.set(value,)
    
    def invalid(self,):
        return self._value is None
    
    def invalidate(self,):
        self._value = None
    
    def __getitem__(self, index):
        return self.value[index]
    
    def __setitem__(self, index, value):
        self._value[index] = value
    
    def get(self,):
        if self.invalid():
            self.fire_upstream()
        return self._value
    
    def set(self, value):
        self.invalidate_downstream()
        self._value = value
    
    @property
    def shape(self):
        return np.asarray(self.get()).shape
    

#how to make a parameter subclass
#class classname(Parameter):
#    _id = Column(Integer, ForeignKey("Parameter._id"), primary_key=True)
#    _value = Column(Float)
#    __mapper_args__={
#        "polymorphic_identity": "classname"
#    }

