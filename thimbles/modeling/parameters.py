import time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from copy import copy

from thimbles.sqlaimports import *
from thimbles.thimblesdb import ThimblesTable, Base
from functools import reduce

mult_func = lambda x, y: x*y
def flat_size(shape_tup):
    if shape_tup == tuple():
        return 1
    return reduce(mult_func, shape_tup)

class ParameterGroup(object):
    
    def __init__(self, parameters=None):
        if parameters is None:
            parameters = []
        self.parameters = parameters
    
    @property
    def free_parameters(self):
        return [param for param in self.parameters if not param.fixed]
    
    def parameter_index(self, parameter):
        return self.parameters.index(parameter)
    
    def get_pvec(self, attr=None, free_only=True):
        if free_only:
            parameters = self.free_parameters
        else:
            parameters = self.parameters
        if attr is None:
            pvals = [np.asarray(p.get()).reshape((-1,)) for p in parameters]
        else:
            pshapes = [p.shape for p in parameters]
            pvals = [np.array(getattr(p, attr)) for p in parameters]
            out_vec = []
            for p_idx in range(len(parameters)):
                pshape = pshapes[p_idx]
                pval = pvals[p_idx]
                if pshape == tuple() or pval.shape != tuple():
                    out_vec.append(pval)
                else:
                    out_vec.append(np.repeat(pval, flat_size(pshape)))
            pvals = out_vec
        if len(pvals) == 0:
            return None
        return np.hstack(pvals)
    
    def set_pvec(self, pvec, attr=None, free_only=True, as_delta=False):
        if free_only:
            parameters = self.free_parameters
        else:
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
    
    def get_pdict(self, value_replacements=None, attr=None, free_only=True, name_as_key=False):
        
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

class FixedParameterException(Exception):
    pass

class Parameter(ThimblesTable, Base):
    parameter_class = Column(String)
    __mapper_args__={
        "polymorphic_identity":"parameter",
        "polymorphic_on": parameter_class
    }
    
    _value = None
    
    #class attributes
    name = "base"
    
    def __init__(self, value=None,):
        pass
    
    @property
    def models(self):
        models = []
        for mod_assoc in self._models:
            models.append(mod_assoc.model)
        return models
    
    @property
    def value(self):
        if self._value is None:
            m_models = self.mapped_models
            if len(m_models) >= 1:
                if len(m_models) > 1:
                    print("warning parameter value regeneration is non-unique consider changing the model hierarchy to use cached parameter values.")
                mod = m_models[0]
                mod.fire() #should populate our self._value attribute
        return self._value
    
    @value.setter
    def value(self, value):
        self.set(value,)
    
    def get(self):
        return self.value
    
    def set(self, value):
        #import pdb; pdb.set_trace()
        self._value = value
        mods = []
        mod_set = set()
        mods.extend(self.models)
        while len(mods) > 0:
            mod = mods.pop(0)
            mod_set.add(mod)
            #execute the model
            mod.fire()
            #mark the donwnstream models inputs as invalid
            downstream = mod.output_p.models
            for dsmod in downstream:
                dsmod.output_p._value = None 
            #TODO: implement incremental updates somehow
            mods.extend([nmod for nmod in mod.output_p.models if nmod not in mod_set])
    
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

